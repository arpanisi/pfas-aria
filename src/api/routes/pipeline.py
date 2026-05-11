"""
Pipeline Routes.
Upload dataset, configure run, trigger pipeline, stream status.
Replaces the old pipeline.py — now uses the full ETL layer.
"""

from __future__ import annotations

import asyncio
import io
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import CurrentUser, DBSession
from src.db.orm import (
    Citation,
    ExperimentSegmentationBatch,
    Hypothesis,
    ModelResult,
    Paper,
    Regime,
    Run,
    ValidationResult,
)
from src.db.redis_client import delete_run_status, get_run_status, set_run_status
from src.ingestion.unified_experimental_sheet import (
    UnifiedSheetMeta,
    load_excel_bytes_with_layout,
)
from src.utils.logging import get_logger
from src.utils.paths import RAW_DIR

logger = get_logger(__name__)
router = APIRouter(prefix="/pipeline", tags=["Pipeline"])


# ── Schemas ───────────────────────────────────────────────────────────────────


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    n_unique: int
    missing_pct: float
    is_numeric: bool
    sample_values: list


class DatasetPreview(BaseModel):
    filename: str
    n_rows: int
    n_cols: int
    columns: list[ColumnInfo]
    # 0-based Excel row used as column headers (0 = first row). Omitted for CSV/TSV.
    excel_header_row: int | None = None


class LegacyRegimeSummaryOut(BaseModel):
    regime_id: int
    n_rows: int
    row_indices_sample: list[str]
    condition_values_sample: list[str]
    non_constant_input_cols: list[str]
    non_constant_output_cols: list[str]


class RegimeRowCountOut(BaseModel):
    """Row count for one regime id (1–5), including zero when no rows match."""

    regime_id: int
    n_rows: int


class LegacySegmentationPreviewOut(BaseModel):
    filename: str
    n_rows: int
    n_cols: int
    n_regimes: int
    pfoa_col: str
    pfba_col: str
    pfbs_col: str
    n_pfas_input_columns: int
    input_cols: list[str]
    output_cols: list[str]
    regimes: list[LegacyRegimeSummaryOut]
    regime_row_counts: list[RegimeRowCountOut]
    warnings: list[str]


def _normalize_dataframe_columns(df: pd.DataFrame) -> None:
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]


def _safe_raw_file_path(filename: str) -> Path:
    if not filename or ".." in filename or filename.startswith(("/", "\\")):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid filename")
    name = Path(filename).name
    if not name:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid filename")
    p = (RAW_DIR / name).resolve()
    try:
        p.relative_to(RAW_DIR.resolve())
    except ValueError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid filename") from None
    if not p.is_file():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Dataset file not found in raw store: {name}. Upload the file first.",
        )
    return p


def _read_normalized_dataframe(
    path: Path,
) -> tuple[pd.DataFrame, int | None, UnifiedSheetMeta | None]:
    content = path.read_bytes()
    suffix = path.suffix.lower().lstrip(".")
    excel_header_row: int | None = None
    unified_meta: UnifiedSheetMeta | None = None
    if suffix == "csv":
        df = pd.read_csv(io.BytesIO(content), low_memory=False)
    elif suffix == "tsv":
        df = pd.read_csv(io.BytesIO(content), sep="\t", low_memory=False)
    elif suffix in {"xlsx", "xls"}:
        df, excel_header_row, unified_meta = load_excel_bytes_with_layout(content)
    else:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Unsupported file type for preview: {suffix}",
        )
    _normalize_dataframe_columns(df)
    return df, excel_header_row, unified_meta


class RunConfig(BaseModel):
    run_name: str
    filename: str
    outcome_variable: str
    feature_columns: list[str]
    exclude_columns: list[str] = []
    max_rounds: int = 10
    convergence_threshold: float = 0.75
    hypotheses_per_round: int = 6
    strict_validation: bool = False
    regime_id: int | None = None


class RunStatus(BaseModel):
    run_id: str
    run_name: str
    status: str
    current_round: int
    final_match_score: float
    converged: bool
    n_rounds_completed: int
    run_kind: str = "pipeline"
    regime_id: int | None = None
    hypotheses_tested: int | None = None
    dataset_filename: str | None = None
    created_at: datetime | None = None


def _snapshot_dict(run: Run) -> dict:
    s = run.config_snapshot
    return s if isinstance(s, dict) else {}


def run_status_from_orm(
    run: Run,
    *,
    hypothesis_count: int = 0,
    status_override: str | None = None,
    current_round_override: int | None = None,
    match_score_override: float | None = None,
    converged_override: bool | None = None,
) -> RunStatus:
    snap = _snapshot_dict(run)
    run_kind = str(snap.get("run_kind") or "pipeline")
    regime_raw = snap.get("regime_id")
    regime_id = int(regime_raw) if regime_raw is not None else None
    fname = snap.get("dataset_filename")
    dataset_filename = str(fname) if fname else None

    if run_kind == "screening":
        ht_raw = snap.get("hypotheses_tested")
        hypotheses_tested = int(ht_raw) if ht_raw is not None else None
    else:
        hypotheses_tested = int(hypothesis_count)

    status = status_override if status_override is not None else run.status
    cur_round = (
        current_round_override
        if current_round_override is not None
        else run.n_rounds_completed
    )
    match = (
        match_score_override
        if match_score_override is not None
        else run.final_match_score
    )
    conv = converged_override if converged_override is not None else run.converged

    return RunStatus(
        run_id=run.id,
        run_name=run.run_name,
        status=status,
        current_round=cur_round,
        final_match_score=match,
        converged=conv,
        n_rounds_completed=cur_round,
        run_kind=run_kind,
        regime_id=regime_id,
        hypotheses_tested=hypotheses_tested,
        dataset_filename=dataset_filename,
        created_at=run.created_at,
    )


def _screening_run_matches_grounding_payload(
    run: Run, *, filename: str, regime_id: int
) -> bool:
    snap = _snapshot_dict(run)
    if snap.get("run_kind") != "screening":
        return False
    if int(snap.get("regime_id", -1)) != int(regime_id):
        return False
    return str(snap.get("dataset_filename", "")) == Path(filename).name


async def _delete_hypothesis_subtree(db: AsyncSession, run_id: str) -> None:
    hyp_ids = (
        (await db.execute(select(Hypothesis.id).where(Hypothesis.run_id == run_id)))
        .scalars()
        .all()
    )
    if not hyp_ids:
        return
    mr_ids = (
        (
            await db.execute(
                select(ModelResult.id).where(ModelResult.hypothesis_id.in_(hyp_ids))
            )
        )
        .scalars()
        .all()
    )
    if mr_ids:
        await db.execute(delete(Citation).where(Citation.model_result_id.in_(mr_ids)))
        await db.execute(
            delete(ValidationResult).where(ValidationResult.model_result_id.in_(mr_ids))
        )
        await db.execute(delete(ModelResult).where(ModelResult.id.in_(mr_ids)))
    await db.execute(delete(Hypothesis).where(Hypothesis.run_id == run_id))


async def _persist_screening_grounded_bundles(
    db: AsyncSession,
    run: Run,
    bundles: list[ScreeningBundleOut],
    *,
    regime_id: int,
) -> None:
    """Replace prior rows and insert hypotheses, models, validation, citations."""
    await _delete_hypothesis_subtree(db, run.id)
    regime_label = f"regime_{regime_id}"
    for bundle in bundles:
        h = bundle.hypothesis
        m = bundle.model_result
        hid = str(uuid.uuid4())
        hyp_row = Hypothesis(
            id=hid,
            run_id=run.id,
            hypothesis_id=h.hypothesis_id,
            round=h.round,
            description=h.description,
            rationale=h.rationale,
            primary_variables=list(h.primary_variables),
            interaction_terms=[],
            control_variables=[],
            model_family=h.model_family,
            priority_score=h.priority_score,
            is_refinement=h.is_refinement,
            rag_support=[],
        )
        db.add(hyp_row)
        await db.flush()

        mid = str(uuid.uuid4())
        mr_row = ModelResult(
            id=mid,
            hypothesis_id=hid,
            model_type=m.model_type,
            regime_label=regime_label,
            r_squared=float(m.r_squared),
            adj_r_squared=float(m.adj_r_squared),
            n_observations=int(m.n_observations),
            coefficients=dict(m.coefficients),
            p_values=dict(m.p_values),
            significant_variables=list(m.significant_variables),
            match_score=float(m.match_score),
            success=True,
            error_message=None,
        )
        db.add(mr_row)
        await db.flush()

        vr_row = ValidationResult(
            id=str(uuid.uuid4()),
            model_result_id=mid,
            overall_passed=bool(m.validation_passed),
            pass_rate=1.0 if m.validation_passed else 0.35,
        )
        db.add(vr_row)

        for c in bundle.citations:
            db.add(
                Citation(
                    id=str(uuid.uuid4()),
                    model_result_id=mid,
                    source=c.source,
                    title=c.title,
                    authors=[],
                    url=c.url,
                    year=c.year,
                    similarity_score=float(c.similarity_score),
                    abstract_snippet=None,
                    variable=c.variable,
                )
            )

    if bundles:
        best_match = max(
            (float(b.model_result.match_score) for b in bundles), default=0.0
        )
        run.final_match_score = best_match
    snap = dict(_snapshot_dict(run))
    snap["grounding_saved_at"] = datetime.utcnow().isoformat() + "Z"
    snap["grounded_hypothesis_count"] = len(bundles)
    run.config_snapshot = snap


async def _delete_run_record(db: AsyncSession, run_id: str) -> bool:
    run = await db.get(Run, run_id)
    if not run:
        return False
    await _delete_hypothesis_subtree(db, run_id)
    await db.execute(delete(Regime).where(Regime.run_id == run_id))
    await db.execute(
        update(ExperimentSegmentationBatch)
        .where(ExperimentSegmentationBatch.run_id == run_id)
        .values(run_id=None)
    )
    await db.delete(run)
    try:
        await delete_run_status(run_id)
    except Exception as e:  # noqa: BLE001
        logger.warning("Redis delete_run_status failed for %s: %s", run_id, e)
    return True


class AutomatedScreeningIterationIn(BaseModel):
    filename: str
    run_name: str = ""
    regime_id: int | None = None
    # Reserved for future screening / ranking thresholds; logged for traceability.
    convergence_threshold: float | None = None


class AutomatedScreeningIterationOut(BaseModel):
    hypotheses_tested: int
    run_id: str | None = None


MIN_CORPUS_PAPERS_FOR_SCREENING_GROUNDING = 3


class ScreeningGroundedIn(BaseModel):
    filename: str
    regime_id: int
    run_name: str = ""
    run_id: str | None = None


class ScreeningHypothesisOut(BaseModel):
    id: str
    hypothesis_id: str
    round: int
    description: str
    rationale: str | None
    primary_variables: list[str]
    model_family: str
    priority_score: float
    is_refinement: bool


class ScreeningModelOut(BaseModel):
    id: str
    hypothesis_id: str
    model_type: str
    r_squared: float
    adj_r_squared: float
    n_observations: int
    coefficients: dict
    p_values: dict
    significant_variables: list[str]
    match_score: float
    validation_passed: bool


class ScreeningCitationOut(BaseModel):
    id: str
    source: str
    title: str
    url: str | None = None
    year: str | None = None
    similarity_score: float
    variable: str | None = None
    hypothesis_id: str | None = None


class ScreeningBundleOut(BaseModel):
    hypothesis: ScreeningHypothesisOut
    model_result: ScreeningModelOut
    citations: list[ScreeningCitationOut]


class ScreeningGroundedOut(BaseModel):
    run_name: str
    filename: str
    display_title: str
    dataset_n_rows: int
    dataset_n_cols: int
    n_corpus_papers: int
    regime_id: int
    regime_n_rows: int
    bundles: list[ScreeningBundleOut]
    warnings: list[str]
    persisted_to_run_id: str | None = None


# ── Upload ────────────────────────────────────────────────────────────────────


@router.post("/upload", response_model=DatasetPreview)
async def upload_dataset(
    user: CurrentUser,
    file: UploadFile = File(...),
) -> DatasetPreview:
    """
    Upload a CSV or Excel dataset.
    Returns column metadata so the frontend can render the column selector.
    """
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No filename provided")

    suffix = file.filename.rsplit(".", 1)[-1].lower()
    if suffix not in {"csv", "xlsx", "xls", "tsv"}:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Unsupported file type: {suffix}. Use CSV or Excel.",
        )

    content = await file.read()

    excel_header_row: int | None = None
    try:
        if suffix == "csv":
            df = pd.read_csv(io.BytesIO(content), low_memory=False)
        elif suffix == "tsv":
            df = pd.read_csv(io.BytesIO(content), sep="\t", low_memory=False)
        else:
            df, excel_header_row, unified_meta = load_excel_bytes_with_layout(content)
            if unified_meta is not None:
                logger.info(
                    "Excel %s: unified layout (independent/dependent row %s, header row %s)",
                    file.filename,
                    unified_meta.role_row_index,
                    unified_meta.name_row_index,
                )
            elif excel_header_row:
                logger.info(
                    "Excel %s: inferred column header row index %s "
                    "(skipped %s preamble row(s))",
                    file.filename,
                    excel_header_row,
                    excel_header_row,
                )
    except Exception as e:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, f"Cannot parse file: {e}"
        ) from e

    # Standardize column names (idempotent if already normalized by Excel loader)
    _normalize_dataframe_columns(df)

    # Save raw file
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / file.filename).write_bytes(content)

    columns = [
        ColumnInfo(
            name=str(col),
            dtype=str(df[col].dtype),
            n_unique=int(df[col].nunique()),
            missing_pct=round(float(df[col].isnull().mean() * 100), 2),
            is_numeric=bool(pd.api.types.is_numeric_dtype(df[col])),
            sample_values=df[col].dropna().unique()[:5].tolist(),
        )
        for col in df.columns
    ]

    logger.info(f"Uploaded: {file.filename} ({len(df)} rows × {len(df.columns)} cols)")

    return DatasetPreview(
        filename=file.filename,
        n_rows=len(df),
        n_cols=len(df.columns),
        columns=columns,
        excel_header_row=excel_header_row if suffix in {"xlsx", "xls"} else None,
    )


@router.get(
    "/legacy-segmentation-preview",
    response_model=LegacySegmentationPreviewOut,
)
async def legacy_segmentation_preview(
    user: CurrentUser,
    filename: str,
    input_start: int = 2,
    input_end: int = 37,
    output_start: int = 37,
) -> LegacySegmentationPreviewOut:
    """
    Run legacy PFAS mask regime assignment on an uploaded file in ``data/raw``.

    Uses the same column normalization as ``/pipeline/upload``. Intended for
    the New Run UI to visualize regime row counts and a sample of conditions.
    """
    from src.ingestion.legacy_pfca_regime_masks import (
        assign_legacy_pfca_regime_masks,
        assign_pfca_regime_masks_from_column_lists,
    )

    path = _safe_raw_file_path(filename)
    df, _hdr, unified_meta = _read_normalized_dataframe(path)

    try:
        if unified_meta is not None:
            res = assign_pfca_regime_masks_from_column_lists(
                df,
                unified_meta.input_cols,
                unified_meta.output_cols,
                replace_no_in_inputs=True,
            )
        else:
            res = assign_legacy_pfca_regime_masks(
                df,
                input_start=input_start,
                input_end=input_end,
                output_start=output_start,
                replace_no_in_inputs=True,
            )
    except ValueError as e:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            str(e),
        ) from e

    cond_col = "condition" if "condition" in df.columns else None
    regime_out: list[LegacyRegimeSummaryOut] = []
    input_list = list(res.input_cols)
    output_list = list(res.output_cols)
    for rid, sub in sorted(res.regimes.items()):
        idx_sample = [str(i) for i in sub.index[:40].tolist()]
        cond_sample: list[str] = []
        if cond_col and cond_col in sub.columns:
            cond_sample = sorted(sub[cond_col].dropna().astype(str).unique().tolist())[
                :25
            ]
        non_constant_input_cols = [
            col
            for col in input_list
            if col in sub.columns and int(sub[col].nunique()) > 1
        ]
        non_constant_output_cols = [
            col
            for col in output_list
            if col in sub.columns and int(sub[col].nunique()) > 1
        ]
        regime_out.append(
            LegacyRegimeSummaryOut(
                regime_id=int(rid),
                n_rows=len(sub),
                row_indices_sample=idx_sample,
                condition_values_sample=cond_sample,
                non_constant_input_cols=non_constant_input_cols,
                non_constant_output_cols=non_constant_output_cols,
            )
        )

    regime_row_counts_out = [
        RegimeRowCountOut(regime_id=rid, n_rows=res.regime_row_counts.get(rid, 0))
        for rid in range(1, 6)
    ]

    return LegacySegmentationPreviewOut(
        filename=Path(filename).name,
        n_rows=len(df),
        n_cols=len(df.columns),
        n_regimes=len(regime_out),
        pfoa_col=res.pfoa_col,
        pfba_col=res.pfba_col,
        pfbs_col=res.pfbs_col,
        n_pfas_input_columns=len(res.pfas_cols),
        input_cols=list(res.input_cols),
        output_cols=list(res.output_cols),
        regimes=regime_out,
        regime_row_counts=regime_row_counts_out,
        warnings=list(res.warnings),
    )


@router.post(
    "/automated-screening-iteration",
    response_model=AutomatedScreeningIterationOut,
)
async def automated_screening_iteration(
    body: AutomatedScreeningIterationIn,
    user: CurrentUser,
    db: DBSession,
) -> AutomatedScreeningIterationOut:
    """
    One synchronous screening pass: input subsets × numeric outputs × model families.
    When ``regime_id`` is set, only that regime is screened; otherwise all nonempty regimes.
    """
    from src.pipeline.automated_screening import run_automated_screening_iteration

    path = _safe_raw_file_path(body.filename)
    logger.info(
        "Automated screening iteration file=%s run_name=%s regime_id=%s conv=%s user=%s",
        body.filename,
        body.run_name,
        body.regime_id,
        body.convergence_threshold,
        user.get("sub", "?"),
    )

    def _sync() -> int:
        df, _hdr, unified_meta = _read_normalized_dataframe(path)
        return run_automated_screening_iteration(
            df,
            unified_meta=unified_meta,
            regime_id=body.regime_id,
        )

    loop = asyncio.get_event_loop()
    n = int(await loop.run_in_executor(None, _sync))

    run_id: str | None = None
    if body.regime_id is not None:
        run_id = str(uuid.uuid4())[:8]
        snap: dict = {
            "run_kind": "screening",
            "regime_id": int(body.regime_id),
            "hypotheses_tested": n,
            "dataset_filename": Path(body.filename).name,
        }
        db_run = Run(
            id=run_id,
            run_name=(body.run_name or "").strip() or "Screening",
            status="screening_complete",
            outcome_variable=None,
            selected_features=[],
            excluded_features=[],
            n_rounds_completed=0,
            final_match_score=0.0,
            converged=False,
            config_snapshot=snap,
        )
        db.add(db_run)
        await db.flush()

    return AutomatedScreeningIterationOut(hypotheses_tested=n, run_id=run_id)


@router.post("/screening-grounded", response_model=ScreeningGroundedOut)
async def screening_grounded(
    body: ScreeningGroundedIn,
    user: CurrentUser,
    db: DBSession,
) -> ScreeningGroundedOut:
    """
    After automated screening, rank a single regime's best linear fits by
    in-sample fit and similarity to the uploaded corpus (minimum 3 papers).
    """
    n_papers = int(await db.scalar(select(func.count(Paper.id))) or 0)
    if n_papers < MIN_CORPUS_PAPERS_FOR_SCREENING_GROUNDING:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "CORPUS_TOO_SMALL",
                "n_papers": n_papers,
                "min_required": MIN_CORPUS_PAPERS_FOR_SCREENING_GROUNDING,
                "message": (
                    f"Upload at least {MIN_CORPUS_PAPERS_FOR_SCREENING_GROUNDING} PDF papers "
                    "to the corpus before viewing literature-grounded screening results."
                ),
            },
        )

    path = _safe_raw_file_path(body.filename)

    def _sync() -> dict:
        from src.pipeline.screening_grounded import run_screening_grounded_for_regime
        from src.rag.pipeline import RAGPipeline

        df, _hdr, unified_meta = _read_normalized_dataframe(path)
        retriever = RAGPipeline().build(force_rebuild=False)
        payload = run_screening_grounded_for_regime(
            df,
            unified_meta=unified_meta,
            regime_id=body.regime_id,
            retriever=retriever,
        )
        payload["dataset_n_rows"] = len(df)
        payload["dataset_n_cols"] = len(df.columns)
        payload["n_corpus_papers"] = n_papers
        payload["filename"] = Path(body.filename).name
        payload["run_name"] = (body.run_name or "").strip() or "Screening run"
        payload["display_title"] = "Literature Grounding of Discovered Signals"
        return payload

    loop = asyncio.get_event_loop()
    raw = await loop.run_in_executor(None, _sync)

    bundles_out: list[ScreeningBundleOut] = []
    for b in raw.get("bundles", []):
        bundles_out.append(
            ScreeningBundleOut(
                hypothesis=ScreeningHypothesisOut(**b["hypothesis"]),
                model_result=ScreeningModelOut(**b["model_result"]),
                citations=[ScreeningCitationOut(**c) for c in b.get("citations", [])],
            )
        )

    logger.info(
        "Screening grounded file=%s regime=%s bundles=%s user=%s",
        body.filename,
        body.regime_id,
        len(bundles_out),
        user.get("sub", "?"),
    )

    persisted_id: str | None = None
    rid_in = (body.run_id or "").strip()
    if rid_in:
        run_row = await db.get(Run, rid_in)
        if not run_row:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"Run {rid_in} not found",
            )
        if not _screening_run_matches_grounding_payload(
            run_row,
            filename=body.filename,
            regime_id=body.regime_id,
        ):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "run_id does not match this screening pass (filename or regime).",
            )
        await _persist_screening_grounded_bundles(
            db,
            run_row,
            bundles_out,
            regime_id=body.regime_id,
        )
        persisted_id = rid_in

    return ScreeningGroundedOut(
        run_name=raw["run_name"],
        filename=raw["filename"],
        display_title=raw["display_title"],
        dataset_n_rows=int(raw["dataset_n_rows"]),
        dataset_n_cols=int(raw["dataset_n_cols"]),
        n_corpus_papers=int(raw["n_corpus_papers"]),
        regime_id=int(raw["regime_id"]),
        regime_n_rows=int(raw["regime_n_rows"]),
        bundles=bundles_out,
        warnings=list(raw.get("warnings", [])),
        persisted_to_run_id=persisted_id,
    )


@router.post("/run", status_code=status.HTTP_202_ACCEPTED)
async def start_run(
    config: RunConfig,
    user: CurrentUser,
    db: DBSession,
) -> JSONResponse:
    """
    Start a pipeline run with user-selected configuration.
    Returns run_id immediately. Poll /status/{run_id} or connect to WebSocket.
    """
    run_id = str(uuid.uuid4())[:8]

    run = Run(
        id=run_id,
        run_name=config.run_name,
        status="initializing",
        outcome_variable=config.outcome_variable,
        selected_features=config.feature_columns,
        excluded_features=config.exclude_columns,
        config_snapshot={
            "run_kind": "pipeline",
            "dataset_filename": config.filename,
            **(
                {"regime_id": int(config.regime_id)}
                if config.regime_id is not None
                else {}
            ),
        },
    )
    db.add(run)
    await db.flush()

    # Cache initial status for WebSocket
    await set_run_status(
        run_id,
        {
            "run_id": run_id,
            "status": "initializing",
            "round": 0,
            "match_score": 0.0,
        },
    )

    # Launch pipeline in background
    asyncio.create_task(_run_pipeline_background(run_id, config))

    logger.info(f"Run {run_id} started: {config.run_name}")
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={"run_id": run_id, "status": "initializing"},
    )


# ── Status ────────────────────────────────────────────────────────────────────


@router.get("/status/{run_id}", response_model=RunStatus)
async def get_run_status_endpoint(
    run_id: str,
    user: CurrentUser,
    db: DBSession,
) -> RunStatus:
    """Get current run status. Checks Redis cache first, then PostgreSQL."""
    run = await db.get(Run, run_id)
    if not run:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Run {run_id} not found")

    hyp_count = int(
        await db.scalar(
            select(func.count(Hypothesis.id)).where(Hypothesis.run_id == run_id)
        )
        or 0
    )
    cached = await get_run_status(run_id)
    if cached:
        return run_status_from_orm(
            run,
            hypothesis_count=hyp_count,
            status_override=str(cached.get("status", run.status)),
            current_round_override=int(cached.get("round", run.n_rounds_completed)),
            match_score_override=float(
                cached.get("match_score", run.final_match_score)
            ),
            converged_override=bool(cached.get("converged", run.converged)),
        )

    return run_status_from_orm(run, hypothesis_count=hyp_count)


@router.get("/runs", response_model=list[RunStatus])
async def list_runs(
    user: CurrentUser,
    db: DBSession,
) -> list[RunStatus]:
    """List all runs (full pipeline and screening passes), most recent first."""
    result = await db.execute(select(Run).order_by(Run.created_at.desc()).limit(50))
    runs = result.scalars().all()
    if not runs:
        return []

    run_ids = [r.id for r in runs]
    count_rows = (
        await db.execute(
            select(Hypothesis.run_id, func.count(Hypothesis.id))
            .where(Hypothesis.run_id.in_(run_ids))
            .group_by(Hypothesis.run_id)
        )
    ).all()
    count_map = {row[0]: int(row[1]) for row in count_rows}

    return [
        run_status_from_orm(r, hypothesis_count=count_map.get(r.id, 0)) for r in runs
    ]


class ClearScreeningRunsOut(BaseModel):
    deleted: int


@router.delete("/runs/screening/all", response_model=ClearScreeningRunsOut)
async def delete_all_screening_runs(
    user: CurrentUser,
    db: DBSession,
) -> ClearScreeningRunsOut:
    """Remove every screening (hypothesis test) run from history."""
    result = await db.execute(select(Run))
    deleted = 0
    for row in result.scalars().all():
        if _snapshot_dict(row).get("run_kind") == "screening":
            await _delete_run_record(db, row.id)
            deleted += 1
    logger.info(
        "Cleared %s screening runs user=%s",
        deleted,
        user.get("sub", "?"),
    )
    return ClearScreeningRunsOut(deleted=deleted)


@router.delete(
    "/runs/{run_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def delete_run(
    run_id: str,
    user: CurrentUser,
    db: DBSession,
) -> None:
    """Delete a run and dependent hypothesis / model / citation rows."""
    if not await _delete_run_record(db, run_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Run {run_id} not found")
    logger.info("Deleted run %s user=%s", run_id, user.get("sub", "?"))


# ── Background task ───────────────────────────────────────────────────────────


async def _run_pipeline_background(run_id: str, config: RunConfig) -> None:
    """Run the full pipeline in a background asyncio task."""
    from src.db.postgres import AsyncSessionFactory
    from src.orchestration.pipeline import run_pipeline

    async def _update_db(status: str, **kwargs: object) -> None:
        async with AsyncSessionFactory() as session:
            run = await session.get(Run, run_id)
            if run:
                run.status = status
                for k, v in kwargs.items():
                    setattr(run, k, v)
                await session.commit()

    try:
        await set_run_status(
            run_id,
            {"run_id": run_id, "status": "running", "round": 0, "match_score": 0.0},
        )
        await _update_db("running")

        # Inject UI config into environment
        import os

        os.environ["ARIA_OUTCOME_VARIABLE"] = config.outcome_variable
        os.environ["ARIA_DATA_FILE"] = str(RAW_DIR / config.filename)
        os.environ["ARIA_MAX_ROUNDS"] = str(config.max_rounds)
        os.environ["ARIA_CONVERGENCE_THRESHOLD"] = str(config.convergence_threshold)
        os.environ["ARIA_HYPOTHESES_PER_ROUND"] = str(config.hypotheses_per_round)

        loop = asyncio.get_event_loop()
        state, report = await loop.run_in_executor(
            None, lambda: run_pipeline(run_name=config.run_name)
        )

        final_status = "converged" if report.converged else "completed"
        await set_run_status(
            run_id,
            {
                "run_id": run_id,
                "status": final_status,
                "round": report.total_rounds,
                "match_score": report.final_score,
                "converged": report.converged,
            },
        )
        await _update_db(
            final_status,
            final_match_score=report.final_score,
            converged=report.converged,
            n_rounds_completed=report.total_rounds,
            stop_reason=report.stop_reason,
        )

    except Exception as e:
        logger.error(f"Run {run_id} failed: {e}")
        await set_run_status(
            run_id, {"run_id": run_id, "status": "failed", "error": str(e)}
        )
        await _update_db("failed", error_message=str(e))
