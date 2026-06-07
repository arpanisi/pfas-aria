"""
Pipeline Routes.
Upload dataset, configure run, trigger pipeline, stream status.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import Any, cast

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from sqlalchemy import func, select

from src.api.deps import CurrentUser, DBSession
from src.api.schemas.pipeline_schemas import (
    AutomatedScreeningIterationIn,
    AutomatedScreeningIterationOut,
    ClearScreeningRunsOut,
    ColumnInfo,
    DatasetPreview,
    GroundingJobProgress,
    GroundingJobStart,
    LegacyRegimeSummaryOut,
    LegacySegmentationPreviewOut,
    RegimeRowCountOut,
    RunStatus,
    ScreeningBundleOut,
    ScreeningCitationOut,
    ScreeningGroundedIn,
    ScreeningGroundedOut,
    ScreeningHypothesisOut,
    ScreeningModelOut,
    ScreeningStatsBundleOut,
    ScreeningStatsIn,
    ScreeningStatsOut,
)
from src.api.tenant import current_user_sub, safe_upload_filename
from src.db.orm import Hypothesis, Paper, Run
from src.db.redis_client import (
    get_grounding_job,
    get_run_status,
    invalidate_db_cache,
    set_grounding_job,
)
from src.ingestion.data_loader import DataLoader
from src.ingestion.unified_experimental_sheet import UnifiedSheetMeta
from src.reporting.narrative import (
    aggregate_system_summary,
    generate_display_title,
    generate_hypothesis_description,
    generate_hypothesis_rationale_from_model_evidence,
    generate_next_steps,
)
from src.services.pipeline_service import (
    coefs_as_floats,
    dataset_preview_rows,
    delete_run_record,
    enrich_bundles_with_rationale,
    load_grounded_bundles,
    load_stats_candidates,
    next_steps_inputs,
    normalize_dataframe_columns,
    persist_grounded_bundles,
    persist_stats_bundles,
    persist_upload_encodings,
    raw_upload_path,
    read_normalized_dataframe,
    run_status_from_orm,
    safe_raw_file_path,
    screening_run_matches_grounding_payload,
    snapshot_dict,
    variable_layout_payload,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/pipeline", tags=["Pipeline"])

MIN_CORPUS_PAPERS_FOR_SCREENING_GROUNDING = 1


async def _set_job(job_id: str, **kwargs: object) -> None:
    await set_grounding_job(job_id, dict(kwargs))


def _public_grounding_error(exc: Exception) -> str:
    raw = str(exc)
    lowered = raw.lower()
    if (
        "resolution lifetime expired" in lowered
        or "dns operation timed out" in lowered
        or "all nameservers failed" in lowered
        or "serverselectiontimeouterror" in lowered
    ):
        return (
            "Literature database temporarily unavailable. MongoDB DNS or network "
            "resolution failed; please retry after the connection recovers."
        )
    return raw


# ── Upload ────────────────────────────────────────────────────────────────────


@router.post("/upload", response_model=DatasetPreview)
async def upload_dataset(
    user: CurrentUser,
    db: DBSession,
    file: UploadFile = File(...),
) -> DatasetPreview:
    """Upload a CSV or Excel dataset. Returns column metadata for the column selector."""
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No filename provided")

    user_sub = current_user_sub(user)
    safe_name = safe_upload_filename(file.filename)
    suffix = safe_name.rsplit(".", 1)[-1].lower()
    if suffix not in {"csv", "xlsx", "xls", "tsv"}:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Unsupported file type: {suffix}. Use CSV or Excel.",
        )

    content = await file.read()
    import io

    excel_header_row: int | None = None
    unified_meta: UnifiedSheetMeta | None = None
    try:
        if suffix == "csv":
            df = pd.read_csv(io.BytesIO(content), low_memory=False)
        elif suffix == "tsv":
            df = pd.read_csv(io.BytesIO(content), sep="\t", low_memory=False)
        else:
            from src.ingestion.unified_experimental_sheet import (
                load_excel_bytes_with_layout,
            )

            df, excel_header_row, unified_meta = load_excel_bytes_with_layout(content)
    except Exception as e:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, f"Cannot parse file: {e}"
        ) from e

    normalize_dataframe_columns(df)
    df = DataLoader()._clean(df)
    from src.ingestion.upload_data_cleaning import apply_upload_data_cleaning

    cleaning_notes, encodings = apply_upload_data_cleaning(
        df, unified_meta=unified_meta
    )

    dest = raw_upload_path(safe_name, user_sub=user_sub)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)

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

    cols_list = list(df.columns)
    if unified_meta is not None:
        upload_input_cols = [c for c in unified_meta.input_cols if c in df.columns]
        upload_output_cols = [c for c in unified_meta.output_cols if c in df.columns]
    else:
        input_start, input_end, output_start = 2, 37, 37
        n = len(cols_list)
        if n > output_start:
            upload_input_cols = cols_list[input_start:input_end]
            upload_output_cols = cols_list[output_start:]
        elif n > input_start:
            upload_input_cols = cols_list[input_start:input_end]
            upload_output_cols = []
        else:
            upload_input_cols, upload_output_cols = [], []

    layout = variable_layout_payload(
        unified_meta,
        upload_input_cols=upload_input_cols,
        upload_output_cols=upload_output_cols,
    )
    label_encoding_record_id = await persist_upload_encodings(
        db,
        user_sub=user_sub,
        filename=safe_name,
        encodings=encodings,
        variable_layout=layout,
    )

    logger.info("Uploaded: %s (%s rows × %s cols)", safe_name, len(df), len(df.columns))
    return DatasetPreview(
        filename=safe_name,
        n_rows=len(df),
        n_cols=len(df.columns),
        columns=columns,
        excel_header_row=excel_header_row if suffix in {"xlsx", "xls"} else None,
        input_cols=upload_input_cols,
        output_cols=upload_output_cols,
        preview_rows=dataset_preview_rows(df, limit=100),
        cleaning_notes=cleaning_notes,
        metadata_cols=list(unified_meta.metadata_cols) if unified_meta else [],
        all_columns=[str(c) for c in df.columns],
        label_encoding_record_id=label_encoding_record_id,
    )


# ── Segmentation preview ──────────────────────────────────────────────────────


@router.get("/legacy-segmentation-preview", response_model=LegacySegmentationPreviewOut)
async def legacy_segmentation_preview(
    user: CurrentUser,
    filename: str,
    input_start: int = 2,
    input_end: int = 37,
    output_start: int = 37,
) -> LegacySegmentationPreviewOut:
    from src.ingestion.dataset_screening_layout import (
        assign_screening_layout_from_column_lists,
        assign_screening_layout_from_legacy_column_slices,
    )

    user_sub = current_user_sub(user)
    path = safe_raw_file_path(filename, user_sub=user_sub)
    df, _hdr, unified_meta = read_normalized_dataframe(path)

    try:
        if unified_meta is not None:
            res = assign_screening_layout_from_column_lists(
                df,
                unified_meta.input_cols,
                unified_meta.output_cols,
                replace_no_in_inputs=True,
            )
        else:
            res = assign_screening_layout_from_legacy_column_slices(
                df,
                input_start=input_start,
                input_end=input_end,
                output_start=output_start,
                replace_no_in_inputs=True,
            )
    except ValueError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e)) from e

    cond_col = "condition" if "condition" in df.columns else None
    input_list = list(res.input_cols)
    output_list = list(res.output_cols)
    regime_out: list[LegacyRegimeSummaryOut] = []
    for rid, sub in sorted(res.regimes.items()):
        idx_sample = [str(i) for i in sub.index[:40].tolist()]
        cond_sample: list[str] = []
        if cond_col and cond_col in sub.columns:
            cond_sample = sorted(sub[cond_col].dropna().astype(str).unique().tolist())[
                :25
            ]
        regime_out.append(
            LegacyRegimeSummaryOut(
                regime_id=int(rid),
                n_rows=len(sub),
                row_indices_sample=idx_sample,
                condition_values_sample=cond_sample,
                non_constant_input_cols=[
                    c
                    for c in input_list
                    if c in sub.columns and int(sub[c].nunique()) > 1
                ],
                non_constant_output_cols=[
                    c
                    for c in output_list
                    if c in sub.columns and int(sub[c].nunique()) > 1
                ],
            )
        )

    return LegacySegmentationPreviewOut(
        filename=Path(filename).name,
        n_rows=len(df),
        n_cols=len(df.columns),
        n_regimes=len(regime_out),
        input_cols=list(res.input_cols),
        output_cols=list(res.output_cols),
        regimes=regime_out,
        regime_row_counts=[
            RegimeRowCountOut(regime_id=rid, n_rows=res.regime_row_counts.get(rid, 0))
            for rid in range(1, 6)
        ],
        warnings=list(res.warnings),
    )


# ── Automated screening ───────────────────────────────────────────────────────


@router.post(
    "/automated-screening-iteration", response_model=AutomatedScreeningIterationOut
)
async def automated_screening_iteration(
    body: AutomatedScreeningIterationIn,
    user: CurrentUser,
    db: DBSession,
) -> AutomatedScreeningIterationOut:
    from src.pipeline.automated_screening import run_automated_screening_iteration

    user_sub = current_user_sub(user)
    path = safe_raw_file_path(body.filename, user_sub=user_sub)

    def _sync() -> int:
        df, _hdr, unified_meta = read_normalized_dataframe(path)
        return int(
            run_automated_screening_iteration(
                df, unified_meta=unified_meta, regime_id=body.regime_id
            )
        )

    n = int(await asyncio.get_event_loop().run_in_executor(None, _sync))

    run_id: str | None = None
    if body.regime_id is not None or n > 0:
        run_id = str(uuid.uuid4())[:8]
        snap: dict[str, Any] = {
            "run_kind": "screening",
            "regime_id": int(body.regime_id) if body.regime_id is not None else None,
            "hypotheses_tested": n,
            "dataset_filename": Path(body.filename).name,
        }
        if body.regime_id is None:
            snap["screening_scope"] = "all_slices"
        db_run = Run(
            id=run_id,
            user_sub=user_sub,
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


# ── Screening grounded (sync) ─────────────────────────────────────────────────


@router.post("/screening-grounded", response_model=ScreeningGroundedOut)
async def screening_grounded(
    body: ScreeningGroundedIn,
    user: CurrentUser,
    db: DBSession,
) -> ScreeningGroundedOut:
    user_sub = current_user_sub(user)
    n_papers = int(
        await db.scalar(select(func.count(Paper.id)).where(Paper.user_sub == user_sub))
        or 0
    )
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

    path = safe_raw_file_path(body.filename, user_sub=user_sub)

    def _sync() -> dict:
        from src.pipeline.screening_grounded import run_screening_grounded_for_regime
        from src.rag.pipeline import RAGPipeline

        df, _hdr, unified_meta = read_normalized_dataframe(path)
        retriever = RAGPipeline(user_sub=user_sub).build(force_rebuild=False)
        payload = run_screening_grounded_for_regime(
            df,
            unified_meta=unified_meta,
            regime_id=body.regime_id,
            retriever=retriever,
        )
        payload.update(
            {
                "dataset_n_rows": len(df),
                "dataset_n_cols": len(df.columns),
                "n_corpus_papers": n_papers,
                "filename": Path(body.filename).name,
                "run_name": (body.run_name or "").strip() or "Screening run",
                "display_title": "Literature Grounding of Discovered Signals",
            }
        )
        return cast(dict[str, Any], payload)

    try:
        raw = await asyncio.get_event_loop().run_in_executor(None, _sync)
    except Exception as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            _public_grounding_error(exc),
        ) from exc
    bundles_out = [
        ScreeningBundleOut(
            hypothesis=ScreeningHypothesisOut(**b["hypothesis"]),
            model_result=ScreeningModelOut(**b["model_result"]),
            citations=[ScreeningCitationOut(**c) for c in b.get("citations", [])],
            output_variable=b.get("y_col"),
        )
        for b in raw.get("bundles", [])
    ]
    bundles_out = await enrich_bundles_with_rationale(bundles_out)

    persisted_id: str | None = None
    rid_in = (body.run_id or "").strip()
    extra_warnings: list[str] = []
    if rid_in:
        run_row = await db.get(Run, rid_in)
        if not run_row:
            extra_warnings.append(
                f"Run id {rid_in!r} was not found; results are shown but not saved."
            )
        elif run_row.user_sub != user_sub:
            extra_warnings.append(
                f"Run id {rid_in!r} was not found; results are shown but not saved."
            )
        elif not screening_run_matches_grounding_payload(
            run_row, filename=body.filename, regime_id=body.regime_id
        ):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "run_id does not match this screening pass.",
            )
        else:
            await persist_grounded_bundles(
                db, run_row, bundles_out, regime_id=body.regime_id
            )
            await invalidate_db_cache(rid_in)
            persisted_id = rid_in

    rationales = [b.hypothesis.rationale for b in bundles_out if b.hypothesis.rationale]
    summary = aggregate_system_summary(rationales)
    sig_vars, non_sig_vars, cit_titles = next_steps_inputs(bundles_out)
    nxt = generate_next_steps(
        significant_vars=sig_vars,
        non_significant_vars=non_sig_vars,
        system_summary=summary,
        citation_titles=cit_titles,
    )
    display_title = generate_display_title(bundles_out, system_summary=summary)

    if persisted_id:
        run_row = await db.get(Run, persisted_id)
        if run_row:
            snap = dict(snapshot_dict(run_row))
            snap.setdefault("grounding_result_meta", {}).update(
                {
                    "run_name": raw["run_name"],
                    "filename": raw["filename"],
                    "display_title": display_title,
                    "dataset_n_rows": raw.get("dataset_n_rows", 0),
                    "dataset_n_cols": raw.get("dataset_n_cols", 0),
                    "n_corpus_papers": raw.get("n_corpus_papers", 0),
                    "regime_n_rows": raw.get("regime_n_rows", 0),
                    "system_summary": summary,
                    "next_steps": nxt,
                }
            )
            run_row.config_snapshot = snap
            await db.commit()

    return ScreeningGroundedOut(
        run_name=raw["run_name"],
        filename=raw["filename"],
        display_title=display_title,
        dataset_n_rows=int(raw["dataset_n_rows"]),
        dataset_n_cols=int(raw["dataset_n_cols"]),
        n_corpus_papers=int(raw["n_corpus_papers"]),
        regime_id=int(raw["regime_id"]),
        regime_n_rows=int(raw["regime_n_rows"]),
        bundles=bundles_out,
        warnings=[*list(raw.get("warnings", [])), *extra_warnings],
        persisted_to_run_id=persisted_id,
        system_summary=summary,
        next_steps=nxt,
    )


# ── Screening grounded (async job) ────────────────────────────────────────────


@router.post("/screening-grounded/start", response_model=GroundingJobStart)
async def screening_grounded_start(
    body: ScreeningGroundedIn,
    user: CurrentUser,
    db: DBSession,
) -> GroundingJobStart:
    """Start a grounding job in the background; poll /progress/{job_id} for updates."""
    user_sub = current_user_sub(user)
    n_papers = int(
        await db.scalar(select(func.count(Paper.id)).where(Paper.user_sub == user_sub))
        or 0
    )
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
    job_id = str(uuid.uuid4())[:16]
    await _set_job(
        job_id,
        pct=0,
        stage="Starting…",
        done=False,
        result=None,
        error=None,
        started_at=time.time(),
        user_sub=user_sub,
    )
    asyncio.create_task(_run_grounding_job(job_id, body, n_papers, user_sub))
    return GroundingJobStart(job_id=job_id)


@router.get("/screening-grounded/saved/{run_id}", response_model=ScreeningGroundedOut)
async def saved_screening_grounded(
    run_id: str,
    user: CurrentUser,
    db: DBSession,
) -> ScreeningGroundedOut:
    """Return a previously saved literature grounding report for a screening run."""
    saved = await load_grounded_bundles(
        db, run_id.strip(), user_sub=current_user_sub(user)
    )
    if not saved:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Saved literature report for run {run_id!r} not found",
        )
    return saved


@router.get(
    "/screening-grounded/progress/{job_id}", response_model=GroundingJobProgress
)
async def screening_grounded_progress(
    job_id: str,
    user: CurrentUser,
) -> GroundingJobProgress:
    """Poll progress of a grounding job started via /start."""
    job = dict(await get_grounding_job(job_id) or {})
    if not job:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Job {job_id!r} not found")
    if job.get("user_sub") != current_user_sub(user):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Job {job_id!r} not found")
    eta: int | None = None
    pct = int(job.get("pct", 0))
    started_at = job.get("started_at")
    if started_at and pct > 5 and not job.get("done"):
        elapsed = time.time() - float(started_at)
        rate = pct / elapsed
        if rate > 0:
            eta = max(1, int((100 - pct) / rate))
    return GroundingJobProgress(
        pct=pct,
        stage=str(job.get("stage", "")),
        done=bool(job.get("done", False)),
        eta_seconds=eta,
        result=job.get("result"),
        error=job.get("error"),
    )


async def _run_grounding_job(
    job_id: str, body: ScreeningGroundedIn, n_papers: int, user_sub: str
) -> None:
    """Background coroutine: run screening grounding and emit progress updates via Redis."""
    from src.db.postgres import AsyncSessionFactory

    try:
        await _set_job(job_id, pct=5, stage="Checking for pre-computed results…")
        rid_in = (body.run_id or "").strip()

        if rid_in:
            async with AsyncSessionFactory() as session:
                saved = await load_grounded_bundles(session, rid_in, user_sub=user_sub)
                if saved:
                    await _set_job(
                        job_id,
                        pct=100,
                        stage="Done",
                        done=True,
                        result=saved.model_dump(),
                    )
                    return

        path = safe_raw_file_path(body.filename, user_sub=user_sub)

        precomputed: list[dict] = []
        if rid_in:
            async with AsyncSessionFactory() as session:
                precomputed = await load_stats_candidates(
                    session, rid_in, body.regime_id, user_sub=user_sub
                )

        loop = asyncio.get_event_loop()

        if precomputed:
            await _set_job(
                job_id, pct=20, stage="Loading pre-computed screening results…"
            )

            def _sync_rag_only() -> dict:
                from src.pipeline.screening_grounded import (
                    run_grounding_from_precomputed,
                )
                from src.rag.pipeline import RAGPipeline

                retriever = RAGPipeline(user_sub=user_sub).build(force_rebuild=False)
                df, _hdr, _meta = read_normalized_dataframe(path)
                payload = run_grounding_from_precomputed(
                    precomputed,
                    retriever,
                    regime_id=body.regime_id,
                    regime_n_rows=len(df),
                )
                payload.update(
                    {
                        "dataset_n_rows": len(df),
                        "dataset_n_cols": len(df.columns),
                        "n_corpus_papers": n_papers,
                        "filename": Path(body.filename).name,
                        "run_name": (body.run_name or "").strip() or "Screening run",
                        "display_title": "Literature Grounding of Discovered Signals",
                    }
                )
                return cast(dict[str, Any], payload)

            await _set_job(job_id, pct=30, stage="Retrieving literature matches…")

            async def _rag_tick(stop: asyncio.Event) -> None:
                pct = 30
                while not stop.is_set():
                    await asyncio.sleep(4)
                    if stop.is_set():
                        break
                    pct = min(pct + 2, 44)
                    job = await get_grounding_job(job_id)
                    if job and not job.get("done") and job.get("pct", 0) < 45:
                        await _set_job(
                            job_id, pct=pct, stage="Retrieving literature matches…"
                        )

            stop_rag = asyncio.Event()
            rag_task = asyncio.create_task(_rag_tick(stop_rag))
            try:
                raw = await loop.run_in_executor(None, _sync_rag_only)
            finally:
                stop_rag.set()
                rag_task.cancel()

        else:
            await _set_job(
                job_id, pct=15, stage="Fitting models & retrieving literature…"
            )

            async def _fitting_tick(stop: asyncio.Event) -> None:
                pct = 15
                while not stop.is_set():
                    await asyncio.sleep(4)
                    if stop.is_set():
                        break
                    pct = (
                        min(pct + 4, 30)
                        if pct < 30
                        else min(pct + 2, 40)
                        if pct < 40
                        else min(pct + 1, 44)
                    )
                    job = await get_grounding_job(job_id)
                    if job and not job.get("done") and job.get("pct", 0) < 45:
                        await _set_job(
                            job_id,
                            pct=pct,
                            stage="Fitting models & retrieving literature…",
                        )

            def _sync_full() -> dict:
                from src.pipeline.screening_grounded import (
                    run_screening_grounded_for_regime,
                )
                from src.rag.pipeline import RAGPipeline

                df, _hdr, unified_meta = read_normalized_dataframe(path)
                retriever = RAGPipeline(user_sub=user_sub).build(force_rebuild=False)
                payload = run_screening_grounded_for_regime(
                    df,
                    unified_meta=unified_meta,
                    regime_id=body.regime_id,
                    retriever=retriever,
                )
                payload.update(
                    {
                        "dataset_n_rows": len(df),
                        "dataset_n_cols": len(df.columns),
                        "n_corpus_papers": n_papers,
                        "filename": Path(body.filename).name,
                        "run_name": (body.run_name or "").strip() or "Screening run",
                        "display_title": "Literature Grounding of Discovered Signals",
                    }
                )
                return cast(dict[str, Any], payload)

            stop_tick = asyncio.Event()
            tick_task = asyncio.create_task(_fitting_tick(stop_tick))
            try:
                raw = await loop.run_in_executor(None, _sync_full)
            finally:
                stop_tick.set()
                tick_task.cancel()

        await _set_job(job_id, pct=45, stage="Ranking hypotheses…")

        bundles_out = [
            ScreeningBundleOut(
                hypothesis=ScreeningHypothesisOut(**b["hypothesis"]),
                model_result=ScreeningModelOut(**b["model_result"]),
                citations=[ScreeningCitationOut(**c) for c in b.get("citations", [])],
                output_variable=b.get("y_col"),
            )
            for b in raw.get("bundles", [])
        ]

        n_bundles = len(bundles_out)
        completed_count = [0]

        async def _one_tracked(b: ScreeningBundleOut) -> ScreeningBundleOut:
            titles = [c.title for c in b.citations if c.title][:3]
            _timeout_s = 55
            description, rationale = await asyncio.gather(
                asyncio.to_thread(
                    generate_hypothesis_description,
                    raw_description=b.hypothesis.description,
                    primary_variables=list(b.hypothesis.primary_variables),
                    significant_variables=list(b.model_result.significant_variables),
                    request_timeout_seconds=_timeout_s,
                ),
                asyncio.to_thread(
                    generate_hypothesis_rationale_from_model_evidence,
                    hypothesis_description=b.hypothesis.description,
                    primary_variables=list(b.hypothesis.primary_variables),
                    model_family=b.hypothesis.model_family,
                    r_squared=float(b.model_result.r_squared),
                    adj_r_squared=float(b.model_result.adj_r_squared),
                    significant_variables=list(b.model_result.significant_variables),
                    coefficients=coefs_as_floats(dict(b.model_result.coefficients)),
                    validation_passed=bool(b.model_result.validation_passed),
                    citation_titles=titles,
                    request_timeout_seconds=_timeout_s,
                ),
            )
            completed_count[0] += 1
            pct = 50 + int(completed_count[0] / max(n_bundles, 1) * 40)
            await _set_job(
                job_id,
                pct=pct,
                stage=f"Interpreting hypothesis {completed_count[0]}/{n_bundles}…",
            )
            return ScreeningBundleOut(
                hypothesis=b.hypothesis.model_copy(
                    update={"description": description, "rationale": rationale}
                ),
                model_result=b.model_result,
                citations=b.citations,
                output_variable=b.output_variable,
            )

        bundles_out = list(
            await asyncio.gather(*(_one_tracked(b) for b in bundles_out))
        )

        await _set_job(job_id, pct=93, stage="Saving results…")

        persisted_id: str | None = None
        extra_warnings: list[str] = []
        rid_in = (body.run_id or "").strip()
        if rid_in:
            async with AsyncSessionFactory() as session:
                run_row = await session.get(Run, rid_in)
                if not run_row:
                    extra_warnings.append(
                        f"Run id {rid_in!r} was not found; results shown but not saved."
                    )
                elif run_row.user_sub != user_sub:
                    extra_warnings.append(
                        f"Run id {rid_in!r} was not found; results shown but not saved."
                    )
                elif not screening_run_matches_grounding_payload(
                    run_row, filename=body.filename, regime_id=body.regime_id
                ):
                    extra_warnings.append(
                        "run_id does not match this screening pass (filename or regime)."
                    )
                else:
                    await persist_grounded_bundles(
                        session,
                        run_row,
                        bundles_out,
                        regime_id=body.regime_id,
                        grounding_meta={
                            "run_name": raw.get("run_name", ""),
                            "filename": raw.get("filename", ""),
                            "display_title": raw.get("display_title", ""),
                            "dataset_n_rows": raw.get("dataset_n_rows", 0),
                            "dataset_n_cols": raw.get("dataset_n_cols", 0),
                            "n_corpus_papers": raw.get("n_corpus_papers", 0),
                            "regime_n_rows": raw.get("regime_n_rows", 0),
                        },
                    )
                    await session.commit()
                    await invalidate_db_cache(rid_in)
                    persisted_id = rid_in

        rationales = [
            b.hypothesis.rationale for b in bundles_out if b.hypothesis.rationale
        ]
        summary = await asyncio.to_thread(aggregate_system_summary, rationales)
        sig_vars, non_sig_vars, cit_titles = next_steps_inputs(bundles_out)
        nxt = await asyncio.to_thread(
            generate_next_steps,
            significant_vars=sig_vars,
            non_significant_vars=non_sig_vars,
            system_summary=summary,
            citation_titles=cit_titles,
        )
        display_title = generate_display_title(bundles_out, system_summary=summary)

        if persisted_id:
            async with AsyncSessionFactory() as session:
                run_row = await session.get(Run, persisted_id)
                if run_row:
                    snap = dict(snapshot_dict(run_row))
                    snap.setdefault("grounding_result_meta", {}).update(
                        {
                            "system_summary": summary,
                            "next_steps": nxt,
                            "display_title": display_title,
                        }
                    )
                    run_row.config_snapshot = snap
                    await session.commit()

        result = ScreeningGroundedOut(
            run_name=raw["run_name"],
            filename=raw["filename"],
            display_title=display_title,
            dataset_n_rows=int(raw["dataset_n_rows"]),
            dataset_n_cols=int(raw["dataset_n_cols"]),
            n_corpus_papers=int(raw["n_corpus_papers"]),
            regime_id=int(raw["regime_id"]),
            regime_n_rows=int(raw["regime_n_rows"]),
            bundles=bundles_out,
            warnings=[*list(raw.get("warnings", [])), *extra_warnings],
            persisted_to_run_id=persisted_id,
            system_summary=summary,
            next_steps=nxt,
        )
        await _set_job(
            job_id, pct=100, stage="Done", done=True, result=result.model_dump()
        )
        logger.info("Grounding job %s complete bundles=%s", job_id, len(bundles_out))

    except Exception as e:
        logger.error("Grounding job %s failed: %s", job_id, e)
        await _set_job(
            job_id,
            pct=0,
            stage="Failed",
            done=True,
            error=_public_grounding_error(e),
        )


# ── Screening stats ───────────────────────────────────────────────────────────


@router.post("/screening-stats", response_model=ScreeningStatsOut)
async def screening_stats(
    body: ScreeningStatsIn,
    user: CurrentUser,
    db: DBSession,
) -> ScreeningStatsOut:
    """Run OLS screening for a single regime. No corpus required."""
    user_sub = current_user_sub(user)
    path = safe_raw_file_path(body.filename, user_sub=user_sub)

    def _sync() -> dict:
        from src.pipeline.screening_grounded import run_screening_stats_for_regime

        df, _hdr, unified_meta = read_normalized_dataframe(path)
        payload = run_screening_stats_for_regime(
            df, unified_meta=unified_meta, regime_id=body.regime_id
        )
        payload.update(
            {
                "dataset_n_rows": len(df),
                "dataset_n_cols": len(df.columns),
                "filename": Path(body.filename).name,
                "run_name": (body.run_name or "").strip() or "Screening stats",
                "display_title": "Statistical Screening Results",
            }
        )
        return cast(dict[str, Any], payload)

    raw = await asyncio.get_event_loop().run_in_executor(None, _sync)
    bundles_out = [
        ScreeningStatsBundleOut(
            hypothesis=ScreeningHypothesisOut(**b["hypothesis"]),
            model_result=ScreeningModelOut(**b["model_result"]),
            diagnostics=b.get("diagnostics"),
            output_variable=b.get("y_col"),
        )
        for b in raw.get("bundles", [])
    ]

    persisted_id: str | None = None
    rid_in = (body.run_id or "").strip()
    if rid_in and bundles_out:
        run_row = await db.get(Run, rid_in)
        if (
            run_row
            and run_row.user_sub == user_sub
            and screening_run_matches_grounding_payload(
                run_row, filename=body.filename, regime_id=body.regime_id
            )
        ):
            await persist_stats_bundles(
                db, run_row, raw.get("bundles", []), regime_id=body.regime_id
            )
            persisted_id = rid_in
            logger.info(
                "Persisted %s stats bundles to run %s", len(bundles_out), rid_in
            )

    logger.info(
        "Screening stats file=%s regime=%s bundles=%s user=%s",
        body.filename,
        body.regime_id,
        len(bundles_out),
        user.get("sub", "?"),
    )
    return ScreeningStatsOut(
        run_name=raw["run_name"],
        filename=raw["filename"],
        display_title=raw["display_title"],
        dataset_n_rows=int(raw["dataset_n_rows"]),
        dataset_n_cols=int(raw["dataset_n_cols"]),
        regime_id=int(raw["regime_id"]),
        regime_n_rows=int(raw["regime_n_rows"]),
        bundles=bundles_out,
        warnings=list(raw.get("warnings", [])),
        persisted_to_run_id=persisted_id,
    )


@router.get("/status/{run_id}", response_model=RunStatus)
async def get_run_status_endpoint(
    run_id: str, user: CurrentUser, db: DBSession
) -> RunStatus:
    user_sub = current_user_sub(user)
    run = await db.get(Run, run_id)
    if not run or run.user_sub != user_sub:
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
async def list_runs(user: CurrentUser, db: DBSession) -> list[RunStatus]:
    user_sub = current_user_sub(user)
    result = await db.execute(
        select(Run)
        .where(Run.user_sub == user_sub)
        .order_by(Run.created_at.desc())
        .limit(50)
    )
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


@router.delete("/runs/screening/all", response_model=ClearScreeningRunsOut)
async def delete_all_screening_runs(
    user: CurrentUser, db: DBSession
) -> ClearScreeningRunsOut:
    user_sub = current_user_sub(user)
    result = await db.execute(select(Run).where(Run.user_sub == user_sub))
    deleted = 0
    for row in result.scalars().all():
        if snapshot_dict(row).get("run_kind") == "screening":
            await delete_run_record(db, row.id, user_sub=user_sub)
            deleted += 1
    logger.info("Cleared %s screening runs user=%s", deleted, user.get("sub", "?"))
    return ClearScreeningRunsOut(deleted=deleted)


@router.delete(
    "/runs/{run_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None
)
async def delete_run(run_id: str, user: CurrentUser, db: DBSession) -> None:
    user_sub = current_user_sub(user)
    if not await delete_run_record(db, run_id, user_sub=user_sub):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Run {run_id} not found")
    logger.info("Deleted run %s user=%s", run_id, user.get("sub", "?"))
