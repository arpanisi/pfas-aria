"""
Pipeline service layer.

Pure helper functions shared across pipeline route handlers — DataFrame ops,
DB persistence, run-lifecycle utilities, and LLM enrichment. No FastAPI route
decorators here; only the route file holds those.
"""

from __future__ import annotations

import asyncio
import io
import math
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from fastapi import HTTPException, status
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas.pipeline_schemas import (
    RunStatus,
    ScreeningBundleOut,
    ScreeningCitationOut,
    ScreeningGroundedOut,
    ScreeningHypothesisOut,
    ScreeningModelOut,
)
from src.db.orm import (
    Citation,
    DatasetUploadEncoding,
    ExperimentSegmentationBatch,
    Hypothesis,
    ModelResult,
    Regime,
    Run,
    ValidationResult,
)
from src.db.redis_client import delete_run_status
from src.ingestion.unified_experimental_sheet import (
    UnifiedSheetMeta,
    load_excel_bytes_with_layout,
)
from src.reporting.narrative import (
    aggregate_system_summary,
    generate_hypothesis_description,
    generate_hypothesis_rationale_from_model_evidence,
    generate_next_steps,
)
from src.utils.logging import get_logger
from src.utils.paths import RAW_DIR

logger = get_logger(__name__)


# ── DataFrame / ingestion helpers ─────────────────────────────────────────────


def normalize_dataframe_columns(df: pd.DataFrame) -> None:
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]


def _preview_cell_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, pd.Timestamp | datetime):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.10g}"
    try:
        fv = float(cast(Any, value))
        if not math.isfinite(fv):
            return ""
        ti = int(fv)
        if ti == fv and abs(ti) < 10**15:
            return str(ti)
        return f"{fv:.10g}"
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.lower() in ("nan", "nat", "none"):
        return ""
    return text


def dataset_preview_rows(df: pd.DataFrame, *, limit: int = 100) -> list[dict[str, str]]:
    if df.empty or limit <= 0:
        return []
    out: list[dict[str, str]] = []
    for rec in df.head(limit).to_dict(orient="records"):
        out.append({str(k): _preview_cell_str(v) for k, v in rec.items()})
    return out


def safe_raw_file_path(filename: str) -> Path:
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


def read_normalized_dataframe(
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
            f"Unsupported file type: {suffix}",
        )
    normalize_dataframe_columns(df)
    from src.ingestion.upload_data_cleaning import apply_upload_data_cleaning

    _notes, _enc = apply_upload_data_cleaning(df, unified_meta=unified_meta)
    return df, excel_header_row, unified_meta


# ── Run lifecycle helpers ──────────────────────────────────────────────────────


def snapshot_dict(run: Run) -> dict:
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
    snap = snapshot_dict(run)
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

    _status = status_override if status_override is not None else run.status
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
    screening_phase = (
        str(snap["screening_phase"]) if snap.get("screening_phase") else None
    )

    return RunStatus(
        run_id=run.id,
        run_name=run.run_name,
        status=_status,
        current_round=cur_round,
        final_match_score=match,
        converged=conv,
        n_rounds_completed=cur_round,
        run_kind=run_kind,
        regime_id=regime_id,
        hypotheses_tested=hypotheses_tested,
        dataset_filename=dataset_filename,
        created_at=run.created_at,
        screening_phase=screening_phase,
    )


def screening_run_matches_grounding_payload(
    run: Run, *, filename: str, regime_id: int
) -> bool:
    snap = snapshot_dict(run)
    if snap.get("run_kind") != "screening":
        return False
    snap_rid = snap.get("regime_id")
    if snap_rid is not None and int(snap_rid) != int(regime_id):
        return False
    return str(snap.get("dataset_filename", "")) == Path(filename).name


# ── DB helpers ────────────────────────────────────────────────────────────────


async def delete_hypothesis_subtree(db: AsyncSession, run_id: str) -> None:
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


def coefs_as_floats(coefs: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in coefs.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def sanitize_float_json(d: dict, *, fallback: float | None = None) -> dict:
    """Replace NaN / Inf with fallback so Postgres JSON doesn't choke."""
    out: dict = {}
    for k, v in d.items():
        try:
            fv = float(v)
            out[str(k)] = fallback if not math.isfinite(fv) else fv
        except (TypeError, ValueError):
            out[str(k)] = fallback
    return out


async def enrich_bundles_with_rationale(
    bundles: list[ScreeningBundleOut],
) -> list[ScreeningBundleOut]:
    """LLM description + rationale for each bundle, gathered concurrently."""
    _timeout_s = 55

    async def _one(b: ScreeningBundleOut) -> ScreeningBundleOut:
        titles = [c.title for c in b.citations if c.title][:3]
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
        return ScreeningBundleOut(
            hypothesis=b.hypothesis.model_copy(
                update={"description": description, "rationale": rationale}
            ),
            model_result=b.model_result,
            citations=b.citations,
        )

    return list(await asyncio.gather(*(_one(b) for b in bundles)))


async def persist_grounded_bundles(
    db: AsyncSession,
    run: Run,
    bundles: list[ScreeningBundleOut],
    *,
    regime_id: int,
    grounding_meta: dict | None = None,
) -> None:
    await delete_hypothesis_subtree(db, run.id)
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
            coefficients=sanitize_float_json(dict(m.coefficients), fallback=0.0),
            p_values=sanitize_float_json(dict(m.p_values), fallback=1.0),
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
                    abstract_snippet=c.abstract_snippet,
                    variable=c.variable,
                )
            )

    if bundles:
        best_match = max(
            (float(b.model_result.match_score) for b in bundles), default=0.0
        )
        run.final_match_score = best_match
    snap = dict(snapshot_dict(run))
    snap["grounding_saved_at"] = datetime.utcnow().isoformat() + "Z"
    snap["grounded_hypothesis_count"] = len(bundles)
    snap["screening_phase"] = "grounding_completed"
    if grounding_meta:
        snap["grounding_result_meta"] = grounding_meta
    run.config_snapshot = snap


async def persist_stats_bundles(
    db: AsyncSession,
    run: Run,
    raw_bundles: list[dict],
    *,
    regime_id: int,
) -> None:
    await delete_hypothesis_subtree(db, run.id)
    regime_label = f"regime_{regime_id}"
    for b in raw_bundles:
        h = b["hypothesis"]
        m = b["model_result"]
        y_col = str(b.get("y_col") or "")
        x_cols = list(h.get("primary_variables") or [])
        hid = str(uuid.uuid4())
        hyp_row = Hypothesis(
            id=hid,
            run_id=run.id,
            hypothesis_id=str(h["hypothesis_id"]),
            round=int(h.get("round", 1)),
            description=str(h.get("description", "")),
            rationale=h.get("rationale"),
            primary_variables=x_cols,
            interaction_terms=[],
            control_variables=[],
            model_family=str(h.get("model_family", "screening_ols")),
            priority_score=float(h.get("priority_score", 0.0)),
            is_refinement=bool(h.get("is_refinement", False)),
            rag_support=[{"_stats_meta": True, "y_col": y_col, "x_cols": x_cols}],
        )
        db.add(hyp_row)
        await db.flush()
        mr_row = ModelResult(
            id=str(uuid.uuid4()),
            hypothesis_id=hid,
            model_type=str(m.get("model_type", "screening_linear")),
            regime_label=regime_label,
            r_squared=float(m.get("r_squared", 0.0)),
            adj_r_squared=float(m.get("adj_r_squared", 0.0)),
            n_observations=int(m.get("n_observations", 0)),
            coefficients=sanitize_float_json(
                dict(m.get("coefficients") or {}), fallback=0.0
            ),
            p_values=sanitize_float_json(dict(m.get("p_values") or {}), fallback=1.0),
            significant_variables=list(m.get("significant_variables") or []),
            match_score=0.0,
            success=True,
            error_message=None,
        )
        db.add(mr_row)
    snap = dict(snapshot_dict(run))
    snap["stats_saved_at"] = datetime.utcnow().isoformat() + "Z"
    snap["stats_bundle_count"] = len(raw_bundles)
    snap["screening_phase"] = "stats_completed"
    run.config_snapshot = snap


async def load_stats_candidates(
    db: AsyncSession, run_id: str, regime_id: int
) -> list[dict]:
    regime_label = f"regime_{regime_id}"
    hyp_rows = (
        (await db.execute(select(Hypothesis).where(Hypothesis.run_id == run_id)))
        .scalars()
        .all()
    )
    if not hyp_rows:
        return []
    out: list[dict] = []
    for hyp in hyp_rows:
        meta = next(
            (
                r
                for r in (hyp.rag_support or [])
                if isinstance(r, dict) and r.get("_stats_meta")
            ),
            None,
        )
        if not meta:
            continue
        mr = (
            (
                await db.execute(
                    select(ModelResult).where(
                        ModelResult.hypothesis_id == hyp.id,
                        ModelResult.regime_label == regime_label,
                    )
                )
            )
            .scalars()
            .first()
        )
        if not mr:
            continue
        out.append(
            {
                "y_col": str(meta.get("y_col") or ""),
                "x_cols": list(meta.get("x_cols") or hyp.primary_variables or []),
                "r_squared": float(mr.r_squared),
                "adj_r_squared": float(mr.adj_r_squared),
                "n_obs": int(mr.n_observations),
                "coefficients": dict(mr.coefficients or {}),
                "p_values": dict(mr.p_values or {}),
                "significant_variables": list(mr.significant_variables or []),
            }
        )
    return out


def next_steps_inputs(
    bundles: list[ScreeningBundleOut],
) -> tuple[list[str], list[str], list[str]]:
    sig: set[str] = set()
    non_sig: set[str] = set()
    titles: list[str] = []
    seen: set[str] = set()
    for b in bundles:
        sig.update(b.model_result.significant_variables or [])
        for v in b.hypothesis.primary_variables or []:
            if v not in sig:
                non_sig.add(v)
        for c in b.citations:
            if c.title and c.title not in seen:
                titles.append(c.title)
                seen.add(c.title)
    non_sig -= sig
    return list(sig), list(non_sig), titles[:4]


async def load_grounded_bundles(
    db: AsyncSession, run_id: str
) -> ScreeningGroundedOut | None:
    run_row = await db.get(Run, run_id)
    if not run_row:
        return None
    snap = snapshot_dict(run_row)
    if snap.get("screening_phase") != "grounding_completed":
        return None

    meta = snap.get("grounding_result_meta") or {}
    hyp_rows = (
        (await db.execute(select(Hypothesis).where(Hypothesis.run_id == run_id)))
        .scalars()
        .all()
    )

    bundles_out: list[ScreeningBundleOut] = []
    for hyp in hyp_rows:
        if any(
            isinstance(s, dict) and s.get("_stats_meta")
            for s in (hyp.rag_support or [])
        ):
            continue
        mr = (
            (
                await db.execute(
                    select(ModelResult).where(ModelResult.hypothesis_id == hyp.id)
                )
            )
            .scalars()
            .first()
        )
        if not mr:
            continue
        vr = (
            (
                await db.execute(
                    select(ValidationResult).where(
                        ValidationResult.model_result_id == mr.id
                    )
                )
            )
            .scalars()
            .first()
        )
        cit_rows = (
            (
                await db.execute(
                    select(Citation).where(Citation.model_result_id == mr.id)
                )
            )
            .scalars()
            .all()
        )
        bundles_out.append(
            ScreeningBundleOut(
                hypothesis=ScreeningHypothesisOut(
                    id=hyp.id,
                    hypothesis_id=hyp.hypothesis_id,
                    round=hyp.round,
                    description=hyp.description,
                    rationale=hyp.rationale,
                    primary_variables=list(hyp.primary_variables),
                    model_family=hyp.model_family,
                    priority_score=hyp.priority_score,
                    is_refinement=hyp.is_refinement,
                ),
                model_result=ScreeningModelOut(
                    id=mr.id,
                    hypothesis_id=hyp.hypothesis_id,
                    model_type=mr.model_type,
                    r_squared=mr.r_squared,
                    adj_r_squared=mr.adj_r_squared,
                    n_observations=mr.n_observations,
                    coefficients=dict(mr.coefficients or {}),
                    p_values=dict(mr.p_values or {}),
                    significant_variables=list(mr.significant_variables or []),
                    match_score=mr.match_score,
                    validation_passed=bool(vr.overall_passed) if vr else False,
                ),
                citations=[
                    ScreeningCitationOut(
                        id=c.id,
                        source=c.source,
                        title=c.title,
                        url=c.url,
                        year=c.year,
                        similarity_score=c.similarity_score,
                        abstract_snippet=c.abstract_snippet,
                        variable=c.variable,
                        hypothesis_id=hyp.hypothesis_id,
                    )
                    for c in cit_rows
                ],
            )
        )

    summary = meta.get("system_summary") or None
    next_steps = meta.get("next_steps") or None
    display_title = (
        str(meta.get("display_title") or "") or "Literature Grounding Results"
    )

    if not summary or not next_steps:
        rationales = [
            b.hypothesis.rationale for b in bundles_out if b.hypothesis.rationale
        ]
        if not summary:
            summary = await asyncio.to_thread(aggregate_system_summary, rationales)
        if not next_steps:
            sig_vars, non_sig_vars, cit_titles = next_steps_inputs(bundles_out)
            next_steps = await asyncio.to_thread(
                generate_next_steps,
                significant_vars=sig_vars,
                non_significant_vars=non_sig_vars,
                system_summary=summary,
                citation_titles=cit_titles,
            )

    return ScreeningGroundedOut(
        run_name=str(meta.get("run_name") or run_row.run_name),
        filename=str(meta.get("filename") or snap.get("dataset_filename") or ""),
        display_title=display_title,
        dataset_n_rows=int(meta.get("dataset_n_rows") or 0),
        dataset_n_cols=int(meta.get("dataset_n_cols") or 0),
        n_corpus_papers=int(meta.get("n_corpus_papers") or 0),
        regime_id=int(snap.get("regime_id") or 0),
        regime_n_rows=int(meta.get("regime_n_rows") or 0),
        bundles=bundles_out,
        warnings=[],
        persisted_to_run_id=run_id,
        system_summary=summary,
        next_steps=next_steps,
    )


async def delete_run_record(db: AsyncSession, run_id: str) -> bool:
    run = await db.get(Run, run_id)
    if not run:
        return False
    await delete_hypothesis_subtree(db, run_id)
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
        logger.warning("Redis delete_run_status failed for {}: {}", run_id, e)
    return True


async def persist_upload_encodings(
    db: AsyncSession,
    *,
    user_sub: str,
    filename: str,
    encodings: dict[str, Any],
    variable_layout: dict[str, Any],
) -> str | None:
    safe_name = Path(filename).name
    try:
        await db.execute(
            delete(DatasetUploadEncoding).where(
                DatasetUploadEncoding.user_sub == user_sub,
                DatasetUploadEncoding.filename == safe_name,
            )
        )
        row = DatasetUploadEncoding(
            user_sub=user_sub,
            filename=safe_name,
            encodings=dict(encodings),
            variable_layout=dict(variable_layout),
        )
        db.add(row)
        await db.flush()
        return str(row.id)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not persist upload encodings: {}", e)
        return None


def variable_layout_payload(
    unified_meta: UnifiedSheetMeta | None,
    *,
    upload_input_cols: list[str],
    upload_output_cols: list[str],
) -> dict[str, Any]:
    if unified_meta is not None:
        return {
            "kind": "unified",
            "metadata_cols": list(unified_meta.metadata_cols),
            "input_cols": list(unified_meta.input_cols),
            "output_cols": list(unified_meta.output_cols),
            "role_row_index": unified_meta.role_row_index,
            "name_row_index": unified_meta.name_row_index,
            "data_start_row_index": unified_meta.data_start_row_index,
            "experiment_id_col": unified_meta.experiment_id_col,
            "time_col": unified_meta.time_col,
        }
    return {
        "kind": "legacy_slices",
        "input_cols": upload_input_cols,
        "output_cols": upload_output_cols,
    }
