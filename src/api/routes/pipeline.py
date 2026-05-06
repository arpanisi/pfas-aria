"""
Pipeline Routes.
Upload dataset, configure run, trigger pipeline, stream status.
Replaces the old pipeline.py — now uses the full ETL layer.
"""

from __future__ import annotations

import asyncio
import io
import uuid

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import select

from src.api.deps import CurrentUser, DBSession
from src.db.orm import Run
from src.db.redis_client import get_run_status, set_run_status
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


class RunStatus(BaseModel):
    run_id: str
    run_name: str
    status: str
    current_round: int
    final_match_score: float
    converged: bool
    n_rounds_completed: int


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

    try:
        if suffix == "csv":
            df = pd.read_csv(io.BytesIO(content), low_memory=False)
        elif suffix == "tsv":
            df = pd.read_csv(io.BytesIO(content), sep="\t", low_memory=False)
        else:
            df = pd.read_excel(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, f"Cannot parse file: {e}"
        ) from e

    # Standardize column names
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

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
    )


# ── Start run ─────────────────────────────────────────────────────────────────


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
    # Try Redis cache first (live updates)
    cached = await get_run_status(run_id)
    if cached:
        return RunStatus(
            run_id=run_id,
            run_name=cached.get("run_name", ""),
            status=cached.get("status", "unknown"),
            current_round=cached.get("round", 0),
            final_match_score=cached.get("match_score", 0.0),
            converged=cached.get("converged", False),
            n_rounds_completed=cached.get("round", 0),
        )

    run = await db.get(Run, run_id)
    if not run:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Run {run_id} not found")

    return RunStatus(
        run_id=run.id,
        run_name=run.run_name,
        status=run.status,
        current_round=run.n_rounds_completed,
        final_match_score=run.final_match_score,
        converged=run.converged,
        n_rounds_completed=run.n_rounds_completed,
    )


@router.get("/runs", response_model=list[RunStatus])
async def list_runs(
    user: CurrentUser,
    db: DBSession,
) -> list[RunStatus]:
    """List all pipeline runs, most recent first."""
    result = await db.execute(select(Run).order_by(Run.created_at.desc()).limit(20))
    runs = result.scalars().all()

    return [
        RunStatus(
            run_id=r.id,
            run_name=r.run_name,
            status=r.status,
            current_round=r.n_rounds_completed,
            final_match_score=r.final_match_score,
            converged=r.converged,
            n_rounds_completed=r.n_rounds_completed,
        )
        for r in runs
    ]


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
