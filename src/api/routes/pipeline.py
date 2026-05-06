"""
Pipeline Routes.
Endpoints for uploading data, configuring and triggering pipeline runs,
and checking run status.
"""

from __future__ import annotations

import asyncio
import io
import uuid

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import CurrentUser
from src.api.database import get_db
from src.api.models.orm import Run
from src.utils.logging import get_logger
from src.utils.paths import RAW_DIR

logger = get_logger(__name__)
router = APIRouter(prefix="/pipeline", tags=["Pipeline"])

# Track active runs in memory (for real-time status)
_active_runs: dict[str, str] = {}  # run_id -> status


# ── Request / Response schemas ────────────────────────────────────────────────


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    n_unique: int
    missing_pct: float
    sample_values: list


class DatasetPreview(BaseModel):
    filename: str
    n_rows: int
    n_cols: int
    columns: list[ColumnInfo]


class RunConfig(BaseModel):
    run_name: str
    outcome_variable: str
    feature_columns: list[str]
    exclude_columns: list[str] = []
    max_rounds: int = 10
    convergence_threshold: float = 0.75
    hypotheses_per_round: int = 6


class RunStatus(BaseModel):
    run_id: str
    run_name: str
    status: str
    current_round: int
    final_match_score: float
    converged: bool
    n_rounds_completed: int


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/upload", response_model=DatasetPreview)
async def upload_dataset(
    user: CurrentUser,
    file: UploadFile = File(...),
) -> DatasetPreview:
    """
    Upload a dataset file (CSV or Excel).
    Returns column metadata for the frontend column selector.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided",
        )

    suffix = file.filename.rsplit(".", 1)[-1].lower()
    if suffix not in {"csv", "xlsx", "xls", "tsv"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {suffix}. Use CSV or Excel.",
        )

    content = await file.read()

    try:
        if suffix == "csv":
            df = pd.read_csv(io.BytesIO(content))
        elif suffix == "tsv":
            df = pd.read_csv(io.BytesIO(content), sep="\t")
        else:
            df = pd.read_excel(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not parse file: {e}",
        ) from e

    # Save to data/raw for pipeline access
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    save_path = RAW_DIR / file.filename
    save_path.write_bytes(content)

    columns = []
    for col in df.columns:
        series = df[col]
        columns.append(
            ColumnInfo(
                name=str(col),
                dtype=str(series.dtype),
                n_unique=int(series.nunique()),
                missing_pct=round(float(series.isnull().mean() * 100), 2),
                sample_values=series.dropna().unique()[:5].tolist(),
            )
        )

    logger.info(f"Dataset uploaded: {file.filename} ({len(df)} rows)")

    return DatasetPreview(
        filename=file.filename,
        n_rows=len(df),
        n_cols=len(df.columns),
        columns=columns,
    )


@router.post("/run", status_code=status.HTTP_202_ACCEPTED)
async def start_run(
    config: RunConfig,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    """
    Start a pipeline run with the selected configuration.
    Returns a run_id immediately — poll /status/{run_id} for updates.
    """
    run_id = str(uuid.uuid4())[:8]

    # Persist run record
    run = Run(
        id=run_id,
        run_name=config.run_name,
        status="initializing",
        outcome_variable=config.outcome_variable,
        selected_features=config.feature_columns,
    )
    db.add(run)
    await db.flush()

    _active_runs[run_id] = "initializing"

    # Launch pipeline in background
    asyncio.create_task(_run_pipeline_async(run_id, config))

    logger.info(f"Run {run_id} started: {config.run_name}")

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={"run_id": run_id, "status": "initializing"},
    )


@router.get("/status/{run_id}", response_model=RunStatus)
async def get_run_status(
    run_id: str,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> RunStatus:
    """Get current status of a pipeline run."""
    run = await db.get(Run, run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

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
    db: AsyncSession = Depends(get_db),
) -> list[RunStatus]:
    """List all pipeline runs."""
    from sqlalchemy import select

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


async def _run_pipeline_async(run_id: str, config: RunConfig) -> None:
    """
    Run the pipeline in a background asyncio task.
    Updates the DB run record at each stage.
    """
    from src.orchestration.pipeline import run_pipeline

    try:
        _active_runs[run_id] = "running"
        await _update_run_status(run_id, "running")

        # Override config from UI selections
        _inject_run_config(config)

        # Run in thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        state, report = await loop.run_in_executor(
            None,
            lambda: run_pipeline(run_name=config.run_name),
        )

        await _update_run_status(
            run_id=run_id,
            status="converged" if report.converged else "completed",
            final_match_score=report.final_score,
            converged=report.converged,
            n_rounds=report.total_rounds,
        )
        _active_runs[run_id] = "completed"

    except Exception as e:
        logger.error(f"Run {run_id} failed: {e}")
        await _update_run_status(run_id, "failed", error_message=str(e))
        _active_runs[run_id] = "failed"


async def _update_run_status(
    run_id: str,
    status: str,
    final_match_score: float = 0.0,
    converged: bool = False,
    n_rounds: int = 0,
    error_message: str | None = None,
) -> None:
    """Update the run record in the database."""
    from src.api.database import AsyncSessionFactory

    async with AsyncSessionFactory() as db:
        run = await db.get(Run, run_id)
        if run:
            run.status = status
            run.final_match_score = final_match_score
            run.converged = converged
            run.n_rounds_completed = n_rounds
            if error_message:
                run.error_message = error_message
            await db.commit()


def _inject_run_config(config: RunConfig) -> None:
    """
    Temporarily override config settings from UI selections.
    In production this would write to a run-specific config file.
    """
    import os

    os.environ["ARIA_OUTCOME_VARIABLE"] = config.outcome_variable
    os.environ["ARIA_MAX_ROUNDS"] = str(config.max_rounds)
    os.environ["ARIA_CONVERGENCE_THRESHOLD"] = str(config.convergence_threshold)
    os.environ["ARIA_HYPOTHESES_PER_ROUND"] = str(config.hypotheses_per_round)
