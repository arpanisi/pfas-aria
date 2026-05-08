"""
ARQ Task Queue Worker.
Replaces asyncio.create_task() for pipeline runs.

ARQ is Redis-backed — jobs survive server restarts.
Each pipeline run is a job in the queue.
Workers pull jobs and execute them.

Run worker: arq src.queue.worker.WorkerSettings
"""

from __future__ import annotations

import os
from typing import Any

from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


async def run_pipeline_job(
    ctx: dict[str, Any],
    run_id: str,
    run_name: str,
    filename: str,
    outcome_variable: str,
    feature_columns: list[str],
    exclude_columns: list[str],
    max_rounds: int,
    convergence_threshold: float,
    hypotheses_per_round: int,
    strict_validation: bool,
) -> dict[str, Any]:
    """
    ARQ job function — runs the full pipeline for one experiment run.
    Called by the ARQ worker process, not the FastAPI server.
    """
    setup_logging()
    logger.info(f"ARQ job starting: run_id={run_id}, run_name={run_name}")

    from src.db.postgres import AsyncSessionFactory
    from src.db.redis_client import set_run_status
    from src.utils.paths import RAW_DIR

    async def update_status(status: str, **kwargs: object) -> None:
        await set_run_status(
            run_id,
            {
                "run_id": run_id,
                "status": status,
                "run_name": run_name,
                **kwargs,
            },
        )
        async with AsyncSessionFactory() as session:
            from src.db.orm import Run

            run = await session.get(Run, run_id)
            if run:
                run.status = status
                for k, v in kwargs.items():
                    if hasattr(run, k):
                        setattr(run, k, v)
                await session.commit()

    try:
        await update_status("running", round=0, match_score=0.0)

        # Inject config into environment for pipeline to pick up
        os.environ["ARIA_OUTCOME_VARIABLE"] = outcome_variable
        os.environ["ARIA_DATA_FILE"] = str(RAW_DIR / filename)
        os.environ["ARIA_MAX_ROUNDS"] = str(max_rounds)
        os.environ["ARIA_CONVERGENCE_THRESHOLD"] = str(convergence_threshold)
        os.environ["ARIA_HYPOTHESES_PER_ROUND"] = str(hypotheses_per_round)

        state, report = await _run_pipeline_in_executor(run_name)

        final_status = "converged" if report.converged else "completed"
        await update_status(
            final_status,
            round=report.total_rounds,
            match_score=report.final_score,
            converged=report.converged,
            n_rounds_completed=report.total_rounds,
            stop_reason=report.stop_reason,
        )

        logger.info(f"ARQ job complete: run_id={run_id}, status={final_status}")
        return {
            "run_id": run_id,
            "status": final_status,
            "total_rounds": report.total_rounds,
            "final_score": report.final_score,
            "converged": report.converged,
        }

    except Exception as e:
        logger.error(f"ARQ job failed: run_id={run_id}, error={e}")
        await update_status("failed", error_message=str(e))
        raise


async def _run_pipeline_in_executor(run_name: str) -> tuple:
    """Run synchronous pipeline in a thread pool."""
    import asyncio

    from src.orchestration.pipeline import run_pipeline

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: run_pipeline(run_name=run_name))


def get_redis_settings() -> Any:

    from arq.connections import RedisSettings
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    return RedisSettings.from_dsn(redis_url)


class WorkerSettings:
    """ARQ worker configuration."""

    functions = [run_pipeline_job]
    redis_settings = get_redis_settings()  # set from REDIS_URL env var at startup
    max_jobs = 4  # concurrent pipeline runs
    job_timeout = 7200  # 2 hours max per run
    keep_result = 3600  # keep result for 1 hour
    retry_jobs = True
    max_tries = 2  # retry once on failure

    @classmethod
    def get_redis_settings(cls) -> Any:
        from arq.connections import RedisSettings

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        return RedisSettings.from_dsn(redis_url)
