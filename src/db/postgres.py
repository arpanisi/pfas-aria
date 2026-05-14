"""
PostgreSQL client.
Async SQLAlchemy engine shared across the ETL layer and API.
Stores: experimental data versions, paper registry, pipeline outputs,
        run metadata, validation results, arXiv/S2 cache metadata.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/pfas_aria",
)

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionFactory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def get_session() -> AsyncGenerator[AsyncSession, None]:  # type: ignore[override]
    """Yield an async DB session. Use as FastAPI dependency or async context."""
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_all_tables() -> None:
    """Create all tables. Called at startup if running without Alembic."""
    import src.db.orm  # noqa: F401 — register all models on Base.metadata

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def create_materialized_views() -> None:
    """
    Create PostgreSQL materialized views for dashboard aggregations.
    These are pre-computed and refreshed after each pipeline round.

    Views:
      mv_run_convergence   — convergence score history per run
      mv_top_variables     — most significant variables across all runs
      mv_top_citations     — highest-scoring citations across all runs
    """
    views = [
        # Convergence history per run — feeds the convergence chart
        """
        CREATE MATERIALIZED VIEW IF NOT EXISTS mv_run_convergence AS
        SELECT
            h.run_id,
            h.round,
            AVG(mr.match_score) AS avg_match_score,
            MAX(mr.r_squared)   AS best_r_squared,
            COUNT(h.id)         AS n_hypotheses,
            SUM(CASE WHEN vr.overall_passed THEN 1 ELSE 0 END) AS n_passed
        FROM hypotheses h
        JOIN model_results mr ON mr.hypothesis_id = h.id
        LEFT JOIN validation_results vr ON vr.model_result_id = mr.id
        GROUP BY h.run_id, h.round
        ORDER BY h.run_id, h.round
        WITH NO DATA;
        """,
        # Index on materialized view
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ix_mv_run_convergence
        ON mv_run_convergence (run_id, round);
        """,
        # Top variables across all runs — feeds the variable importance panel
        """
        CREATE MATERIALIZED VIEW IF NOT EXISTS mv_top_variables AS
        SELECT
            var_name,
            COUNT(*)            AS frequency,
            AVG(mr.match_score) AS avg_match_score,
            AVG(mr.r_squared)   AS avg_r_squared
        FROM model_results mr,
             json_array_elements_text(mr.significant_variables::json) AS var_name
        GROUP BY var_name
        ORDER BY frequency DESC
        WITH NO DATA;
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ix_mv_top_variables
        ON mv_top_variables (var_name);
        """,
        # Top citations across all runs — feeds the citations panel
        """
        CREATE MATERIALIZED VIEW IF NOT EXISTS mv_top_citations AS
        SELECT
            c.title,
            c.url,
            c.source,
            c.year,
            COUNT(*)              AS citation_count,
            AVG(c.similarity_score) AS avg_similarity
        FROM citations c
        GROUP BY c.title, c.url, c.source, c.year
        ORDER BY avg_similarity DESC
        WITH NO DATA;
        """,
    ]

    async with engine.begin() as conn:
        for sql in views:
            try:
                await conn.execute(__import__("sqlalchemy").text(sql.strip()))
            except Exception:
                # Views may already exist — not an error
                pass


async def refresh_materialized_views() -> None:
    """
    Refresh all materialized views concurrently.
    Call after each pipeline round completes.
    CONCURRENTLY means the dashboard can still read stale data while refresh runs.
    """
    from sqlalchemy import text

    view_names = [
        "mv_run_convergence",
        "mv_top_variables",
        "mv_top_citations",
    ]

    async with engine.begin() as conn:
        for view in view_names:
            try:
                await conn.execute(
                    text(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view}")
                )
            except Exception:
                # Non-critical — dashboard falls back to live queries
                pass
