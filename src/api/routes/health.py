"""
Health Routes.
Liveness and readiness checks for Render deployment and monitoring.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/")
async def root() -> dict:
    return {"status": "ok", "service": "pfas-aria-api", "version": "0.1.0"}


@router.get("/health")
async def health() -> dict:
    """Liveness probe — just confirms the process is alive."""
    return {"status": "healthy"}


@router.get("/ready")
async def readiness() -> dict:
    """
    Readiness probe — checks all backing services are reachable.
    Returns 200 only if PostgreSQL, MongoDB, and Redis are all up.
    """
    import sqlalchemy

    checks: dict[str, str] = {}

    # PostgreSQL
    try:
        from src.db.postgres import engine

        async with engine.connect() as conn:
            await conn.execute(sqlalchemy.text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"error: {e}"

    # MongoDB
    try:
        from src.db.mongodb import get_mongo_client

        await get_mongo_client().admin.command("ping")
        checks["mongodb"] = "ok"
    except Exception as e:
        checks["mongodb"] = f"error: {e}"

    # Redis
    try:
        from src.db.redis_client import ping

        ok = await ping()
        checks["redis"] = "ok" if ok else "error: ping failed"
    except Exception as e:
        checks["redis"] = f"error: {e}"

    all_ok = all(v == "ok" for v in checks.values())
    return {
        "status": "ready" if all_ok else "degraded",
        "checks": checks,
    }
