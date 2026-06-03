"""
FastAPI Application — PFAS-ARIA Backend.
Entry point for the full backend API.

Architecture:
  - Auth: Clerk JWT on all protected routes
  - DB: PostgreSQL (SQLAlchemy async) + MongoDB (Motor) + Redis
  - Routes: pipeline, results, corpus, health
  - WebSocket: real-time pipeline status
  - Docs: auto-generated at /docs
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.api.routes import corpus, health, pipeline, results
from src.api.websocket import router as ws_router
from src.utils.logging import get_logger, setup_logging
from src.utils.paths import ensure_dirs

logger = get_logger(__name__)

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
AUTO_MIGRATE = os.getenv("AUTO_MIGRATE", "true").lower() in ("1", "true", "yes")
IS_PRODUCTION = ENVIRONMENT.lower() == "production"

SECURITY_HEADERS = {
    "Content-Security-Policy": os.getenv(
        "SECURITY_CSP",
        "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": (
        "camera=(), microphone=(), geolocation=(), payment=(), usb=(), "
        "accelerometer=(), gyroscope=(), magnetometer=()"
    ),
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle."""
    setup_logging()
    ensure_dirs()
    logger.info(f"PFAS-ARIA API starting — {ENVIRONMENT}")

    # Ensure DB tables exist (gated by AUTO_MIGRATE; disable in prod if using Alembic)
    if AUTO_MIGRATE:
        try:
            from src.db.postgres import (
                create_all_tables,
                create_materialized_views,
                ensure_tenant_isolation_schema,
            )

            await create_all_tables()
            await ensure_tenant_isolation_schema()
            await create_materialized_views()
            logger.info("PostgreSQL tables and materialized views ready")
        except Exception as e:
            logger.warning(f"DB setup warning: {e}")

    # Ensure MongoDB indexes
    try:
        from src.db.mongodb import ensure_indexes

        await ensure_indexes()
        logger.info("MongoDB indexes ready")
    except Exception as e:
        logger.warning(f"MongoDB setup warning: {e}")

    yield
    logger.info("PFAS-ARIA API shutting down")


app = FastAPI(
    title="PFAS-ARIA API",
    description="Autonomous Research Intelligence Agent — PFAS Degradation Analysis",
    version="0.1.0",
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None if IS_PRODUCTION else "/redoc",
    openapi_url=None if IS_PRODUCTION else "/openapi.json",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next) -> Response:
    """Attach a conservative browser security baseline to every API response."""
    response: Response = await call_next(request)
    for header, value in SECURITY_HEADERS.items():
        response.headers.setdefault(header, value)
    if IS_PRODUCTION:
        response.headers.setdefault(
            "Strict-Transport-Security",
            "max-age=31536000; includeSubDomains; preload",
        )
    return response


# ── CORS ──────────────────────────────────────────────────────────────────────
# FRONTEND_URL is set per-platform in environment variables (e.g. render.yaml).
# Security is handled by Clerk JWT, not origin filtering.

origins = [FRONTEND_URL.rstrip("/")]
if ENVIRONMENT == "development":
    origins += ["http://localhost:3000", "http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
)

# ── Routes ────────────────────────────────────────────────────────────────────

app.include_router(health.router)
app.include_router(pipeline.router)
app.include_router(results.router)
app.include_router(corpus.router)
app.include_router(ws_router)
