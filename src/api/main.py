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

from src.api.routes import corpus, health, pipeline, results
from src.api.websocket import router as ws_router
from src.utils.logging import get_logger, setup_logging
from src.utils.paths import ensure_dirs

logger = get_logger(__name__)

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
AUTO_MIGRATE = os.getenv("AUTO_MIGRATE", "true").lower() in ("1", "true", "yes")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle."""
    setup_logging()
    ensure_dirs()
    logger.info(f"PFAS-ARIA API starting — {ENVIRONMENT}")

    # Ensure DB tables exist (gated by AUTO_MIGRATE; disable in prod if using Alembic)
    if AUTO_MIGRATE:
        try:
            from src.db.postgres import create_all_tables, create_materialized_views

            await create_all_tables()
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
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

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
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
)

# ── Routes ────────────────────────────────────────────────────────────────────

app.include_router(health.router)
app.include_router(pipeline.router)
app.include_router(results.router)
app.include_router(corpus.router)
app.include_router(ws_router)
