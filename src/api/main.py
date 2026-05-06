"""
FastAPI Application.
Main entry point for the PFAS-ARIA backend API.

Stages:
  - CORS configured for React frontend
  - Clerk JWT auth on all protected routes
  - REST endpoints for pipeline, results, reports
  - WebSocket for real-time status
  - Auto-generated docs at /docs
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import pipeline, results
from src.api.websocket import router as ws_router
from src.utils.logging import get_logger, setup_logging
from src.utils.paths import ensure_dirs

logger = get_logger(__name__)

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown logic."""
    setup_logging()
    ensure_dirs()
    logger.info(f"PFAS-ARIA API starting — environment: {ENVIRONMENT}")
    yield
    logger.info("PFAS-ARIA API shutting down")


app = FastAPI(
    title="PFAS-ARIA API",
    description="Autonomous Research Intelligence Agent for PFAS Degradation Analysis",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────

allowed_origins = [FRONTEND_URL]
if ENVIRONMENT == "development":
    allowed_origins += ["http://localhost:3000", "http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────

app.include_router(pipeline.router)
app.include_router(results.router)
app.include_router(ws_router)


# ── Health check ──────────────────────────────────────────────────────────────


@app.get("/", tags=["Health"])
async def root() -> dict:
    return {"status": "ok", "service": "pfas-aria-api"}


@app.get("/health", tags=["Health"])
async def health() -> dict:
    return {
        "status": "healthy",
        "environment": ENVIRONMENT,
        "version": "0.1.0",
    }
