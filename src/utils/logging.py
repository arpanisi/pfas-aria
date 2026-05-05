"""
Logging setup using Loguru.
Every module imports `get_logger(__name__)` — that's it.
All logs are structured, timestamped, and written to file + stdout.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger as _loguru_logger

from src.utils.config import get_settings

_configured = False


def setup_logging() -> None:
    """Configure Loguru. Call once at application startup."""
    global _configured
    if _configured:
        return

    settings = get_settings()
    log_cfg = settings.pipeline.logging

    # Remove default handler
    _loguru_logger.remove()

    # Stdout handler
    _loguru_logger.add(
        sys.stdout,
        level=log_cfg.level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
            "{message}"
        ),
    )

    # File handler
    if log_cfg.log_to_file:
        log_dir = Path(log_cfg.log_directory)
        log_dir.mkdir(parents=True, exist_ok=True)
        _loguru_logger.add(
            log_dir / "pfas_aria_{time:YYYY-MM-DD}.log",
            level=log_cfg.level,
            rotation=log_cfg.rotation,
            retention=log_cfg.retention,
            compression="zip",
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}"
            ),
            serialize=False,
        )

    _configured = True


def get_logger(name: str):
    """Return a module-level logger bound to the given name.

    Usage:
        from src.utils.logging import get_logger
        logger = get_logger(__name__)
    """
    setup_logging()
    return _loguru_logger.bind(name=name)
