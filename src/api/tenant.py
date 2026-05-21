"""Tenant helpers for per-user data isolation."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


def current_user_sub(user: dict) -> str:
    """Return the authenticated Clerk subject used for ownership checks."""
    sub = str(user.get("sub") or "").strip()
    return sub or "anonymous"


def tenant_storage_key(user_sub: str) -> str:
    """Filesystem-safe, non-reversible key for user-scoped local storage."""
    digest = hashlib.sha256(user_sub.encode()).hexdigest()[:24]
    return f"user_{digest}"


def safe_upload_filename(filename: str) -> str:
    """Normalize an uploaded filename without allowing path traversal."""
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._()\- ]+", "_", name).strip()
    return name or "upload"
