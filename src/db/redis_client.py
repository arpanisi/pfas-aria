"""
Redis client.
Used for arXiv and Semantic Scholar response caching with TTL.
Prevents redundant API calls across pipeline runs.

Key patterns:
  arxiv:{arxiv_id}     — cached arXiv paper JSON, TTL=30 days
  s2:{paper_id}        — cached S2 paper JSON, TTL=30 days
  query:{hash}         — cached search results for a query, TTL=7 days
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import redis.asyncio as aioredis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

ARXIV_TTL = 60 * 60 * 24 * 30  # 30 days
S2_TTL = 60 * 60 * 24 * 30  # 30 days
QUERY_TTL = 60 * 60 * 24 * 7  # 7 days

_client: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    """Return the singleton Redis client."""
    global _client
    if _client is None:
        _client = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _client


# ── arXiv cache ───────────────────────────────────────────────────────────────


async def get_arxiv_paper(arxiv_id: str) -> dict | None:
    """Retrieve a cached arXiv paper by ID. Returns None if not cached."""
    r = get_redis()
    data = await r.get(f"arxiv:{arxiv_id}")
    return json.loads(data) if data else None


async def set_arxiv_paper(arxiv_id: str, paper: dict) -> None:
    """Cache an arXiv paper with TTL."""
    r = get_redis()
    await r.setex(f"arxiv:{arxiv_id}", ARXIV_TTL, json.dumps(paper))


async def get_arxiv_query(query: str) -> list[dict] | None:
    """Retrieve cached arXiv search results for a query."""
    r = get_redis()
    key = f"arxiv_query:{_hash(query)}"
    data = await r.get(key)
    return json.loads(data) if data else None


async def set_arxiv_query(query: str, results: list[dict]) -> None:
    """Cache arXiv search results for a query."""
    r = get_redis()
    key = f"arxiv_query:{_hash(query)}"
    await r.setex(key, QUERY_TTL, json.dumps(results))


# ── Semantic Scholar cache ────────────────────────────────────────────────────


async def get_s2_paper(paper_id: str) -> dict | None:
    """Retrieve a cached S2 paper by ID."""
    r = get_redis()
    data = await r.get(f"s2:{paper_id}")
    return json.loads(data) if data else None


async def set_s2_paper(paper_id: str, paper: dict) -> None:
    """Cache a Semantic Scholar paper with TTL."""
    r = get_redis()
    await r.setex(f"s2:{paper_id}", S2_TTL, json.dumps(paper))


async def get_s2_query(query: str) -> list[dict] | None:
    """Retrieve cached S2 search results."""
    r = get_redis()
    key = f"s2_query:{_hash(query)}"
    data = await r.get(key)
    return json.loads(data) if data else None


async def set_s2_query(query: str, results: list[dict]) -> None:
    """Cache S2 search results."""
    r = get_redis()
    key = f"s2_query:{_hash(query)}"
    await r.setex(key, QUERY_TTL, json.dumps(results))


# ── Pipeline status cache ─────────────────────────────────────────────────────


async def set_run_status(run_id: str, status: dict) -> None:
    """Cache live run status for WebSocket polling."""
    r = get_redis()
    await r.setex(f"run:{run_id}:status", 3600, json.dumps(status))


async def get_run_status(run_id: str) -> dict | None:
    """Get live run status from cache."""
    r = get_redis()
    data = await r.get(f"run:{run_id}:status")
    return json.loads(data) if data else None


async def delete_run_status(run_id: str) -> None:
    """Remove cached status when a run is deleted."""
    r = get_redis()
    await r.delete(f"run:{run_id}:status")


# ── Utilities ─────────────────────────────────────────────────────────────────


def _hash(text: str) -> str:
    """Short hash for cache keys."""
    return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:16]


async def ping() -> bool:
    """Check Redis connectivity."""
    try:
        r = get_redis()
        return bool(await r.ping())
    except Exception:
        return False


async def flush_literature_cache() -> int:
    """Clear all cached literature results. Returns number of keys deleted."""
    r = get_redis()
    keys = []
    async for key in r.scan_iter("arxiv:*"):
        keys.append(key)
    async for key in r.scan_iter("s2:*"):
        keys.append(key)
    async for key in r.scan_iter("arxiv_query:*"):
        keys.append(key)
    async for key in r.scan_iter("s2_query:*"):
        keys.append(key)
    if keys:
        await r.delete(*keys)
    return len(keys)


# ── DB query cache ────────────────────────────────────────────────────────────

DB_QUERY_TTL = 60 * 5  # 5 minutes — short TTL, data changes after each round


async def get_db_query(cache_key: str) -> Any | None:
    """Retrieve a cached DB query result."""
    r = get_redis()
    data = await r.get(f"db:{cache_key}")
    return json.loads(data) if data else None


async def set_db_query(cache_key: str, result: Any, ttl: int = DB_QUERY_TTL) -> None:
    """Cache a DB query result with TTL."""
    r = get_redis()
    await r.setex(f"db:{cache_key}", ttl, json.dumps(result))


async def invalidate_db_cache(run_id: str) -> None:
    """Invalidate all DB cache entries for a run. Call after each round."""
    r = get_redis()
    keys = []
    async for key in r.scan_iter(f"db:{run_id}:*"):
        keys.append(key)
    if keys:
        await r.delete(*keys)


# ── Grounding job state ───────────────────────────────────────────────────────

GROUNDING_JOB_TTL = 60 * 60 * 2  # 2 hours — survives a slow LLM enrichment pass


async def set_grounding_job(job_id: str, fields: dict) -> None:
    """Upsert grounding job state. Merges `fields` into the existing dict."""
    r = get_redis()
    key = f"grounding_job:{job_id}"
    existing_raw = await r.get(key)
    existing: dict = json.loads(existing_raw) if existing_raw else {}
    existing.update(fields)
    await r.setex(key, GROUNDING_JOB_TTL, json.dumps(existing))


async def get_grounding_job(job_id: str) -> dict | None:
    """Return grounding job state or None if not found / expired."""
    r = get_redis()
    data = await r.get(f"grounding_job:{job_id}")
    return json.loads(data) if data else None
