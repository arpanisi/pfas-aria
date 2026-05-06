"""
Literature Cache.
Redis-backed cache for arXiv and Semantic Scholar API responses.
Prevents redundant API calls across pipeline runs.

Falls back gracefully if Redis is unavailable — pipeline still works,
just without caching (live API calls every time).
"""

from __future__ import annotations

from src.utils.logging import get_logger

logger = get_logger(__name__)

ARXIV_TTL_DAYS = 30
S2_TTL_DAYS = 30
QUERY_TTL_DAYS = 7


class LiteratureCache:
    """
    Unified cache interface for all literature API responses.
    Wraps Redis with graceful degradation if Redis is unavailable.
    """

    def __init__(self) -> None:
        self._available = False
        self._client = None
        self._connect()

    def _connect(self) -> None:
        try:
            import asyncio

            from src.db.redis_client import ping

            # Test connection synchronously during init
            async def _test() -> bool:
                return await ping()

            loop = asyncio.new_event_loop()
            self._available = loop.run_until_complete(_test())
            loop.close()

            if self._available:
                logger.info("Literature cache: Redis connected")
            else:
                logger.warning(
                    "Literature cache: Redis unavailable — running without cache"
                )
        except Exception as e:
            logger.warning(f"Literature cache: Redis init failed ({e}) — no cache")
            self._available = False

    # ── arXiv ─────────────────────────────────────────────────────────────────

    async def get_arxiv(self, arxiv_id: str) -> dict | None:
        if not self._available:
            return None
        try:
            from src.db.redis_client import get_arxiv_paper

            return await get_arxiv_paper(arxiv_id)
        except Exception:
            return None

    async def set_arxiv(self, arxiv_id: str, paper: dict) -> None:
        if not self._available:
            return
        try:
            from src.db.redis_client import set_arxiv_paper

            await set_arxiv_paper(arxiv_id, paper)
        except Exception:
            pass

    async def get_arxiv_query(self, query: str) -> list[dict] | None:
        if not self._available:
            return None
        try:
            from src.db.redis_client import get_arxiv_query

            return await get_arxiv_query(query)
        except Exception:
            return None

    async def set_arxiv_query(self, query: str, results: list[dict]) -> None:
        if not self._available:
            return
        try:
            from src.db.redis_client import set_arxiv_query

            await set_arxiv_query(query, results)
        except Exception:
            pass

    # ── Semantic Scholar ──────────────────────────────────────────────────────

    async def get_s2(self, paper_id: str) -> dict | None:
        if not self._available:
            return None
        try:
            from src.db.redis_client import get_s2_paper

            return await get_s2_paper(paper_id)
        except Exception:
            return None

    async def set_s2(self, paper_id: str, paper: dict) -> None:
        if not self._available:
            return
        try:
            from src.db.redis_client import set_s2_paper

            await set_s2_paper(paper_id, paper)
        except Exception:
            pass

    async def get_s2_query(self, query: str) -> list[dict] | None:
        if not self._available:
            return None
        try:
            from src.db.redis_client import get_s2_query

            return await get_s2_query(query)
        except Exception:
            return None

    async def set_s2_query(self, query: str, results: list[dict]) -> None:
        if not self._available:
            return
        try:
            from src.db.redis_client import set_s2_query

            await set_s2_query(query, results)
        except Exception:
            pass

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        return self._available

    async def flush(self) -> int:
        """Clear all cached literature. Returns number of keys deleted."""
        if not self._available:
            return 0
        try:
            from src.db.redis_client import flush_literature_cache

            return await flush_literature_cache()
        except Exception:
            return 0
