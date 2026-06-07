"""
Semantic Scholar Client.
Searches Semantic Scholar for open-access papers.
Free tier available — optional API key for higher rate limits.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

import requests

from src.utils.logging import get_logger

logger = get_logger(__name__)

BASE_URL = "https://api.semanticscholar.org/graph/v1"
RATE_LIMIT_SECONDS = 1.0

# Process-wide rate limiter — unauthenticated S2 limit is ~100 req/5 min.
# Enforcing 1.5s between calls keeps us safely under.
_s2_lock = threading.Lock()
_s2_last_call: float = 0.0
_S2_MIN_INTERVAL = 1.5


def _s2_rate_limit() -> None:
    global _s2_last_call
    with _s2_lock:
        gap = _S2_MIN_INTERVAL - (time.monotonic() - _s2_last_call)
        if gap > 0:
            time.sleep(gap)
        _s2_last_call = time.monotonic()


# ── Output schema ─────────────────────────────────────────────────────────────


@dataclass
class S2Paper:
    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    year: int | None
    citation_count: int
    url: str
    open_access_url: str | None
    fields_of_study: list[str]


# ── Client ────────────────────────────────────────────────────────────────────


class SemanticScholarClient:
    """
    Searches Semantic Scholar API for relevant papers.
    Uses open-access filter to stay within the spirit of the project.
    """

    FIELDS = "paperId,title,authors,abstract,year,citationCount,externalIds,openAccessPdf,fieldsOfStudy"

    def __init__(self) -> None:
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
        self._headers: dict[str, str] = {}
        if api_key:
            self._headers["x-api-key"] = api_key
            logger.debug("Semantic Scholar: using API key")
        else:
            logger.debug("Semantic Scholar: no API key, rate limits apply")

    def search(self, query: str, max_results: int = 10) -> list[S2Paper]:
        """Search Semantic Scholar for papers matching query."""
        logger.debug(f"Semantic Scholar search: '{query[:80]}'")
        _s2_rate_limit()
        try:
            response = requests.get(
                f"{BASE_URL}/paper/search",
                params={
                    "query": query,
                    "limit": str(max_results),
                    "fields": self.FIELDS,
                },
                headers=self._headers,
                timeout=30,
            )

            if response.status_code == 429:
                for wait in (5, 10, 15):
                    logger.warning(
                        f"Semantic Scholar rate limited — retrying in {wait}s"
                    )
                    time.sleep(wait)
                    response = requests.get(
                        f"{BASE_URL}/paper/search",
                        params={
                            "query": query,
                            "limit": str(max_results),
                            "fields": self.FIELDS,
                        },
                        headers=self._headers,
                        timeout=30,
                    )
                    if response.status_code != 429:
                        break
                if response.status_code != 200:
                    logger.warning(
                        f"Semantic Scholar returned {response.status_code} after retries"
                    )
                    return []

            if response.status_code != 200:
                logger.warning(f"Semantic Scholar returned {response.status_code}")
                return []

            data = response.json()
            papers = [
                self._to_paper(p)
                for p in data.get("data", [])
                if p.get("abstract")  # Skip papers without abstracts
            ]
            logger.debug(f"Semantic Scholar: {len(papers)} results")
            return papers

        except Exception as e:
            logger.warning(f"Semantic Scholar search failed: {e}")
            return []

    def search_for_finding(
        self,
        significant_variables: list[str],
        domain_context: str,
        max_results: int = 10,
    ) -> list[S2Paper]:
        """Search using domain context and significant variables."""
        query = self._build_query(significant_variables, domain_context)
        papers = self.search(query, max_results=max_results)
        time.sleep(RATE_LIMIT_SECONDS)
        return papers

    def _build_query(
        self, significant_variables: list[str], domain_context: str
    ) -> str:
        var_str = " ".join(significant_variables[:3])
        return f"{domain_context} {var_str}".strip()

    def _to_paper(self, data: dict) -> S2Paper:
        authors = [a.get("name", "") for a in data.get("authors", [])[:5]]
        open_access = data.get("openAccessPdf")
        open_url = open_access.get("url") if isinstance(open_access, dict) else None
        ext_ids = data.get("externalIds") or {}
        url = f"https://www.semanticscholar.org/paper/{data.get('paperId', '')}"
        if "ArXiv" in ext_ids:
            url = f"https://arxiv.org/abs/{ext_ids['ArXiv']}"

        return S2Paper(
            paper_id=str(data.get("paperId", "")),
            title=str(data.get("title", "")),
            authors=authors,
            abstract=str(data.get("abstract", "")),
            year=data.get("year"),
            citation_count=int(data.get("citationCount", 0)),
            url=url,
            open_access_url=open_url,
            fields_of_study=data.get("fieldsOfStudy") or [],
        )
