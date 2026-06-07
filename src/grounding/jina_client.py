"""
Jina Search Client.
Replaces the direct arXiv API for literature discovery — searches the open web
and returns results from journals, PMC, EPA, ACS, etc. instead of arXiv only.
Returns ArxivPaper objects so the rest of the grounding pipeline is unchanged.
"""

from __future__ import annotations

import os

import httpx

from src.grounding.arxiv_client import ArxivPaper
from src.utils.logging import get_logger

logger = get_logger(__name__)

_JINA_SEARCH_URL = "https://s.jina.ai/"


class JinaSearchClient:
    def __init__(self, max_results: int = 8, timeout: int = 30) -> None:
        self.max_results = max_results
        self.timeout = timeout
        api_key = os.getenv("JINA_API_KEY", "").strip()
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }
        self._enabled = bool(api_key)
        if not self._enabled:
            logger.warning("JINA_API_KEY not set — Jina search disabled")

    def search_for_finding(
        self,
        finding: str,
        significant_variables: list[str],
        domain_context: str,
    ) -> list[ArxivPaper]:
        var_str = " ".join(significant_variables[:3])
        query = f"{domain_context} {var_str} {finding}"[:200].strip()
        return self.search(query)

    def search(self, query: str) -> list[ArxivPaper]:
        if not self._enabled:
            return []
        logger.debug("Jina search: '{}'", query[:80])
        try:
            resp = httpx.get(
                f"{_JINA_SEARCH_URL}{query}",
                headers=self._headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return self._parse(resp.json())
        except Exception as e:
            logger.warning("Jina search failed: {}", e)
            return []

    def _parse(self, data: dict) -> list[ArxivPaper]:
        papers: list[ArxivPaper] = []
        for item in (data.get("data") or [])[:self.max_results]:
            title = (item.get("title") or "").strip()
            abstract = (item.get("description") or item.get("content") or "").strip()[:500]
            url = item.get("url") or ""
            published = (item.get("publishedTime") or "")[:10]
            if not title:
                continue
            papers.append(
                ArxivPaper(
                    arxiv_id="",
                    title=title,
                    authors=[],
                    abstract=abstract,
                    url=url,
                    published=published,
                    categories=[],
                )
            )
        return papers
