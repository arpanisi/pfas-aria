"""Crossref metadata client."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

import requests

from src.utils.logging import get_logger

logger = get_logger(__name__)

BASE_URL = "https://api.crossref.org/works"
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class CrossrefWork:
    doi: str
    title: str
    authors: list[str]
    abstract: str
    year: int | None
    url: str
    citation_count: int
    venue: str | None


class CrossrefClient:
    """Search Crossref works for DOI-backed journal/preprint metadata."""

    def __init__(self, timeout: int = 8) -> None:
        self.timeout = timeout
        self.mailto = os.getenv("CROSSREF_MAILTO", "").strip()
        self.api_key = os.getenv("CROSSREF_API_KEY", "").strip()

    def search(self, query: str, max_results: int = 10) -> list[CrossrefWork]:
        try:
            params: dict[str, Any] = {
                "query.bibliographic": query,
                "rows": max_results,
                "select": (
                    "DOI,title,author,abstract,published-print,published-online,"
                    "published,URL,is-referenced-by-count,container-title,type"
                ),
            }
            if self.mailto:
                params["mailto"] = self.mailto
            headers = {}
            if self.api_key:
                headers["crossref-api-key"] = f"Bearer {self.api_key}"
            r = requests.get(
                BASE_URL, params=params, headers=headers, timeout=self.timeout
            )
            if r.status_code != 200:
                logger.warning("Crossref returned {}", r.status_code)
                return []
            items = (r.json().get("message") or {}).get("items") or []
            return [self._to_work(item) for item in items if item.get("title")]
        except Exception as e:  # noqa: BLE001
            logger.warning("Crossref search failed: {}", e)
            return []

    def _to_work(self, item: dict) -> CrossrefWork:
        authors = []
        for a in (item.get("author") or [])[:5]:
            name = " ".join(
                part for part in [a.get("given"), a.get("family")] if part
            ).strip()
            if name:
                authors.append(name)
        title = item.get("title") or [""]
        container = item.get("container-title") or [None]
        doi = str(item.get("DOI") or "")
        return CrossrefWork(
            doi=doi,
            title=str(title[0] if title else ""),
            authors=authors,
            abstract=_clean_abstract(item.get("abstract")),
            year=_published_year(item),
            url=str(item.get("URL") or (f"https://doi.org/{doi}" if doi else "")),
            citation_count=int(item.get("is-referenced-by-count") or 0),
            venue=container[0] if container else None,
        )


def _clean_abstract(value: object) -> str:
    if not value:
        return ""
    return _TAG_RE.sub(" ", str(value)).strip()


def _published_year(item: dict) -> int | None:
    for key in ("published-print", "published-online", "published"):
        parts = (item.get(key) or {}).get("date-parts") or []
        if parts and parts[0]:
            try:
                return int(parts[0][0])
            except (TypeError, ValueError):
                return None
    return None
