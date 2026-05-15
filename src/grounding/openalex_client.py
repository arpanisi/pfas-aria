"""OpenAlex scholarly works client."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests

from src.utils.logging import get_logger

logger = get_logger(__name__)

BASE_URL = "https://api.openalex.org/works"


@dataclass
class OpenAlexWork:
    work_id: str
    title: str
    authors: list[str]
    abstract: str
    year: int | None
    doi: str | None
    url: str
    citation_count: int
    venue: str | None
    open_access_url: str | None


class OpenAlexClient:
    """Search OpenAlex works and normalize metadata for grounding."""

    def __init__(self, timeout: int = 8) -> None:
        self.timeout = timeout
        self.api_key = os.getenv("OPENALEX_API_KEY", "").strip()
        self.mailto = os.getenv("OPENALEX_MAILTO", "").strip()

    def search(self, query: str, max_results: int = 10) -> list[OpenAlexWork]:
        try:
            params: dict[str, Any] = {
                "search": query,
                "per-page": max_results,
                "sort": "relevance_score:desc",
            }
            if self.api_key:
                params["api_key"] = self.api_key
            if self.mailto:
                params["mailto"] = self.mailto
            r = requests.get(BASE_URL, params=params, timeout=self.timeout)
            if r.status_code != 200:
                logger.warning("OpenAlex returned {}", r.status_code)
                return []
            data = r.json()
            return [self._to_work(item) for item in data.get("results", [])]
        except Exception as e:  # noqa: BLE001
            logger.warning("OpenAlex search failed: {}", e)
            return []

    def _to_work(self, item: dict) -> OpenAlexWork:
        authors = []
        for auth in (item.get("authorships") or [])[:5]:
            author = auth.get("author") or {}
            name = author.get("display_name")
            if name:
                authors.append(str(name))

        open_access = item.get("open_access") or {}
        primary_location = item.get("primary_location") or {}
        source = primary_location.get("source") or {}

        return OpenAlexWork(
            work_id=str(item.get("id") or ""),
            title=str(item.get("display_name") or ""),
            authors=authors,
            abstract=_abstract_from_inverted_index(item.get("abstract_inverted_index")),
            year=item.get("publication_year"),
            doi=_normalize_doi(item.get("doi")),
            url=str(item.get("doi") or item.get("id") or ""),
            citation_count=int(item.get("cited_by_count") or 0),
            venue=source.get("display_name"),
            open_access_url=open_access.get("oa_url"),
        )


def _normalize_doi(value: object) -> str | None:
    if not value:
        return None
    doi = str(value)
    return doi.removeprefix("https://doi.org/").strip() or None


def _abstract_from_inverted_index(index: object) -> str:
    if not isinstance(index, dict):
        return ""
    positions: list[tuple[int, str]] = []
    for word, locs in index.items():
        if not isinstance(locs, list):
            continue
        for loc in locs:
            try:
                positions.append((int(loc), str(word)))
            except (TypeError, ValueError):
                continue
    return " ".join(word for _, word in sorted(positions))
