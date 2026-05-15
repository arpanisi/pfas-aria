"""
arXiv Client.
Uses the arXiv Atom API directly (requests) — the arxiv Python library
adds ~60-120s of overhead per call due to internal rate-limit sleeps.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # noqa: N817
from dataclasses import dataclass

import requests

from src.utils.logging import get_logger

logger = get_logger(__name__)

_ATOM_NS = "http://www.w3.org/2005/Atom"
_ARXIV_NS = "http://arxiv.org/schemas/atom"
_BASE_URL = "http://export.arxiv.org/api/query"


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    published: str
    categories: list[str]


class ArxivClient:
    def __init__(self, max_results: int = 8, timeout: int = 15) -> None:
        self.max_results = max_results
        self.timeout = timeout

    def search(self, query: str) -> list[ArxivPaper]:
        logger.debug("arXiv search: '{}'", query[:80])
        try:
            r = requests.get(
                _BASE_URL,
                params={
                    "search_query": f"all:{query}",
                    "max_results": str(self.max_results),
                    "sortBy": "relevance",
                },
                timeout=self.timeout,
            )
            r.raise_for_status()
            return self._parse(r.text)
        except Exception as e:
            logger.warning("arXiv search failed: {}", e)
            return []

    def _parse(self, xml_text: str) -> list[ArxivPaper]:
        root = ET.fromstring(xml_text)
        papers: list[ArxivPaper] = []
        for entry in root.findall(f"{{{_ATOM_NS}}}entry"):
            try:
                arxiv_id = (entry.findtext(f"{{{_ATOM_NS}}}id") or "").split("/")[-1]
                title = (entry.findtext(f"{{{_ATOM_NS}}}title") or "").strip()
                abstract = (entry.findtext(f"{{{_ATOM_NS}}}summary") or "").strip()
                published = (entry.findtext(f"{{{_ATOM_NS}}}published") or "")[:10]
                url = entry.findtext(f"{{{_ATOM_NS}}}id") or ""
                authors = [
                    (a.findtext(f"{{{_ATOM_NS}}}name") or "")
                    for a in entry.findall(f"{{{_ATOM_NS}}}author")[:5]
                ]
                categories = [
                    t.get("term", "")
                    for t in entry.findall(f"{{{_ATOM_NS}}}category")
                ]
                if title and abstract:
                    papers.append(ArxivPaper(
                        arxiv_id=arxiv_id,
                        title=title,
                        authors=authors,
                        abstract=abstract,
                        url=url,
                        published=published,
                        categories=categories,
                    ))
            except Exception:
                continue
        return papers
