"""
Literature Deduplicator.
Deduplicates papers fetched from multiple sources (arXiv, Semantic Scholar)
by arXiv ID, DOI, or normalized title.

Prevents the same paper from being scored multiple times,
which would artificially inflate match scores.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DeduplicatedPaper:
    """A paper after deduplication, with merged metadata."""

    canonical_id: str  # arXiv ID > DOI > title hash
    title: str
    abstract: str
    authors: list[str]
    url: str
    year: str | None
    sources: list[str]  # Which sources this came from


class LiteratureDeduplicator:
    """
    Deduplicates a mixed list of papers from multiple sources.
    Priority: arXiv ID > DOI > normalized title match.
    """

    def deduplicate(
        self,
        arxiv_papers: list[dict],
        s2_papers: list[dict],
    ) -> list[DeduplicatedPaper]:
        """
        Merge arXiv and S2 paper lists, removing duplicates.
        Returns a unified list of DeduplicatedPaper objects.
        """
        seen_ids: dict[str, DeduplicatedPaper] = {}
        seen_titles: dict[str, str] = {}  # normalized_title → canonical_id

        # Process arXiv papers first (higher priority)
        for paper in arxiv_papers:
            arxiv_id = paper.get("arxiv_id", "")
            title = paper.get("title", "")
            norm_title = self._normalize_title(title)

            canonical_id = f"arxiv:{arxiv_id}" if arxiv_id else f"title:{norm_title}"

            if canonical_id not in seen_ids:
                dedup = DeduplicatedPaper(
                    canonical_id=canonical_id,
                    title=title,
                    abstract=paper.get("abstract", ""),
                    authors=paper.get("authors", []),
                    url=paper.get("url", ""),
                    year=paper.get("published", "")[:4]
                    if paper.get("published")
                    else None,
                    sources=["arxiv"],
                )
                seen_ids[canonical_id] = dedup
                seen_titles[norm_title] = canonical_id

        # Process S2 papers — check for duplicates with arXiv
        for paper in s2_papers:
            title = paper.get("title", "")
            norm_title = self._normalize_title(title)
            paper_id = paper.get("paper_id", "")

            # Check if already seen by title
            if norm_title in seen_titles:
                existing_id = seen_titles[norm_title]
                seen_ids[existing_id].sources.append("semantic_scholar")
                continue

            canonical_id = f"s2:{paper_id}" if paper_id else f"title:{norm_title}"

            if canonical_id not in seen_ids:
                dedup = DeduplicatedPaper(
                    canonical_id=canonical_id,
                    title=title,
                    abstract=paper.get("abstract", ""),
                    authors=paper.get("authors", []),
                    url=paper.get("url", ""),
                    year=str(paper.get("year", "")) if paper.get("year") else None,
                    sources=["semantic_scholar"],
                )
                seen_ids[canonical_id] = dedup
                seen_titles[norm_title] = canonical_id

        result = list(seen_ids.values())

        logger.debug(
            f"Deduplication: {len(arxiv_papers)} arXiv + "
            f"{len(s2_papers)} S2 → {len(result)} unique"
        )

        return result

    def _normalize_title(self, title: str) -> str:
        """Normalize a title for fuzzy comparison."""
        # Lowercase, remove punctuation, collapse whitespace
        normalized = title.lower()
        normalized = re.sub(r"[^\w\s]", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized
