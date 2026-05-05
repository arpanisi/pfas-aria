"""
arXiv Client.
Searches arXiv for open-access papers relevant to model findings.
Extracts keywords from significant coefficients and searches accordingly.
No API key required — arXiv is fully open.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import arxiv

from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ── Output schema ─────────────────────────────────────────────────────────────


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    published: str
    categories: list[str]


# ── Client ────────────────────────────────────────────────────────────────────


class ArxivClient:
    """
    Searches arXiv using keyword queries derived from model findings.
    Returns structured paper metadata for similarity scoring.
    """

    BASE_CATEGORIES = ["physics.chem-ph", "q-bio", "cond-mat", "eess.SP"]
    RATE_LIMIT_SECONDS = 3.0  # arXiv asks for polite delays between requests

    def __init__(self) -> None:
        self.settings = get_settings()
        self.max_results = (
            self.settings.agent_config_arxiv_max_results
            if hasattr(self.settings, "agent_config_arxiv_max_results")
            else 10
        )
        self._client = arxiv.Client(
            page_size=self.max_results,
            delay_seconds=self.RATE_LIMIT_SECONDS,
            num_retries=3,
        )

    def search(self, query: str) -> list[ArxivPaper]:
        """Search arXiv with a plain-language query."""
        logger.debug(f"arXiv search: '{query[:80]}'")
        try:
            search = arxiv.Search(
                query=query,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            results = list(self._client.results(search))
            papers = [self._to_paper(r) for r in results]
            logger.debug(f"arXiv returned {len(papers)} results")
            return papers
        except Exception as e:
            logger.warning(f"arXiv search failed for '{query[:60]}': {e}")
            return []

    def search_for_finding(
        self,
        finding: str,
        significant_variables: list[str],
        domain_context: str,
    ) -> list[ArxivPaper]:
        """
        Build targeted queries from a model finding and search arXiv.
        Runs multiple queries and deduplicates results.
        """
        queries = self._build_queries(finding, significant_variables, domain_context)
        seen_ids: set[str] = set()
        all_papers: list[ArxivPaper] = []

        for query in queries:
            papers = self.search(query)
            for paper in papers:
                if paper.arxiv_id not in seen_ids:
                    all_papers.append(paper)
                    seen_ids.add(paper.arxiv_id)
            time.sleep(self.RATE_LIMIT_SECONDS)

        logger.info(
            f"arXiv: {len(all_papers)} unique papers from {len(queries)} queries"
        )
        return all_papers

    def _build_queries(
        self,
        finding: str,
        significant_variables: list[str],
        domain_context: str,
    ) -> list[str]:
        """Build 2-3 targeted search queries."""
        queries = []

        # Query 1: domain + top significant variables
        if significant_variables:
            var_str = " ".join(significant_variables[:3])
            queries.append(f"{domain_context} {var_str}")

        # Query 2: domain context alone
        queries.append(domain_context)

        # Query 3: finding-derived keywords
        keywords = self._extract_keywords(finding)
        if keywords:
            queries.append(keywords)

        return queries[:3]

    def _extract_keywords(self, text: str) -> str:
        """Extract meaningful keywords from a finding description."""
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "of",
            "in",
            "on",
            "at",
            "to",
            "for",
            "with",
            "by",
            "from",
            "and",
            "or",
            "but",
            "not",
            "that",
            "this",
            "which",
            "when",
            "where",
            "how",
            "what",
            "significantly",
            "positively",
            "negatively",
            "associated",
            "effect",
            "impact",
            "relationship",
            "between",
            "among",
        }
        words = text.lower().split()
        keywords = [
            w.strip(".,()[]")
            for w in words
            if w.strip(".,()[]") not in stop_words and len(w.strip(".,()[]")) > 3
        ]
        return " ".join(keywords[:8])

    def _to_paper(self, result: arxiv.Result) -> ArxivPaper:
        return ArxivPaper(
            arxiv_id=result.entry_id.split("/")[-1],
            title=result.title,
            authors=[str(a) for a in result.authors[:5]],
            abstract=result.summary,
            url=result.entry_id,
            published=result.published.strftime("%Y-%m-%d")
            if result.published
            else "unknown",
            categories=result.categories,
        )
