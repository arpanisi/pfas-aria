"""Europe PMC publication search client."""

from __future__ import annotations

from dataclasses import dataclass

import requests

from src.utils.logging import get_logger

logger = get_logger(__name__)

BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


@dataclass
class EuropePMCPaper:
    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    year: int | None
    doi: str | None
    url: str
    citation_count: int
    source: str


class EuropePMCClient:
    """Search Europe PMC articles and preprints."""

    def __init__(self, timeout: int = 8) -> None:
        self.timeout = timeout

    def search(self, query: str, max_results: int = 10) -> list[EuropePMCPaper]:
        try:
            r = requests.get(
                BASE_URL,
                params={
                    "query": query,
                    "format": "json",
                    "pageSize": str(max_results),
                    "resultType": "core",
                    "synonym": "true",
                },
                timeout=self.timeout,
            )
            if r.status_code != 200:
                logger.warning("Europe PMC returned {}", r.status_code)
                return []
            results = ((r.json().get("resultList") or {}).get("result") or [])
            return [self._to_paper(item) for item in results if item.get("title")]
        except Exception as e:  # noqa: BLE001
            logger.warning("Europe PMC search failed: {}", e)
            return []

    def _to_paper(self, item: dict) -> EuropePMCPaper:
        doi = item.get("doi")
        pmid = item.get("pmid")
        pmcid = item.get("pmcid")
        source = str(item.get("source") or "")
        paper_id = str(doi or pmcid or pmid or item.get("id") or "")
        url = ""
        if doi:
            url = f"https://doi.org/{doi}"
        elif pmcid:
            url = f"https://europepmc.org/article/PMC/{pmcid}"
        elif pmid:
            url = f"https://europepmc.org/article/MED/{pmid}"
        return EuropePMCPaper(
            paper_id=paper_id,
            title=str(item.get("title") or ""),
            authors=_authors(item.get("authorString")),
            abstract=str(item.get("abstractText") or ""),
            year=_year(item.get("pubYear")),
            doi=str(doi) if doi else None,
            url=url or str(item.get("fullTextUrlList", {}).get("fullTextUrl", [{}])[0].get("url", "")),
            citation_count=_int(item.get("citedByCount")),
            source=source,
        )


def _authors(value: object) -> list[str]:
    if not value:
        return []
    return [a.strip() for a in str(value).split(",")[:5] if a.strip()]


def _year(value: object) -> int | None:
    try:
        return int(str(value)[:4])
    except (TypeError, ValueError):
        return None


def _int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
