"""Unit tests for the Literature Grounding Agent and supporting components."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.grounding.arxiv_client import ArxivClient, ArxivPaper
from src.grounding.scorer import (
    GroundingResult,
    GroundingScorer,
)
from src.grounding.semantic_scholar import S2Paper, SemanticScholarClient

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_arxiv_paper(i: int = 1) -> ArxivPaper:
    return ArxivPaper(
        arxiv_id=f"2401.{i:05d}",
        title=f"PFAS Photodegradation Study {i}",
        authors=["Author A", "Author B"],
        abstract="UV irradiation drives PFAS degradation through C-F bond cleavage. "
        "Higher UV intensity increases the pseudo-first-order rate constant.",
        url=f"https://arxiv.org/abs/2401.{i:05d}",
        published="2024-01-15",
        categories=["physics.chem-ph"],
    )


def _make_s2_paper(i: int = 1) -> S2Paper:
    return S2Paper(
        paper_id=f"s2_{i}",
        title=f"PFAS Treatment Review {i}",
        authors=["Researcher C"],
        abstract="Advanced oxidation processes for PFAS removal. "
        "UV intensity and pH are primary controlling factors.",
        year=2023,
        citation_count=42,
        url=f"https://www.semanticscholar.org/paper/s2_{i}",
        open_access_url=None,
        fields_of_study=["Environmental Science"],
    )


def _make_model_result() -> MagicMock:
    result = MagicMock()
    result.hypothesis_id = "H1_global"
    result.model_type = "ols"
    result.outcome_variable = "degradation_rate"
    result.significant_variables = ["uv_intensity", "ph"]
    result.highly_significant = ["uv_intensity"]
    result.coefficients = {"uv_intensity": 0.012, "ph": -0.04}
    result.p_values = {"uv_intensity": 0.001, "ph": 0.023}
    result.r_squared = 0.72
    result.success = True
    return result


def _make_validation_report() -> MagicMock:
    report = MagicMock()
    report.pass_rate = 0.8
    report.overall_passed = True
    return report


# ── ArxivClient Tests ─────────────────────────────────────────────────────────


class TestArxivClient:
    def test_extract_keywords_removes_stopwords(self):
        with patch("src.grounding.arxiv_client.get_settings"):
            client = ArxivClient()
        text = "UV intensity significantly increases the degradation rate of PFAS"
        keywords = client._extract_keywords(text)
        assert "the" not in keywords
        assert "significantly" not in keywords
        assert "intensity" in keywords or "degradation" in keywords

    def test_build_queries_returns_list(self):
        with patch("src.grounding.arxiv_client.get_settings"):
            client = ArxivClient()
        queries = client._build_queries(
            finding="UV drives degradation",
            significant_variables=["uv_intensity", "ph"],
            domain_context="PFAS photochemical degradation",
        )
        assert isinstance(queries, list)
        assert len(queries) <= 3
        assert len(queries) >= 1

    def test_build_queries_includes_domain(self):
        with patch("src.grounding.arxiv_client.get_settings"):
            client = ArxivClient()
        queries = client._build_queries(
            finding="UV drives degradation",
            significant_variables=["uv_intensity"],
            domain_context="PFAS photochemical degradation",
        )
        assert any("PFAS" in q for q in queries)

    @patch("src.grounding.arxiv_client.arxiv")
    def test_search_returns_empty_on_failure(self, mock_arxiv):
        with patch("src.grounding.arxiv_client.get_settings"):
            client = ArxivClient()
        mock_arxiv.Search.side_effect = Exception("API error")
        results = client.search("PFAS degradation")
        assert results == []


# ── SemanticScholarClient Tests ───────────────────────────────────────────────


class TestSemanticScholarClient:
    def test_build_query_combines_domain_and_vars(self):
        client = SemanticScholarClient()
        query = client._build_query(
            significant_variables=["uv_intensity", "ph"],
            domain_context="PFAS degradation",
        )
        assert "PFAS" in query
        assert "uv_intensity" in query

    def test_search_handles_rate_limit(self):
        client = SemanticScholarClient()
        with patch("src.grounding.semantic_scholar.requests.get") as mock_get:
            mock_get.return_value.status_code = 429
            results = client.search("PFAS degradation")
        assert results == []

    def test_search_handles_api_error(self):
        client = SemanticScholarClient()
        with patch("src.grounding.semantic_scholar.requests.get") as mock_get:
            mock_get.side_effect = Exception("Connection error")
            results = client.search("PFAS degradation")
        assert results == []

    def test_to_paper_extracts_fields(self):
        client = SemanticScholarClient()
        data = {
            "paperId": "abc123",
            "title": "PFAS Study",
            "authors": [{"name": "Author A"}],
            "abstract": "Study abstract here",
            "year": 2023,
            "citationCount": 10,
            "externalIds": {},
            "openAccessPdf": None,
            "fieldsOfStudy": ["Chemistry"],
        }
        paper = client._to_paper(data)
        assert paper.paper_id == "abc123"
        assert paper.title == "PFAS Study"
        assert paper.year == 2023
        assert paper.citation_count == 10

    def test_to_paper_uses_arxiv_url(self):
        client = SemanticScholarClient()
        data = {
            "paperId": "abc123",
            "title": "Test",
            "authors": [],
            "abstract": "Abstract",
            "year": 2023,
            "citationCount": 0,
            "externalIds": {"ArXiv": "2401.00001"},
            "openAccessPdf": None,
            "fieldsOfStudy": [],
        }
        paper = client._to_paper(data)
        assert "arxiv.org" in paper.url


# ── GroundingScorer Tests ─────────────────────────────────────────────────────


class TestGroundingScorer:
    @pytest.fixture
    def mock_retriever(self):
        retriever = MagicMock()
        retriever.retrieve_for_grounding.return_value = []
        return retriever

    @pytest.fixture
    def mock_embedder(self):
        embedder = MagicMock()
        embedder.embed_one.return_value = np.random.rand(384).astype(np.float32)
        embedder.embed.return_value = np.random.rand(3, 384).astype(np.float32)
        embedder.batch_similarity.return_value = np.array([0.75, 0.60, 0.45])
        return embedder

    def test_build_finding_text_positive_coeff(self, mock_retriever):
        with (
            patch("src.grounding.scorer.get_settings"),
            patch("src.grounding.scorer.get_embedder"),
        ):
            scorer = GroundingScorer(mock_retriever)
        text = scorer._build_finding_text("uv_intensity", 0.012, "degradation_rate")
        assert "increases" in text
        assert "uv_intensity" in text
        assert "degradation_rate" in text

    def test_build_finding_text_negative_coeff(self, mock_retriever):
        with (
            patch("src.grounding.scorer.get_settings"),
            patch("src.grounding.scorer.get_embedder"),
        ):
            scorer = GroundingScorer(mock_retriever)
        text = scorer._build_finding_text("ph", -0.04, "degradation_rate")
        assert "decreases" in text

    def test_score_returns_grounding_result(self, mock_retriever, mock_embedder):
        model_result = _make_model_result()
        validation_report = _make_validation_report()
        arxiv_papers = [_make_arxiv_paper(i) for i in range(3)]
        s2_papers = [_make_s2_paper(i) for i in range(2)]

        mock_settings = MagicMock()
        mock_settings.convergence.weights.rag_similarity = 0.5
        mock_settings.convergence.weights.arxiv_citation = 0.3
        mock_settings.convergence.weights.validation_pass_rate = 0.2

        with (
            patch("src.grounding.scorer.get_settings", return_value=mock_settings),
            patch("src.grounding.scorer.get_embedder", return_value=mock_embedder),
        ):
            scorer = GroundingScorer(mock_retriever)
            result = scorer.score(
                model_result=model_result,
                validation_report=validation_report,
                arxiv_papers=arxiv_papers,
                s2_papers=s2_papers,
            )

        assert isinstance(result, GroundingResult)
        assert result.hypothesis_id == "H1_global"
        assert 0.0 <= result.global_match_score <= 1.0

    def test_match_score_in_valid_range(self, mock_retriever, mock_embedder):
        model_result = _make_model_result()
        validation_report = _make_validation_report()

        mock_settings = MagicMock()
        mock_settings.convergence.weights.rag_similarity = 0.5
        mock_settings.convergence.weights.arxiv_citation = 0.3
        mock_settings.convergence.weights.validation_pass_rate = 0.2

        with (
            patch("src.grounding.scorer.get_settings", return_value=mock_settings),
            patch("src.grounding.scorer.get_embedder", return_value=mock_embedder),
        ):
            scorer = GroundingScorer(mock_retriever)
            result = scorer.score(
                model_result=model_result,
                validation_report=validation_report,
                arxiv_papers=[],
                s2_papers=[],
            )

        for fs in result.finding_scores:
            assert 0.0 <= fs.match_score <= 1.0

    def test_empty_papers_returns_zero_arxiv_score(self, mock_retriever, mock_embedder):
        mock_settings = MagicMock()
        mock_settings.convergence.weights.rag_similarity = 0.5
        mock_settings.convergence.weights.arxiv_citation = 0.3
        mock_settings.convergence.weights.validation_pass_rate = 0.2

        with (
            patch("src.grounding.scorer.get_settings", return_value=mock_settings),
            patch("src.grounding.scorer.get_embedder", return_value=mock_embedder),
        ):
            scorer = GroundingScorer(mock_retriever)
            score, citations = scorer._score_against_papers(
                "UV drives degradation", [], source="arxiv"
            )

        assert score == 0.0
        assert citations == []

    def test_citations_deduplicated_in_result(self, mock_retriever, mock_embedder):
        model_result = _make_model_result()
        validation_report = _make_validation_report()
        # Same paper twice
        arxiv_papers = [_make_arxiv_paper(1), _make_arxiv_paper(1)]

        mock_settings = MagicMock()
        mock_settings.convergence.weights.rag_similarity = 0.5
        mock_settings.convergence.weights.arxiv_citation = 0.3
        mock_settings.convergence.weights.validation_pass_rate = 0.2

        with (
            patch("src.grounding.scorer.get_settings", return_value=mock_settings),
            patch("src.grounding.scorer.get_embedder", return_value=mock_embedder),
        ):
            scorer = GroundingScorer(mock_retriever)
            result = scorer.score(
                model_result=model_result,
                validation_report=validation_report,
                arxiv_papers=arxiv_papers,
                s2_papers=[],
            )

        urls = [c.url for c in result.all_citations]
        assert len(urls) == len(set(urls))
