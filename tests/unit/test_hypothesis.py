"""Unit tests for the Hypothesis Agent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents.data_intelligence import (
    DataIntelligenceReport,
    FeatureProfile,
    PanelStructure,
)
from src.agents.hypothesis import HypothesisAgent, HypothesisSet

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_feature_profile(name: str, corr: float = 0.5) -> FeatureProfile:
    return FeatureProfile(
        name=name,
        is_numeric=True,
        distribution="normal",
        skewness=0.1,
        kurtosis=0.2,
        outlier_count=0,
        correlation_with_outcome=corr,
        vif_score=None,
        recommended_transform="none",
        importance_rank=1,
    )


@pytest.fixture
def mock_report() -> DataIntelligenceReport:
    return DataIntelligenceReport(
        panel_structure=PanelStructure(
            is_panel=True,
            entity_column="experiment_id",
            time_column="time_hours",
            n_entities=60,
            n_time_periods=60,
            balanced=True,
            recommended_model="fixed_effects",
            hausman_applicable=True,
            reasoning="Panel structure detected",
        ),
        feature_profiles=[
            _make_feature_profile("uv_intensity", 0.82),
            _make_feature_profile("temperature_c", 0.61),
            _make_feature_profile("ph", -0.45),
            FeatureProfile(
                name="catalyst",
                is_numeric=False,
                distribution="categorical",
                skewness=None,
                kurtosis=None,
                outlier_count=0,
                correlation_with_outcome=None,
                vif_score=None,
                recommended_transform="encode",
                importance_rank=4,
            ),
        ],
        top_features=["uv_intensity", "temperature_c", "ph"],
        schema_summary="Test dataset: 60 rows, outcome=degradation_rate",
        llm_interpretation="Two regimes detected based on UV intensity.",
        warnings=[],
    )


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.retrieve_for_hypothesis.return_value = []
    retriever.format_context.return_value = "Relevant literature context about PFAS."
    return retriever


@pytest.fixture
def agent() -> HypothesisAgent:
    mock_settings = MagicMock()
    mock_settings.llm.provider = "ollama"
    mock_settings.llm.model = "llama3.1:8b"
    mock_settings.llm.temperature = 0.1
    mock_settings.llm.max_tokens = 4096
    mock_settings.llm.request_timeout = 120
    mock_settings.domain.context = "PFAS photochemical degradation"
    mock_settings.supervisor.hypotheses_per_round = 3
    mock_settings.hypothesis.max_variables_per_model = 6
    mock_settings.hypothesis.max_interaction_terms = 2

    with patch("src.agents.base.get_settings", return_value=mock_settings):
        return HypothesisAgent()


def _make_llm_response(n: int = 3) -> str:
    hypotheses = []
    for i in range(1, n + 1):
        hypotheses.append(
            f"""{{
  "id": "H{i}",
  "description": "UV intensity drives PFAS degradation rate in hypothesis {i}",
  "rationale": "UV photons cleave C-F bonds directly. Higher intensity increases quantum yield.",
  "primary_variables": ["uv_intensity", "ph"],
  "interaction_terms": [["uv_intensity", "temperature_c"]],
  "control_variables": ["temperature_c"],
  "model_family": "fixed_effects",
  "suggested_transforms": {{"uv_intensity": "log", "ph": "none"}},
  "priority_score": {round(0.9 - i * 0.1, 1)}
}}"""
        )
    return "[" + ",".join(hypotheses) + "]"


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestHypothesisAgent:
    def test_run_returns_hypothesis_set(self, agent, mock_report, mock_retriever):
        with patch.object(agent, "call_llm", return_value=_make_llm_response(3)):
            result = agent.run(mock_report, mock_retriever, round_number=1)
        assert isinstance(result, HypothesisSet)

    def test_correct_number_of_hypotheses(self, agent, mock_report, mock_retriever):
        with patch.object(agent, "call_llm", return_value=_make_llm_response(3)):
            result = agent.run(mock_report, mock_retriever, round_number=1)
        assert result.n_hypotheses == 3

    def test_hypotheses_are_sorted_by_priority(
        self, agent, mock_report, mock_retriever
    ):
        with patch.object(agent, "call_llm", return_value=_make_llm_response(3)):
            result = agent.run(mock_report, mock_retriever, round_number=1)
        scores = [h.priority_score for h in result.hypotheses]
        assert scores == sorted(scores, reverse=True)

    def test_hypothesis_has_required_fields(self, agent, mock_report, mock_retriever):
        with patch.object(agent, "call_llm", return_value=_make_llm_response(1)):
            result = agent.run(mock_report, mock_retriever, round_number=1)
        h = result.hypotheses[0]
        assert h.id
        assert h.description
        assert h.rationale
        assert len(h.primary_variables) > 0
        assert h.model_family in {
            "ols",
            "fixed_effects",
            "random_effects",
            "lasso",
            "gradient_boosting",
        }
        assert 0.0 <= h.priority_score <= 1.0

    def test_invalid_variables_filtered_out(self, agent, mock_report, mock_retriever):
        """Variables not in the dataset should be filtered."""
        response = """[{
            "id": "H1",
            "description": "Test hypothesis",
            "rationale": "Test rationale",
            "primary_variables": ["uv_intensity", "NONEXISTENT_VAR"],
            "interaction_terms": [],
            "control_variables": [],
            "model_family": "ols",
            "suggested_transforms": {},
            "priority_score": 0.7
        }]"""
        with patch.object(agent, "call_llm", return_value=response):
            result = agent.run(mock_report, mock_retriever, round_number=1)
        h = result.hypotheses[0]
        assert "NONEXISTENT_VAR" not in h.primary_variables
        assert "uv_intensity" in h.primary_variables

    def test_invalid_model_family_corrected(self, agent, mock_report, mock_retriever):
        response = """[{
            "id": "H1",
            "description": "Test",
            "rationale": "Test",
            "primary_variables": ["uv_intensity"],
            "interaction_terms": [],
            "control_variables": [],
            "model_family": "INVALID_MODEL",
            "suggested_transforms": {},
            "priority_score": 0.5
        }]"""
        with patch.object(agent, "call_llm", return_value=response):
            result = agent.run(mock_report, mock_retriever, round_number=1)
        # Invalid family corrected to ols, then panel enforcement upgrades to fixed_effects
        assert result.hypotheses[0].model_family in {
            "ols",
            "fixed_effects",
            "random_effects",
        }

    def test_panel_model_enforced(self, agent, mock_report, mock_retriever):
        """OLS should be upgraded to fixed_effects for panel data."""
        response = """[{
            "id": "H1",
            "description": "Test",
            "rationale": "Test",
            "primary_variables": ["uv_intensity"],
            "interaction_terms": [],
            "control_variables": [],
            "model_family": "ols",
            "suggested_transforms": {},
            "priority_score": 0.5
        }]"""
        with patch.object(agent, "call_llm", return_value=response):
            result = agent.run(mock_report, mock_retriever, round_number=1)
        # Panel data detected → should be upgraded
        assert result.hypotheses[0].model_family == "fixed_effects"

    def test_max_variables_constraint(self, agent, mock_report, mock_retriever):
        """Should not exceed max_variables_per_model."""
        response = """[{
            "id": "H1",
            "description": "Test",
            "rationale": "Test",
            "primary_variables": ["uv_intensity", "ph", "temperature_c"],
            "interaction_terms": [],
            "control_variables": ["uv_intensity", "ph", "temperature_c"],
            "model_family": "fixed_effects",
            "suggested_transforms": {},
            "priority_score": 0.5
        }]"""
        with patch.object(agent, "call_llm", return_value=response):
            result = agent.run(mock_report, mock_retriever, round_number=1)
        h = result.hypotheses[0]
        total = len(h.primary_variables) + len(h.control_variables)
        assert total <= 6

    def test_max_interactions_constraint(self, agent, mock_report, mock_retriever):
        """Should not exceed max_interaction_terms."""
        response = """[{
            "id": "H1",
            "description": "Test",
            "rationale": "Test",
            "primary_variables": ["uv_intensity", "ph"],
            "interaction_terms": [
                ["uv_intensity", "ph"],
                ["uv_intensity", "temperature_c"],
                ["ph", "temperature_c"]
            ],
            "control_variables": [],
            "model_family": "fixed_effects",
            "suggested_transforms": {},
            "priority_score": 0.5
        }]"""
        with patch.object(agent, "call_llm", return_value=response):
            result = agent.run(mock_report, mock_retriever, round_number=1)
        assert len(result.hypotheses[0].interaction_terms) <= 2

    def test_round_2_marks_refinement(self, agent, mock_report, mock_retriever):
        previous = [
            {
                "hypothesis_id": "H1",
                "r_squared": 0.45,
                "validation_passed": True,
                "match_score": 0.5,
                "significant_variables": ["uv_intensity"],
            }
        ]
        with patch.object(agent, "call_llm", return_value=_make_llm_response(2)):
            result = agent.run(
                mock_report, mock_retriever, round_number=2, previous_results=previous
            )
        assert all(h.is_refinement for h in result.hypotheses)
        assert result.refinement_notes != ""

    def test_no_valid_primary_vars_skipped(self, agent, mock_report, mock_retriever):
        """Hypothesis with no valid variables should be skipped gracefully."""
        response = """[
            {
                "id": "H1",
                "description": "Bad hypothesis",
                "rationale": "No valid vars",
                "primary_variables": ["FAKE_VAR_1", "FAKE_VAR_2"],
                "interaction_terms": [],
                "control_variables": [],
                "model_family": "ols",
                "suggested_transforms": {},
                "priority_score": 0.9
            },
            {
                "id": "H2",
                "description": "Good hypothesis",
                "rationale": "Valid var",
                "primary_variables": ["uv_intensity"],
                "interaction_terms": [],
                "control_variables": [],
                "model_family": "ols",
                "suggested_transforms": {},
                "priority_score": 0.7
            }
        ]"""
        with patch.object(agent, "call_llm", return_value=response):
            result = agent.run(mock_report, mock_retriever, round_number=1)
        assert result.n_hypotheses == 1
        assert result.hypotheses[0].id == "H2"
