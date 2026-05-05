"""
Integration test for the full pipeline loop.
Uses synthetic data and mocked LLM — no real model calls.
Tests that the supervisor correctly wires agents and manages state.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.orchestration.state import PipelineState, PipelineStatus


@pytest.fixture
def synthetic_bundle():
    """Minimal synthetic DataBundle for pipeline testing."""
    from src.ingestion.data_loader import ColumnProfile, DataBundle

    np.random.seed(42)
    n = 60
    uv = np.random.uniform(5, 40, n)
    ph = np.random.uniform(6.0, 9.0, n)
    temp = np.random.uniform(15, 35, n)
    deg = 0.3 + 0.012 * uv - 0.04 * ph + np.random.normal(0, 0.05, n)

    df = pd.DataFrame(
        {
            "experiment_id": range(1, n + 1),
            "time_hours": np.linspace(0, 10, n),
            "degradation_rate": deg,
            "uv_intensity": uv,
            "ph": ph,
            "temperature_c": temp,
            "catalyst": ["TiO2"] * (n // 2) + ["ZnO"] * (n // 2),
        }
    )

    profiles = {
        col: ColumnProfile(
            name=col,
            dtype="float64",
            n_unique=n,
            n_missing=0,
            missing_pct=0.0,
            min=float(df[col].min()) if col != "catalyst" else None,
            max=float(df[col].max()) if col != "catalyst" else None,
            mean=float(df[col].mean()) if col != "catalyst" else None,
            std=float(df[col].std()) if col != "catalyst" else None,
            is_numeric=col != "catalyst",
            is_categorical=col == "catalyst",
            is_datetime=False,
            sample_values=df[col].unique()[:3].tolist(),
        )
        for col in df.columns
    }

    return DataBundle(
        df=df,
        outcome_variable="degradation_rate",
        entity_id_column="experiment_id",
        time_column="time_hours",
        numeric_columns=[
            "degradation_rate",
            "uv_intensity",
            "ph",
            "temperature_c",
            "time_hours",
        ],
        categorical_columns=["catalyst"],
        datetime_columns=[],
        feature_columns=["uv_intensity", "ph", "temperature_c", "catalyst"],
        column_profiles=profiles,
        n_rows=n,
        n_cols=len(df.columns),
        has_missing=False,
        source_path="synthetic",
        schema_summary="Synthetic PFAS dataset for testing",
    )


class TestPipelineStateTransitions:
    """Test state machine transitions without running real agents."""

    def test_initial_state(self):
        state = PipelineState(
            run_id="test",
            max_rounds=3,
            convergence_threshold=0.75,
        )
        assert state.status == PipelineStatus.INITIALIZING
        assert state.current_round == 0
        assert not state.should_stop

    def test_failed_state_propagates(self):
        state = PipelineState()
        state.status = PipelineStatus.FAILED
        state.errors.append("Test error")
        assert state.status == PipelineStatus.FAILED
        assert "Test error" in state.errors

    def test_convergence_after_one_round(self):
        state = PipelineState(convergence_threshold=0.75, max_rounds=10)
        state.current_round = 1

        mock_g = MagicMock()
        mock_g.global_match_score = 0.85
        mock_g.convergence_ready = True
        mock_g.top_citations = []
        mock_g.structured_results = []

        mock_m = MagicMock()
        mock_m.n_hypotheses = 3
        mock_m.n_passed = 2
        mock_m.best_r_squared = 0.75
        mock_m.top_variables = ["uv_intensity"]
        mock_m.structured_results = []

        state.add_round_summary(mock_g, mock_m)
        state.final_match_score = 0.85

        assert state.should_stop is True
        assert "converged" in state.stop_reason

    def test_no_convergence_continues_loop(self):
        state = PipelineState(convergence_threshold=0.75, max_rounds=5)

        for i, score in enumerate([0.40, 0.55], start=1):
            state.current_round = i
            mock_g = MagicMock()
            mock_g.global_match_score = score
            mock_g.convergence_ready = False
            mock_g.top_citations = []
            mock_g.structured_results = []
            mock_m = MagicMock()
            mock_m.n_hypotheses = 3
            mock_m.n_passed = 2
            mock_m.best_r_squared = 0.6
            mock_m.top_variables = []
            mock_m.structured_results = []
            state.add_round_summary(mock_g, mock_m)

        assert state.should_stop is False

    def test_hard_stop_at_max_rounds(self):
        state = PipelineState(convergence_threshold=0.75, max_rounds=2)

        for i, score in enumerate([0.40, 0.55], start=1):
            state.current_round = i
            mock_g = MagicMock()
            mock_g.global_match_score = score
            mock_g.convergence_ready = False
            mock_g.top_citations = []
            mock_g.structured_results = []
            mock_m = MagicMock()
            mock_m.n_hypotheses = 3
            mock_m.n_passed = 1
            mock_m.best_r_squared = 0.5
            mock_m.top_variables = []
            mock_m.structured_results = []
            state.add_round_summary(mock_g, mock_m)

        assert state.should_stop is True
        assert "max rounds" in state.stop_reason

    def test_round_summaries_accumulate(self):
        state = PipelineState(max_rounds=5)

        for i, score in enumerate([0.4, 0.6, 0.5], start=1):
            state.current_round = i
            mock_g = MagicMock()
            mock_g.global_match_score = score
            mock_g.convergence_ready = False
            mock_g.top_citations = []
            mock_g.structured_results = [{"hypothesis_id": f"H{i}"}]
            mock_m = MagicMock()
            mock_m.n_hypotheses = 3
            mock_m.n_passed = 2
            mock_m.best_r_squared = 0.6
            mock_m.top_variables = ["uv_intensity"]
            mock_m.structured_results = []
            state.add_round_summary(mock_g, mock_m)

        assert len(state.round_summaries) == 3
        assert len(state.convergence_scores) == 3
        assert len(state.all_structured_results) == 3
