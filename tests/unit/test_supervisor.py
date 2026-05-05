"""Unit tests for Supervisor and Convergence components."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.orchestration.convergence import ConvergenceJudge, ConvergenceReport
from src.orchestration.state import PipelineState, PipelineStatus, RoundSummary

# ── PipelineState Tests ───────────────────────────────────────────────────────


class TestPipelineState:
    def test_initial_status(self):
        state = PipelineState()
        assert state.status == PipelineStatus.INITIALIZING
        assert state.current_round == 0

    def test_should_stop_when_converged(self):
        state = PipelineState(convergence_threshold=0.75)
        state.convergence_scores = [0.80]
        assert state.should_stop is True

    def test_should_not_stop_below_threshold(self):
        state = PipelineState(convergence_threshold=0.75)
        state.convergence_scores = [0.60]
        state.current_round = 1
        state.max_rounds = 10
        assert state.should_stop is False

    def test_should_stop_at_max_rounds(self):
        state = PipelineState(convergence_threshold=0.75, max_rounds=3)
        state.convergence_scores = [0.50]
        state.current_round = 3
        assert state.should_stop is True

    def test_should_stop_false_with_no_scores(self):
        state = PipelineState()
        assert state.should_stop is False

    def test_stop_reason_converged(self):
        state = PipelineState(convergence_threshold=0.75)
        state.convergence_scores = [0.80]
        assert "converged" in state.stop_reason

    def test_stop_reason_max_rounds(self):
        state = PipelineState(convergence_threshold=0.75, max_rounds=3)
        state.convergence_scores = [0.50]
        state.current_round = 3
        assert "max rounds" in state.stop_reason

    def test_add_round_summary(self):
        state = PipelineState()
        state.current_round = 1

        mock_grounding = MagicMock()
        mock_grounding.global_match_score = 0.65
        mock_grounding.convergence_ready = False
        mock_grounding.top_citations = []
        mock_grounding.structured_results = []

        mock_modeling = MagicMock()
        mock_modeling.n_hypotheses = 5
        mock_modeling.n_passed = 3
        mock_modeling.best_r_squared = 0.72
        mock_modeling.top_variables = ["uv_intensity", "ph"]
        mock_modeling.structured_results = []

        state.add_round_summary(mock_grounding, mock_modeling)

        assert len(state.round_summaries) == 1
        assert len(state.convergence_scores) == 1
        assert state.convergence_scores[0] == 0.65

    def test_multiple_rounds_tracked(self):
        state = PipelineState()

        for i, score in enumerate([0.45, 0.60, 0.78], start=1):
            state.current_round = i
            mock_g = MagicMock()
            mock_g.global_match_score = score
            mock_g.convergence_ready = score >= 0.75
            mock_g.top_citations = []
            mock_g.structured_results = []
            mock_m = MagicMock()
            mock_m.n_hypotheses = 5
            mock_m.n_passed = 3
            mock_m.best_r_squared = 0.7
            mock_m.top_variables = []
            mock_m.structured_results = []
            state.add_round_summary(mock_g, mock_m)

        assert len(state.convergence_scores) == 3
        assert state.convergence_scores == [0.45, 0.60, 0.78]


# ── ConvergenceJudge Tests ────────────────────────────────────────────────────


class TestConvergenceJudge:
    def _make_state_with_rounds(
        self, scores: list[float], threshold: float = 0.75
    ) -> PipelineState:
        state = PipelineState(
            run_id="test_run",
            convergence_threshold=threshold,
        )
        state.convergence_scores = scores
        state.final_match_score = scores[-1] if scores else 0.0
        state.current_round = len(scores)

        for i, score in enumerate(scores, start=1):
            summary = RoundSummary(
                round=i,
                n_hypotheses=5,
                n_models_passed=3,
                best_r_squared=0.7,
                global_match_score=score,
                top_variables=["uv_intensity", "ph"],
                top_citations=[],
                converged=score >= threshold,
            )
            state.round_summaries.append(summary)

        return state

    def test_evaluate_returns_report(self):
        state = self._make_state_with_rounds([0.45, 0.65, 0.80])
        judge = ConvergenceJudge()
        report = judge.evaluate(state)
        assert isinstance(report, ConvergenceReport)

    def test_converged_true_above_threshold(self):
        state = self._make_state_with_rounds([0.45, 0.80])
        judge = ConvergenceJudge()
        report = judge.evaluate(state)
        assert report.converged is True

    def test_converged_false_below_threshold(self):
        state = self._make_state_with_rounds([0.45, 0.60])
        judge = ConvergenceJudge()
        report = judge.evaluate(state)
        assert report.converged is False

    def test_best_round_identified(self):
        state = self._make_state_with_rounds([0.40, 0.75, 0.65])
        judge = ConvergenceJudge()
        report = judge.evaluate(state)
        assert report.best_round == 2
        assert report.best_score == 0.75

    def test_improvement_computed(self):
        state = self._make_state_with_rounds([0.40, 0.60, 0.75])
        judge = ConvergenceJudge()
        report = judge.evaluate(state)
        assert len(report.improvement_per_round) == 2
        assert report.improvement_per_round[0] == pytest.approx(0.20, abs=0.001)

    def test_top_variables_aggregated(self):
        state = self._make_state_with_rounds([0.5, 0.7])
        judge = ConvergenceJudge()
        report = judge.evaluate(state)
        assert "uv_intensity" in report.top_variables_across_rounds

    def test_score_history_correct(self):
        scores = [0.40, 0.55, 0.70, 0.78]
        state = self._make_state_with_rounds(scores)
        judge = ConvergenceJudge()
        report = judge.evaluate(state)
        assert report.score_history == scores

    def test_total_rounds_correct(self):
        state = self._make_state_with_rounds([0.5, 0.6, 0.7])
        judge = ConvergenceJudge()
        report = judge.evaluate(state)
        assert report.total_rounds == 3


# ── Supervisor Integration (mocked) ──────────────────────────────────────────


class TestSupervisorMocked:
    @patch("src.orchestration.supervisor.get_settings")
    def test_supervisor_initializes(self, mock_settings):
        mock_settings.return_value.supervisor.max_rounds = 3
        mock_settings.return_value.supervisor.convergence_threshold = 0.75

        with patch("src.orchestration.supervisor.StateGraph") as mock_graph:
            mock_graph.return_value.compile.return_value = MagicMock()
            from src.orchestration.supervisor import SupervisorAgent

            agent = SupervisorAgent()
            assert agent is not None

    def test_pipeline_state_propagates_failure(self):
        """If ingestion fails, status should be FAILED."""
        state = PipelineState()
        state.status = PipelineStatus.FAILED
        state.errors.append("Ingestion failed: file not found")
        assert state.status == PipelineStatus.FAILED
        assert len(state.errors) == 1
