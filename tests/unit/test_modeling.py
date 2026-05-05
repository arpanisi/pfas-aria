"""Unit tests for the Modeling Engine and Validation Engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.agents.hypothesis import Hypothesis
from src.modeling.engine import ModelingEngine, ModelResult
from src.validation.validator import ValidationEngine, ValidationReport

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 80
    uv = np.random.uniform(5, 40, n)
    ph = np.random.uniform(6.0, 9.0, n)
    temp = np.random.uniform(15, 35, n)
    noise = np.random.normal(0, 0.05, n)
    degradation = 0.3 + 0.012 * uv - 0.04 * ph + 0.005 * temp + noise

    return pd.DataFrame(
        {
            "experiment_id": range(1, n + 1),
            "time_hours": np.linspace(0, 10, n),
            "degradation_rate": degradation,
            "uv_intensity": uv,
            "ph": ph,
            "temperature_c": temp,
            "catalyst": ["TiO2"] * (n // 2) + ["ZnO"] * (n // 2),
            "regime": ["regime_0"] * (n // 2) + ["regime_1"] * (n // 2),
        }
    )


def _make_hypothesis(
    hyp_id: str = "H1",
    primary_vars: list[str] | None = None,
    model_family: str = "ols",
    interactions: list[tuple[str, str]] | None = None,
) -> Hypothesis:
    return Hypothesis(
        id=hyp_id,
        round=1,
        description="UV intensity drives degradation",
        rationale="Photochemical mechanism",
        outcome_variable="degradation_rate",
        primary_variables=primary_vars or ["uv_intensity", "ph"],
        interaction_terms=interactions or [],
        control_variables=["temperature_c"],
        model_family=model_family,
        suggested_transforms={},
        priority_score=0.8,
        rag_support=[],
    )


# ── Modeling Engine Tests ─────────────────────────────────────────────────────


class TestModelingEngine:
    def test_ols_returns_model_result(self, sample_df):
        engine = ModelingEngine()
        h = _make_hypothesis(model_family="ols")
        results = engine.run(h, sample_df)
        assert len(results) >= 1
        assert isinstance(results[0], ModelResult)

    def test_ols_success(self, sample_df):
        engine = ModelingEngine()
        h = _make_hypothesis(model_family="ols")
        results = engine.run(h, sample_df)
        assert results[0].success is True

    def test_ols_r_squared_reasonable(self, sample_df):
        engine = ModelingEngine()
        h = _make_hypothesis(model_family="ols")
        results = engine.run(h, sample_df)
        assert 0.0 <= results[0].r_squared <= 1.0

    def test_ols_has_coefficients(self, sample_df):
        engine = ModelingEngine()
        h = _make_hypothesis(model_family="ols")
        results = engine.run(h, sample_df)
        assert len(results[0].coefficients) > 0

    def test_ols_has_p_values(self, sample_df):
        engine = ModelingEngine()
        h = _make_hypothesis(model_family="ols")
        results = engine.run(h, sample_df)
        assert len(results[0].p_values) > 0

    def test_uv_intensity_significant(self, sample_df):
        """UV intensity is the true driver — should be significant."""
        engine = ModelingEngine()
        h = _make_hypothesis(model_family="ols", primary_vars=["uv_intensity"])
        results = engine.run(h, sample_df)
        assert "uv_intensity" in results[0].significant_variables

    def test_lasso_returns_result(self, sample_df):
        engine = ModelingEngine()
        h = _make_hypothesis(model_family="lasso")
        results = engine.run(h, sample_df)
        assert results[0].success is True
        assert results[0].model_type == "lasso"

    def test_gradient_boosting_returns_result(self, sample_df):
        engine = ModelingEngine()
        h = _make_hypothesis(model_family="gradient_boosting")
        results = engine.run(h, sample_df)
        assert results[0].success is True
        assert results[0].r_squared > 0.0

    def test_per_regime_produces_multiple_results(self, sample_df):
        engine = ModelingEngine()
        h = _make_hypothesis(model_family="ols")
        results = engine.run(h, sample_df, per_regime=True)
        # Should have global + one per regime
        assert len(results) >= 2

    def test_interaction_term_included(self, sample_df):
        engine = ModelingEngine()
        h = _make_hypothesis(
            model_family="ols",
            primary_vars=["uv_intensity", "ph"],
            interactions=[("uv_intensity", "ph")],
        )
        results = engine.run(h, sample_df)
        assert results[0].success is True
        col_names = results[0].variables_used
        assert any("x" in c for c in col_names)

    def test_invalid_variables_handled(self, sample_df):
        engine = ModelingEngine()
        h = _make_hypothesis(model_family="ols", primary_vars=["NONEXISTENT_VAR"])
        results = engine.run(h, sample_df)
        assert results[0].success is False

    def test_result_has_residuals(self, sample_df):
        engine = ModelingEngine()
        h = _make_hypothesis(model_family="ols")
        results = engine.run(h, sample_df)
        assert results[0].residuals is not None
        assert len(results[0].residuals) == len(sample_df)

    def test_n_observations_correct(self, sample_df):
        engine = ModelingEngine()
        h = _make_hypothesis(model_family="ols")
        results = engine.run(h, sample_df)
        assert results[0].n_observations == len(sample_df)


# ── Validation Engine Tests ───────────────────────────────────────────────────


class TestValidationEngine:
    def _run_ols_and_get_result(self, sample_df: pd.DataFrame) -> ModelResult:
        engine = ModelingEngine()
        h = _make_hypothesis(model_family="ols")
        return engine.run(h, sample_df)[0]

    def test_validation_returns_report(self, sample_df):
        validator = ValidationEngine()
        result = self._run_ols_and_get_result(sample_df)
        report = validator.validate(result, sample_df)
        assert isinstance(report, ValidationReport)

    def test_vif_runs_for_ols(self, sample_df):
        validator = ValidationEngine()
        result = self._run_ols_and_get_result(sample_df)
        report = validator.validate(result, sample_df)
        assert len(report.vif_results) > 0

    def test_normality_test_runs(self, sample_df):
        validator = ValidationEngine()
        result = self._run_ols_and_get_result(sample_df)
        report = validator.validate(result, sample_df)
        assert report.shapiro_p_value >= 0.0

    def test_anova_runs_with_regimes(self, sample_df):
        validator = ValidationEngine()
        result = self._run_ols_and_get_result(sample_df)
        report = validator.validate(result, sample_df)
        assert report.anova_applicable is True

    def test_cv_score_in_range(self, sample_df):
        validator = ValidationEngine()
        result = self._run_ols_and_get_result(sample_df)
        report = validator.validate(result, sample_df)
        assert len(report.cv_r2_scores) == 5

    def test_effect_size_computed(self, sample_df):
        validator = ValidationEngine()
        result = self._run_ols_and_get_result(sample_df)
        report = validator.validate(result, sample_df)
        assert report.effect_size_label in (
            "negligible",
            "small",
            "medium",
            "large",
            "none",
        )

    def test_pass_rate_in_range(self, sample_df):
        validator = ValidationEngine()
        result = self._run_ols_and_get_result(sample_df)
        report = validator.validate(result, sample_df)
        assert 0.0 <= report.pass_rate <= 1.0

    def test_failed_model_fails_validation(self, sample_df):
        validator = ValidationEngine()
        engine = ModelingEngine()
        h = _make_hypothesis(primary_vars=["FAKE_VAR"])
        failed_result = engine.run(h, sample_df)[0]
        report = validator.validate(failed_result, sample_df)
        assert report.overall_passed is False

    def test_good_model_passes_validation(self, sample_df):
        """A well-specified model should pass most validation tests."""
        validator = ValidationEngine()
        engine = ModelingEngine()
        h = _make_hypothesis(
            primary_vars=["uv_intensity", "ph", "temperature_c"],
            model_family="ols",
        )
        result = engine.run(h, sample_df)[0]
        report = validator.validate(result, sample_df)
        # Should pass at least half the tests
        assert report.pass_rate >= 0.5

    def test_vif_acceptable_for_uncorrelated_vars(self, sample_df):
        """UV, pH, temp are not highly correlated — VIF should be low."""
        validator = ValidationEngine()
        engine = ModelingEngine()
        h = _make_hypothesis(
            primary_vars=["uv_intensity", "ph", "temperature_c"],
            model_family="ols",
        )
        result = engine.run(h, sample_df)[0]
        report = validator.validate(result, sample_df)
        assert report.vif_passed is True
