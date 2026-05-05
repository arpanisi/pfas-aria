"""
Modeling Runner.
Orchestrates the full modeling + validation pipeline for a HypothesisSet.
For each hypothesis: fit model → validate → collect results.
Returns structured output ready for the Literature Grounding Agent.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.agents.hypothesis import Hypothesis, HypothesisSet
from src.modeling.engine import ModelingEngine, ModelResult
from src.utils.logging import get_logger
from src.validation.validator import ValidationEngine, ValidationReport

logger = get_logger(__name__)


# ── Output schema ─────────────────────────────────────────────────────────────


@dataclass
class HypothesisModelingResult:
    """Modeling + validation results for one hypothesis."""

    hypothesis: Hypothesis
    model_results: list[ModelResult]
    validation_reports: list[ValidationReport]
    best_model: ModelResult | None  # Highest R² that passed validation
    validation_passed: bool
    pass_rate: float
    summary: dict  # Structured dict for downstream agents


@dataclass
class ModelingRoundResult:
    """Full results for all hypotheses in one round."""

    round: int
    hypothesis_results: list[HypothesisModelingResult]
    n_hypotheses: int
    n_passed: int
    n_failed: int
    best_r_squared: float
    top_variables: list[str]  # Most significant across all models
    structured_results: list[dict]  # For Hypothesis Agent refinement


# ── Runner ────────────────────────────────────────────────────────────────────


class ModelingRunner:
    """
    Runs the full modeling + validation cycle for a hypothesis set.
    """

    def __init__(self) -> None:
        self.engine = ModelingEngine()
        self.validator = ValidationEngine()

    def run(
        self,
        hypothesis_set: HypothesisSet,
        df: pd.DataFrame,
        per_regime: bool = True,
    ) -> ModelingRoundResult:
        """
        For each hypothesis in the set:
          1. Fit model(s)
          2. Validate each model
          3. Collect and rank results
        """
        logger.info(
            f"=== Modeling Runner: Round {hypothesis_set.round} — "
            f"{hypothesis_set.n_hypotheses} hypotheses ==="
        )

        hypothesis_results = []

        for hypothesis in hypothesis_set.hypotheses:
            logger.info(f"Processing hypothesis {hypothesis.id}...")
            result = self._process_hypothesis(hypothesis, df, per_regime)
            hypothesis_results.append(result)

        # Aggregate
        n_passed = sum(1 for r in hypothesis_results if r.validation_passed)
        n_failed = len(hypothesis_results) - n_passed

        best_r2 = max(
            (
                r.best_model.r_squared
                for r in hypothesis_results
                if r.best_model is not None
            ),
            default=0.0,
        )

        top_vars = self._aggregate_top_variables(hypothesis_results)
        structured = [r.summary for r in hypothesis_results]

        round_result = ModelingRoundResult(
            round=hypothesis_set.round,
            hypothesis_results=hypothesis_results,
            n_hypotheses=len(hypothesis_results),
            n_passed=n_passed,
            n_failed=n_failed,
            best_r_squared=best_r2,
            top_variables=top_vars,
            structured_results=structured,
        )

        logger.info(
            f"=== Modeling Runner: Complete — "
            f"{n_passed}/{len(hypothesis_results)} passed validation, "
            f"best R²={best_r2:.4f} ==="
        )

        return round_result

    # ── Per-hypothesis processing ─────────────────────────────────────────────

    def _process_hypothesis(
        self,
        hypothesis: Hypothesis,
        df: pd.DataFrame,
        per_regime: bool,
    ) -> HypothesisModelingResult:
        # Fit models
        model_results = self.engine.run(hypothesis, df, per_regime=per_regime)

        # Validate each model
        validation_reports = []
        for model_result in model_results:
            report = self.validator.validate(model_result, df)
            validation_reports.append(report)

        # Find best passing model
        passing = [
            (mr, vr)
            for mr, vr in zip(model_results, validation_reports)
            if mr.success and vr.overall_passed
        ]

        best_model: ModelResult | None
        if passing:
            best_model, _ = max(passing, key=lambda x: x[0].r_squared)
            validation_passed = True
        else:
            # Fall back to best model even if validation failed
            successful = [mr for mr in model_results if mr.success]
            best_model = (
                max(successful, key=lambda m: m.r_squared) if successful else None
            )
            validation_passed = False

        pass_rate = (
            len(passing) / len(validation_reports) if validation_reports else 0.0
        )

        summary = self._build_summary(
            hypothesis=hypothesis,
            model_results=model_results,
            validation_reports=validation_reports,
            best_model=best_model,
            validation_passed=validation_passed,
        )

        return HypothesisModelingResult(
            hypothesis=hypothesis,
            model_results=model_results,
            validation_reports=validation_reports,
            best_model=best_model,
            validation_passed=validation_passed,
            pass_rate=pass_rate,
            summary=summary,
        )

    def _build_summary(
        self,
        hypothesis: Hypothesis,
        model_results: list[ModelResult],
        validation_reports: list[ValidationReport],
        best_model: ModelResult | None,
        validation_passed: bool,
    ) -> dict:
        """Build structured summary for downstream agents."""
        significant_vars: list[str] = []
        if best_model:
            significant_vars = best_model.significant_variables

        avg_pass_rate = (
            sum(vr.pass_rate for vr in validation_reports) / len(validation_reports)
            if validation_reports
            else 0.0
        )

        return {
            "hypothesis_id": hypothesis.id,
            "description": hypothesis.description,
            "model_type": hypothesis.model_family,
            "r_squared": best_model.r_squared if best_model else 0.0,
            "adj_r_squared": best_model.adj_r_squared if best_model else 0.0,
            "significant_variables": significant_vars,
            "highly_significant_variables": (
                best_model.highly_significant if best_model else []
            ),
            "coefficients": best_model.coefficients if best_model else {},
            "p_values": best_model.p_values if best_model else {},
            "validation_passed": validation_passed,
            "validation_pass_rate": round(avg_pass_rate, 3),
            "n_observations": best_model.n_observations if best_model else 0,
            "match_score": 0.0,  # Filled by Grounding Agent
            "primary_variables": hypothesis.primary_variables,
            "interaction_terms": [list(pair) for pair in hypothesis.interaction_terms],
            "anova_p_value": (
                validation_reports[0].anova_p_value if validation_reports else 1.0
            ),
            "cv_r2_mean": (
                validation_reports[0].cv_r2_mean if validation_reports else 0.0
            ),
            "max_vif": (validation_reports[0].max_vif if validation_reports else 0.0),
        }

    def _aggregate_top_variables(
        self, hypothesis_results: list[HypothesisModelingResult]
    ) -> list[str]:
        """Rank variables by frequency of significance across all models."""
        var_counts: dict[str, int] = {}
        for hr in hypothesis_results:
            if hr.best_model:
                for var in hr.best_model.significant_variables:
                    var_counts[var] = var_counts.get(var, 0) + 1

        return sorted(var_counts, key=lambda v: var_counts[v], reverse=True)
