"""
Output Persister.
Writes all pipeline outputs to PostgreSQL after each round.
Single responsibility: take in-memory pipeline objects → write to DB.

Called by the supervisor after each round completes.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.orm import (
    Citation,
    Hypothesis,
    ModelResult,
    Regime,
    Run,
    ValidationResult,
)
from src.orchestration.state import PipelineState
from src.utils.logging import get_logger

logger = get_logger(__name__)


class OutputPersister:
    """
    Persists pipeline outputs to PostgreSQL.
    All writes are idempotent — safe to call multiple times.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_run(self, state: PipelineState) -> Run:
        """Create or update the Run record."""
        run = Run(
            id=state.run_id,
            run_name=state.run_name,
            status=str(state.status),
            outcome_variable=getattr(state.data_bundle, "outcome_variable", None)
            if state.data_bundle
            else None,
            selected_features=getattr(state.data_bundle, "feature_columns", [])
            if state.data_bundle
            else [],
        )
        self._session.add(run)
        await self._session.flush()
        logger.debug(f"Created run: {state.run_id}")
        return run

    async def update_run_status(
        self,
        run_id: str,
        status: str,
        final_match_score: float = 0.0,
        converged: bool = False,
        n_rounds: int = 0,
        stop_reason: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update run status after pipeline completes or fails."""
        run = await self._session.get(Run, run_id)
        if run:
            run.status = status
            run.final_match_score = final_match_score
            run.converged = converged
            run.n_rounds_completed = n_rounds
            run.stop_reason = stop_reason
            run.error_message = error_message
            await self._session.flush()

    async def persist_regimes(self, run_id: str, intelligence_report: object) -> None:
        """Persist detected regimes from DataIntelligenceReport."""
        regimes = getattr(intelligence_report, "regimes", [])
        for r in regimes:
            regime = Regime(
                run_id=run_id,
                label=r.label,
                method=getattr(intelligence_report, "regime_method", "unknown"),
                size=r.size,
                description=r.description,
                summary_stats=r.summary_stats,
            )
            self._session.add(regime)
        await self._session.flush()
        logger.debug(f"Persisted {len(regimes)} regimes for run {run_id}")

    async def persist_round(
        self,
        run_id: str,
        round_number: int,
        hypothesis_set: object,
        modeling_result: object,
        grounding_result: object,
    ) -> None:
        """Persist all outputs from one pipeline round."""
        hypotheses_list = getattr(hypothesis_set, "hypotheses", [])
        hypothesis_results = getattr(modeling_result, "hypothesis_results", [])

        for hyp_obj in hypotheses_list:
            # Persist hypothesis
            hyp_record = Hypothesis(
                run_id=run_id,
                hypothesis_id=hyp_obj.id,
                round=round_number,
                description=hyp_obj.description,
                rationale=hyp_obj.rationale,
                primary_variables=hyp_obj.primary_variables,
                interaction_terms=[list(t) for t in hyp_obj.interaction_terms],
                control_variables=hyp_obj.control_variables,
                model_family=hyp_obj.model_family,
                priority_score=hyp_obj.priority_score,
                is_refinement=hyp_obj.is_refinement,
                rag_support=hyp_obj.rag_support,
            )
            self._session.add(hyp_record)
            await self._session.flush()

            # Find matching modeling result
            matching = next(
                (hr for hr in hypothesis_results if hr.hypothesis.id == hyp_obj.id),
                None,
            )
            if not matching or not matching.best_model:
                continue

            model = matching.best_model

            # Find matching grounding result
            grounding_scores = getattr(grounding_result, "hypothesis_grounding", [])
            grounding = next(
                (g for g in grounding_scores if g.hypothesis_id == model.hypothesis_id),
                None,
            )
            match_score = getattr(grounding, "global_match_score", 0.0)

            # Persist model result
            model_record = ModelResult(
                hypothesis_id=hyp_record.id,
                model_type=model.model_type,
                r_squared=model.r_squared,
                adj_r_squared=model.adj_r_squared,
                aic=model.aic,
                bic=model.bic,
                n_observations=model.n_observations,
                coefficients=model.coefficients,
                p_values=model.p_values,
                confidence_intervals=model.confidence_intervals,
                standard_errors=model.standard_errors,
                significant_variables=model.significant_variables,
                highly_significant=model.highly_significant,
                match_score=match_score,
                success=model.success,
                error_message=model.error_message if not model.success else None,
            )
            self._session.add(model_record)
            await self._session.flush()

            # Persist validation result
            if matching.validation_reports:
                vr = matching.validation_reports[0]
                val_record = ValidationResult(
                    model_result_id=model_record.id,
                    vif_passed=vr.vif_passed,
                    max_vif=vr.max_vif,
                    shapiro_p=vr.shapiro_p_value,
                    normality_passed=vr.normality_passed,
                    bp_p=vr.bp_p_value,
                    homoscedasticity_passed=vr.homoscedasticity_passed,
                    anova_f=vr.anova_f_statistic,
                    anova_p=vr.anova_p_value,
                    anova_passed=vr.anova_passed,
                    cv_r2_mean=vr.cv_r2_mean,
                    cv_r2_std=vr.cv_r2_std,
                    cv_passed=vr.cv_passed,
                    eta_squared=vr.eta_squared,
                    effect_size_label=vr.effect_size_label,
                    overall_passed=vr.overall_passed,
                    pass_rate=vr.pass_rate,
                    n_tests_run=vr.n_tests_run,
                    n_tests_passed=vr.n_tests_passed,
                    warnings=vr.warnings,
                )
                self._session.add(val_record)
                await self._session.flush()

            # Persist citations
            if grounding:
                for fs in getattr(grounding, "finding_scores", []):
                    for citation in getattr(fs, "citations", [])[:5]:
                        cit_record = Citation(
                            model_result_id=model_record.id,
                            source=citation.source,
                            title=citation.title,
                            authors=citation.authors,
                            url=citation.url,
                            year=citation.year,
                            similarity_score=citation.similarity_score,
                            abstract_snippet=citation.abstract_snippet,
                            variable=fs.variable,
                        )
                        self._session.add(cit_record)

        await self._session.flush()
        logger.info(
            f"Persisted round {round_number} for run {run_id}: "
            f"{len(hypotheses_list)} hypotheses"
        )
