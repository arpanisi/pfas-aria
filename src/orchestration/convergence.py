"""
Convergence Judge.
Computes and tracks the global match score across rounds.
Provides convergence history and stopping decision logic.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.orchestration.state import PipelineState
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConvergenceReport:
    """Summary of convergence across all rounds."""

    run_id: str
    total_rounds: int
    final_score: float
    threshold: float
    converged: bool
    stop_reason: str
    score_history: list[float]
    best_round: int
    best_score: float
    improvement_per_round: list[float]
    top_variables_across_rounds: list[str]
    top_citations_across_rounds: list[dict]


class ConvergenceJudge:
    """
    Analyses pipeline state to produce a convergence report.
    Called once after the pipeline finishes.
    """

    def evaluate(self, state: PipelineState) -> ConvergenceReport:
        scores = state.convergence_scores
        summaries = state.round_summaries

        best_round = int(scores.index(max(scores))) + 1 if scores else 0
        best_score = max(scores) if scores else 0.0

        improvement = []
        for i in range(1, len(scores)):
            improvement.append(round(scores[i] - scores[i - 1], 4))

        # Aggregate top variables across all rounds
        var_counts: dict[str, int] = {}
        for s in summaries:
            for var in s.top_variables:
                var_counts[var] = var_counts.get(var, 0) + 1
        top_vars = sorted(var_counts, key=lambda v: var_counts[v], reverse=True)[:10]

        # Aggregate top citations across all rounds
        seen_urls: set[str] = set()
        top_citations: list[dict] = []
        for s in summaries:
            for c in s.top_citations:
                if c.get("url") not in seen_urls:
                    top_citations.append(c)
                    seen_urls.add(str(c.get("url", "")))

        report = ConvergenceReport(
            run_id=state.run_id,
            total_rounds=state.current_round,
            final_score=state.final_match_score,
            threshold=state.convergence_threshold,
            converged=state.final_match_score >= state.convergence_threshold,
            stop_reason=state.stop_reason,
            score_history=scores,
            best_round=best_round,
            best_score=best_score,
            improvement_per_round=improvement,
            top_variables_across_rounds=top_vars,
            top_citations_across_rounds=top_citations[:15],
        )

        self._log_report(report)
        return report

    def _log_report(self, report: ConvergenceReport) -> None:
        logger.info(
            f"\n{'=' * 50}\n"
            f"CONVERGENCE REPORT — Run {report.run_id}\n"
            f"{'=' * 50}\n"
            f"Rounds completed : {report.total_rounds}\n"
            f"Final score      : {report.final_score:.4f}\n"
            f"Threshold        : {report.threshold}\n"
            f"Converged        : {'YES' if report.converged else 'NO'}\n"
            f"Stop reason      : {report.stop_reason}\n"
            f"Best round       : {report.best_round} (score={report.best_score:.4f})\n"
            f"Score history    : {[round(s, 3) for s in report.score_history]}\n"
            f"Top variables    : {report.top_variables_across_rounds[:5]}\n"
            f"{'=' * 50}"
        )
