"""
Pipeline Entrypoint.
The single function you call to run PFAS-ARIA end-to-end.

Usage:
    python -m src.orchestration.pipeline
    # or
    from src.orchestration.pipeline import run_pipeline
    state, report = run_pipeline()
"""

from __future__ import annotations

import sys

from src.orchestration.convergence import ConvergenceJudge, ConvergenceReport
from src.orchestration.state import PipelineState, PipelineStatus
from src.orchestration.supervisor import SupervisorAgent
from src.utils.logging import get_logger, setup_logging
from src.utils.paths import ensure_dirs

logger = get_logger(__name__)


def run_pipeline(
    run_name: str | None = None,
) -> tuple[PipelineState, ConvergenceReport]:
    """
    Run the full PFAS-ARIA pipeline end-to-end.

    Returns:
        (PipelineState, ConvergenceReport) — full state and convergence summary
    """
    setup_logging()
    ensure_dirs()

    logger.info("PFAS-ARIA pipeline initializing...")

    supervisor = SupervisorAgent()
    final_state = supervisor.run(run_name=run_name)

    if final_state.status == PipelineStatus.FAILED:
        logger.error(f"Pipeline failed: {final_state.errors}")
        raise RuntimeError(
            f"Pipeline failed after round {final_state.current_round}.\n"
            f"Errors: {final_state.errors}"
        )

    convergence_report = ConvergenceJudge().evaluate(final_state)

    logger.info(
        f"Pipeline complete — "
        f"{'CONVERGED' if convergence_report.converged else 'MAX ROUNDS'} "
        f"in {convergence_report.total_rounds} rounds"
    )

    return final_state, convergence_report


def print_summary(state: PipelineState, report: ConvergenceReport) -> None:
    """Print a human-readable summary to stdout."""
    print("\n" + "=" * 60)
    print("PFAS-ARIA RUN SUMMARY")
    print("=" * 60)
    print(f"Run ID          : {report.run_id}")
    print(f"Rounds          : {report.total_rounds}")
    print(f"Final score     : {report.final_score:.4f}")
    print(f"Converged       : {'YES ✓' if report.converged else 'NO ✗'}")
    print(f"Stop reason     : {report.stop_reason}")
    print(f"Score history   : {[round(s, 3) for s in report.score_history]}")
    print()
    print("Top variables found significant:")
    for var in report.top_variables_across_rounds[:5]:
        print(f"  • {var}")
    print()
    print("Top citations:")
    for c in report.top_citations_across_rounds[:5]:
        print(f"  • {c.get('title', 'Unknown')[:70]}")
        print(f"    Score: {c.get('score', 0):.3f} | {c.get('url', '')}")
    print("=" * 60)


if __name__ == "__main__":
    run_name = sys.argv[1] if len(sys.argv) > 1 else None
    state, report = run_pipeline(run_name=run_name)
    print_summary(state, report)
