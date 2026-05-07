"""
Report Generator.
Orchestrates the full report build at pipeline convergence.

Usage:
    from src.reporting.generator import generate_report
    md_path, pdf_path = generate_report(pipeline_state, convergence_report)
"""

from __future__ import annotations

from pathlib import Path

from src.orchestration.convergence import ConvergenceReport
from src.orchestration.state import PipelineState
from src.reporting.exporter import export_markdown, export_pdf
from src.reporting.narrative import NarrativeGenerator
from src.reporting.sections import build_sections
from src.utils.logging import get_logger

logger = get_logger(__name__)


def generate_report(
    pipeline_state: PipelineState,
    convergence_report: ConvergenceReport,
    output_dir: Path | None = None,
    skip_pdf: bool = False,
) -> tuple[Path, Path | None]:
    """
    Generate the full convergence report.
    Returns (markdown_path, pdf_path).
    pdf_path is None if skip_pdf=True or PDF export fails.
    """
    logger.info(f"Generating report for run {pipeline_state.run_id}...")

    sections = build_sections(pipeline_state, convergence_report)

    try:
        sections = NarrativeGenerator().generate(sections)
    except Exception as e:
        logger.warning(f"Narrative generation failed: {e} — using fallbacks")

    md_path = export_markdown(
        sections,
        output_dir / f"{pipeline_state.run_id}_report.md" if output_dir else None,
    )

    pdf_path: Path | None = None
    if not skip_pdf:
        try:
            pdf_path = export_pdf(
                sections,
                output_dir / f"{pipeline_state.run_id}_report.pdf"
                if output_dir
                else None,
            )
        except Exception as e:
            logger.warning(f"PDF export failed: {e} — Markdown still available")

    logger.info(
        f"Report complete — "
        f"{'CONVERGED' if convergence_report.converged else 'MAX ROUNDS'}, "
        f"{convergence_report.total_rounds} rounds, "
        f"score={convergence_report.final_score:.4f}"
    )
    return md_path, pdf_path
