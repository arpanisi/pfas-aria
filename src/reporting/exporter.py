"""
Report Exporter.
Renders ReportSections to:
  1. Markdown (.md) — machine-readable, version-controllable
  2. PDF (.pdf) — for sharing and publication, via reportlab

The Markdown is generated first — the PDF is rendered from the same data.
"""

from __future__ import annotations

from pathlib import Path

from src.reporting.sections import ReportSections
from src.utils.logging import get_logger
from src.utils.paths import REPORTS_DIR

logger = get_logger(__name__)


# ── Markdown exporter ─────────────────────────────────────────────────────────


def export_markdown(sections: ReportSections, output_path: Path | None = None) -> Path:
    """Render report to Markdown. Returns path to the written file."""
    path = output_path or REPORTS_DIR / f"{sections.run_id}_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    def h1(text: str) -> None:
        lines.extend([f"# {text}", ""])

    def h2(text: str) -> None:
        lines.extend([f"## {text}", ""])

    def h3(text: str) -> None:
        lines.extend([f"### {text}", ""])

    def para(text: str) -> None:
        if text:
            lines.extend([text, ""])

    def table(headers: list[str], rows: list[list[str]]) -> None:
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        lines.append("")

    # Title and metadata
    h1(sections.title)
    lines.extend(
        [
            f"**Run ID:** `{sections.run_id}`",
            f"**Generated:** {sections.generated_at}",
            "",
        ]
    )

    # Executive summary
    h2("Executive Summary")
    para(sections.executive_summary)

    # 1. System classification
    h2("1. System Classification")
    c = sections.classification
    table(
        ["Property", "Value"],
        [
            ["Experimental group", c.experimental_group],
            ["Dataset size", f"{c.n_rows:,} rows"],
            ["Experiments", str(c.n_experiments)],
            ["Timepoints per experiment", str(c.n_timepoints)],
            ["Outcome variable", c.outcome_variable],
            ["Panel structure detected", "Yes" if c.has_panel_structure else "No"],
            [
                "Inflection point",
                f"{c.inflection_timepoint} min"
                if c.inflection_timepoint
                else "Not detected",
            ],
        ],
    )
    lines.append(f"**Active outputs modeled:** {', '.join(c.active_outputs)}")
    lines.append("")
    lines.append(f"**Active predictors:** {', '.join(c.active_predictors[:8])}")
    lines.append("")

    # 2. Time structure
    h2("2. Time Structure")
    para(
        f"Time-dependent pathway evolution explains **{sections.time_structure.time_explains_pct:.1f}%** "
        f"of within-experiment variation on average. "
        "This means experimental conditions produce systematic but smaller deviations "
        "around a dominant time-driven trajectory."
    )
    if sections.time_structure.within_condition_r2:
        table(
            ["Output", "Within-condition R²", "Interpretation"],
            [
                [
                    name,
                    f"{r2:.4f}",
                    "High — time is dominant"
                    if r2 > 0.7
                    else "Moderate"
                    if r2 > 0.4
                    else "Low — parameters matter more",
                ]
                for name, r2 in sections.time_structure.within_condition_r2.items()
            ],
        )
    if sections.time_structure.inflection_detected:
        para(
            f"**Temporal inflection detected at {sections.time_structure.inflection_detected:.0f} minutes.** "
            "Pathway behavior diverges between experimental conditions after this point."
        )

    # 3. Parameter effects
    h2("3. Parameter Effects")
    sig = [e for e in sections.parameter_effects if e.is_significant]
    if sig:
        table(
            [
                "Variable",
                "Output",
                "Coefficient",
                "95% CI",
                "p-value",
                "Effect size",
                "Direction",
            ],
            [
                [
                    e.variable,
                    e.output,
                    f"{e.coefficient:.4f}",
                    f"[{e.ci_lower:.4f}, {e.ci_upper:.4f}]",
                    f"{e.p_value:.4f}",
                    e.effect_size,
                    e.direction,
                ]
                for e in sorted(sig, key=lambda x: abs(x.coefficient), reverse=True)
            ],
        )
    else:
        para("No statistically significant parameter effects detected (p < 0.05).")

    # 4. Cross-output consistency
    h2("4. Cross-Output Consistency")
    para(
        "Variables significant across multiple response variables are more robust findings. "
        "A consistency score of 1.0 means the variable was significant in all modeled outputs."
    )
    consistent = [
        x for x in sections.cross_output_consistency if x.consistency_score > 0
    ]
    if consistent:
        table(
            ["Variable", "Significant in", "Consistency", "Direction"],
            [
                [
                    x.variable,
                    ", ".join(x.outputs_significant) or "none",
                    f"{x.consistency_score:.2f}",
                    x.dominant_direction,
                ]
                for x in consistent[:10]
            ],
        )

    # 5. Finding narratives
    if sections.finding_narratives:
        h2("5. Key Findings")
        for i, narrative in enumerate(sections.finding_narratives, 1):
            h3(f"Finding {i}")
            para(narrative)

    # 6. Pathway metrics
    h2("6. Pathway Metrics")
    pm = sections.pathway_metrics
    para(pm.interpretation)
    if pm.has_entropy or pm.has_kl:
        rows = []
        if pm.has_entropy:
            rows.append(
                [
                    "Shannon Entropy",
                    "Computed",
                    f"Peak at {pm.entropy_peak_time:.0f} min"
                    if pm.entropy_peak_time
                    else "—",
                ]
            )
        if pm.has_kl:
            rows.append(
                ["KL Divergence", "Computed", "Tracks shift from initial composition"]
            )
        if pm.surfactant_effect_on_entropy is not None:
            rows.append(
                [
                    "Surfactant → Entropy",
                    f"β = {pm.surfactant_effect_on_entropy:.4f}",
                    "Reduces pathway spread",
                ]
            )
        table(["Metric", "Status", "Notes"], rows)

    # 7. Literature citations
    h2("7. Literature Support")
    if sections.citations:
        table(
            ["Rank", "Source", "Year", "Similarity", "Variable", "Title"],
            [
                [
                    str(c.rank),
                    c.source.replace("semantic_scholar", "S2"),
                    c.year or "—",
                    f"{c.similarity_score:.2f}",
                    c.supporting_variable or "—",
                    c.title[:60] + "..." if len(c.title) > 60 else c.title,
                ]
                for c in sections.citations[:15]
            ],
        )
    else:
        para("No citations found.")

    # 8. Convergence history
    h2("8. Convergence History")
    cv = sections.convergence
    table(
        ["Property", "Value"],
        [
            ["Total rounds", str(cv.total_rounds)],
            ["Final match score", f"{cv.final_score:.4f}"],
            ["Threshold", str(cv.threshold)],
            ["Converged", "YES ✓" if cv.converged else "NO ✗"],
            ["Stop reason", cv.stop_reason],
            ["Best round", f"Round {cv.best_round} (score={cv.best_score:.4f})"],
        ],
    )
    if cv.score_history:
        lines.append("**Score per round:**")
        lines.append("")
        for i, score in enumerate(cv.score_history, 1):
            bar = "█" * int(score * 20)
            lines.append(f"  Round {i}: {score:.4f} {bar}")
        lines.append("")

    # 9. Unresolved findings
    h2("9. What the System Could Not Resolve")
    u = sections.unresolved
    if u.low_r2_outputs:
        para(f"**Low explanatory power:** {', '.join(u.low_r2_outputs)}")
    if u.contradicted_findings:
        para(
            f"**Contradictions with literature:** {'; '.join(u.contradicted_findings)}"
        )
    if u.caveats:
        for caveat in u.caveats:
            para(f"⚠ {caveat}")
    if not any([u.low_r2_outputs, u.contradicted_findings, u.caveats]):
        para("No major unresolved issues identified.")

    # 10. Reproducibility
    h2("10. Reproducibility")
    r = sections.reproducibility
    table(
        ["Property", "Value"],
        [
            ["Run ID", r.run_id],
            ["Data file", r.data_filename],
            ["Data hash", r.data_hash or "not recorded"],
            ["Generated at", r.generated_at],
        ]
        + [[k, v] for k, v in r.software_versions.items()],
    )
    lines.append(f"**To re-run:** `{r.rerun_command}`")
    lines.append("")

    # Write file
    content = "\n".join(lines)
    path.write_text(content, encoding="utf-8")
    logger.info(f"Markdown report written: {path}")
    return path


# ── PDF exporter ──────────────────────────────────────────────────────────────


def export_pdf(sections: ReportSections, output_path: Path | None = None) -> Path:
    """Render report to PDF using reportlab. Returns path to PDF file."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    path = output_path or REPORTS_DIR / f"{sections.run_id}_report.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    base = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=base["Title"],
        fontSize=18,
        spaceAfter=6,
        textColor=colors.HexColor("#0a0d12"),
    )
    h1_style = ParagraphStyle(
        "H1",
        parent=base["Heading1"],
        fontSize=14,
        spaceBefore=14,
        spaceAfter=6,
        textColor=colors.HexColor("#00a888"),
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=base["Heading2"],
        fontSize=11,
        spaceBefore=10,
        spaceAfter=4,
        textColor=colors.HexColor("#1a1f2e"),
    )
    body_style = ParagraphStyle(
        "Body",
        parent=base["Normal"],
        fontSize=9,
        spaceAfter=6,
        leading=13,
    )
    meta_style = ParagraphStyle(
        "Meta",
        parent=base["Normal"],
        fontSize=8,
        textColor=colors.grey,
        spaceAfter=4,
    )

    def tbl(
        headers: list[str], rows: list[list[str]], col_widths: list | None = None
    ) -> Table:
        data = [headers] + rows
        t = Table(data, colWidths=col_widths)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f2f5")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1a1f2e")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f9fafb")],
                    ),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e6ed")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("PADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return t

    story: list = []
    c = sections.classification
    cv = sections.convergence

    # Title page
    story.append(Paragraph(sections.title, title_style))
    story.append(
        Paragraph(
            f"Run ID: {sections.run_id} | Generated: {sections.generated_at}",
            meta_style,
        )
    )
    story.append(Spacer(1, 0.2 * inch))

    # Executive summary
    story.append(Paragraph("Executive Summary", h1_style))
    story.append(
        Paragraph(sections.executive_summary or "See full report below.", body_style)
    )
    story.append(Spacer(1, 0.1 * inch))

    # System classification
    story.append(Paragraph("1. System Classification", h1_style))
    story.append(
        tbl(
            ["Property", "Value"],
            [
                ["Experimental group", c.experimental_group],
                ["Dataset", f"{c.n_rows:,} rows, {c.n_experiments} experiments"],
                ["Outcome variable", c.outcome_variable],
                ["Panel structure", "Yes" if c.has_panel_structure else "No"],
                ["Active outputs", ", ".join(c.active_outputs)],
            ],
            col_widths=[2.5 * inch, 4 * inch],
        )
    )

    # Time structure
    story.append(Paragraph("2. Time Structure", h1_style))
    story.append(
        Paragraph(
            f"Time-dependent pathway evolution explains {sections.time_structure.time_explains_pct:.1f}% "
            "of within-experiment variation on average.",
            body_style,
        )
    )
    if sections.time_structure.within_condition_r2:
        story.append(
            tbl(
                ["Output", "Within R²"],
                [
                    [name, f"{r2:.4f}"]
                    for name, r2 in sections.time_structure.within_condition_r2.items()
                ],
                col_widths=[3 * inch, 2 * inch],
            )
        )

    # Parameter effects
    story.append(Paragraph("3. Parameter Effects", h1_style))
    sig = [e for e in sections.parameter_effects if e.is_significant]
    if sig:
        story.append(
            tbl(
                ["Variable", "Output", "β", "p-value", "Effect", "Direction"],
                [
                    [
                        e.variable[:20],
                        e.output[:20],
                        f"{e.coefficient:.4f}",
                        f"{e.p_value:.4f}",
                        e.effect_size,
                        e.direction,
                    ]
                    for e in sorted(
                        sig, key=lambda x: abs(x.coefficient), reverse=True
                    )[:15]
                ],
                col_widths=[
                    1.5 * inch,
                    1.5 * inch,
                    0.8 * inch,
                    0.8 * inch,
                    0.8 * inch,
                    0.8 * inch,
                ],
            )
        )
    else:
        story.append(
            Paragraph("No statistically significant effects detected.", body_style)
        )

    # Finding narratives
    if sections.finding_narratives:
        story.append(Paragraph("4. Key Findings", h1_style))
        for i, narrative in enumerate(sections.finding_narratives, 1):
            story.append(Paragraph(f"Finding {i}", h2_style))
            story.append(Paragraph(narrative, body_style))

    # Convergence
    story.append(PageBreak())
    story.append(Paragraph("5. Convergence History", h1_style))
    story.append(
        tbl(
            ["Property", "Value"],
            [
                ["Total rounds", str(cv.total_rounds)],
                ["Final score", f"{cv.final_score:.4f}"],
                ["Threshold", str(cv.threshold)],
                ["Converged", "YES" if cv.converged else "NO"],
                ["Stop reason", cv.stop_reason],
                ["Best round", f"Round {cv.best_round} (score={cv.best_score:.4f})"],
            ],
            col_widths=[2.5 * inch, 4 * inch],
        )
    )

    # Citations
    story.append(Paragraph("6. Literature Support", h1_style))
    if sections.citations:
        story.append(
            tbl(
                ["Rank", "Source", "Similarity", "Title"],
                [
                    [
                        str(c.rank),
                        c.source.replace("semantic_scholar", "S2"),
                        f"{c.similarity_score:.2f}",
                        c.title[:50] + "..." if len(c.title) > 50 else c.title,
                    ]
                    for c in sections.citations[:10]
                ],
                col_widths=[0.5 * inch, 0.8 * inch, 0.8 * inch, 4.5 * inch],
            )
        )

    # Reproducibility
    story.append(Paragraph("7. Reproducibility", h1_style))
    r = sections.reproducibility
    story.append(
        tbl(
            ["Property", "Value"],
            [
                ["Run ID", r.run_id],
                ["Data file", r.data_filename],
                ["Data hash", r.data_hash or "not recorded"],
            ]
            + [[k, v] for k, v in r.software_versions.items()]
            + [["Re-run command", r.rerun_command]],
            col_widths=[2 * inch, 4.5 * inch],
        )
    )

    doc.build(story)
    logger.info(f"PDF report written: {path}")
    return path
