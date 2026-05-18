"""
Report Sections.
Each function takes pipeline state objects and returns
structured section data — dicts and lists, not strings.
The narrative.py module adds LLM-written prose.
The exporter.py module renders everything to Markdown and PDF.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class SystemClassification:
    experimental_group: str
    n_rows: int
    n_experiments: int
    n_timepoints: int
    outcome_variable: str
    active_outputs: list[str]
    active_predictors: list[str]
    has_panel_structure: bool
    inflection_timepoint: float | None


@dataclass
class TimeStructureSection:
    within_condition_r2: dict[str, float]
    dominant_timepoints: list[float]
    inflection_detected: float | None
    time_explains_pct: float


@dataclass
class ParameterEffectRow:
    variable: str
    output: str
    coefficient: float
    std_error: float
    p_value: float
    ci_lower: float
    ci_upper: float
    effect_size: str
    is_significant: bool
    direction: str = field(init=False)

    def __post_init__(self) -> None:
        self.direction = "positive" if self.coefficient > 0 else "negative"


@dataclass
class CrossOutputConsistency:
    variable: str
    outputs_significant: list[str]
    outputs_tested: list[str]
    consistency_score: float
    dominant_direction: str


@dataclass
class PathwayMetricsSection:
    has_entropy: bool
    has_kl: bool
    entropy_peak_time: float | None
    entropy_collapse_time: float | None
    kl_terminal_value: float | None
    surfactant_effect_on_entropy: float | None
    interpretation: str


@dataclass
class CitationEntry:
    rank: int
    title: str
    source: str
    year: str | None
    url: str | None
    similarity_score: float
    supporting_variable: str | None
    agrees_with_finding: bool | None


@dataclass
class ConvergenceSection:
    total_rounds: int
    final_score: float
    threshold: float
    converged: bool
    stop_reason: str
    score_history: list[float]
    best_round: int
    best_score: float
    improvement_per_round: list[float]


@dataclass
class UnresolvedFindings:
    low_match_findings: list[str]
    contradicted_findings: list[str]
    low_r2_outputs: list[str]
    caveats: list[str]


@dataclass
class ReproducibilitySection:
    run_id: str
    run_name: str
    generated_at: str
    data_filename: str
    data_hash: str | None
    software_versions: dict[str, str]
    config_snapshot: dict[str, Any]
    rerun_command: str


@dataclass
class ReportSections:
    title: str
    run_id: str
    generated_at: str
    classification: SystemClassification
    time_structure: TimeStructureSection
    parameter_effects: list[ParameterEffectRow]
    cross_output_consistency: list[CrossOutputConsistency]
    pathway_metrics: PathwayMetricsSection
    citations: list[CitationEntry]
    convergence: ConvergenceSection
    unresolved: UnresolvedFindings
    reproducibility: ReproducibilitySection
    executive_summary: str = ""
    finding_narratives: list[str] = field(default_factory=list)
    contradiction_notes: list[str] = field(default_factory=list)


def build_sections(pipeline_state: Any, convergence_report: Any) -> ReportSections:
    """Build all report sections from pipeline state."""
    import importlib.metadata
    import sys

    now = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    db = getattr(pipeline_state, "data_bundle", None)
    profile = getattr(pipeline_state, "dataset_profile", None)
    panel_results = getattr(pipeline_state, "panel_results", {})

    # Classification
    n_rows = getattr(db, "n_rows", 0) if db else 0
    outcome = getattr(db, "outcome_variable", "unknown") if db else "unknown"
    active_outputs = [o.name for o in profile.active_outputs] if profile else [outcome]
    active_predictors = profile.active_predictors if profile else []
    is_panel = profile.is_panel if profile else False
    inflection = profile.split_timepoint if profile else None
    n_entities = profile.n_entities if profile else 0
    n_timepoints = profile.n_timepoints if profile else 0
    composition_cols = getattr(profile, "composition_columns", []) if profile else []

    group = _classify_group(composition_cols)

    classification = SystemClassification(
        experimental_group=group,
        n_rows=n_rows,
        n_experiments=n_entities,
        n_timepoints=n_timepoints,
        outcome_variable=outcome,
        active_outputs=active_outputs,
        active_predictors=active_predictors[:10],
        has_panel_structure=is_panel,
        inflection_timepoint=inflection,
    )

    # Time structure
    within_r2 = {
        name: getattr(res, "within_r2", 0.0) for name, res in panel_results.items()
    }
    avg_within = sum(within_r2.values()) / max(len(within_r2), 1)
    all_sig_times: list[float] = []
    for res in panel_results.values():
        for te in getattr(res, "time_effects", []):
            if te.is_significant:
                all_sig_times.append(te.time_value)
    dominant_times = [t for t, _ in Counter(all_sig_times).most_common(5)]

    time_structure = TimeStructureSection(
        within_condition_r2=within_r2,
        dominant_timepoints=sorted(dominant_times),
        inflection_detected=inflection,
        time_explains_pct=round(avg_within * 100, 1),
    )

    # Parameter effects
    param_effects: list[ParameterEffectRow] = []
    for output_name, res in panel_results.items():
        for pe in getattr(res, "parameter_effects", []):
            param_effects.append(
                ParameterEffectRow(
                    variable=pe.variable,
                    output=output_name,
                    coefficient=pe.coefficient,
                    std_error=pe.std_error,
                    p_value=pe.p_value,
                    ci_lower=pe.ci_lower,
                    ci_upper=pe.ci_upper,
                    effect_size=pe.effect_size,
                    is_significant=pe.is_significant,
                )
            )

    consistency = _cross_output_consistency(param_effects, list(panel_results.keys()))
    pathway = _pathway_section(pipeline_state, panel_results)
    raw_gr = getattr(pipeline_state, "grounding_result", None)
    citations = _citation_entries(raw_gr)

    convergence = ConvergenceSection(
        total_rounds=convergence_report.total_rounds,
        final_score=convergence_report.final_score,
        threshold=convergence_report.threshold,
        converged=convergence_report.converged,
        stop_reason=convergence_report.stop_reason,
        score_history=convergence_report.score_history,
        best_round=convergence_report.best_round,
        best_score=convergence_report.best_score,
        improvement_per_round=convergence_report.improvement_per_round,
    )

    unresolved = _unresolved(panel_results, citations, convergence_report)

    try:
        versions: dict[str, str] = {
            "python": sys.version.split()[0],
            "pfas-aria": "0.1.0",
            "linearmodels": importlib.metadata.version("linearmodels"),
            "statsmodels": importlib.metadata.version("statsmodels"),
        }
    except Exception:
        versions = {"pfas-aria": "0.1.0"}

    data_file = getattr(db, "source_path", "unknown") if db else "unknown"
    config = getattr(pipeline_state, "config", {})

    reproducibility = ReproducibilitySection(
        run_id=pipeline_state.run_id,
        run_name=pipeline_state.run_name,
        generated_at=now,
        data_filename=data_file,
        data_hash=getattr(db, "content_hash", None) if db else None,
        software_versions=versions,
        config_snapshot=config,
        rerun_command=f"make run-pipeline RUN_NAME={pipeline_state.run_name}",
    )

    return ReportSections(
        title=f"PFAS-ARIA Convergence Report — {pipeline_state.run_name}",
        run_id=pipeline_state.run_id,
        generated_at=now,
        classification=classification,
        time_structure=time_structure,
        parameter_effects=param_effects,
        cross_output_consistency=consistency,
        pathway_metrics=pathway,
        citations=citations,
        convergence=convergence,
        unresolved=unresolved,
        reproducibility=reproducibility,
    )


def _classify_group(composition_cols: list[str]) -> str:
    n = len(composition_cols)
    if n >= 5:
        return "Composition Series (5+ components)"
    if n == 1:
        return f"Single Component ({composition_cols[0]})"
    if n == 0:
        return "No Composition Series"
    return f"Partial Composition Series ({n} components)"


def _cross_output_consistency(
    effects: list[ParameterEffectRow],
    all_outputs: list[str],
) -> list[CrossOutputConsistency]:
    var_outputs: dict[str, list[ParameterEffectRow]] = defaultdict(list)
    for e in effects:
        var_outputs[e.variable].append(e)
    results = []
    for var, rows in var_outputs.items():
        sig = [r for r in rows if r.is_significant]
        pos = sum(1 for r in sig if r.coefficient > 0)
        neg = len(sig) - pos
        direction = "positive" if pos > neg else ("negative" if neg > pos else "mixed")
        results.append(
            CrossOutputConsistency(
                variable=var,
                outputs_significant=[r.output for r in sig],
                outputs_tested=all_outputs,
                consistency_score=round(len(sig) / max(len(rows), 1), 2),
                dominant_direction=direction,
            )
        )
    return sorted(results, key=lambda x: x.consistency_score, reverse=True)


def _pathway_section(pipeline_state: Any, panel_results: dict) -> PathwayMetricsSection:
    has_entropy = "shannon_entropy" in panel_results
    has_kl = "kl_divergence" in panel_results
    surfactant_effect = None
    if has_entropy:
        for pe in getattr(panel_results["shannon_entropy"], "parameter_effects", []):
            if "surfactant" in pe.variable.lower() and pe.is_significant:
                surfactant_effect = pe.coefficient
                break
    interp = (
        "Shannon entropy and KL divergence computed from composition distribution."
        if has_entropy and has_kl
        else "Shannon entropy computed — tracks pathway spread."
        if has_entropy
        else "KL divergence computed — tracks shift from initial composition."
        if has_kl
        else "No composition series detected — pathway metrics not computed."
    )
    return PathwayMetricsSection(
        has_entropy=has_entropy,
        has_kl=has_kl,
        entropy_peak_time=getattr(pipeline_state, "inflection_timepoint", None),
        entropy_collapse_time=None,
        kl_terminal_value=None,
        surfactant_effect_on_entropy=surfactant_effect,
        interpretation=interp,
    )


def _citation_entries(grounding_result: Any) -> list[CitationEntry]:
    if grounding_result is None:
        return []
    return [
        CitationEntry(
            rank=i + 1,
            title=getattr(c, "title", "Unknown"),
            source=getattr(c, "source", "unknown"),
            year=getattr(c, "year", None),
            url=getattr(c, "url", None),
            similarity_score=getattr(c, "similarity_score", 0.0),
            supporting_variable=getattr(c, "variable", None),
            agrees_with_finding=None,
        )
        for i, c in enumerate(getattr(grounding_result, "top_citations", [])[:20])
    ]


def _unresolved(
    panel_results: dict,
    citations: list[CitationEntry],
    convergence_report: Any,
) -> UnresolvedFindings:
    low_r2 = [
        f"{name} (R²={getattr(res, 'overall_r2', 0):.3f})"
        for name, res in panel_results.items()
        if getattr(res, "overall_r2", 1.0) < 0.4
    ]
    caveats = []
    if not convergence_report.converged:
        caveats.append(
            f"Pipeline did not converge — stopped at max rounds "
            f"(score={convergence_report.final_score:.3f}, threshold={convergence_report.threshold})"
        )
    low_sim = [c for c in citations if c.similarity_score < 0.5]
    if low_sim:
        caveats.append(
            f"{len(low_sim)} citations have similarity < 0.50 — weak literature support"
        )
    return UnresolvedFindings(
        low_match_findings=[],
        contradicted_findings=[],
        low_r2_outputs=low_r2,
        caveats=caveats,
    )
