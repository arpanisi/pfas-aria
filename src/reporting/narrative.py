"""
Narrative Generator.
LLM writes three things — everything else is templated:
  1. Executive summary (150-200 words, plain English)
  2. Finding narratives (one paragraph per significant finding)
  3. Contradiction notes (if findings contradict cited papers)

The LLM never generates numbers. It only interprets them.
"""

from __future__ import annotations

import json

from src.reporting.sections import (
    CrossOutputConsistency,
    ParameterEffectRow,
    ReportSections,
)
from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class NarrativeGenerator:
    def __init__(self) -> None:
        self.settings = get_settings()

    def generate(self, sections: ReportSections) -> ReportSections:
        logger.info("Generating report narratives...")
        sections.executive_summary = self._executive_summary(sections)
        sections.finding_narratives = self._finding_narratives(sections)
        sections.contradiction_notes = self._contradiction_notes(sections)
        logger.info("Narratives complete")
        return sections

    def _executive_summary(self, sections: ReportSections) -> str:
        c = sections.classification
        cv = sections.convergence
        n_sig = sum(1 for e in sections.parameter_effects if e.is_significant)
        top_vars = [
            x.variable
            for x in sections.cross_output_consistency[:3]
            if x.consistency_score > 0.5
        ]
        context = {
            "experimental_group": c.experimental_group,
            "n_experiments": c.n_experiments,
            "n_rows": c.n_rows,
            "outcome_variable": c.outcome_variable,
            "active_outputs": c.active_outputs,
            "converged": cv.converged,
            "total_rounds": cv.total_rounds,
            "final_score": round(cv.final_score, 3),
            "stop_reason": cv.stop_reason,
            "n_significant_findings": n_sig,
            "top_variables": top_vars,
            "time_explains_pct": sections.time_structure.time_explains_pct,
            "n_citations": len(sections.citations),
            "pathway_metrics": sections.pathway_metrics.interpretation,
        }
        prompt = (
            "You are writing the executive summary of a scientific convergence report "
            "for a plasma-based PFAS degradation study.\n\n"
            f"Dataset context:\n{json.dumps(context, indent=2)}\n\n"
            "Write a 150-200 word executive summary for a domain scientist.\n"
            "Requirements:\n"
            "- Plain English, no statistical jargon\n"
            "- State which variables matter and in which direction\n"
            "- State convergence status\n"
            "- State literature agreement\n"
            "- Do NOT invent numbers not provided\n"
            "- Do NOT mention OLS, fixed effects, LASSO, or model families\n"
            "- Third person past tense\n"
            "Write only the paragraph, no heading."
        )
        return self._call_llm(
            prompt, max_tokens=300, fallback=self._fallback_summary(sections)
        )

    def _finding_narratives(self, sections: ReportSections) -> list[str]:
        narratives = []
        seen: set[str] = set()
        sig = [e for e in sections.parameter_effects if e.is_significant]
        for consistency in sections.cross_output_consistency[:5]:
            if consistency.consistency_score < 0.3 or consistency.variable in seen:
                continue
            seen.add(consistency.variable)
            var_effects = [e for e in sig if e.variable == consistency.variable]
            if not var_effects:
                continue
            supporting = next(
                (
                    c
                    for c in sections.citations
                    if c.supporting_variable == consistency.variable
                ),
                None,
            )
            context = {
                "variable": consistency.variable,
                "direction": consistency.dominant_direction,
                "outputs_significant": consistency.outputs_significant,
                "consistency_score": consistency.consistency_score,
                "effects": [
                    {
                        "output": e.output,
                        "coefficient": round(e.coefficient, 4),
                        "p_value": round(e.p_value, 4),
                        "effect_size": e.effect_size,
                        "ci": [round(e.ci_lower, 4), round(e.ci_upper, 4)],
                    }
                    for e in var_effects[:3]
                ],
                "literature_support": supporting.title if supporting else None,
                "similarity": supporting.similarity_score if supporting else None,
            }
            prompt = (
                "Write a 2-3 sentence scientific finding narrative for a PFAS degradation study.\n\n"
                f"Context:\n{json.dumps(context, indent=2)}\n\n"
                "Requirements:\n"
                "- 2-3 sentences only\n"
                "- Written for a domain scientist\n"
                "- State what the variable does, how strongly, whether literature agrees\n"
                "- Do NOT mention model families\n"
                "- Start with the variable name\n"
                "Write only the narrative."
            )
            narratives.append(
                self._call_llm(
                    prompt,
                    max_tokens=150,
                    fallback=self._fallback_finding(consistency, var_effects),
                )
            )
        return narratives

    def _contradiction_notes(self, sections: ReportSections) -> list[str]:
        notes = []
        for finding in sections.unresolved.contradicted_findings:
            prompt = (
                "Write one sentence noting a contradiction between a model finding "
                "and published literature in a PFAS degradation study.\n\n"
                f"Finding: {finding}\n\n"
                "Be factual. One sentence only."
            )
            notes.append(
                self._call_llm(prompt, max_tokens=80, fallback=f"Note: {finding}")
            )
        return notes

    def _call_llm(self, prompt: str, max_tokens: int = 200, fallback: str = "") -> str:
        try:
            import httpx

            from src.utils.resilience import get_llm_circuit

            def _do() -> str:
                r = httpx.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.settings.llm.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"num_predict": max_tokens, "temperature": 0.3},
                    },
                    timeout=60.0,
                )
                r.raise_for_status()
                return str(r.json().get("response", "").strip())

            return get_llm_circuit().call(_do) or fallback
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return fallback

    def _fallback_summary(self, sections: ReportSections) -> str:
        c = sections.classification
        cv = sections.convergence
        top = [
            x.variable
            for x in sections.cross_output_consistency[:3]
            if x.consistency_score > 0.5
        ]
        n_sig = sum(1 for e in sections.parameter_effects if e.is_significant)
        status = "converged" if cv.converged else "reached maximum rounds"
        return (
            f"The PFAS-ARIA system analyzed {c.n_experiments} experiments "
            f"({c.n_rows} observations) from a {c.experimental_group} dataset "
            f"across {cv.total_rounds} rounds of autonomous hypothesis testing. "
            f"The pipeline {status} with a match score of {cv.final_score:.2f}. "
            f"{n_sig} statistically significant parameter effects were identified "
            f"across {len(c.active_outputs)} response variables. "
            + (f"Key variables: {', '.join(top)}. " if top else "")
            + f"Results were grounded against {len(sections.citations)} literature sources."
        )

    def _fallback_finding(
        self, consistency: CrossOutputConsistency, effects: list[ParameterEffectRow]
    ) -> str:
        return (
            f"{consistency.variable} showed a {consistency.dominant_direction} and statistically significant "
            f"effect on {', '.join(consistency.outputs_significant) if consistency.outputs_significant else 'the outcome'}, "
            f"consistent across {consistency.consistency_score:.0%} of tested outputs."
        )
