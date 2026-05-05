"""
Literature Grounding Agent.
Orchestrates arXiv search, Semantic Scholar search, and similarity scoring
for every validated model result in a round.

For each hypothesis that passed validation:
  1. Extract significant findings
  2. Search arXiv and Semantic Scholar
  3. Score against RAG corpus + external papers
  4. Attach citations to findings
  5. Return per-hypothesis match scores for convergence judge
"""

from __future__ import annotations

from dataclasses import dataclass

from src.grounding.arxiv_client import ArxivClient
from src.grounding.scorer import Citation, GroundingResult, GroundingScorer
from src.grounding.semantic_scholar import SemanticScholarClient
from src.modeling.engine import ModelResult
from src.modeling.runner import ModelingRoundResult
from src.rag.retriever import Retriever
from src.utils.logging import get_logger
from src.validation.validator import ValidationReport

logger = get_logger(__name__)


# ── Output schema ─────────────────────────────────────────────────────────────


@dataclass
class RoundGroundingResult:
    """All grounding results for one full pipeline round."""

    round: int
    hypothesis_grounding: list[GroundingResult]
    global_match_score: float  # Mean across all hypotheses
    n_citations_found: int
    top_citations: list[Citation]  # Best across all hypotheses
    convergence_ready: bool  # match_score >= threshold
    structured_results: list[dict]  # For supervisor loop


# ── Agent ─────────────────────────────────────────────────────────────────────


class GroundingAgent:
    """
    Grounds each validated model result in open scientific literature.
    Returns per-hypothesis match scores used by the convergence judge.
    """

    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever
        self.arxiv = ArxivClient()
        self.s2 = SemanticScholarClient()
        self.scorer = GroundingScorer(retriever)

    def run(
        self,
        modeling_result: ModelingRoundResult,
        convergence_threshold: float = 0.75,
    ) -> RoundGroundingResult:
        """
        Ground all hypothesis results from a modeling round.
        Only grounds hypotheses that have at least one model result.
        """
        logger.info(
            f"=== Grounding Agent: Round {modeling_result.round} — "
            f"{modeling_result.n_hypotheses} hypotheses ==="
        )

        grounding_results: list[GroundingResult] = []

        for hr in modeling_result.hypothesis_results:
            if hr.best_model is None:
                logger.warning(f"Skipping {hr.hypothesis.id} — no model result")
                continue

            logger.info(f"Grounding hypothesis {hr.hypothesis.id}...")
            result = self._ground_hypothesis(
                hypothesis_id=hr.hypothesis.id,
                model_result=hr.best_model,
                validation_report=hr.validation_reports[0]
                if hr.validation_reports
                else None,
                significant_variables=hr.best_model.significant_variables,
                domain_context=self._get_domain_context(),
            )
            grounding_results.append(result)

        # Aggregate
        if grounding_results:
            global_score = sum(r.global_match_score for r in grounding_results) / len(
                grounding_results
            )
        else:
            global_score = 0.0

        all_citations = [c for r in grounding_results for c in r.top_citations]
        seen: set[str] = set()
        top_citations: list[Citation] = []
        for c in sorted(all_citations, key=lambda x: x.similarity_score, reverse=True):
            if c.url not in seen:
                top_citations.append(c)
                seen.add(c.url)
            if len(top_citations) >= 15:
                break

        structured = self._build_structured_results(grounding_results, modeling_result)

        round_result = RoundGroundingResult(
            round=modeling_result.round,
            hypothesis_grounding=grounding_results,
            global_match_score=round(global_score, 4),
            n_citations_found=len(seen),
            top_citations=top_citations,
            convergence_ready=global_score >= convergence_threshold,
            structured_results=structured,
        )

        logger.info(
            f"=== Grounding Agent: Complete — "
            f"global match={global_score:.4f}, "
            f"citations={len(seen)}, "
            f"convergence={'YES' if round_result.convergence_ready else 'NO'} ==="
        )

        return round_result

    # ── Per-hypothesis grounding ──────────────────────────────────────────────

    def _ground_hypothesis(
        self,
        hypothesis_id: str,
        model_result: ModelResult,
        validation_report: ValidationReport | None,
        significant_variables: list[str],
        domain_context: str,
    ) -> GroundingResult:
        # Build finding description for search
        finding = self._describe_finding(model_result, significant_variables)

        # Search arXiv
        logger.debug(f"  Searching arXiv for {hypothesis_id}...")
        arxiv_papers = self.arxiv.search_for_finding(
            finding=finding,
            significant_variables=significant_variables[:5],
            domain_context=domain_context,
        )

        # Search Semantic Scholar
        logger.debug(f"  Searching Semantic Scholar for {hypothesis_id}...")
        s2_papers = self.s2.search_for_finding(
            significant_variables=significant_variables[:5],
            domain_context=domain_context,
        )

        # Create a dummy validation report if none provided
        if validation_report is None:
            from src.validation.validator import ValidationReport

            validation_report = ValidationReport(
                hypothesis_id=hypothesis_id,
                model_type=model_result.model_type,
                pass_rate=0.5,
            )

        # Score everything
        result = self.scorer.score(
            model_result=model_result,
            validation_report=validation_report,
            arxiv_papers=arxiv_papers,
            s2_papers=s2_papers,
        )

        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _describe_finding(
        self, model_result: ModelResult, significant_variables: list[str]
    ) -> str:
        """Build a plain-language description of the key model finding."""
        if not significant_variables:
            return f"model of {model_result.outcome_variable}"

        top_var = significant_variables[0]
        coeff = model_result.coefficients.get(top_var, 0.0)
        direction = "increases" if coeff > 0 else "decreases"

        return (
            f"{top_var} {direction} {model_result.outcome_variable} "
            f"in PFAS photochemical degradation"
        )

    def _get_domain_context(self) -> str:
        try:
            from src.utils.config import get_settings

            return get_settings().domain.context
        except Exception:
            return "PFAS photochemical degradation"

    def _build_structured_results(
        self,
        grounding_results: list[GroundingResult],
        modeling_result: ModelingRoundResult,
    ) -> list[dict]:
        """Build structured dicts for the supervisor loop."""
        structured = []
        for gr in grounding_results:
            # Find matching modeling summary
            modeling_summary = next(
                (
                    s
                    for s in modeling_result.structured_results
                    if s.get("hypothesis_id") == gr.hypothesis_id
                ),
                {},
            )
            merged = dict(modeling_summary)
            merged["match_score"] = gr.global_match_score
            merged["n_citations"] = len(gr.top_citations)
            merged["top_citations"] = [
                {
                    "title": c.title,
                    "url": c.url,
                    "similarity": c.similarity_score,
                    "source": c.source,
                }
                for c in gr.top_citations[:3]
            ]
            structured.append(merged)

        return structured
