"""
Pipeline State.
Defines the global state object that flows through the LangGraph supervisor loop.
Every agent reads from and writes to this state — nothing is passed directly between agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class PipelineStatus(StrEnum):
    INITIALIZING = "initializing"
    INGESTING = "ingesting"
    BUILDING_RAG = "building_rag"
    ANALYZING_DATA = "analyzing_data"
    GENERATING_HYPOTHESES = "generating_hypotheses"
    MODELING = "modeling"
    GROUNDING = "grounding"
    CONVERGED = "converged"
    MAX_ROUNDS_REACHED = "max_rounds_reached"
    FAILED = "failed"


@dataclass
class RoundSummary:
    """Summary of one complete pipeline round."""

    round: int
    n_hypotheses: int
    n_models_passed: int
    best_r_squared: float
    global_match_score: float
    top_variables: list[str]
    top_citations: list[dict]
    converged: bool


@dataclass
class PipelineState:
    """
    Global state that flows through the entire LangGraph pipeline.
    Initialized once and updated by each agent in sequence.
    """

    # ── Run metadata ──────────────────────────────────────────────────────────
    run_id: str = ""
    run_name: str = ""
    status: PipelineStatus = PipelineStatus.INITIALIZING
    current_round: int = 0
    max_rounds: int = 10
    convergence_threshold: float = 0.75

    # ── Config ────────────────────────────────────────────────────────────────
    config: dict[str, Any] = field(default_factory=dict)

    # ── Ingestion outputs ─────────────────────────────────────────────────────
    data_bundle: Any | None = None  # DataBundle
    corpus_bundle: Any | None = None  # CorpusBundle
    retriever: Any | None = None  # Retriever

    # ── Agent outputs (updated each round) ───────────────────────────────────
    intelligence_report: Any | None = None  # DataIntelligenceReport
    hypothesis_set: Any | None = None  # HypothesisSet
    modeling_result: Any | None = None  # ModelingRoundResult
    grounding_result: Any | None = None  # RoundGroundingResult

    # ── History across all rounds ─────────────────────────────────────────────
    round_summaries: list[RoundSummary] = field(default_factory=list)
    all_structured_results: list[dict] = field(default_factory=list)
    convergence_scores: list[float] = field(default_factory=list)

    # ── Final outputs ─────────────────────────────────────────────────────────
    final_match_score: float = 0.0
    final_report_path: str = ""

    # ── Error tracking ────────────────────────────────────────────────────────
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_round_summary(
        self,
        grounding_result: Any,
        modeling_result: Any,
    ) -> None:
        """Record a completed round."""
        summary = RoundSummary(
            round=self.current_round,
            n_hypotheses=getattr(modeling_result, "n_hypotheses", 0),
            n_models_passed=getattr(modeling_result, "n_passed", 0),
            best_r_squared=getattr(modeling_result, "best_r_squared", 0.0),
            global_match_score=getattr(grounding_result, "global_match_score", 0.0),
            top_variables=getattr(modeling_result, "top_variables", []),
            top_citations=[
                {"title": c.title, "url": c.url, "score": c.similarity_score}
                for c in getattr(grounding_result, "top_citations", [])[:5]
            ],
            converged=getattr(grounding_result, "convergence_ready", False),
        )
        self.round_summaries.append(summary)
        self.convergence_scores.append(summary.global_match_score)

        structured = getattr(grounding_result, "structured_results", [])
        self.all_structured_results.extend(structured)

    @property
    def should_stop(self) -> bool:
        """True if pipeline should stop after current round."""
        if not self.convergence_scores:
            return False
        latest_score = self.convergence_scores[-1]
        return (
            latest_score >= self.convergence_threshold
            or self.current_round >= self.max_rounds
        )

    @property
    def stop_reason(self) -> str:
        if not self.convergence_scores:
            return "no rounds completed"
        if self.convergence_scores[-1] >= self.convergence_threshold:
            return f"converged (score={self.convergence_scores[-1]:.3f})"
        return f"max rounds reached ({self.max_rounds})"
