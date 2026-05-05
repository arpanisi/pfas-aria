"""
Hypothesis Agent.
Generates structured, ranked scientific hypotheses from:
  - Data Intelligence report (regimes, feature profiles, top features)
  - RAG context (relevant chunks from the 50 papers)
  - Previous round results (for loop refinement)

Each hypothesis is a fully specified modeling instruction —
variable selection, interaction terms, model family, and priority score.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.agents.base import BaseAgent
from src.agents.data_intelligence import DataIntelligenceReport
from src.rag.retriever import Retriever
from src.utils.exceptions import HypothesisGenerationError
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ── Output schema ─────────────────────────────────────────────────────────────


@dataclass
class Hypothesis:
    """A single fully-specified modeling hypothesis."""

    id: str  # e.g. "H1", "H2"
    round: int  # Which loop iteration generated this
    description: str  # Plain-language scientific statement
    rationale: str  # Why this hypothesis is scientifically plausible
    outcome_variable: str  # What we are modeling
    primary_variables: list[str]  # Main predictors
    interaction_terms: list[tuple[str, str]]  # Variable pairs to interact
    control_variables: list[str]  # Variables to include as controls
    model_family: str  # "ols" | "fixed_effects" | "random_effects" | "lasso" | "gradient_boosting"
    suggested_transforms: dict[str, str]  # var_name -> "log" | "sqrt" | "none"
    priority_score: float  # 0.0-1.0, higher = test first
    rag_support: list[str]  # Source titles that support this hypothesis
    is_refinement: bool = False  # True if generated from previous round failure


@dataclass
class HypothesisSet:
    """Full output of the Hypothesis Agent for one round."""

    round: int
    hypotheses: list[Hypothesis]
    n_hypotheses: int
    domain_context: str
    rag_context_used: str  # Summary of what literature was retrieved
    refinement_notes: str  # If round > 1, what changed and why


# ── Agent ─────────────────────────────────────────────────────────────────────


class HypothesisAgent(BaseAgent):
    """
    Generates structured hypotheses by combining:
      1. Feature intelligence from DataIntelligenceReport
      2. Domain knowledge from RAG (your 50 papers)
      3. Feedback from previous modeling rounds (if looping)

    Output is a prioritized list of Hypothesis objects ready
    for the Modeling Agent to execute.
    """

    SYSTEM_PROMPT = """You are an expert environmental chemist and statistician
specializing in PFAS (per- and polyfluoroalkyl substances) degradation research.
You generate precise, testable statistical hypotheses grounded in physical chemistry.
You think carefully about which variables to include, which interactions matter,
and which model families are appropriate for panel data.
Always respond with valid JSON only — no preamble, no markdown."""

    def run(
        self,
        report: DataIntelligenceReport,
        retriever: Retriever,
        round_number: int = 1,
        previous_results: list[dict] | None = None,
    ) -> HypothesisSet:
        """
        Generate a prioritized hypothesis set for this round.

        Args:
            report: Output from DataIntelligenceAgent
            retriever: RAG retriever for literature context
            round_number: Current loop iteration (1-indexed)
            previous_results: Structured results from previous round's models
        """
        logger.info(f"=== Hypothesis Agent: Round {round_number} ===")

        # Step 1: Retrieve relevant literature
        logger.info("Step 1: Retrieving RAG context...")
        rag_chunks = retriever.retrieve_for_hypothesis(
            variable_names=report.top_features[:10],
            domain_context=self.settings.domain.context,
            top_k=10,
        )
        rag_context = retriever.format_context(rag_chunks, max_chars=3000)
        rag_titles = list({c.title for c in rag_chunks})

        # Step 2: Build prompt
        logger.info("Step 2: Generating hypotheses with LLM...")
        prompt = self._build_prompt(
            report=report,
            rag_context=rag_context,
            round_number=round_number,
            previous_results=previous_results,
        )

        # Step 3: Call LLM
        raw_response = self.call_llm(prompt, system=self.SYSTEM_PROMPT)

        # Step 4: Parse and validate
        hypotheses = self._parse_hypotheses(
            raw_response=raw_response,
            report=report,
            round_number=round_number,
            rag_titles=rag_titles,
        )

        if not hypotheses:
            raise HypothesisGenerationError(
                f"Hypothesis Agent produced no valid hypotheses in round {round_number}"
            )

        # Step 5: Apply constraints and rank
        hypotheses = self._apply_constraints(hypotheses, report)
        hypotheses = sorted(hypotheses, key=lambda h: h.priority_score, reverse=True)

        refinement_notes = ""
        if round_number > 1 and previous_results:
            refinement_notes = self._summarize_refinement(previous_results)

        hypothesis_set = HypothesisSet(
            round=round_number,
            hypotheses=hypotheses,
            n_hypotheses=len(hypotheses),
            domain_context=self.settings.domain.context,
            rag_context_used=f"{len(rag_chunks)} chunks from {len(rag_titles)} papers",
            refinement_notes=refinement_notes,
        )

        logger.info(
            f"=== Hypothesis Agent: Complete — "
            f"{len(hypotheses)} hypotheses generated ==="
        )
        return hypothesis_set

    # ── Prompt Building ───────────────────────────────────────────────────────

    def _build_prompt(
        self,
        report: DataIntelligenceReport,
        rag_context: str,
        round_number: int,
        previous_results: list[dict] | None,
    ) -> str:
        cfg = self.settings

        # Feature summary for prompt
        feature_lines = []
        for fp in report.feature_profiles:
            if fp.is_numeric and fp.correlation_with_outcome is not None:
                feature_lines.append(
                    f"  {fp.name}: r={fp.correlation_with_outcome:.3f}, "
                    f"dist={fp.distribution}, transform={fp.recommended_transform}"
                )
            elif not fp.is_numeric:
                feature_lines.append(f"  {fp.name}: categorical")

        # Regime summary
        regime_lines = []
        for r in report.regimes:
            regime_lines.append(f"  {r.label} (n={r.size}): {r.description[:100]}")

        # Previous results summary (for refinement rounds)
        prev_summary = ""
        if round_number > 1 and previous_results:
            prev_summary = self._format_previous_results(previous_results)

        n_hyp = cfg.supervisor.hypotheses_per_round
        max_vars = cfg.hypothesis.max_variables_per_model
        max_interactions = cfg.hypothesis.max_interaction_terms

        prompt = f"""You are generating statistical hypotheses for round {round_number} of analysis.

DOMAIN: {cfg.domain.context}

DATASET OVERVIEW:
{report.schema_summary}

DETECTED REGIMES ({report.n_regimes}):
{chr(10).join(regime_lines)}

PANEL STRUCTURE:
- Type: {report.panel_structure.recommended_model}
- Reasoning: {report.panel_structure.reasoning}

FEATURE PROFILES (ranked by correlation with outcome):
{chr(10).join(feature_lines[:15])}

TOP FEATURES: {", ".join(report.top_features[:8])}

RELEVANT LITERATURE CONTEXT:
{rag_context}

{f"PREVIOUS ROUND RESULTS (use to refine):{chr(10)}{prev_summary}" if prev_summary else ""}

TASK:
Generate exactly {n_hyp} distinct, testable statistical hypotheses.
Each hypothesis must be scientifically grounded and testable with regression or ML.

CONSTRAINTS:
- Maximum {max_vars} variables per model (including controls)
- Maximum {max_interactions} interaction terms per model
- Prefer interpretable models unless accuracy requires complexity
- Recommended panel model: {report.panel_structure.recommended_model}
- Available model families: ols, fixed_effects, random_effects, lasso, gradient_boosting
- Consider regime-specific effects where scientifically justified

Respond with a JSON array of exactly {n_hyp} hypothesis objects.
Each object must have these exact keys:
{{
  "id": "H1",
  "description": "One sentence scientific hypothesis statement",
  "rationale": "Why this is physically/chemically plausible (2-3 sentences)",
  "primary_variables": ["var1", "var2"],
  "interaction_terms": [["var1", "var2"]],
  "control_variables": ["var3"],
  "model_family": "fixed_effects",
  "suggested_transforms": {{"var1": "log", "var2": "none"}},
  "priority_score": 0.85
}}

Return ONLY the JSON array. No other text."""

        return prompt

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse_hypotheses(
        self,
        raw_response: str,
        report: DataIntelligenceReport,
        round_number: int,
        rag_titles: list[str],
    ) -> list[Hypothesis]:
        """Parse LLM JSON response into Hypothesis objects."""
        try:
            parsed = self.parse_json(raw_response)
        except Exception as e:
            raise HypothesisGenerationError(
                f"Failed to parse hypothesis JSON: {e}\nRaw: {raw_response[:500]}"
            ) from e

        if not isinstance(parsed, list):
            parsed = [parsed]

        hypotheses = []
        valid_columns = set(
            report.feature_profiles[i].name for i in range(len(report.feature_profiles))
        ) | {
            report.panel_structure.entity_column or "",
            report.panel_structure.time_column or "",
        }

        for item in parsed:
            try:
                h = self._parse_single(
                    item=item,
                    round_number=round_number,
                    outcome=report.labeled_df.columns[0]
                    if report.labeled_df is not None
                    else "outcome",
                    rag_titles=rag_titles,
                    valid_columns=valid_columns,
                )
                if h:
                    hypotheses.append(h)
            except Exception as e:
                logger.warning(f"Skipping malformed hypothesis: {e}")
                continue

        return hypotheses

    def _parse_single(
        self,
        item: dict,
        round_number: int,
        outcome: str,
        rag_titles: list[str],
        valid_columns: set[str],
    ) -> Hypothesis | None:
        """Parse and validate a single hypothesis dict."""
        if not isinstance(item, dict):
            return None

        # Extract fields with safe defaults
        hyp_id = str(item.get("id", f"H{round_number}_auto"))
        description = str(item.get("description", ""))
        rationale = str(item.get("rationale", ""))
        primary_vars = [
            str(v) for v in item.get("primary_variables", []) if str(v) in valid_columns
        ]
        interaction_raw = item.get("interaction_terms", [])
        interactions = [
            (str(pair[0]), str(pair[1]))
            for pair in interaction_raw
            if isinstance(pair, (list, tuple)) and len(pair) == 2
        ]
        control_vars = [
            str(v) for v in item.get("control_variables", []) if str(v) in valid_columns
        ]
        model_family = str(item.get("model_family", "ols"))
        transforms_raw = item.get("suggested_transforms", {})
        transforms = {str(k): str(v) for k, v in transforms_raw.items()}
        priority = float(item.get("priority_score", 0.5))

        # Must have at least one primary variable
        if not primary_vars:
            logger.warning(
                f"Hypothesis {hyp_id} has no valid primary variables — skipping"
            )
            return None

        # Validate model family
        valid_families = {
            "ols",
            "fixed_effects",
            "random_effects",
            "lasso",
            "gradient_boosting",
        }
        if model_family not in valid_families:
            model_family = "ols"

        # Clamp priority
        priority = max(0.0, min(1.0, priority))

        return Hypothesis(
            id=hyp_id,
            round=round_number,
            description=description,
            rationale=rationale,
            outcome_variable=outcome,
            primary_variables=primary_vars,
            interaction_terms=interactions,
            control_variables=control_vars,
            model_family=model_family,
            suggested_transforms=transforms,
            priority_score=priority,
            rag_support=rag_titles[:3],
            is_refinement=round_number > 1,
        )

    # ── Constraints ───────────────────────────────────────────────────────────

    def _apply_constraints(
        self,
        hypotheses: list[Hypothesis],
        report: DataIntelligenceReport,
    ) -> list[Hypothesis]:
        """Enforce config constraints on each hypothesis."""
        max_vars = self.settings.hypothesis.max_variables_per_model
        max_interactions = self.settings.hypothesis.max_interaction_terms

        for h in hypotheses:
            # Enforce max variables
            total_vars = h.primary_variables + h.control_variables
            if len(total_vars) > max_vars:
                # Keep top-correlated variables
                corr_map = {
                    fp.name: abs(fp.correlation_with_outcome or 0.0)
                    for fp in report.feature_profiles
                    if fp.is_numeric
                }
                total_vars = sorted(
                    total_vars, key=lambda v: corr_map.get(v, 0.0), reverse=True
                )[:max_vars]
                h.primary_variables = [
                    v for v in h.primary_variables if v in total_vars
                ]
                h.control_variables = [
                    v for v in h.control_variables if v in total_vars
                ]

            # Enforce max interactions
            if len(h.interaction_terms) > max_interactions:
                h.interaction_terms = h.interaction_terms[:max_interactions]

            # Enforce panel model recommendation
            if report.panel_structure.is_panel:
                if h.model_family == "ols":
                    h.model_family = report.panel_structure.recommended_model

        return hypotheses

    # ── Refinement helpers ────────────────────────────────────────────────────

    def _format_previous_results(self, previous_results: list[dict]) -> str:
        """Format previous round results for the LLM prompt."""
        lines = []
        for result in previous_results[:5]:  # Top 5 results
            hyp_id = result.get("hypothesis_id", "?")
            r2 = result.get("r_squared", "?")
            passed = result.get("validation_passed", False)
            match_score = result.get("match_score", 0.0)
            sig_vars = result.get("significant_variables", [])
            lines.append(
                f"  {hyp_id}: R²={r2}, validation={'PASS' if passed else 'FAIL'}, "
                f"lit_match={match_score:.2f}, significant={sig_vars}"
            )
        return "\n".join(lines)

    def _summarize_refinement(self, previous_results: list[dict]) -> str:
        """Generate a plain-language refinement summary."""
        passed = [r for r in previous_results if r.get("validation_passed")]
        failed = [r for r in previous_results if not r.get("validation_passed")]
        high_match = [r for r in previous_results if r.get("match_score", 0) > 0.6]

        return (
            f"Previous round: {len(passed)} passed validation, "
            f"{len(failed)} failed. "
            f"{len(high_match)} had high literature match (>0.6). "
            f"Refine by exploring untested variable combinations "
            f"or alternative model families."
        )
