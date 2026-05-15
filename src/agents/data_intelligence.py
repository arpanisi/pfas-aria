"""
Data Intelligence Agent.
Autonomously analyzes the experimental dataset to:
  1. Discover panel structure (entity/time, FE vs RE)
  2. Profile features (distributions, correlations, outliers)
  3. Produce a structured DataIntelligenceReport for downstream agents
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy import stats

from src.agents.base import BaseAgent
from src.ingestion.data_loader import DataBundle
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ── Output schemas ────────────────────────────────────────────────────────────


@dataclass
class PanelStructure:
    is_panel: bool
    entity_column: str | None
    time_column: str | None
    n_entities: int
    n_time_periods: int
    balanced: bool  # Same number of obs per entity?
    recommended_model: str  # "fixed_effects" | "random_effects" | "ols"
    hausman_applicable: bool
    reasoning: str


@dataclass
class FeatureProfile:
    name: str
    is_numeric: bool
    distribution: str  # "normal" | "skewed" | "bimodal" | "uniform"
    skewness: float | None
    kurtosis: float | None
    outlier_count: int
    correlation_with_outcome: float | None
    vif_score: float | None  # Populated later during modeling
    recommended_transform: str  # "none" | "log" | "sqrt" | "standardize"
    importance_rank: int  # 1 = most important


@dataclass
class DataIntelligenceReport:
    """Full output of the Data Intelligence Agent. Fed to all downstream agents."""

    panel_structure: PanelStructure
    feature_profiles: list[FeatureProfile]
    top_features: list[str]  # Ranked by correlation with outcome
    schema_summary: str  # Passed through from DataBundle
    llm_interpretation: str  # Agent's plain-language summary
    warnings: list[str]  # Data quality warnings


# ── Agent ─────────────────────────────────────────────────────────────────────


class DataIntelligenceAgent(BaseAgent):
    """
    Autonomously discovers structure in the experimental dataset.
    Profiles features and panel structure, then asks the LLM to interpret.
    """

    SYSTEM_PROMPT = """You are a data scientist specializing in experimental
physical chemistry and environmental science. You analyze experimental datasets
and identify meaningful structure — patterns, and relationships —
grounded in scientific reasoning. Always respond with valid JSON."""

    def run(self, bundle: DataBundle) -> DataIntelligenceReport:
        logger.info("=== Data Intelligence Agent: Starting ===")

        warnings: list[str] = []

        # Step 1: Discover panel structure
        logger.info("Step 1: Discovering panel structure...")
        panel = self._discover_panel_structure(bundle)

        # Step 2: Profile features
        logger.info("Step 2: Profiling features...")
        feature_profiles = self._profile_features(bundle)

        # Step 3: Rank features
        top_features = self._rank_features(feature_profiles)

        # Step 4: LLM interpretation
        logger.info("Step 4: Generating LLM interpretation...")
        llm_interpretation = self._interpret_with_llm(
            bundle, panel, feature_profiles, top_features
        )

        report = DataIntelligenceReport(
            panel_structure=panel,
            feature_profiles=feature_profiles,
            top_features=top_features,
            schema_summary=bundle.schema_summary,
            llm_interpretation=llm_interpretation,
            warnings=warnings,
        )

        logger.info(
            f"=== Data Intelligence Agent: Complete — "
            f"{len(feature_profiles)} features profiled ==="
        )
        return report

    # ── Panel Structure ───────────────────────────────────────────────────────

    def _discover_panel_structure(self, bundle: DataBundle) -> PanelStructure:
        """Determine if data is panel data and recommend the model type."""
        df = bundle.df
        entity_col = bundle.entity_id_column
        time_col = bundle.time_column

        if not entity_col or not time_col:
            return PanelStructure(
                is_panel=False,
                entity_column=entity_col,
                time_column=time_col,
                n_entities=0,
                n_time_periods=0,
                balanced=False,
                recommended_model="ols",
                hausman_applicable=False,
                reasoning="No entity or time column detected — treating as cross-sectional data.",
            )

        n_entities = df[entity_col].nunique()
        n_periods = df[time_col].nunique()
        obs_per_entity = df.groupby(entity_col).size()
        balanced = bool(obs_per_entity.nunique() == 1)

        # Within-group variance test: if entity explains variance → fixed effects
        within_var = self._compute_within_variance(
            df, entity_col, bundle.outcome_variable
        )
        between_var = self._compute_between_variance(
            df, entity_col, bundle.outcome_variable
        )

        if within_var > 0 and between_var > 0:
            within_between_ratio = within_var / between_var
        else:
            within_between_ratio = 1.0

        # Heuristic: high within-variation → fixed effects
        if within_between_ratio > 0.3:
            recommended = "fixed_effects"
            reasoning = (
                f"Panel data detected ({n_entities} entities × {n_periods} periods). "
                f"Within/between variance ratio={within_between_ratio:.3f} suggests "
                f"entity-specific effects — fixed effects model recommended."
            )
        else:
            recommended = "random_effects"
            reasoning = (
                f"Panel data detected ({n_entities} entities × {n_periods} periods). "
                f"Low within-variance ratio={within_between_ratio:.3f} suggests "
                f"random effects may be appropriate — verify with Hausman test."
            )

        return PanelStructure(
            is_panel=True,
            entity_column=entity_col,
            time_column=time_col,
            n_entities=n_entities,
            n_time_periods=n_periods,
            balanced=balanced,
            recommended_model=recommended,
            hausman_applicable=n_entities > 2 and n_periods > 2,
            reasoning=reasoning,
        )

    def _compute_within_variance(
        self, df: pd.DataFrame, entity_col: str, outcome: str
    ) -> float:
        """Within-entity variance of the outcome variable."""
        try:
            group_means = df.groupby(entity_col)[outcome].transform("mean")
            return float((df[outcome] - group_means).var())
        except Exception:
            return 0.0

    def _compute_between_variance(
        self, df: pd.DataFrame, entity_col: str, outcome: str
    ) -> float:
        """Between-entity variance of the outcome variable."""
        try:
            return float(df.groupby(entity_col)[outcome].mean().var())
        except Exception:
            return 0.0

    # ── Feature Profiling ─────────────────────────────────────────────────────

    def _profile_features(self, bundle: DataBundle) -> list[FeatureProfile]:
        """Profile each feature column."""
        profiles = []
        outcome = bundle.outcome_variable
        df = bundle.df

        for i, col in enumerate(bundle.feature_columns):
            series = df[col].dropna()

            if bundle.column_profiles[col].is_numeric:
                skewness = float(stats.skew(series))
                kurt = float(stats.kurtosis(series))
                distribution = self._classify_distribution(series, skewness, kurt)
                outliers = self._count_outliers(series)
                corr = self._correlation_with_outcome(df[col], df[outcome])
                transform = self._recommend_transform(series, skewness)

                profiles.append(
                    FeatureProfile(
                        name=col,
                        is_numeric=True,
                        distribution=distribution,
                        skewness=round(skewness, 4),
                        kurtosis=round(kurt, 4),
                        outlier_count=outliers,
                        correlation_with_outcome=round(corr, 4)
                        if corr is not None
                        else None,
                        vif_score=None,
                        recommended_transform=transform,
                        importance_rank=i + 1,
                    )
                )
            else:
                profiles.append(
                    FeatureProfile(
                        name=col,
                        is_numeric=False,
                        distribution="categorical",
                        skewness=None,
                        kurtosis=None,
                        outlier_count=0,
                        correlation_with_outcome=None,
                        vif_score=None,
                        recommended_transform="encode",
                        importance_rank=i + 1,
                    )
                )

        return profiles

    def _classify_distribution(
        self, series: pd.Series, skewness: float, kurtosis: float
    ) -> str:
        _, p_value = stats.normaltest(series)
        if p_value > 0.05:
            return "normal"
        if abs(skewness) > 1.5:
            return "skewed"
        if kurtosis > 3:
            return "heavy_tailed"
        return "non_normal"

    def _count_outliers(self, series: pd.Series) -> int:
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return int(((series < lower) | (series > upper)).sum())

    def _correlation_with_outcome(
        self, feature: pd.Series, outcome: pd.Series
    ) -> float | None:
        try:
            mask = feature.notna() & outcome.notna()
            if mask.sum() < 3:
                return None
            corr, _ = stats.pearsonr(feature[mask], outcome[mask])
            return float(corr)
        except Exception:
            return None

    def _recommend_transform(self, series: pd.Series, skewness: float) -> str:
        if series.min() > 0 and abs(skewness) > 1.5:
            return "log"
        if series.min() >= 0 and abs(skewness) > 0.8:
            return "sqrt"
        return "none"

    def _rank_features(self, profiles: list[FeatureProfile]) -> list[str]:
        """Rank numeric features by absolute correlation with outcome."""
        numeric = [
            p
            for p in profiles
            if p.is_numeric and p.correlation_with_outcome is not None
        ]
        ranked = sorted(
            numeric,
            key=lambda p: abs(p.correlation_with_outcome or 0.0),
            reverse=True,
        )
        for i, p in enumerate(ranked):
            p.importance_rank = i + 1

        return [p.name for p in ranked]

    # ── LLM Interpretation ────────────────────────────────────────────────────

    def _interpret_with_llm(
        self,
        bundle: DataBundle,
        panel: PanelStructure,
        feature_profiles: list[FeatureProfile],
        top_features: list[str],
    ) -> str:
        """Ask the LLM to interpret the discovered structure in scientific terms."""

        top_corr = []
        for p in feature_profiles:
            if p.correlation_with_outcome is not None:
                top_corr.append(
                    f"  {p.name}: r={p.correlation_with_outcome:.3f}, "
                    f"distribution={p.distribution}, "
                    f"recommended_transform={p.recommended_transform}"
                )

        prompt = f"""You are analyzing an experimental scientific dataset.

Dataset overview:
{bundle.schema_summary}

Panel structure:
- Is panel data: {panel.is_panel}
- Recommended model: {panel.recommended_model}
- Reasoning: {panel.reasoning}

Top features by correlation with outcome ({bundle.outcome_variable}):
{chr(10).join(top_corr[:10])}

Please provide:
1. Which features are most scientifically significant and why
2. Any data quality concerns you notice

Respond as a JSON object with keys:
- "scientific_interpretation": overall paragraph interpretation
- "key_features": list of the 5 most important feature names with brief reason each
- "warnings": list of any data quality concerns
"""

        try:
            response = self.call_llm(prompt, system=self.SYSTEM_PROMPT)
            parsed = self.parse_json(response)
            return str(parsed.get("scientific_interpretation", response))
        except Exception as e:
            logger.warning(f"LLM interpretation failed: {e} — using fallback")
            return (
                f"Panel structure: {panel.recommended_model}. "
                f"Top features: {', '.join(top_features[:5])}."
            )
