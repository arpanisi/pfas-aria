"""
Data Intelligence Agent.
Autonomously analyzes the experimental dataset to:
  1. Detect data regimes (changepoints + clustering)
  2. Discover panel structure (entity/time, FE vs RE)
  3. Profile features (distributions, correlations, outliers)
  4. Produce a structured DataIntelligenceReport for downstream agents
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

from src.agents.base import BaseAgent
from src.ingestion.data_loader import DataBundle
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ── Output schemas ────────────────────────────────────────────────────────────


@dataclass
class Regime:
    label: str  # e.g. "regime_0", "regime_1"
    indices: list[int]  # Row indices belonging to this regime
    size: int
    description: str  # LLM-generated plain-language description
    summary_stats: dict[str, Any]  # Mean/std of key variables in this regime


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

    regimes: list[Regime]
    n_regimes: int
    regime_method: str  # Which method was chosen
    panel_structure: PanelStructure
    feature_profiles: list[FeatureProfile]
    top_features: list[str]  # Ranked by correlation with outcome
    labeled_df: pd.DataFrame  # Original df with 'regime' column added
    schema_summary: str  # Passed through from DataBundle
    llm_interpretation: str  # Agent's plain-language summary
    warnings: list[str]  # Data quality warnings


# ── Agent ─────────────────────────────────────────────────────────────────────


class DataIntelligenceAgent(BaseAgent):
    """
    Autonomously discovers structure in the experimental dataset.
    Runs multiple regime detection methods and uses the LLM to arbitrate
    which segmentation is most scientifically meaningful.
    """

    SYSTEM_PROMPT = """You are a data scientist specializing in experimental
physical chemistry and environmental science. You analyze experimental datasets
and identify meaningful structure — regimes, patterns, and relationships —
grounded in scientific reasoning. Always respond with valid JSON."""

    def run(self, bundle: DataBundle) -> DataIntelligenceReport:
        logger.info("=== Data Intelligence Agent: Starting ===")

        warnings: list[str] = []

        # Step 1: Detect regimes
        logger.info("Step 1: Detecting regimes...")
        regimes, regime_method, labeled_df = self._detect_regimes(bundle, warnings)

        # Step 2: Discover panel structure
        logger.info("Step 2: Discovering panel structure...")
        panel = self._discover_panel_structure(bundle)

        # Step 3: Profile features
        logger.info("Step 3: Profiling features...")
        feature_profiles = self._profile_features(bundle, labeled_df)

        # Step 4: Rank features
        top_features = self._rank_features(feature_profiles)

        # Step 5: LLM interpretation
        logger.info("Step 5: Generating LLM interpretation...")
        llm_interpretation = self._interpret_with_llm(
            bundle, regimes, panel, feature_profiles, top_features
        )

        report = DataIntelligenceReport(
            regimes=regimes,
            n_regimes=len(regimes),
            regime_method=regime_method,
            panel_structure=panel,
            feature_profiles=feature_profiles,
            top_features=top_features,
            labeled_df=labeled_df,
            schema_summary=bundle.schema_summary,
            llm_interpretation=llm_interpretation,
            warnings=warnings,
        )

        logger.info(
            f"=== Data Intelligence Agent: Complete — "
            f"{len(regimes)} regimes, {len(feature_profiles)} features profiled ==="
        )
        return report

    # ── Regime Detection ──────────────────────────────────────────────────────

    def _detect_regimes(
        self,
        bundle: DataBundle,
        warnings: list[str],
    ) -> tuple[list[Regime], str, pd.DataFrame]:
        """Run all regime detection methods and pick the best one."""

        df = bundle.df.copy()
        numeric_cols = [
            c
            for c in bundle.numeric_columns
            if c != bundle.outcome_variable
            and c not in (bundle.entity_id_column or "")
            and df[c].notna().sum() > 0
        ]

        if len(numeric_cols) < 2:
            warnings.append(
                "Too few numeric columns for regime detection — using single regime"
            )
            df["regime"] = "regime_0"
            return self._build_regimes(df, bundle, ["regime_0"]), "single", df

        results = {}

        # Method 1: PELT changepoint detection (time-based)
        if bundle.time_column:
            try:
                pelt_regimes = self._pelt_detection(df, bundle, numeric_cols)
                results["pelt"] = pelt_regimes
                logger.info(f"  PELT: {len(set(pelt_regimes))} regimes")
            except Exception as e:
                warnings.append(f"PELT failed: {e}")

        # Method 2: HDBSCAN clustering (feature-based)
        try:
            hdbscan_regimes = self._hdbscan_detection(df, bundle, numeric_cols)
            results["hdbscan"] = hdbscan_regimes
            logger.info(f"  HDBSCAN: {len(set(hdbscan_regimes))} regimes")
        except Exception as e:
            warnings.append(f"HDBSCAN failed: {e}")

        # Method 3: K-means fallback
        try:
            kmeans_regimes = self._kmeans_detection(df, bundle, numeric_cols)
            results["kmeans"] = kmeans_regimes
            logger.info(f"  K-means: {len(set(kmeans_regimes))} regimes")
        except Exception as e:
            warnings.append(f"K-means failed: {e}")

        if not results:
            warnings.append("All regime detection methods failed — using single regime")
            df["regime"] = "regime_0"
            return self._build_regimes(df, bundle, ["regime_0"]), "single", df

        # Arbitrate: pick method with best silhouette or LLM-guided choice
        best_method, best_labels = self._arbitrate_regimes(results, df, numeric_cols)
        df["regime"] = best_labels

        regime_labels = sorted(df["regime"].unique().tolist())
        regimes = self._build_regimes(df, bundle, regime_labels)

        return regimes, best_method, df

    def _pelt_detection(
        self,
        df: pd.DataFrame,
        bundle: DataBundle,
        numeric_cols: list[str],
    ) -> list[str]:
        """Changepoint detection using ruptures PELT algorithm."""
        import ruptures as rpt

        time_col = bundle.time_column
        df_sorted = df.sort_values(time_col).reset_index(drop=True)

        # Use outcome variable as signal
        signal = (
            df_sorted[bundle.outcome_variable]
            .fillna(df_sorted[bundle.outcome_variable].mean())
            .values.reshape(-1, 1)
        )

        model = rpt.Pelt(model="rbf").fit(signal)
        # Penalty controls number of breakpoints
        breakpoints = model.predict(pen=10)

        labels = []
        regime_idx = 0
        prev = 0
        for bp in breakpoints:
            labels.extend([f"regime_{regime_idx}"] * (bp - prev))
            regime_idx += 1
            prev = bp

        # Align back to original df order
        result = ["regime_0"] * len(df)
        for orig_idx, sorted_idx in enumerate(df_sorted.index):
            if orig_idx < len(labels):
                result[sorted_idx] = labels[orig_idx]

        return result

    def _hdbscan_detection(
        self,
        df: pd.DataFrame,
        bundle: DataBundle,
        numeric_cols: list[str],
    ) -> list[str]:
        """Density-based clustering using HDBSCAN."""
        import hdbscan

        features = df[numeric_cols].fillna(df[numeric_cols].mean())
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)

        min_size = max(
            self.settings.agent_config_min_regime_size
            if hasattr(self.settings, "agent_config_min_regime_size")
            else 5,
            max(5, len(df) // 10),
        )

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size, gen_min_span_tree=True)
        cluster_labels = clusterer.fit_predict(scaled)

        # Map -1 (noise) to nearest cluster
        if -1 in cluster_labels:
            noise_mask = cluster_labels == -1
            cluster_labels[noise_mask] = 0

        return [f"regime_{int(lbl)}" for lbl in cluster_labels]

    def _kmeans_detection(
        self,
        df: pd.DataFrame,
        bundle: DataBundle,
        numeric_cols: list[str],
    ) -> list[str]:
        """K-means clustering with optimal k via silhouette score."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        features = df[numeric_cols].fillna(df[numeric_cols].mean())
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)

        best_k = 2
        best_score = -1.0

        for k in range(2, min(7, len(df) // 5 + 1)):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(scaled)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(scaled, labels)
            if score > best_score:
                best_score = score
                best_k = k

        km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled)
        return [f"regime_{int(lbl)}" for lbl in labels]

    def _arbitrate_regimes(
        self,
        results: dict[str, list[str]],
        df: pd.DataFrame,
        numeric_cols: list[str],
    ) -> tuple[str, list[str]]:
        """Pick the best regime detection result using silhouette scores."""
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import LabelEncoder

        features = df[numeric_cols].fillna(df[numeric_cols].mean())
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)

        best_method = list(results.keys())[0]
        best_score = -1.0

        for method, labels in results.items():
            encoded = LabelEncoder().fit_transform(labels)
            if len(set(encoded)) < 2:
                continue
            try:
                score = silhouette_score(scaled, encoded)
                logger.info(f"  Silhouette [{method}]: {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_method = method
            except Exception:
                continue

        logger.info(f"  Best method: {best_method} (score={best_score:.4f})")
        return best_method, results[best_method]

    def _build_regimes(
        self,
        df: pd.DataFrame,
        bundle: DataBundle,
        regime_labels: list[str],
    ) -> list[Regime]:
        """Build Regime objects from labeled dataframe."""
        regimes = []
        min_size = (
            self.settings.agent_config_min_regime_size
            if hasattr(self.settings, "agent_config_min_regime_size")
            else 5
        )

        for label in regime_labels:
            mask = df["regime"] == label
            subset = df[mask]

            if len(subset) < min_size:
                continue

            # Summary stats for numeric columns
            summary = {}
            for col in bundle.numeric_columns[:8]:  # Top 8 for brevity
                if col in subset.columns:
                    summary[col] = {
                        "mean": round(float(subset[col].mean()), 4),
                        "std": round(float(subset[col].std()), 4),
                    }

            regimes.append(
                Regime(
                    label=label,
                    indices=subset.index.tolist(),
                    size=len(subset),
                    description="",  # Filled by LLM interpretation
                    summary_stats=summary,
                )
            )

        return regimes

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

    def _profile_features(
        self, bundle: DataBundle, labeled_df: pd.DataFrame
    ) -> list[FeatureProfile]:
        """Profile each feature column."""
        profiles = []
        outcome = bundle.outcome_variable

        for i, col in enumerate(bundle.feature_columns):
            series = labeled_df[col].dropna()

            if bundle.column_profiles[col].is_numeric:
                skewness = float(stats.skew(series))
                kurt = float(stats.kurtosis(series))
                distribution = self._classify_distribution(series, skewness, kurt)
                outliers = self._count_outliers(series)
                corr = self._correlation_with_outcome(
                    labeled_df[col], labeled_df[outcome]
                )
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
            key=lambda p: abs(p.correlation_with_outcome),
            reverse=True,
        )
        for i, p in enumerate(ranked):
            p.importance_rank = i + 1

        return [p.name for p in ranked]

    # ── LLM Interpretation ────────────────────────────────────────────────────

    def _interpret_with_llm(
        self,
        bundle: DataBundle,
        regimes: list[Regime],
        panel: PanelStructure,
        feature_profiles: list[FeatureProfile],
        top_features: list[str],
    ) -> str:
        """Ask the LLM to interpret the discovered structure in scientific terms."""

        regime_summaries = []
        for r in regimes:
            stats_str = "; ".join(
                f"{k}: mean={v['mean']}, std={v['std']}"
                for k, v in list(r.summary_stats.items())[:4]
            )
            regime_summaries.append(f"  {r.label} (n={r.size}): {stats_str}")

        top_corr = []
        for p in feature_profiles:
            if p.correlation_with_outcome is not None:
                top_corr.append(
                    f"  {p.name}: r={p.correlation_with_outcome:.3f}, "
                    f"distribution={p.distribution}, "
                    f"recommended_transform={p.recommended_transform}"
                )

        prompt = f"""You are analyzing an experimental PFAS degradation dataset.

Domain context: {self.settings.domain.context}

Dataset overview:
{bundle.schema_summary}

Detected regimes ({len(regimes)} total):
{chr(10).join(regime_summaries)}

Panel structure:
- Is panel data: {panel.is_panel}
- Recommended model: {panel.recommended_model}
- Reasoning: {panel.reasoning}

Top features by correlation with outcome ({bundle.outcome_variable}):
{chr(10).join(top_corr[:10])}

Please provide:
1. A plain-language scientific interpretation of each detected regime
2. What these regimes likely represent physically/chemically in PFAS degradation
3. Which features are most scientifically significant and why
4. Any data quality concerns you notice

Respond as a JSON object with keys:
- "regime_descriptions": dict mapping regime label to plain-language description
- "scientific_interpretation": overall paragraph interpretation
- "key_features": list of the 5 most important feature names with brief reason each
- "warnings": list of any data quality concerns
"""

        try:
            response = self.call_llm(prompt, system=self.SYSTEM_PROMPT)
            parsed = self.parse_json(response)

            # Enrich regime descriptions from LLM output
            regime_desc = parsed.get("regime_descriptions", {})
            for regime in regimes:
                if regime.label in regime_desc:
                    regime.description = regime_desc[regime.label]

            return parsed.get("scientific_interpretation", response)
        except Exception as e:
            logger.warning(f"LLM interpretation failed: {e} — using fallback")
            return (
                f"Automated analysis detected {len(regimes)} regimes. "
                f"Panel structure: {panel.recommended_model}. "
                f"Top features: {', '.join(top_features[:5])}."
            )
