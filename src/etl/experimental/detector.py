"""
Dataset Structure Detector.
The most important new component in Phase 10A.

Examines the uploaded dataset and autonomously decides:
  1. Is this panel data (repeated measurements over time)?
  2. Which outputs are active and modelable?
  3. Can derived outputs (entropy, KL, rate constant k) be computed?
  4. Which modeling approach applies per output?
  5. What are the non-constant predictors?

This runs before any hypothesis is generated.
The decisions made here determine the entire modeling strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import cast

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

# PFCA chain-length column patterns
PFCA_PATTERNS = ["c2", "c3", "c4", "c5", "c6", "c7", "c8"]
PFCA_LONG_CHAIN = ["c6", "c7", "c8"]
PFCA_SHORT_CHAIN = ["c2", "c3", "c4", "c5"]


class ModelingApproach(StrEnum):
    PANEL_REGRESSION = "panel_regression"  # repeated measures, time indicators
    FIRST_ORDER_DECAY = "first_order_decay"  # log-linear OLS for rate constant k
    CROSS_SECTIONAL = "cross_sectional"  # standard OLS, no time structure


class OutputType(StrEnum):
    DIRECT = "direct"  # column exists in data, model directly
    DERIVED_ENTROPY = "derived_entropy"  # computed from PFCA species
    DERIVED_KL = "derived_kl"  # computed from PFCA species
    DERIVED_RATE = "derived_rate"  # fitted from first-order decay
    DERIVED_TOTAL = "derived_total"  # sum of PFCA species


@dataclass
class ActiveOutput:
    """One modelable output with its detection rationale."""

    name: str
    output_type: OutputType
    modeling_approach: ModelingApproach
    source_columns: list[str]
    n_nonzero: int
    variance: float
    rationale: str


@dataclass
class DatasetProfile:
    """
    Full profile of the uploaded dataset.
    Determined once, used by all downstream agents.
    """

    # Time structure
    is_panel: bool
    time_column: str | None
    entity_column: str | None
    n_timepoints: int
    n_entities: int
    time_inflection: float | None  # detected inflection point (e.g. 30 min)

    # PFCA structure
    has_pfca_species: bool
    pfca_columns: list[str]
    has_long_chain: bool
    has_short_chain: bool
    parent_compound: str | None  # single decaying parent (e.g. "c8")

    # Active outputs
    active_outputs: list[ActiveOutput]

    # Active predictors (non-constant across experiments)
    active_predictors: list[str]
    constant_predictors: list[str]  # excluded — same value everywhere

    # Detected inflection — used to split early vs late stage models
    split_timepoint: float | None

    # Summary
    n_active_outputs: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_active_outputs = len(self.active_outputs)

    def describe(self) -> str:
        lines = [
            f"Panel data: {self.is_panel}",
            f"Time column: {self.time_column}",
            f"Entity column: {self.entity_column}",
            f"Timepoints: {self.n_timepoints}, Entities: {self.n_entities}",
            f"PFCA species detected: {self.pfca_columns}",
            f"Parent compound: {self.parent_compound}",
            f"Active outputs ({self.n_active_outputs}):",
        ]
        for o in self.active_outputs:
            lines.append(f"  {o.name} [{o.modeling_approach}] — {o.rationale}")
        lines.append(f"Active predictors: {self.active_predictors[:8]}")
        if self.split_timepoint:
            lines.append(f"Detected inflection at: {self.split_timepoint} min")
        return "\n".join(lines)


class DatasetDetector:
    """
    Autonomously profiles a dataset and determines modeling strategy.
    Called once per run before hypothesis generation.
    """

    MIN_VARIANCE = 1e-6
    MIN_NONZERO_FRACTION = 0.1

    def detect(
        self,
        df: pd.DataFrame,
        outcome_variable: str,
        feature_columns: list[str],
    ) -> DatasetProfile:
        """Run full detection pipeline. Returns DatasetProfile."""
        logger.info("Dataset detection starting...")

        time_col = self._detect_time_column(df)
        entity_col = self._detect_entity_column(df)
        is_panel = self._check_panel_structure(df, time_col, entity_col)
        n_timepoints = df[time_col].nunique() if time_col else 1
        n_entities = df[entity_col].nunique() if entity_col else len(df)

        pfca_cols = self._detect_pfca_columns(df)
        has_pfca = len(pfca_cols) >= 2
        parent = self._detect_parent_compound(df, pfca_cols) if has_pfca else None

        active_predictors, constant_predictors = self._split_predictors(
            df, feature_columns, entity_col, time_col
        )

        active_outputs = self._detect_active_outputs(
            df=df,
            outcome_variable=outcome_variable,
            pfca_cols=pfca_cols,
            has_pfca=has_pfca,
            parent=parent,
            is_panel=is_panel,
            time_col=time_col,
        )

        inflection = (
            self._detect_inflection(df, pfca_cols, time_col)
            if has_pfca and time_col
            else None
        )

        profile = DatasetProfile(
            is_panel=is_panel,
            time_column=time_col,
            entity_column=entity_col,
            n_timepoints=n_timepoints,
            n_entities=n_entities,
            time_inflection=inflection,
            has_pfca_species=has_pfca,
            pfca_columns=pfca_cols,
            has_long_chain=any(c in pfca_cols for c in PFCA_LONG_CHAIN),
            has_short_chain=any(c in pfca_cols for c in PFCA_SHORT_CHAIN),
            parent_compound=parent,
            active_outputs=active_outputs,
            active_predictors=active_predictors,
            constant_predictors=constant_predictors,
            split_timepoint=inflection,
        )

        logger.info(f"Detection complete:\n{profile.describe()}")
        return profile

    # ── Detection methods ──────────────────────────────────────────────────────

    def _detect_time_column(self, df: pd.DataFrame) -> str | None:
        """Find the time column by name pattern and content."""
        candidates = [
            c
            for c in df.columns
            if any(kw in c.lower() for kw in ["time", "hour", "min", "day", "t_"])
        ]
        for col in candidates:
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() >= 3:
                logger.debug(f"Time column detected: {col}")
                return str(col)
        return None

    def _detect_entity_column(self, df: pd.DataFrame) -> str | None:
        """Find the entity/experiment ID column."""
        candidates = [
            c
            for c in df.columns
            if any(
                kw in c.lower()
                for kw in ["id", "experiment", "run", "sample", "entity"]
            )
        ]
        for col in candidates:
            if df[col].nunique() > 1:
                logger.debug(f"Entity column detected: {col}")
                return str(col)
        return None

    def _check_panel_structure(
        self, df: pd.DataFrame, time_col: str | None, entity_col: str | None
    ) -> bool:
        """
        True if dataset has repeated measurements over time per entity.
        Requires: time column, entity column, multiple timepoints per entity.
        """
        if not time_col or not entity_col:
            return False
        counts = df.groupby(entity_col)[time_col].nunique()
        median_timepoints = counts.median()
        is_panel = bool(median_timepoints >= 3)
        logger.debug(
            f"Panel check: median {median_timepoints:.0f} timepoints per entity "
            f"→ {'PANEL' if is_panel else 'CROSS-SECTIONAL'}"
        )
        return is_panel

    def _detect_pfca_columns(self, df: pd.DataFrame) -> list[str]:
        """Find PFCA chain-length columns (c2–c8)."""
        found = []
        for col in df.columns:
            col_lower = col.lower().replace("_", "").replace("-", "")
            for pattern in PFCA_PATTERNS:
                if (
                    col_lower == pattern
                    or col_lower == f"pfca{pattern}"
                    or col_lower == f"conc{pattern}"
                ):
                    found.append(col)
                    break
        logger.debug(f"PFCA columns detected: {found}")
        return found

    def _detect_parent_compound(
        self, df: pd.DataFrame, pfca_cols: list[str]
    ) -> str | None:
        """
        Detect the parent compound — the column that starts highest
        and decays most monotonically.
        """
        best_col = None
        best_score = -1.0
        for col in pfca_cols:
            series = df[col].dropna()
            if len(series) < 4 or series.max() < 0.01:
                continue
            # Score = initial concentration × monotonic decay tendency
            initial = series.iloc[:3].mean()
            final = series.iloc[-3:].mean()
            decay_ratio = (initial - final) / (initial + 1e-9)
            if decay_ratio > best_score:
                best_score = decay_ratio
                best_col = col
        logger.debug(
            f"Parent compound detected: {best_col} (decay ratio={best_score:.3f})"
        )
        return best_col

    def _split_predictors(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        entity_col: str | None,
        time_col: str | None,
    ) -> tuple[list[str], list[str]]:
        """
        Split feature columns into active (vary across experiments)
        and constant (same value everywhere — useless as predictors).
        """
        exclude = {entity_col, time_col}
        active, constant = [], []
        for col in feature_columns:
            if col in exclude or col not in df.columns:
                continue
            variance = df[col].var() if pd.api.types.is_numeric_dtype(df[col]) else None
            if variance is not None and variance < self.MIN_VARIANCE:
                constant.append(col)
            elif df[col].nunique() <= 1:
                constant.append(col)
            else:
                active.append(col)
        logger.debug(f"Predictors: {len(active)} active, {len(constant)} constant")
        return active, constant

    def _detect_active_outputs(
        self,
        df: pd.DataFrame,
        outcome_variable: str,
        pfca_cols: list[str],
        has_pfca: bool,
        parent: str | None,
        is_panel: bool,
        time_col: str | None,
    ) -> list[ActiveOutput]:
        """
        Detect all modelable outputs from the dataset.
        Returns only outputs with sufficient variance and non-zero values.
        """
        outputs: list[ActiveOutput] = []

        # 1. Primary outcome variable (direct)
        if outcome_variable in df.columns:
            col = df[outcome_variable]
            nonzero = int((col > 0).sum())
            var = float(col.var())
            if (
                nonzero > len(df) * self.MIN_NONZERO_FRACTION
                and var > self.MIN_VARIANCE
            ):
                outputs.append(
                    ActiveOutput(
                        name=outcome_variable,
                        output_type=OutputType.DIRECT,
                        modeling_approach=ModelingApproach.PANEL_REGRESSION
                        if is_panel
                        else ModelingApproach.CROSS_SECTIONAL,
                        source_columns=[outcome_variable],
                        n_nonzero=nonzero,
                        variance=var,
                        rationale="Primary outcome variable — sufficient variance and non-zero values",
                    )
                )

        # 2. Fluoride yield (if separate from outcome)
        for col_name in df.columns:
            if "fluoride" in col_name.lower() and col_name != outcome_variable:
                col = df[col_name]
                nonzero = int((col > 0).sum())
                var = float(col.var())
                if (
                    nonzero > len(df) * self.MIN_NONZERO_FRACTION
                    and var > self.MIN_VARIANCE
                ):
                    outputs.append(
                        ActiveOutput(
                            name=col_name,
                            output_type=OutputType.DIRECT,
                            modeling_approach=ModelingApproach.PANEL_REGRESSION
                            if is_panel
                            else ModelingApproach.CROSS_SECTIONAL,
                            source_columns=[col_name],
                            n_nonzero=nonzero,
                            variance=var,
                            rationale="Fluoride yield — defluorination metric",
                        )
                    )

        # 3. Shannon entropy (derived from PFCA species)
        if has_pfca and len(pfca_cols) >= 3:
            outputs.append(
                ActiveOutput(
                    name="shannon_entropy",
                    output_type=OutputType.DERIVED_ENTROPY,
                    modeling_approach=ModelingApproach.PANEL_REGRESSION
                    if is_panel
                    else ModelingApproach.CROSS_SECTIONAL,
                    source_columns=pfca_cols,
                    n_nonzero=len(df),
                    variance=1.0,  # will be computed
                    rationale=f"Derived from {len(pfca_cols)} PFCA species — pathway spread metric",
                )
            )

        # 4. KL divergence (derived from PFCA species)
        if has_pfca and len(pfca_cols) >= 3 and time_col:
            outputs.append(
                ActiveOutput(
                    name="kl_divergence",
                    output_type=OutputType.DERIVED_KL,
                    modeling_approach=ModelingApproach.PANEL_REGRESSION
                    if is_panel
                    else ModelingApproach.CROSS_SECTIONAL,
                    source_columns=pfca_cols,
                    n_nonzero=len(df),
                    variance=1.0,  # will be computed
                    rationale="Derived from PFCA species — pathway shift from initial composition",
                )
            )

        # 5. First-order rate constant k (derived from parent decay)
        if parent and time_col and is_panel:
            outputs.append(
                ActiveOutput(
                    name="degradation_rate_k",
                    output_type=OutputType.DERIVED_RATE,
                    modeling_approach=ModelingApproach.FIRST_ORDER_DECAY,
                    source_columns=[parent, time_col],
                    n_nonzero=len(df),
                    variance=1.0,  # will be computed per experiment
                    rationale=f"First-order rate constant fitted from {parent} decay — one k per experiment",
                )
            )

        # 6. Total PFCA concentration (if multiple species present)
        if has_pfca and len(pfca_cols) >= 3:
            total = df[pfca_cols].sum(axis=1)
            nonzero = int((total > 0).sum())
            var = float(total.var())
            if (
                nonzero > len(df) * self.MIN_NONZERO_FRACTION
                and var > self.MIN_VARIANCE
            ):
                outputs.append(
                    ActiveOutput(
                        name="total_pfca_concentration",
                        output_type=OutputType.DERIVED_TOTAL,
                        modeling_approach=ModelingApproach.PANEL_REGRESSION
                        if is_panel
                        else ModelingApproach.CROSS_SECTIONAL,
                        source_columns=pfca_cols,
                        n_nonzero=nonzero,
                        variance=var,
                        rationale="Sum of all PFCA species — mixture-level degradation kinetics",
                    )
                )

        logger.info(f"Active outputs detected: {[o.name for o in outputs]}")
        return outputs

    def _detect_inflection(
        self,
        df: pd.DataFrame,
        pfca_cols: list[str],
        time_col: str,
    ) -> float | None:
        """
        Detect the temporal inflection point in entropy trajectory.
        The point where intermediate spread peaks — typically ~30 min in PFCA systems.
        """
        try:
            times = sorted(df[time_col].dropna().unique())
            if len(times) < 4:
                return None

            # Compute average entropy per timepoint
            entropies: list[tuple[float, float]] = []
            for t in times:
                subset = df[df[time_col] == t][pfca_cols].values
                if len(subset) == 0:
                    continue
                avg = np.asarray(subset.mean(axis=0), dtype=np.float64)
                total = float(avg.sum())
                if total < 1e-9:
                    entropies.append((float(t), 0.0))
                    continue
                probabilities = avg / total
                probabilities = cast(np.ndarray, probabilities[probabilities > 0])
                entropy = float(np.sum(probabilities * np.log(probabilities)))
                h = -entropy
                entropies.append((float(t), h))

            if len(entropies) < 3:
                return None

            # Find the time with peak entropy
            peak_time = max(entropies, key=lambda x: x[1])[0]
            logger.info(f"Inflection point detected at t={peak_time}")
            return float(peak_time)

        except Exception as e:
            logger.debug(f"Inflection detection failed: {e}")
            return None
