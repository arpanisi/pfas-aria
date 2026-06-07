"""
Experimental Data Transformer.
Handles cleaning, unit normalization, imputation, and feature engineering
before data enters the pipeline.

All transformations are logged and reversible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TransformLog:
    """Audit trail of all transformations applied."""

    steps: list[str] = field(default_factory=list)
    columns_dropped: list[str] = field(default_factory=list)
    columns_imputed: list[str] = field(default_factory=list)
    columns_transformed: list[str] = field(default_factory=list)
    rows_before: int = 0
    rows_after: int = 0


class ExperimentalTransformer:
    """
    Applies a configurable transformation pipeline to experimental data.
    Each step is logged for reproducibility.
    """

    # Columns with these keywords get log-transform candidates
    LOG_CANDIDATES = ["concentration", "dose", "rate", "flux", "intensity"]

    # Known unit conversions for common experimental variables.
    UNIT_PATTERNS = {
        "ug_l": 1e-3,  # µg/L → mg/L
        "ng_l": 1e-6,  # ng/L → mg/L
        "nm": 1e-6,  # nM → mM
        "um": 1e-3,  # µM → mM
    }

    def transform(
        self,
        df: pd.DataFrame,
        outcome_variable: str,
        feature_columns: list[str],
        imputation_strategy: str = "median",
        apply_log_transforms: bool = True,
        drop_high_missing: float = 0.5,
    ) -> tuple[pd.DataFrame, TransformLog]:
        """
        Apply full transformation pipeline.

        Returns:
            (transformed_df, transform_log)
        """
        log = TransformLog(rows_before=len(df))
        df = df.copy()

        # Step 1: Standardize column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        log.steps.append("Standardized column names to snake_case")

        # Step 2: Drop fully-empty columns
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            df = df.drop(columns=empty_cols)
            log.columns_dropped.extend(empty_cols)
            log.steps.append(f"Dropped {len(empty_cols)} fully-empty columns")

        # Step 3: Drop columns with too many missing values
        missing_rates = df.isnull().mean()
        high_missing = [
            c
            for c in feature_columns
            if c in df.columns and missing_rates.get(c, 0) > drop_high_missing
        ]
        if high_missing:
            df = df.drop(columns=high_missing)
            log.columns_dropped.extend(high_missing)
            log.steps.append(
                f"Dropped {len(high_missing)} columns with >{drop_high_missing * 100:.0f}% missing"
            )

        # Step 4: Impute numeric columns
        numeric_features = [
            c
            for c in feature_columns
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]
        for col in numeric_features:
            if df[col].isnull().any():
                if imputation_strategy == "median":
                    fill_val = df[col].median()
                elif imputation_strategy == "mean":
                    fill_val = df[col].mean()
                else:
                    fill_val = df[col].mode().iloc[0] if not df[col].mode().empty else 0
                df[col] = df[col].fillna(fill_val)
                log.columns_imputed.append(col)

        if log.columns_imputed:
            log.steps.append(
                f"Imputed {len(log.columns_imputed)} columns using {imputation_strategy}"
            )

        # Step 5: Remove rows where outcome is missing
        before = len(df)
        df = df.dropna(subset=[outcome_variable])
        dropped = before - len(df)
        if dropped > 0:
            log.steps.append(f"Dropped {dropped} rows with missing outcome variable")

        # Step 6: Optional log transforms for skewed columns
        if apply_log_transforms:
            for col in numeric_features:
                if col not in df.columns or col == outcome_variable:
                    continue
                if any(kw in col.lower() for kw in self.LOG_CANDIDATES):
                    if (df[col] > 0).all():
                        skewness = float(stats.skew(df[col].dropna()))
                        if abs(skewness) > 1.5:
                            df[f"{col}_log"] = np.log(df[col])
                            log.columns_transformed.append(f"{col} → log({col})")

            if log.columns_transformed:
                log.steps.append(f"Applied log transforms: {log.columns_transformed}")

        # Step 7: Reset index
        df = df.reset_index(drop=True)
        log.rows_after = len(df)

        logger.info(
            f"Transformation complete: "
            f"{log.rows_before} → {log.rows_after} rows, "
            f"{len(log.steps)} steps applied"
        )

        return df, log

    def standardize_units(
        self, df: pd.DataFrame, column_units: dict[str, str]
    ) -> pd.DataFrame:
        """
        Normalize units across columns.
        column_units: {column_name: unit_suffix} e.g. {"analyte_conc": "ug_l"}
        """
        df = df.copy()
        for col, unit in column_units.items():
            if col in df.columns and unit in self.UNIT_PATTERNS:
                factor = self.UNIT_PATTERNS[unit]
                df[col] = df[col] * factor
                logger.debug(f"Unit conversion: {col} × {factor} ({unit} → base unit)")
        return df
