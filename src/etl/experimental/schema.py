"""
Experimental Data Schema.
Pandera schema definitions for validating PFAS experimental datasets.

Two layers:
  1. BaseSchema  — minimal required structure (always enforced)
  2. StrictSchema — optional strict validation with domain-specific ranges

The schema is intentionally flexible to handle diverse experimental designs.
Column names are configurable via data_config.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pandera as pa

from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ── Validation result ─────────────────────────────────────────────────────────


@dataclass
class ValidationReport:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_rows_valid: int = 0
    n_rows_total: int = 0
    schema_version: str = "1.0"


# ── Schema definitions ────────────────────────────────────────────────────────


def build_base_schema(outcome_variable: str) -> pa.DataFrameSchema:
    """
    Build a minimal Pandera schema that enforces:
    - Outcome variable exists and is numeric
    - No fully-empty columns
    - At least one numeric feature column

    This runs on every dataset regardless of domain.
    """
    return pa.DataFrameSchema(
        columns={
            outcome_variable: pa.Column(
                float,
                nullable=False,
                description=f"Outcome variable: {outcome_variable}",
            ),
        },
        coerce=True,
        strict=False,  # Allow extra columns
        name="BaseSchema",
    )


def build_pfas_schema(
    outcome_variable: str,
    feature_columns: list[str],
) -> pa.DataFrameSchema:
    """
    PFAS-specific schema with domain knowledge.
    Enforces reasonable ranges for common PFAS degradation variables.
    All range checks are soft — warnings not errors.
    """
    column_defs: dict[str, pa.Column] = {
        outcome_variable: pa.Column(
            float,
            pa.Check.between(0.0, 1.0, error="Degradation rate must be 0-1"),
            nullable=False,
        ),
    }

    # Add soft checks for known PFAS experimental variables
    pfas_ranges = {
        "ph": pa.Check.between(0.0, 14.0),
        "temperature": pa.Check.between(-10.0, 100.0),
        "temperature_c": pa.Check.between(-10.0, 100.0),
        "uv_intensity": pa.Check.greater_than_or_equal_to(0.0),
        "concentration": pa.Check.greater_than_or_equal_to(0.0),
        "time": pa.Check.greater_than_or_equal_to(0.0),
        "time_hours": pa.Check.greater_than_or_equal_to(0.0),
        "dose": pa.Check.greater_than_or_equal_to(0.0),
    }

    for col in feature_columns:
        col_lower = col.lower()
        check = None
        for key, range_check in pfas_ranges.items():
            if key in col_lower:
                check = range_check
                break

        if check is not None:
            column_defs[col] = pa.Column(float, check, nullable=True, coerce=True)
        else:
            column_defs[col] = pa.Column(nullable=True, coerce=True)

    return pa.DataFrameSchema(
        columns=column_defs,
        coerce=True,
        strict=False,
        name="PFASSchema",
    )


# ── Validator ─────────────────────────────────────────────────────────────────


class ExperimentalDataValidator:
    """
    Validates experimental DataFrames against Pandera schemas.
    Always runs base validation.
    Runs domain schema if PFAS context is detected.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    def validate(
        self,
        df: pd.DataFrame,
        feature_columns: list[str] | None = None,
        strict: bool = False,
    ) -> ValidationReport:
        outcome = self.settings.data.outcome_variable
        features = feature_columns or []
        errors: list[str] = []
        warnings: list[str] = []

        # Check 1: Outcome variable exists
        if outcome not in df.columns:
            return ValidationReport(
                passed=False,
                errors=[
                    f"Outcome variable '{outcome}' not found. "
                    f"Columns: {list(df.columns)[:10]}"
                ],
                n_rows_total=len(df),
            )

        # Check 2: Base schema
        try:
            base_schema = build_base_schema(outcome)
            base_schema.validate(df, lazy=True)
        except pa.errors.SchemaErrors as e:
            for _, row in e.failure_cases.iterrows():
                errors.append(
                    f"Column '{row.get('column')}': {row.get('failure_case')}"
                )

        # Check 3: Missing values
        missing = df.isnull().mean()
        for col in df.columns:
            pct = missing[col] * 100
            if pct > 50:
                warnings.append(f"Column '{col}' has {pct:.1f}% missing values")
            elif pct > 20:
                warnings.append(f"Column '{col}' has {pct:.1f}% missing values")

        # Check 4: Domain schema (if strict mode)
        if strict and features:
            try:
                pfas_schema = build_pfas_schema(outcome, features)
                pfas_schema.validate(df, lazy=True)
            except pa.errors.SchemaErrors as e:
                for _, row in e.failure_cases.iterrows():
                    warnings.append(
                        f"Range warning — '{row.get('column')}': "
                        f"{row.get('failure_case')}"
                    )

        # Check 5: Minimum row count
        if len(df) < 20:
            warnings.append(
                f"Dataset has only {len(df)} rows — statistical power may be limited"
            )

        # Check 6: Outcome variance
        if df[outcome].std() < 1e-6:
            errors.append(
                f"Outcome variable '{outcome}' has near-zero variance — "
                f"modeling will fail"
            )

        passed = len(errors) == 0

        if passed:
            logger.info(
                f"Schema validation passed — {len(df)} rows, {len(warnings)} warnings"
            )
        else:
            logger.warning(
                f"Schema validation failed — "
                f"{len(errors)} errors, {len(warnings)} warnings"
            )

        return ValidationReport(
            passed=passed,
            errors=errors,
            warnings=warnings,
            n_rows_valid=len(df.dropna(subset=[outcome])),
            n_rows_total=len(df),
        )
