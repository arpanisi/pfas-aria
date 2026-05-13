"""
Schema validator for the two-row header spreadsheet format.

Row 1: block labels
  - INPUT block:  INPUT, INDEPENDENT, INDEPENDENT VARIABLES, INDEPENDENT VARIABLE
  - OUTPUT block: OUTPUT, DEPENDENT, DEPENDENT VARIABLES, DEPENDENT VARIABLE
  - METADATA block: optional

Row 2: variable names

Required columns (flexible naming):
  - Experiment identifier: experiment_id, run_id, condition, cond, exp_id, sample_id
  - Time column: time_min, time (min), time, t_min, timepoint

No METADATA block required. condition in Row 2 is acceptable as the identifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.ingestion.unified_experimental_sheet import (
    UnifiedSheetMeta,
    parse_unified_experimental_excel,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ── Result types ──────────────────────────────────────────────────────────────


@dataclass
class ValidationError:
    code: str
    message: str
    fix: str = ""


@dataclass
class ValidationResult:
    passed: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    df: pd.DataFrame | None = None
    meta: UnifiedSheetMeta | None = None

    @property
    def first_error_message(self) -> str:
        return self.errors[0].message if self.errors else ""

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "errors": [
                {"code": e.code, "message": e.message, "fix": e.fix}
                for e in self.errors
            ],
            "warnings": self.warnings,
        }


# ── Validator ─────────────────────────────────────────────────────────────────


class SchemaValidator:
    MIN_DATA_ROWS = 5

    def validate_bytes(self, content: bytes, filename: str = "") -> ValidationResult:
        suffix = Path(filename).suffix.lower()
        if suffix in {".csv", ".tsv"}:
            return self._validate_csv(content, suffix)
        if suffix not in {".xlsx", ".xls"}:
            return ValidationResult(
                passed=False,
                errors=[
                    ValidationError(
                        code="UNSUPPORTED_FORMAT",
                        message=f"Unsupported file type '{suffix}'.",
                        fix="Upload a .xlsx, .xls, .csv, or .tsv file.",
                    )
                ],
            )
        return self._validate_excel(content)

    def _validate_excel(self, content: bytes) -> ValidationResult:
        errors: list[ValidationError] = []
        warnings: list[str] = []

        parsed = parse_unified_experimental_excel(content)

        if parsed is None:
            errors.append(
                ValidationError(
                    code="MISSING_ROLE_ROW",
                    message=("Could not detect the two-row header format."),
                    fix=(
                        "Row 1 must contain block labels for each column:\n"
                        "  • INDEPENDENT VARIABLES (or INPUT) for predictor columns\n"
                        "  • DEPENDENT VARIABLES (or OUTPUT) for response columns\n"
                        "Row 2 must contain the actual variable names.\n"
                        "Data starts from Row 3.\n\n"
                        "Your spreadsheet already has 'INDEPENDENT VARIABLES' and "
                        "'DEPENDENT VARIABLES' in Row 1 — make sure these labels "
                        "are present in the cells (not as merged-cell titles that "
                        "only occupy one column while the rest are blank).\n"
                        "Each column must have its own label in Row 1."
                    ),
                )
            )
            return ValidationResult(passed=False, errors=errors)

        df, meta = parsed

        # ── Check experiment identifier ───────────────────────────────────────
        if meta.experiment_id_col is None:
            errors.append(
                ValidationError(
                    code="MISSING_EXPERIMENT_ID",
                    message=(
                        "No experiment identifier column found. "
                        "Expected a column named one of: "
                        "experiment_id, run_id, condition, exp_id, sample_id."
                    ),
                    fix=(
                        "Name one of your columns 'experiment_id', 'run_id', or 'condition' "
                        "in Row 2. It can be under any block (INDEPENDENT VARIABLES or otherwise)."
                    ),
                )
            )

        # ── Check time column ─────────────────────────────────────────────────
        if meta.time_col is None:
            errors.append(
                ValidationError(
                    code="MISSING_TIME_COLUMN",
                    message=(
                        "No time column found. "
                        "Expected a column named one of: "
                        "time_min, time (min), time, t_min."
                    ),
                    fix=(
                        "Name one of your INPUT columns 'time (min)' or 'time_min' in Row 2."
                    ),
                )
            )

        # ── Check at least one predictor beyond time ──────────────────────────
        if meta.time_col is not None:
            non_time_inputs = [
                c
                for c in meta.input_cols
                if c != meta.time_col and c != meta.experiment_id_col
            ]
            if not non_time_inputs:
                errors.append(
                    ValidationError(
                        code="NO_INPUT_PREDICTORS",
                        message="No predictor columns found under INDEPENDENT VARIABLES beyond time.",
                        fix="Add at least one INPUT column beyond time (e.g. pH, voltage, temperature).",
                    )
                )

        # ── Check at least one output ─────────────────────────────────────────
        if not meta.output_cols:
            errors.append(
                ValidationError(
                    code="NO_OUTPUT_COLUMNS",
                    message="No DEPENDENT VARIABLES columns found.",
                    fix=(
                        "Label at least one column as DEPENDENT VARIABLES (or OUTPUT) in Row 1 "
                        "for your response variable (e.g. fluoride yield, concentration)."
                    ),
                )
            )

        # ── Check minimum data rows ───────────────────────────────────────────
        if len(df) < self.MIN_DATA_ROWS:
            errors.append(
                ValidationError(
                    code="INSUFFICIENT_DATA",
                    message=f"Only {len(df)} data rows found. At least {self.MIN_DATA_ROWS} required.",
                    fix="Ensure data starts at Row 3 and contains enough observations.",
                )
            )

        if errors:
            return ValidationResult(passed=False, errors=errors, warnings=warnings)

        # ── Soft warnings ─────────────────────────────────────────────────────
        n_constant = len(meta.constant_input_cols(df))
        if n_constant > 0:
            warnings.append(
                f"{n_constant} INPUT column(s) are constant across the entire dataset "
                "and will be excluded from hypothesis testing automatically."
            )

        if meta.time_col and meta.time_col in df.columns:
            time_numeric = pd.to_numeric(df[meta.time_col], errors="coerce")
            pct_null = time_numeric.isna().mean()
            if pct_null > 0.1:
                warnings.append(
                    f"Time column '{meta.time_col}' has {pct_null:.0%} non-numeric values."
                )

        if meta.experiment_id_col and meta.experiment_id_col in df.columns:
            n_exp = df[meta.experiment_id_col].nunique()
            if n_exp < 2:
                warnings.append(
                    f"Only {n_exp} unique value(s) in '{meta.experiment_id_col}'. "
                    "Panel modeling requires multiple experiments."
                )

        logger.info(
            "Schema validation passed: %s rows, %s inputs, %s outputs, exp_id=%s, time=%s",
            len(df),
            len(meta.input_cols),
            len(meta.output_cols),
            meta.experiment_id_col,
            meta.time_col,
        )
        return ValidationResult(passed=True, warnings=warnings, df=df, meta=meta)

    def _validate_csv(self, content: bytes, suffix: str) -> ValidationResult:
        import io

        sep = "\t" if suffix == ".tsv" else ","
        try:
            df = pd.read_csv(io.BytesIO(content), sep=sep, low_memory=False)
        except Exception as e:
            return ValidationResult(
                passed=False,
                errors=[
                    ValidationError(
                        code="PARSE_ERROR",
                        message=f"Could not parse file: {e}",
                        fix="Check that the file is a valid CSV or TSV.",
                    )
                ],
            )

        errors: list[ValidationError] = []
        warnings: list[str] = []

        df.columns = [
            str(c).strip().lower().replace(" ", "_").replace("-", "_")
            for c in df.columns
        ]

        id_candidates = (
            "experiment_id",
            "exp_id",
            "run_id",
            "sample_id",
            "condition",
            "cond",
        )
        time_candidates = (
            "time_min",
            "time_mins",
            "time",
            "t_min",
            "time_(min)",
            "time(min)",
        )

        id_col = next((c for c in df.columns if c in id_candidates), None)
        time_col = next((c for c in df.columns if c in time_candidates), None)

        if id_col is None:
            errors.append(
                ValidationError(
                    code="MISSING_EXPERIMENT_ID",
                    message="No experiment identifier column found.",
                    fix="Add a column named 'experiment_id', 'run_id', or 'condition'.",
                )
            )
        if time_col is None:
            errors.append(
                ValidationError(
                    code="MISSING_TIME_COLUMN",
                    message="No time column found.",
                    fix="Add a column named 'time_min' or 'time'.",
                )
            )
        if len(df) < self.MIN_DATA_ROWS:
            errors.append(
                ValidationError(
                    code="INSUFFICIENT_DATA",
                    message=f"Only {len(df)} data rows found.",
                    fix=f"At least {self.MIN_DATA_ROWS} rows are required.",
                )
            )

        if errors:
            return ValidationResult(passed=False, errors=errors, warnings=warnings)

        warnings.append(
            "CSV format: column block labels (INDEPENDENT/DEPENDENT VARIABLES) "
            "are not supported. All non-id, non-time columns treated as INPUT. "
            "Use Excel format for full schema control."
        )

        exclude = {id_col, time_col}
        input_cols = [c for c in df.columns if c not in exclude]
        meta = UnifiedSheetMeta(
            role_row_index=0,
            name_row_index=0,
            data_start_row_index=1,
            metadata_cols=[id_col] if id_col else [],
            input_cols=([time_col] if time_col else []) + input_cols,
            output_cols=[],
            experiment_id_col=id_col,
            time_col=time_col,
        )
        return ValidationResult(passed=True, warnings=warnings, df=df, meta=meta)


def validate_upload(content: bytes, filename: str) -> ValidationResult:
    """Top-level entry point called by the upload API route."""
    return SchemaValidator().validate_bytes(content, filename=filename)
