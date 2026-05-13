"""
LabelEncoder on **truly categorical** object/string columns only.

Pipeline per object/string column with nunique > 1:
  1. Try pd.to_numeric(errors='coerce').
  2. If every non-null value either converts cleanly OR is a known
     placeholder string (dash, 'N/A', 'ND', etc.), the column is
     treated as numeric — placeholders become NaN, no encoding.
  3. Otherwise the column is genuinely categorical and gets LabelEncoded.

This prevents concentration columns that contain '-' placeholder cells
from being mistakenly encoded as integer categories.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Strings that represent "not measured / not applicable" in spreadsheets.
_PLACEHOLDERS = frozenset({
    "-", "--", "---",
    "na", "n/a", "n.a.",
    "nan", "none", "null",
    "nd", "n.d.", "nr", "n.r.",
    "bdl", "bl", "<dl", "<bdl",
    "not detected", "not measured",
})


def _is_numeric_with_placeholders(series: pd.Series) -> bool:
    """Return True if every non-null value is either numeric or a placeholder."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    coerced = pd.to_numeric(non_null, errors="coerce")
    failed = non_null[coerced.isna()]
    if len(failed) == 0:
        return True
    return bool(failed.astype(str).str.strip().str.lower().isin(_PLACEHOLDERS).all())


def apply_upload_data_cleaning(
    df: pd.DataFrame,
    *,
    unified_meta: object | None = None,
) -> tuple[list[str], dict[str, Any]]:
    encodings: dict[str, Any] = {}
    if df.empty:
        return [], encodings

    non_unique_cols = [c for c in df.columns if int(df[c].nunique(dropna=True)) > 1]
    if non_unique_cols:
        sub = df[non_unique_cols]
        candidate_cols = list(sub.select_dtypes(include=["object", "string"]).columns)
    else:
        candidate_cols = []

    numeric_coerced: list[str] = []
    cat_cols: list[str] = []
    for col in candidate_cols:
        if _is_numeric_with_placeholders(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            numeric_coerced.append(col)
        else:
            cat_cols.append(col)

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str)).astype(np.float64)
        encodings[col] = {
            "kind": "label_encode",
            "classes": [str(c) for c in le.classes_.tolist()],
        }

    notes: list[str] = []
    if cat_cols:
        notes.append(
            f"LabelEncoder applied to {len(cat_cols)} categorical column(s): "
            + ", ".join(cat_cols[:8])
            + ("…" if len(cat_cols) > 8 else "")
        )
    if numeric_coerced:
        notes.append(
            f"{len(numeric_coerced)} numeric column(s) had placeholder strings (e.g. '-') "
            "converted to NaN: "
            + ", ".join(numeric_coerced[:8])
            + ("…" if len(numeric_coerced) > 8 else "")
        )

    logger.info(
        "Upload data cleaning: rows=%s cols=%s numeric_coerced=%s cat_encoded=%s unified_meta=%s",
        len(df),
        len(df.columns),
        len(numeric_coerced),
        len(cat_cols),
        unified_meta is not None,
    )
    return notes, encodings


def infer_layout_column_lists(
    df: pd.DataFrame, unified_meta: object | None
) -> tuple[list[str], list[str]]:
    cols_list = list(df.columns)
    if unified_meta is not None:
        ic = [c for c in unified_meta.input_cols if c in df.columns]
        oc = [c for c in unified_meta.output_cols if c in df.columns]
        return ic, oc
    input_start, input_end, output_start = 2, 37, 37
    n = len(cols_list)
    if n > output_start:
        return cols_list[input_start:input_end], cols_list[output_start:]
    if n > input_start:
        return cols_list[input_start:input_end], []
    return [], []
