"""
Dataset layout for automated screening — one segment (full table).

No species-specific masks or spreadsheet column inference beyond explicit
input/output column lists (unified sheet meta) or fixed index slices (legacy CSV).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class ScreeningLayoutResult:
    """Screening uses ``regimes[1]`` as the full normalized dataframe."""

    regimes: dict[int, pd.DataFrame]
    input_cols: list[str]
    output_cols: list[str]
    warnings: list[str] = field(default_factory=list)
    regime_row_counts: dict[int, int] = field(default_factory=dict)


def assign_screening_layout_from_column_lists(
    df: pd.DataFrame,
    input_cols: list[str],
    output_cols: list[str],
    *,
    replace_no_in_inputs: bool = True,
) -> ScreeningLayoutResult:
    missing = [c for c in input_cols + output_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Unknown columns in dataframe: {missing[:12]}")

    df_work = df.copy()
    if replace_no_in_inputs and input_cols:
        df_work[input_cols] = df_work[input_cols].replace("no", "No")

    rid = 1
    n = len(df_work)
    return ScreeningLayoutResult(
        regimes={rid: df_work},
        input_cols=list(input_cols),
        output_cols=list(output_cols),
        warnings=[],
        regime_row_counts={rid: n},
    )


def assign_screening_layout_from_legacy_column_slices(
    df: pd.DataFrame,
    *,
    input_start: int = 2,
    input_end: int = 37,
    output_start: int = 37,
    replace_no_in_inputs: bool = True,
) -> ScreeningLayoutResult:
    cols = list(df.columns)
    n = len(cols)
    if input_end > n or output_start > n:
        raise ValueError(
            f"DataFrame has {n} columns; need at least max(input_end, output_start) "
            f"(got input_end={input_end}, output_start={output_start})."
        )
    input_cols = cols[input_start:input_end]
    output_cols = cols[output_start:]
    return assign_screening_layout_from_column_lists(
        df,
        input_cols,
        output_cols,
        replace_no_in_inputs=replace_no_in_inputs,
    )


def screening_layout_to_result_iter_payload(
    result: ScreeningLayoutResult,
) -> dict[str, Any]:
    """Shape expected by :func:`persist_legacy_result_iter` (PFAS keys left empty)."""
    return {
        "regimes": result.regimes,
        "regime_frames": result.regimes,
        "input_cols": result.input_cols,
        "output_cols": result.output_cols,
        "pfas_input_mg_l_cols": [],
        "pfas_cols": [],
        "pfas_out_cols": [],
        "pfbs_out_col": None,
        "regime_row_counts": {str(k): v for k, v in result.regime_row_counts.items()},
    }
