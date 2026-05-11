"""
PFAS notebook regime assignment.

Notebook cell structure: mg/L lists from input/output columns, ``"-"`` → NaN → 0
on those mg/L inputs, then drivers ``pfas_in_cols[0]``, ``[4]``, ``[5]``, ``pfca_rest``,
``X`` / ``mask``, ``r1``–``r5``.

Normalized API column names use ``mg/l`` / ``pfbs``; helpers mirror the notebook's
``'mg/L'`` / ``'PFBS'`` substring checks including those spellings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


def _notebook_mg_l(col: str) -> bool:
    return "mg/L" in col or "mg/l" in col.lower().replace(" ", "")


def _notebook_pfbs_in_col(col: str) -> bool:
    return "PFBS" in col or "pfbs" in col.lower()


@dataclass
class LegacyPfcaRegimeResult:
    """Output of regime assignment functions."""

    regimes: dict[int, pd.DataFrame]
    input_cols: list[str]
    output_cols: list[str]
    pfas_input_mg_l_cols: list[str]
    pfas_cols: list[str]
    pfas_out_cols: list[str]
    pfbs_out_col: str | None
    pfoa_col: str
    pfbs_col: str
    pfba_col: str
    warnings: list[str] = field(default_factory=list)
    regime_row_counts: dict[int, int] = field(default_factory=dict)


def assign_pfca_regime_masks_from_column_lists(
    df: pd.DataFrame,
    input_cols: list[str],
    output_cols: list[str],
    *,
    replace_no_in_inputs: bool = True,
) -> LegacyPfcaRegimeResult:
    warnings: list[str] = []
    missing = [c for c in input_cols + output_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Unknown columns in dataframe: {missing[:12]}")

    df_work = df.copy()
    if replace_no_in_inputs and input_cols:
        sub = df_work[input_cols].replace("no", "No")
        df_work[input_cols] = sub

    print(
        "[PFAS regime] input_cols (independent):",
        len(input_cols),
        input_cols,
        flush=True,
    )
    print(
        "[PFAS regime] output_cols (dependent):",
        len(output_cols),
        output_cols,
        flush=True,
    )

    pfas_in_cols = [col for col in input_cols if _notebook_mg_l(col)]
    pfas_out_cols = [
        col for col in output_cols if _notebook_mg_l(col) and not _notebook_pfbs_in_col(col)
    ]
    pfbs_cands = [col for col in output_cols if _notebook_mg_l(col) and _notebook_pfbs_in_col(col)]
    pfbs_out_col = pfbs_cands[0] if pfbs_cands else None
    if pfbs_out_col is None:
        warnings.append("No output column with mg/L and PFBS in name — pfbs_out_col is None.")

    pfas_input_mg_l_cols = list(pfas_in_cols)
    pfas_cols = list(pfas_in_cols)

    pfoa_in = pfas_in_cols[0]
    pfbs_in = pfas_in_cols[4]
    pfba_in = pfas_in_cols[5]
    pfca_rest = [c for c in pfas_in_cols if c not in [pfbs_in]]
    _ = pfca_rest

    print("[PFAS regime] pfas_in_cols:", pfas_in_cols, flush=True)
    print(
        "[PFAS regime] pfoa_in=[0]",
        pfoa_in,
        "pfbs_in=[4]",
        pfbs_in,
        "pfba_in=[5]",
        pfba_in,
        flush=True,
    )
    print(
        "[PFAS regime] pfas_out_cols=",
        pfas_out_cols,
        "pfbs_out_col=",
        pfbs_out_col,
        flush=True,
    )

    # Notebook: treat "-" as missing concentration, then numeric zeros for regime mask.
    if pfas_in_cols:
        df_work[pfas_in_cols] = df_work[pfas_in_cols].replace("-", np.nan).fillna(0)

    pfas_conc = df_work[pfas_in_cols]
    mask = pfas_conc.notna() & (pfas_conc != 0)

    r1 = df_work[(mask[pfoa_in]) & (~mask.drop(columns=[pfoa_in]).any(axis=1))]
    r2 = df_work[
        (mask[pfoa_in])
        & (mask[pfba_in])
        & (~mask.drop(columns=[pfoa_in, pfba_in]).any(axis=1))
    ]
    r3 = df_work[(~mask[pfbs_in]) & (mask.drop(columns=[pfbs_in]).all(axis=1))]
    r4 = df_work[(mask[pfbs_in]) & (~mask.drop(columns=[pfbs_in]).any(axis=1))]

    assigned_idx = r1.index.union(r2.index).union(r3.index).union(r4.index)
    r5 = df_work.drop(index=assigned_idx)

    regimes_raw = {1: r1, 2: r2, 3: r3, 4: r4, 5: r5}
    regime_row_counts = {rid: len(sub) for rid, sub in regimes_raw.items()}
    regimes = {k: v for k, v in regimes_raw.items() if len(v) > 0}

    print("[PFAS regime] regime_row_counts:", regime_row_counts, flush=True)
    print(
        "[PFAS regime] nonempty regime_ids:",
        list(regimes.keys()),
        {k: len(v) for k, v in regimes.items()},
        flush=True,
    )

    return LegacyPfcaRegimeResult(
        regimes=regimes,
        input_cols=input_cols,
        output_cols=output_cols,
        pfas_input_mg_l_cols=pfas_input_mg_l_cols,
        pfas_cols=pfas_cols,
        pfas_out_cols=pfas_out_cols,
        pfbs_out_col=pfbs_out_col,
        pfoa_col=pfoa_in,
        pfbs_col=pfbs_in,
        pfba_col=pfba_in,
        warnings=warnings,
        regime_row_counts=regime_row_counts,
    )


def assign_legacy_pfca_regime_masks(
    df: pd.DataFrame,
    *,
    input_start: int = 2,
    input_end: int = 37,
    output_start: int = 37,
    replace_no_in_inputs: bool = True,
) -> LegacyPfcaRegimeResult:
    cols = list(df.columns)
    n = len(cols)

    if input_end > n or output_start > n:
        raise ValueError(
            f"DataFrame has {n} columns; need at least max(input_end, output_start) "
            f"(got input_end={input_end}, output_start={output_start})."
        )

    input_cols = cols[input_start:input_end]
    output_cols = cols[output_start:]
    return assign_pfca_regime_masks_from_column_lists(
        df,
        input_cols,
        output_cols,
        replace_no_in_inputs=replace_no_in_inputs,
    )


def legacy_regime_result_to_result_iter_payload(result: LegacyPfcaRegimeResult) -> dict[str, Any]:
    return {
        "regimes": result.regimes,
        "regime_frames": result.regimes,
        "input_cols": result.input_cols,
        "output_cols": result.output_cols,
        "pfas_input_mg_l_cols": result.pfas_input_mg_l_cols,
        "pfas_cols": result.pfas_cols,
        "pfas_out_cols": result.pfas_out_cols,
        "pfbs_out_col": result.pfbs_out_col,
        "regime_row_counts": {str(k): v for k, v in result.regime_row_counts.items()},
    }
