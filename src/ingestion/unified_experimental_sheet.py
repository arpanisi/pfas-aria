"""
Unified experimental spreadsheet layout.

After optional title preamble, one row marks each column as independent (input)
or dependent (output); the next row holds column names; following rows are data.

PFAS driver columns for regime masks (PFOA, PFBA, PFBS) are resolved by name on
mg/L input columns, with positional fallbacks when names are ambiguous.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO
from typing import Literal

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

Role = Literal["input", "output", "unknown"]

_INDEP_KW = (
    "independent",
    "indep",
    "input",
    "predictor",
    "covariate",
    "explanatory",
)
_DEP_KW = (
    "dependent",
    "dep",
    "output",
    "response",
    "target",
)


def _norm_header_cell(v: object) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


def normalize_dataframe_columns_inplace(df: pd.DataFrame) -> None:
    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns
    ]


def classify_indep_dep_cell(cell: object) -> Role:
    if pd.isna(cell):
        return "unknown"
    s = str(cell).strip().lower()
    if any(kw in s for kw in _INDEP_KW):
        return "input"
    if any(kw in s for kw in _DEP_KW):
        return "output"
    return "unknown"


def _role_row_score(row: pd.Series) -> int:
    return sum(1 for v in row.tolist() if classify_indep_dep_cell(v) != "unknown")


def _role_row_has_input_and_output(row: pd.Series) -> bool:
    ins = outs = 0
    for v in row.tolist():
        r = classify_indep_dep_cell(v)
        if r == "input":
            ins += 1
        elif r == "output":
            outs += 1
    return ins >= 4 and outs >= 2


def _likely_name_row(row: pd.Series) -> bool:
    texts = [str(v).strip() for v in row.tolist() if pd.notna(v) and str(v).strip()]
    if len(texts) < 4:
        return False
    if _role_row_score(row) >= max(4, len(texts) // 2):
        return False
    return len(texts) >= max(5, len(row) // 5)


def find_unified_layout_row_indices(raw: pd.DataFrame, scan: int = 16) -> tuple[int, int] | None:
    """
    Return ``(role_row_idx, name_row_idx)`` if a unified layout is found, else None.
    """
    n = min(scan, len(raw))
    for i in range(n - 1):
        r0 = raw.iloc[i]
        r1 = raw.iloc[i + 1]
        if not _role_row_has_input_and_output(r0):
            continue
        if _role_row_score(r0) < 8:
            continue
        if not _likely_name_row(r1):
            continue
        return (i, i + 1)
    return None


def _split_columns_by_role(types_row: pd.Series, n_cols: int) -> tuple[list[int], list[int]]:
    in_ix: list[int] = []
    out_ix: list[int] = []
    for j in range(min(n_cols, len(types_row))):
        role = classify_indep_dep_cell(types_row.iloc[j])
        if role == "input":
            in_ix.append(j)
        elif role == "output":
            out_ix.append(j)
    for j in range(n_cols):
        if j >= len(types_row):
            if j not in in_ix and j not in out_ix:
                in_ix.append(j)
            continue
        if classify_indep_dep_cell(types_row.iloc[j]) == "unknown":
            if j not in in_ix and j not in out_ix:
                in_ix.append(j)
    return sorted(set(in_ix)), sorted(set(out_ix))


def _make_unique_column_names(raw_names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for raw in raw_names:
        base = _norm_header_cell(raw) or "col"
        base_snake = (
            base.strip().lower().replace(" ", "_").replace("-", "_") or "col"
        )
        if base_snake not in seen:
            seen[base_snake] = 0
            out.append(base_snake)
        else:
            seen[base_snake] += 1
            out.append(f"{base_snake}__{seen[base_snake]}")
    return out


@dataclass
class UnifiedSheetMeta:
    role_row_index: int
    name_row_index: int
    data_start_row_index: int
    input_cols: list[str]
    output_cols: list[str]


def parse_unified_experimental_excel(content: bytes) -> tuple[pd.DataFrame, UnifiedSheetMeta] | None:
    """
    If the workbook matches the unified layout, return ``(df, meta)``; else ``None``.
    """
    preview = pd.read_excel(BytesIO(content), header=None, nrows=min(40, 5000))
    layout = find_unified_layout_row_indices(preview)
    if layout is None:
        return None

    role_i, name_i = layout
    full = pd.read_excel(BytesIO(content), header=None)
    if name_i + 1 >= len(full):
        return None

    types_row = full.iloc[role_i]
    names_row = full.iloc[name_i]
    data = full.iloc[name_i + 1 :].copy()

    n_cols = int(data.shape[1])
    raw_headers = [
        names_row.iloc[j] if j < len(names_row) else f"col_{j}" for j in range(n_cols)
    ]
    data.columns = _make_unique_column_names([str(x) for x in raw_headers])
    normalize_dataframe_columns_inplace(data)

    in_ix, out_ix = _split_columns_by_role(types_row, n_cols)
    input_cols = [data.columns[j] for j in in_ix if j < len(data.columns)]
    output_cols = [data.columns[j] for j in out_ix if j < len(data.columns)]

    meta = UnifiedSheetMeta(
        role_row_index=role_i,
        name_row_index=name_i,
        data_start_row_index=name_i + 1,
        input_cols=input_cols,
        output_cols=output_cols,
    )
    logger.info(
        "Unified experimental layout: role_row=%s name_row=%s inputs=%s outputs=%s",
        role_i,
        name_i,
        len(input_cols),
        len(output_cols),
    )
    return data.reset_index(drop=True), meta


def load_excel_bytes_with_layout(content: bytes) -> tuple[pd.DataFrame, int | None, UnifiedSheetMeta | None]:
    """
    Load the first Excel sheet: unified independent/dependent layout if detected,
    otherwise :func:`read_excel_smart` + normalized column names.

    Returns ``(df, excel_header_row, unified_meta_or_none)``.
    ``excel_header_row`` is the 0-based row index used as column names (same
    convention as :func:`read_excel_smart`).
    """
    parsed = parse_unified_experimental_excel(content)
    if parsed is not None:
        df, meta = parsed
        return df, meta.name_row_index, meta

    from src.ingestion.spreadsheet_infer import read_excel_smart

    df, header_i = read_excel_smart(content)
    normalize_dataframe_columns_inplace(df)
    return df, header_i, None


def _is_mg_l_concentration(col: str) -> bool:
    cl = col.lower().replace(" ", "")
    return "mg/l" in cl


def list_pfas_like_output_columns(output_cols: list[str]) -> tuple[list[str], str | None]:
    pfas_out = [
        c
        for c in output_cols
        if _is_mg_l_concentration(c) and "pfbs" not in c.lower()
    ]
    pfbs_cands = [
        c for c in output_cols if _is_mg_l_concentration(c) and "pfbs" in c.lower()
    ]
    return pfas_out, (pfbs_cands[0] if pfbs_cands else None)


def list_mg_l_column_names_in_order(col_names: list[str]) -> list[str]:
    """Every ``col_names`` entry that looks like mg/L, preserving order."""
    return [c for c in col_names if _is_mg_l_concentration(c)]


_INITIAL_INPUT_HINTS = ("initial", "inlet", "feedstock")


def select_regime_pfas_input_mg_l_columns(
    input_cols: list[str],
    *,
    warnings: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """
    Split independent-side mg/L columns for regime masks vs full listing.

    Returns ``(pfas_input_all_mg_l, pfas_regime_mask_cols)``.

    * ``pfas_input_all_mg_l`` — every mg/L column listed on the **input /
      independent** side (never includes dependent/output columns).

    * ``pfas_regime_mask_cols`` — columns used to build the species presence
      ``mask``. When at least three input mg/L names contain *initial*, *inlet*,
      or *feedstock*, only those columns are used so **initial feedstock PFAS**
      are separated from other mg/L inputs (e.g. bulk reagents). Otherwise all
      input-side mg/L columns are used (legacy behavior).

    Dependent-side mg/L must be listed only in ``output_cols`` so they never
    enter this function.
    """
    w = warnings if warnings is not None else []
    all_mg = list_mg_l_column_names_in_order(input_cols)
    initial_tagged = [
        c for c in all_mg if any(h in c.lower() for h in _INITIAL_INPUT_HINTS)
    ]
    if len(initial_tagged) >= 3:
        w.append(
            "Regime species matrix: using only initial/inlet/feedstock-tagged "
            f"mg/L *input* columns ({len(initial_tagged)} of {len(all_mg)} mg/L inputs)."
        )
        return all_mg, initial_tagged
    if initial_tagged:
        w.append(
            f"{len(initial_tagged)} initial-like mg/L input column(s) but fewer than 3; "
            "using all mg/L input columns for the regime species matrix."
        )
    return all_mg, all_mg


list_pfas_mg_l_output_columns = list_pfas_like_output_columns


def detect_pfoa_pfba_pfbs_driver_columns(
    pfas_cols: list[str],
    *,
    warnings: list[str] | None = None,
) -> tuple[str, str, str]:
    """
    Pick PFOA-, PFBA-, and PFBS-input columns by name, with legacy positional
    fallbacks (indices 0, 4, 5 among mg/L inputs) when labels are missing.
    """
    w = warnings if warnings is not None else []
    if not pfas_cols:
        raise ValueError("No mg/L input columns were found for regime masks.")

    pfoa: str | None = None
    pfba: str | None = None
    pfbs: str | None = None

    for c in pfas_cols:
        cl = c.lower()
        if pfbs is None and "pfbs" in cl:
            pfbs = c
        if pfba is None and "pfba" in cl and "pfbs" not in cl:
            pfba = c
        if pfoa is None and "pfoa" in cl and "pfbs" not in cl:
            pfoa = c

    if pfoa is None:
        for c in pfas_cols:
            cl = c.lower()
            if "pfbs" in cl or "pfba" in cl:
                continue
            if re.search(r"\bc8\b", cl) or re.search(r"\(c8\)", cl):
                pfoa = c
                break

    if pfoa is None and len(pfas_cols) >= 1:
        pfoa = pfas_cols[0]
        w.append("PFOA driver column: fallback to first mg/L input column.")

    if pfbs is None:
        for c in pfas_cols:
            if "pfbs" in c.lower():
                pfbs = c
                break
    if pfbs is None and len(pfas_cols) >= 5:
        pfbs = pfas_cols[4]
        w.append("PFBS driver column: fallback to index 4 among mg/L inputs (legacy).")

    if pfba is None:
        for c in pfas_cols:
            cl = c.lower()
            if "pfba" in cl and "pfbs" not in cl:
                pfba = c
                break
    if pfba is None and len(pfas_cols) >= 6:
        pfba = pfas_cols[5]
        w.append("PFBA driver column: fallback to index 5 among mg/L inputs (legacy).")

    if pfbs is None or pfoa is None or pfba is None:
        raise ValueError(
            "Could not resolve PFOA / PFBA / PFBS driver columns. "
            f"mg/L input columns: {pfas_cols}"
        )

    return pfoa, pfba, pfbs
