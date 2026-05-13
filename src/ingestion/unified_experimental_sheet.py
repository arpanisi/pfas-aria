"""
Unified experimental spreadsheet layout.

Required format — two-row header:
  Row 1: block labels — INPUT/INDEPENDENT/INDEPENDENT VARIABLES | OUTPUT/DEPENDENT/DEPENDENT VARIABLES
          METADATA block is optional.
  Row 2: actual variable names
  Row 3+: data

Required columns:
  - One column named: experiment_id, run_id, or condition (under any block)
  - One column named: time_min, time (min), time, or t_min (under INPUT block)

Excel files that do not match this layout are rejected — no fallback inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import Literal

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

Role = Literal["metadata", "input", "output", "unknown"]

# ── Keyword lists — generic, no domain terms ──────────────────────────────────

_METADATA_KW = ("metadata", "meta", "identifier", "info")

_INPUT_KW = (
    "independent",
    "indep",
    "input",
    "predictor",
    "covariate",
    "explanatory",
    "feature",
    "variable",
)

_OUTPUT_KW = (
    "dependent",
    "dep",
    "output",
    "response",
    "target",
    "outcome",
    "result",
    "measure",
)

# Required column name fragments — all acceptable variants
_EXPERIMENT_ID_CANDIDATES = (
    "experiment_id",
    "exp_id",
    "run_id",
    "sample_id",
    "condition",
    "cond",
)

_TIME_CANDIDATES = (
    "time_min",
    "time_mins",
    "time_(min)",
    "time(min)",
    "time",
    "t_min",
    "timepoint",
    "time_point",
)


# ── Cell classification ───────────────────────────────────────────────────────


def _norm(v: object) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip().lower().replace(" ", "_").replace("-", "_")


def classify_role_cell(cell: object) -> Role:
    s = _norm(cell)
    if not s:
        return "unknown"
    # Check metadata first (most specific)
    if any(kw == s or s.startswith(kw) for kw in _METADATA_KW):
        return "metadata"
    # Output before input — "dependent" must not match "independent"
    if any(kw in s for kw in _OUTPUT_KW) and "independent" not in s:
        return "output"
    if any(kw in s for kw in _INPUT_KW):
        return "input"
    return "unknown"


# Keep old name as alias so existing callers don't break
classify_indep_dep_cell = classify_role_cell


def _role_row_score(row: pd.Series) -> int:
    return sum(1 for v in row if classify_role_cell(v) != "unknown")


def _role_row_has_required_blocks(row: pd.Series) -> bool:
    """
    A valid role row must have at least one INPUT-like and one OUTPUT-like cell.
    METADATA is optional.
    """
    has_input = has_output = False
    for v in row:
        r = classify_role_cell(v)
        if r == "input":
            has_input = True
        elif r == "output":
            has_output = True
    return has_input and has_output


def _likely_name_row(row: pd.Series) -> bool:
    texts = [str(v).strip() for v in row if pd.notna(v) and str(v).strip()]
    if len(texts) < 2:
        return False
    if _role_row_score(row) >= max(2, len(texts) // 2):
        return False
    return len(texts) >= max(2, len(row) // 5)


def normalize_dataframe_columns_inplace(df: pd.DataFrame) -> None:
    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_").replace("(", "_").replace(")", "_")
        for c in df.columns
    ]


def _make_unique_column_names(raw_names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for raw in raw_names:
        base = str(raw).strip() or "col"
        snake = (
            base.lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("(", "_")
            .replace(")", "_")
            .replace("/", "_")
            .replace(",", "_")
            .replace("°", "")
            .replace("μ", "u")
        ) or "col"
        # Collapse multiple underscores
        while "__" in snake:
            snake = snake.replace("__", "_")
        snake = snake.strip("_")
        if snake not in seen:
            seen[snake] = 0
            out.append(snake)
        else:
            seen[snake] += 1
            out.append(f"{snake}__{seen[snake]}")
    return out


# ── Layout detection ──────────────────────────────────────────────────────────


def find_unified_layout_row_indices(
    raw: pd.DataFrame, scan: int = 16
) -> tuple[int, int] | None:
    """
    Scan the first ``scan`` rows for a (role_row, name_row) pair.
    Returns ``(role_row_idx, name_row_idx)`` or None.
    """
    n = min(scan, len(raw) - 1)
    for i in range(n):
        r0 = raw.iloc[i]
        r1 = raw.iloc[i + 1]
        if not _role_row_has_required_blocks(r0):
            continue
        if _role_row_score(r0) < 2:
            continue
        if not _likely_name_row(r1):
            continue
        return (i, i + 1)
    return None


def _split_columns_by_role(
    types_row: pd.Series, n_cols: int
) -> tuple[list[int], list[int], list[int]]:
    """
    Returns (metadata_indices, input_indices, output_indices).
    Unknown cells and NaN cells inherit the last known role (forward-fill).
    This handles merged cells where only the first cell has the label.
    """
    meta_ix: list[int] = []
    in_ix: list[int] = []
    out_ix: list[int] = []

    last_role: Role = "input"  # default if nothing seen yet

    for j in range(n_cols):
        cell = types_row.iloc[j] if j < len(types_row) else None
        role = classify_role_cell(cell)

        if role == "unknown":
            # Forward-fill from last known role
            role = last_role
        else:
            last_role = role

        if role == "metadata":
            meta_ix.append(j)
        elif role == "output":
            out_ix.append(j)
        else:
            in_ix.append(j)

    return sorted(meta_ix), sorted(in_ix), sorted(out_ix)


# ── Metadata ──────────────────────────────────────────────────────────────────


@dataclass
class UnifiedSheetMeta:
    role_row_index: int
    name_row_index: int
    data_start_row_index: int
    metadata_cols: list[str] = field(default_factory=list)
    input_cols: list[str] = field(default_factory=list)
    output_cols: list[str] = field(default_factory=list)
    experiment_id_col: str | None = None
    time_col: str | None = None

    @property
    def all_cols(self) -> list[str]:
        return self.metadata_cols + self.input_cols + self.output_cols

    def candidate_predictor_cols(self, df: pd.DataFrame) -> list[str]:
        """
        INPUT columns that vary (nunique > 1) in df, excluding time and id cols.
        Uses raw values (before numeric coercion) so text columns like
        'yes/no', surfactant names etc. are correctly treated as non-constant.
        """
        exclude = {self.experiment_id_col, self.time_col}
        return [
            c for c in self.input_cols
            if c not in exclude
            and c in df.columns
            and df[c].nunique(dropna=True) > 1
        ]

    def constant_input_cols(self, df: pd.DataFrame) -> list[str]:
        """
        INPUT columns that do not vary in df — excluded from modeling.
        A column is constant if it has ≤1 unique non-null value in its
        raw form (before numeric coercion).
        """
        exclude = {self.experiment_id_col, self.time_col}
        return [
            c for c in self.input_cols
            if c not in exclude
            and c in df.columns
            and df[c].nunique(dropna=True) <= 1
        ]


# ── Required column detection ─────────────────────────────────────────────────


def _find_required_col(
    candidates: tuple[str, ...], available: list[str]
) -> str | None:
    """
    Return the first column whose normalised name matches any candidate.
    Tries exact match first, then startswith.
    """
    available_norm = {c: _norm(c) for c in available}
    # Exact match
    for col, col_n in available_norm.items():
        if col_n in candidates:
            return col
    # Startswith match
    for col, col_n in available_norm.items():
        for cand in candidates:
            if col_n.startswith(cand):
                return col
    return None


# ── Parser ────────────────────────────────────────────────────────────────────


def parse_unified_experimental_excel(
    content: bytes,
) -> tuple[pd.DataFrame, UnifiedSheetMeta] | None:
    """
    If the workbook matches the unified layout, return ``(df, meta)``; else ``None``.
    """
    try:
        preview = pd.read_excel(BytesIO(content), header=None, nrows=min(40, 5000))
    except Exception:
        return None

    layout = find_unified_layout_row_indices(preview)
    if layout is None:
        return None

    role_i, name_i = layout
    try:
        full = pd.read_excel(BytesIO(content), header=None)
    except Exception:
        return None

    if name_i + 1 >= len(full):
        return None

    types_row = full.iloc[role_i]
    names_row = full.iloc[name_i]
    data = full.iloc[name_i + 1:].copy()

    n_cols = int(data.shape[1])
    raw_headers = [
        names_row.iloc[j] if j < len(names_row) else f"col_{j}"
        for j in range(n_cols)
    ]
    col_names = _make_unique_column_names([str(x) for x in raw_headers])
    data.columns = col_names

    meta_ix, in_ix, out_ix = _split_columns_by_role(types_row, n_cols)
    metadata_cols = [col_names[j] for j in meta_ix if j < len(col_names)]
    input_cols = [col_names[j] for j in in_ix if j < len(col_names)]
    output_cols = [col_names[j] for j in out_ix if j < len(col_names)]

    # Search for required columns across metadata + input + all cols
    all_cols = metadata_cols + input_cols + output_cols
    id_col = _find_required_col(_EXPERIMENT_ID_CANDIDATES, metadata_cols + input_cols + all_cols)
    time_col = _find_required_col(_TIME_CANDIDATES, input_cols + all_cols)

    # Experiment id and time are layout keys, not regression inputs; keep them out of input_cols
    # so previews and screening use true covariates only.
    reserved = {c for c in (id_col, time_col) if c}
    input_cols = [c for c in input_cols if c not in reserved]
    # If another column is the experiment key, "condition" in the independent block is still a
    # run / benchmark label, not a covariate.
    input_cols = [c for c in input_cols if _norm(c) != "condition"]

    meta = UnifiedSheetMeta(
        role_row_index=role_i,
        name_row_index=name_i,
        data_start_row_index=name_i + 1,
        metadata_cols=metadata_cols,
        input_cols=input_cols,
        output_cols=output_cols,
        experiment_id_col=id_col,
        time_col=time_col,
    )
    logger.info(
        "Unified layout: role_row=%s name_row=%s metadata=%s inputs=%s outputs=%s exp_id=%s time=%s",
        role_i, name_i,
        len(metadata_cols), len(input_cols), len(output_cols),
        id_col, time_col,
    )
    return data.reset_index(drop=True), meta


def load_excel_bytes_with_layout(
    content: bytes,
) -> tuple[pd.DataFrame, int, UnifiedSheetMeta]:
    """
    Load the first Excel sheet using the unified two-row header layout.

    Raises ``IngestionError`` if the workbook does not match the required format.
    Returns ``(df, name_row_index, meta)``.
    """
    from src.utils.exceptions import IngestionError

    parsed = parse_unified_experimental_excel(content)
    if parsed is None:
        raise IngestionError(
            "Excel file does not match the required two-row header format.\n"
            "Row 1 must contain block labels (INDEPENDENT VARIABLES / DEPENDENT VARIABLES).\n"
            "Row 2 must contain variable names. Data starts at Row 3."
        )

    df, meta = parsed
    return df, meta.name_row_index, meta
