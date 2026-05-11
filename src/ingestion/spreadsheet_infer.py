"""
Heuristics for spreadsheets with optional preamble rows (titles, notes, units).

Used by API upload preview and DataLoader so UI and pipeline agree on layout.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd


def _cell_str(v: object) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


def _header_row_score(row: pd.Series, next_row: pd.Series | None) -> float:
    """Higher = more likely this row is the true column-header row."""
    vals = [_cell_str(v) for v in row.tolist() if pd.notna(v)]
    if len(vals) < 2:
        return -1000.0

    score = min(len(vals), 24) * 2.0
    short_labels = 0
    for v in vals:
        if not v:
            continue
        if len(v) > 100:
            score -= 8.0
        elif len(v) <= 64:
            short_labels += 1
        if re.fullmatch(r"-?\d+(\.\d+)?([eE][+-]?\d+)?", v):
            score -= 1.2

    score += short_labels * 0.35

    if len(vals) <= 2:
        score -= 6.0

    if next_row is not None:
        nn_next = int(next_row.notna().sum())
        if nn_next >= max(2, len(vals) * 0.45):
            score += 2.0
        next_vals = [_cell_str(v) for v in next_row.tolist() if pd.notna(v)][
            : min(16, len(next_row))
        ]
        if not next_vals:
            return score
        numeric_like = sum(
            1
            for v in next_vals
            if v and re.fullmatch(r"-?\d+(\.\d+)?([eE][+-]?\d+)?", v)
        )
        if numeric_like >= max(2, len(next_vals) // 3):
            score += 4.0

    return score


def infer_excel_header_row_index(
    raw_preview: pd.DataFrame, max_candidates: int = 12
) -> int:
    """
    Return 0-based row index in a header=None frame to use as ``header=`` for read_excel.

    Rows above this index are preamble (skipped from the data table).
    """
    if raw_preview.empty or len(raw_preview) == 1:
        return 0

    n = min(max_candidates, len(raw_preview))
    best_i = 0
    best_score = _header_row_score(
        raw_preview.iloc[0],
        raw_preview.iloc[1] if len(raw_preview) > 1 else None,
    )

    for i in range(1, n):
        nxt = raw_preview.iloc[i + 1] if i + 1 < len(raw_preview) else None
        s = _header_row_score(raw_preview.iloc[i], nxt)
        if s > best_score + 0.01:
            best_score = s
            best_i = i

    if best_score < 2.0 and len(raw_preview) > 1:
        s0 = _header_row_score(
            raw_preview.iloc[0],
            raw_preview.iloc[1] if len(raw_preview) > 1 else None,
        )
        s1 = _header_row_score(
            raw_preview.iloc[1],
            raw_preview.iloc[2] if len(raw_preview) > 2 else None,
        )
        if s1 > s0 + 3.0:
            return 1

    return best_i


def read_excel_smart(content_or_path: bytes | Path, *, nrows_preview: int = 50) -> tuple[pd.DataFrame, int]:
    """
    Read the first sheet from Excel bytes or filesystem path.

    Returns ``(dataframe, header_row_index)`` where ``header_row_index`` is the
    row used as column names (0 = first row is header, 1 = skip one preamble row, ...).
    """
    if isinstance(content_or_path, Path):
        preview = pd.read_excel(content_or_path, header=None, nrows=nrows_preview)
        header_i = infer_excel_header_row_index(preview)
        df = pd.read_excel(content_or_path, header=header_i)
        return df, header_i

    buf = io.BytesIO(content_or_path)
    preview = pd.read_excel(buf, header=None, nrows=nrows_preview)
    header_i = infer_excel_header_row_index(preview)
    df = pd.read_excel(io.BytesIO(content_or_path), header=header_i)
    return df, header_i
