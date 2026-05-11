"""Tests for Excel preamble / header-row inference."""

from __future__ import annotations

import pandas as pd

from src.ingestion.spreadsheet_infer import infer_excel_header_row_index, read_excel_smart


def test_infer_header_row_first_row_is_header() -> None:
    raw = pd.DataFrame(
        [
            ["id", "value", "rate"],
            [1, 2.0, 0.5],
            [2, 3.0, 0.6],
        ]
    )
    assert infer_excel_header_row_index(raw) == 0


def test_infer_header_row_skip_title_row() -> None:
    raw = pd.DataFrame(
        [
            ["PFAS batch study Q3", None, None],
            ["id", "value", "degradation_rate"],
            [1, 2.0, 0.5],
            [2, 3.0, 0.6],
        ]
    )
    assert infer_excel_header_row_index(raw) == 1


def test_read_excel_smart_round_trip(tmp_path) -> None:
    path = tmp_path / "t.xlsx"
    title = pd.DataFrame([["Title only", None, None]])
    header = pd.DataFrame([["a", "b", "c"]])
    data = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
    combined = pd.concat([title, header, data], ignore_index=True)
    combined.to_excel(path, index=False, header=False)

    df, h = read_excel_smart(path)
    assert h == 1
    assert list(df.columns) == ["a", "b", "c"]
    assert len(df) == 2


def test_read_excel_smart_from_bytes_matches(tmp_path) -> None:
    path = tmp_path / "t2.xlsx"
    title = pd.DataFrame([["Report", None, None]])
    header = pd.DataFrame([["h1", "h2", "h3"]])
    body = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
    pd.concat([title, header, body], ignore_index=True).to_excel(
        path, index=False, header=False
    )
    content = path.read_bytes()
    df, h = read_excel_smart(content)
    assert h == 1
    assert list(df.columns) == ["h1", "h2", "h3"]
    assert len(df) == 2
