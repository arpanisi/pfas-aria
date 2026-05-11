"""Unified independent/dependent row + header row Excel parsing."""

from __future__ import annotations

import io

import pandas as pd

from src.ingestion.legacy_pfca_regime_masks import (
    assign_pfca_regime_masks_from_column_lists,
)
from src.ingestion.unified_experimental_sheet import (
    load_excel_bytes_with_layout,
    parse_unified_experimental_excel,
    select_regime_pfas_input_mg_l_columns,
)


def _minimal_unified_workbook_bytes() -> bytes:
    """11 columns: 8 independent (incl. 6 mg/L species) + 3 dependent."""
    n_in, n_out = 8, 3
    row_role = ["Independent"] * n_in + ["Dependent"] * n_out
    row_names = [
        "condition",
        "time_min",
        "spare mg/L",
        "pfoa mg/L",
        "c4 mg/L",
        "c5 mg/L",
        "pfbs mg/L",
        "pfba mg/L",
        "out_fl mg/L",
        "out_h2 mg/L",
        "pfbs product mg/L",
    ]
    assert len(row_names) == len(row_role)

    z = 0.0
    data_rows = []
    # r1: only PFOA
    r = [z] * len(row_names)
    r[3] = 1.0  # pfoa mg/L column
    data_rows.append(r)
    # r2: PFOA + PFBA
    r = [z] * len(row_names)
    r[3] = 1.0
    r[7] = 1.0  # pfba
    data_rows.append(r)
    # r3: all except PFBS nonzero
    r = [z] * len(row_names)
    for i in (3, 2, 4, 5, 7):
        r[i] = 1.0
    r[6] = 0.0  # pfbs
    data_rows.append(r)
    # r4: only PFBS
    r = [z] * len(row_names)
    r[6] = 1.0
    data_rows.append(r)
    # r5: mixed
    r = [z] * len(row_names)
    r[3] = 1.0
    r[2] = 1.0
    r[6] = 1.0
    data_rows.append(r)

    raw = pd.DataFrame([row_role, row_names, *data_rows])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        raw.to_excel(writer, index=False, header=False)
    return buf.getvalue()


def test_parse_unified_experimental_excel_detects_layout() -> None:
    content = _minimal_unified_workbook_bytes()
    parsed = parse_unified_experimental_excel(content)
    assert parsed is not None
    df, meta = parsed
    assert len(df) == 5
    assert len(meta.input_cols) == 8
    assert len(meta.output_cols) == 3
    assert "pfoa_mg/l" in meta.input_cols
    assert "out_fl_mg/l" in meta.output_cols


def test_unified_layout_regime_assignment_matches_notebook_masks() -> None:
    content = _minimal_unified_workbook_bytes()
    parsed = parse_unified_experimental_excel(content)
    assert parsed is not None
    df, meta = parsed
    res = assign_pfca_regime_masks_from_column_lists(
        df,
        meta.input_cols,
        meta.output_cols,
        replace_no_in_inputs=False,
    )
    assert res.regime_row_counts == {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}
    assert sum(res.regime_row_counts.values()) == len(df)
    assert res.regimes[1].index.tolist() == [0]
    assert res.regimes[2].index.tolist() == [1]
    assert res.regimes[3].index.tolist() == [2]
    assert res.regimes[4].index.tolist() == [3]
    assert res.regimes[5].index.tolist() == [4]


def test_load_excel_bytes_with_layout_falls_back_without_role_row(tmp_path) -> None:
    p = tmp_path / "plain.xlsx"
    pd.DataFrame({"a": [1], "b": [2]}).to_excel(p, index=False)
    content = p.read_bytes()
    df, hdr, meta = load_excel_bytes_with_layout(content)
    assert meta is None
    assert hdr is not None
    assert "a" in [str(c).lower() for c in df.columns]


def test_title_preamble_before_role_row() -> None:
    """Unified layout can start after a non-data title row."""
    n_in, n_out = 8, 3
    row_title = ["Dataset title"] * (n_in + n_out)
    row_role = ["Independent"] * n_in + ["Dependent"] * n_out
    row_names = [
        "condition",
        "time_min",
        "spare mg/L",
        "pfoa mg/L",
        "c4 mg/L",
        "c5 mg/L",
        "pfbs mg/L",
        "pfba mg/L",
        "out_fl mg/L",
        "out_h2 mg/L",
        "pfbs product mg/L",
    ]
    z = 0.0
    r1 = [z] * len(row_names)
    r1[3] = 1.0
    raw = pd.DataFrame([row_title, row_role, row_names, r1])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        raw.to_excel(writer, index=False, header=False)
    parsed = parse_unified_experimental_excel(buf.getvalue())
    assert parsed is not None
    df, meta = parsed
    assert len(df) == 1
    assert meta.role_row_index == 1
    assert meta.name_row_index == 2


def test_select_regime_pfas_input_prefers_initial_when_enough_tagged() -> None:
    inputs = [
        "reagent mg/L",
        "pfoa_initial mg/L",
        "pfba_initial mg/L",
        "pfbs_initial mg/L",
        "spare_initial mg/L",
    ]
    w: list[str] = []
    all_mg, mask_cols = select_regime_pfas_input_mg_l_columns(inputs, warnings=w)
    assert len(all_mg) == 5
    assert len(mask_cols) == 4
    assert "reagent mg/L" in all_mg
    assert "reagent mg/L" not in mask_cols
    assert all("initial" in c.lower() for c in mask_cols)


def test_select_regime_pfas_input_falls_back_when_few_initial_tags() -> None:
    inputs = ["pfoa mg/L", "pfba_initial mg/L", "other mg/L", "pfbs mg/L"]
    w: list[str] = []
    all_mg, mask_cols = select_regime_pfas_input_mg_l_columns(inputs, warnings=w)
    assert all_mg == mask_cols
    assert len(mask_cols) == 4
