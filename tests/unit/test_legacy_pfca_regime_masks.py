"""Tests for legacy PFAS regime mask assignment."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.legacy_pfca_regime_masks import assign_legacy_pfca_regime_masks


def _minimal_unified_like_df() -> pd.DataFrame:
    """Build a narrow table: 2 skip + 8 inputs + 2 outputs; 6 PFAS inputs with mg/L."""
    skip = ["skip0", "skip1"]
    inputs = [
        "PFOA mg/L",
        "C3 mg/L",
        "C4 mg/L",
        "C5 mg/L",
        "PFBS mg/L",
        "PFBA mg/L",
        "C7 mg/L",
        "air",
    ]
    outs = ["Yield fluoride mg/L", "PFBS out mg/L"]
    cols = skip + inputs + outs

    n = 5
    z = 0.0
    data = {c: [z] * n for c in cols}
    df = pd.DataFrame(data)
    # Row 0: only PFOA nonzero -> r1
    df.loc[0, "PFOA mg/L"] = 1.0
    # Row 1: PFOA + PFBA nonzero, others zero
    df.loc[1, "PFOA mg/L"] = 1.0
    df.loc[1, "PFBA mg/L"] = 1.0
    # Row 2: all except PFBS nonzero (PFBS 0)
    for c in ["PFOA mg/L", "C3 mg/L", "C4 mg/L", "C5 mg/L", "PFBA mg/L", "C7 mg/L"]:
        df.loc[2, c] = 1.0
    df.loc[2, "PFBS mg/L"] = 0.0
    # Row 3: only PFBS nonzero
    df.loc[3, "PFBS mg/L"] = 1.0
    # Row 4: mixed leftover -> r5
    df.loc[4, "PFOA mg/L"] = 1.0
    df.loc[4, "C3 mg/L"] = 1.0
    df.loc[4, "PFBS mg/L"] = 1.0
    return df


def test_assign_legacy_regimes_r1_r4_and_r5() -> None:
    df = _minimal_unified_like_df()
    # columns: 0-1 skip, 2-9 input (8 cols), 10-11 output
    res = assign_legacy_pfca_regime_masks(
        df, input_start=2, input_end=10, output_start=10, replace_no_in_inputs=False
    )
    assert 1 in res.regimes and len(res.regimes[1]) == 1
    assert res.regimes[1].index.tolist() == [0]
    assert 2 in res.regimes and res.regimes[2].index.tolist() == [1]
    assert 3 in res.regimes and res.regimes[3].index.tolist() == [2]
    assert 4 in res.regimes and res.regimes[4].index.tolist() == [3]
    assert 5 in res.regimes and res.regimes[5].index.tolist() == [4]


def test_assign_legacy_requires_mg_l_input_columns() -> None:
    df = pd.DataFrame({"a": [1], "PFOA mg/L": [1]})
    with pytest.raises(ValueError, match="at least 3 mg/L"):
        assign_legacy_pfca_regime_masks(df, input_start=0, input_end=2, output_start=2)
