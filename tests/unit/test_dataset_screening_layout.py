"""Full-dataset screening layout (no species-specific segmentation)."""

from __future__ import annotations

import pandas as pd

from src.ingestion.dataset_screening_layout import (
    assign_screening_layout_from_column_lists,
    assign_screening_layout_from_legacy_column_slices,
)


def test_assign_from_column_lists_one_segment() -> None:
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [0.1, 0.2, 0.3, 0.4, 0.5],
            "y": [1.0, 1.1, 1.2, 1.3, 1.4],
        }
    )
    res = assign_screening_layout_from_column_lists(
        df, ["a", "b"], ["y"], replace_no_in_inputs=False
    )
    assert res.regimes.keys() == {1}
    assert len(res.regimes[1]) == 5
    assert res.regime_row_counts == {1: 5}


def test_assign_from_legacy_slices_uses_column_range() -> None:
    df = pd.DataFrame({f"c{i}": [i] for i in range(40)})
    res = assign_screening_layout_from_legacy_column_slices(
        df, input_start=2, input_end=5, output_start=5
    )
    assert res.input_cols == ["c2", "c3", "c4"]
    assert res.output_cols[0] == "c5"


def test_assign_from_legacy_slices_rejects_short_frame() -> None:
    df = pd.DataFrame({"x": [1]})
    try:
        assign_screening_layout_from_legacy_column_slices(
            df, input_start=2, input_end=10, output_start=10
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError")
