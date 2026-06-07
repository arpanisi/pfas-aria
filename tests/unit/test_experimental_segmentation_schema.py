"""Regression tests for experimental segmentation SQL schema registration."""

from __future__ import annotations


def test_experiment_segmentation_tables_registered() -> None:
    import src.db.orm  # noqa: F401 — register models on Base.metadata
    from src.db.postgres import Base

    names = set(Base.metadata.tables)
    assert "experiment_segmentation_batches" in names
    assert "experiment_segmented_regimes" in names
    assert "experiment_regime_rows" in names
    assert "experiment_regime_column_stats" in names
    assert "experiment_regime_regression_specs" in names


def test_coalesce_dict_key_variants() -> None:
    from src.db.experimental_segmentation_persist import _coalesce_key

    d = {1: "a", "2": "b"}
    assert _coalesce_key(d, 1) == "a"
    assert _coalesce_key(d, "2") == "b"
    assert _coalesce_key(d, 2) == "b"
    assert _coalesce_key(d, 99) is None
