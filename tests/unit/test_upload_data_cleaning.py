"""Tests for upload cleaning (LabelEncoder on object / string categorical columns)."""

from __future__ import annotations

import pandas as pd

from src.ingestion.upload_data_cleaning import apply_upload_data_cleaning


def test_numeric_columns_not_label_encoded():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
    _notes, enc = apply_upload_data_cleaning(df, unified_meta=None)
    assert df["a"].tolist() == [1.0, 2.0, 3.0]
    assert df["b"].dtype == float
    assert "a" not in enc
    assert "classes" in enc["b"] and len(enc["b"]["classes"]) == 3


def test_constant_object_column_not_encoded():
    df = pd.DataFrame({"x": ["-", "-", "-"], "y": [1, 2, 3]})
    _notes, enc = apply_upload_data_cleaning(df, unified_meta=None)
    assert df["x"].tolist() == ["-", "-", "-"]
    assert "x" not in enc


def test_non_constant_object_column_encoded():
    df = pd.DataFrame({"lab": ["low", "high", "low", "high"]})
    apply_upload_data_cleaning(df, unified_meta=None)
    assert df["lab"].dtype == float
    assert df["lab"].iloc[0] == df["lab"].iloc[2]
    assert df["lab"].iloc[1] == df["lab"].iloc[3]


def test_encoding_metadata_only_for_cat_cols():
    df = pd.DataFrame({"lab": ["a", "b", "a"], "n": [1.0, 2.0, 3.0]})
    _notes, enc = apply_upload_data_cleaning(df, unified_meta=None)
    assert enc["lab"]["kind"] == "label_encode"
    assert set(enc["lab"]["classes"]) == {"a", "b"}
    assert "n" not in enc


def test_nullable_string_dtype_column_encoded():
    df = pd.DataFrame({"s": pd.array(["x", "y", "x"], dtype="string"), "n": [1, 2, 3]})
    _notes, enc = apply_upload_data_cleaning(df, unified_meta=None)
    assert df["s"].dtype == float
    assert "s" in enc
    assert df["n"].tolist() == [1, 2, 3]
