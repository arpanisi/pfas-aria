"""Unit tests for DataLoader."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ingestion.data_loader import DataLoader, DataBundle
from src.utils.exceptions import DataFileNotFoundError, IngestionError


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "experiment_id": [1, 2, 3, 4, 5],
        "time_hours": [0.5, 1.0, 2.0, 4.0, 8.0],
        "degradation_rate": [0.12, 0.25, 0.48, 0.71, 0.89],
        "uv_intensity": [10.0, 10.0, 20.0, 20.0, 30.0],
        "ph": [7.0, 7.5, 7.0, 7.5, 8.0],
        "temperature_c": [20.0, 20.0, 25.0, 25.0, 30.0],
        "catalyst": ["TiO2", "TiO2", "ZnO", "ZnO", "TiO2"],
    })


@pytest.fixture
def loader_with_mock(tmp_path, sample_df):
    """DataLoader with mocked config pointing to a temp CSV."""
    csv_path = tmp_path / "pfas_data.csv"
    sample_df.to_csv(csv_path, index=False)

    mock_settings = MagicMock()
    mock_settings.data.file_path = str(csv_path)
    mock_settings.data.outcome_variable = "degradation_rate"
    mock_settings.data.entity_id_column = None
    mock_settings.data.time_column = None
    mock_settings.data.exclude_columns = []

    with patch("src.ingestion.data_loader.get_settings", return_value=mock_settings):
        yield DataLoader()


class TestDataLoader:

    def test_load_returns_bundle(self, loader_with_mock):
        bundle = loader_with_mock.load()
        assert isinstance(bundle, DataBundle)

    def test_correct_row_count(self, loader_with_mock):
        bundle = loader_with_mock.load()
        assert bundle.n_rows == 5

    def test_outcome_variable_identified(self, loader_with_mock):
        bundle = loader_with_mock.load()
        assert bundle.outcome_variable == "degradation_rate"

    def test_outcome_not_in_features(self, loader_with_mock):
        bundle = loader_with_mock.load()
        assert "degradation_rate" not in bundle.feature_columns

    def test_numeric_columns_detected(self, loader_with_mock):
        bundle = loader_with_mock.load()
        assert "uv_intensity" in bundle.numeric_columns
        assert "ph" in bundle.numeric_columns

    def test_categorical_columns_detected(self, loader_with_mock):
        bundle = loader_with_mock.load()
        assert "catalyst" in bundle.categorical_columns

    def test_entity_column_auto_detected(self, loader_with_mock):
        bundle = loader_with_mock.load()
        assert bundle.entity_id_column == "experiment_id"

    def test_time_column_auto_detected(self, loader_with_mock):
        bundle = loader_with_mock.load()
        assert bundle.time_column == "time_hours"

    def test_schema_summary_not_empty(self, loader_with_mock):
        bundle = loader_with_mock.load()
        assert len(bundle.schema_summary) > 100

    def test_column_profiles_complete(self, loader_with_mock):
        bundle = loader_with_mock.load()
        for col in ["uv_intensity", "ph", "temperature_c"]:
            assert col in bundle.column_profiles
            profile = bundle.column_profiles[col]
            assert profile.is_numeric
            assert profile.mean is not None

    def test_file_not_found_raises(self, tmp_path):
        mock_settings = MagicMock()
        mock_settings.data.file_path = str(tmp_path / "nonexistent.csv")
        mock_settings.data.outcome_variable = "degradation_rate"
        mock_settings.data.entity_id_column = None
        mock_settings.data.time_column = None
        mock_settings.data.exclude_columns = []

        with patch("src.ingestion.data_loader.get_settings", return_value=mock_settings):
            loader = DataLoader()
            with pytest.raises(DataFileNotFoundError):
                loader.load()

    def test_missing_outcome_raises(self, tmp_path, sample_df):
        csv_path = tmp_path / "pfas_data.csv"
        sample_df.to_csv(csv_path, index=False)

        mock_settings = MagicMock()
        mock_settings.data.file_path = str(csv_path)
        mock_settings.data.outcome_variable = "nonexistent_col"
        mock_settings.data.entity_id_column = None
        mock_settings.data.time_column = None
        mock_settings.data.exclude_columns = []

        with patch("src.ingestion.data_loader.get_settings", return_value=mock_settings):
            loader = DataLoader()
            with pytest.raises(IngestionError):
                loader.load()
