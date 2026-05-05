"""Unit tests for the Data Intelligence Agent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.agents.data_intelligence import DataIntelligenceAgent, DataIntelligenceReport
from src.ingestion.data_loader import ColumnProfile, DataBundle  # noqa: E402

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_profile(name: str, is_numeric: bool = True) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        dtype="float64" if is_numeric else "object",
        n_unique=10,
        n_missing=0,
        missing_pct=0.0,
        min=0.0 if is_numeric else None,
        max=1.0 if is_numeric else None,
        mean=0.5 if is_numeric else None,
        std=0.2 if is_numeric else None,
        is_numeric=is_numeric,
        is_categorical=not is_numeric,
        is_datetime=False,
        sample_values=[0.1, 0.5, 0.9] if is_numeric else ["A", "B"],
    )


@pytest.fixture
def sample_bundle() -> DataBundle:
    np.random.seed(42)
    n = 60

    # Two clear groups
    group1 = pd.DataFrame({
        "experiment_id": range(1, 31),
        "time_hours": np.linspace(0, 5, 30),
        "degradation_rate": np.random.normal(0.3, 0.05, 30),
        "uv_intensity": np.random.normal(10, 1, 30),
        "ph": np.random.normal(7.0, 0.2, 30),
        "temperature_c": np.random.normal(20, 1, 30),
        "catalyst": ["TiO2"] * 30,
    })
    group2 = pd.DataFrame({
        "experiment_id": range(31, 61),
        "time_hours": np.linspace(5, 10, 30),
        "degradation_rate": np.random.normal(0.7, 0.05, 30),
        "uv_intensity": np.random.normal(30, 1, 30),
        "ph": np.random.normal(8.0, 0.2, 30),
        "temperature_c": np.random.normal(30, 1, 30),
        "catalyst": ["ZnO"] * 30,
    })
    df = pd.concat([group1, group2], ignore_index=True)

    profiles = {col: _make_profile(col, col != "catalyst") for col in df.columns}

    return DataBundle(
        df=df,
        outcome_variable="degradation_rate",
        entity_id_column="experiment_id",
        time_column="time_hours",
        numeric_columns=["degradation_rate", "uv_intensity", "ph", "temperature_c", "time_hours"],
        categorical_columns=["catalyst"],
        datetime_columns=[],
        feature_columns=["uv_intensity", "ph", "temperature_c", "catalyst"],
        column_profiles=profiles,
        n_rows=n,
        n_cols=len(df.columns),
        has_missing=False,
        source_path="data/raw/test.csv",
        schema_summary="Test dataset with 60 rows",
    )


@pytest.fixture
def agent() -> DataIntelligenceAgent:
    mock_settings = MagicMock()
    mock_settings.llm.provider = "ollama"
    mock_settings.llm.model = "llama3.1:8b"
    mock_settings.llm.temperature = 0.1
    mock_settings.llm.max_tokens = 4096
    mock_settings.llm.request_timeout = 120
    mock_settings.domain.context = "PFAS degradation"
    mock_settings.agent_config_min_regime_size = 5

    with patch("src.agents.base.get_settings", return_value=mock_settings), \
         patch("src.agents.data_intelligence.get_settings", return_value=mock_settings):
        return DataIntelligenceAgent()


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestDataIntelligenceAgent:

    def test_kmeans_detection_returns_labels(self, agent, sample_bundle):
        df = sample_bundle.df
        numeric_cols = ["uv_intensity", "ph", "temperature_c"]
        labels = agent._kmeans_detection(df, sample_bundle, numeric_cols)
        assert len(labels) == len(df)
        assert all(lbl.startswith("regime_") for lbl in labels)

    def test_kmeans_detects_two_clusters(self, agent, sample_bundle):
        """With clearly separated data, k-means should find 2 clusters."""
        df = sample_bundle.df
        numeric_cols = ["uv_intensity", "temperature_c"]
        labels = agent._kmeans_detection(df, sample_bundle, numeric_cols)
        assert len(set(labels)) >= 2

    def test_hdbscan_detection_returns_labels(self, agent, sample_bundle):
        df = sample_bundle.df
        numeric_cols = ["uv_intensity", "ph", "temperature_c"]
        labels = agent._hdbscan_detection(df, sample_bundle, numeric_cols)
        assert len(labels) == len(df)

    def test_build_regimes_correct_count(self, agent, sample_bundle):
        df = sample_bundle.df.copy()
        df["regime"] = ["regime_0"] * 30 + ["regime_1"] * 30
        regimes = agent._build_regimes(df, sample_bundle, ["regime_0", "regime_1"])
        assert len(regimes) == 2
        assert regimes[0].size == 30
        assert regimes[1].size == 30

    def test_panel_structure_detected(self, agent, sample_bundle):
        panel = agent._discover_panel_structure(sample_bundle)
        assert panel.is_panel is True
        assert panel.entity_column == "experiment_id"
        assert panel.time_column == "time_hours"
        assert panel.recommended_model in ("fixed_effects", "random_effects", "ols")

    def test_no_panel_without_columns(self, agent, sample_bundle):
        sample_bundle.entity_id_column = None
        sample_bundle.time_column = None
        panel = agent._discover_panel_structure(sample_bundle)
        assert panel.is_panel is False
        assert panel.recommended_model == "ols"

    def test_feature_profiling_numeric(self, agent, sample_bundle):
        labeled_df = sample_bundle.df.copy()
        labeled_df["regime"] = "regime_0"
        profiles = agent._profile_features(sample_bundle, labeled_df)
        numeric_profiles = [p for p in profiles if p.is_numeric]
        assert len(numeric_profiles) > 0
        for p in numeric_profiles:
            assert p.distribution in ("normal", "skewed", "heavy_tailed", "non_normal")
            assert p.skewness is not None

    def test_feature_profiles_have_correlations(self, agent, sample_bundle):
        labeled_df = sample_bundle.df.copy()
        labeled_df["regime"] = "regime_0"
        profiles = agent._profile_features(sample_bundle, labeled_df)
        numeric_profiles = [p for p in profiles if p.is_numeric]
        corrs = [p.correlation_with_outcome for p in numeric_profiles]
        assert any(c is not None for c in corrs)

    def test_rank_features_returns_sorted_list(self, agent, sample_bundle):
        labeled_df = sample_bundle.df.copy()
        labeled_df["regime"] = "regime_0"
        profiles = agent._profile_features(sample_bundle, labeled_df)
        ranked = agent._rank_features(profiles)
        assert isinstance(ranked, list)
        assert len(ranked) > 0

    def test_classify_distribution_normal(self, agent):
        series = pd.Series(np.random.normal(0, 1, 500))
        result = agent._classify_distribution(series, 0.1, 0.2)
        assert result in ("normal", "non_normal", "heavy_tailed", "skewed")

    def test_count_outliers_detects_extremes(self, agent):
        series = pd.Series([1.0] * 98 + [100.0, -100.0])
        count = agent._count_outliers(series)
        assert count >= 2

    def test_run_with_mocked_llm(self, agent, sample_bundle):
        """Full run with LLM mocked out."""
        mock_response = """{
            "regime_descriptions": {
                "regime_0": "Low UV intensity regime",
                "regime_1": "High UV intensity regime"
            },
            "scientific_interpretation": "Two distinct photochemical regimes detected.",
            "key_features": ["uv_intensity", "temperature_c"],
            "warnings": []
        }"""

        with patch.object(agent, "call_llm", return_value=mock_response):
            report = agent.run(sample_bundle)

        assert isinstance(report, DataIntelligenceReport)
        assert report.n_regimes >= 1
        assert report.panel_structure is not None
        assert len(report.feature_profiles) > 0
        assert report.labeled_df is not None
        assert "regime" in report.labeled_df.columns
