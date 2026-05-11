"""
Data Loader.
Loads the experimental PFAS dataset, infers schema, casts types,
and returns a validated DataBundle ready for downstream agents.
No assumptions about column names — everything is inferred or config-driven.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import get_settings
from src.utils.exceptions import DataFileNotFoundError, IngestionError
from src.utils.logging import get_logger
from src.utils.paths import PROJECT_ROOT

logger = get_logger(__name__)


# ── Output schema ─────────────────────────────────────────────────────────────


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    n_unique: int
    n_missing: int
    missing_pct: float
    min: Any | None
    max: Any | None
    mean: float | None
    std: float | None
    is_numeric: bool
    is_categorical: bool
    is_datetime: bool
    sample_values: list[Any] = field(default_factory=list)


@dataclass
class DataBundle:
    """Everything downstream agents need to know about the dataset."""

    df: pd.DataFrame
    outcome_variable: str
    entity_id_column: str | None
    time_column: str | None
    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    feature_columns: list[str]  # All usable predictors
    column_profiles: dict[str, ColumnProfile]
    n_rows: int
    n_cols: int
    has_missing: bool
    source_path: str
    schema_summary: str  # Human-readable for LLM context


# ── Loader ────────────────────────────────────────────────────────────────────


class DataLoader:
    """
    Loads and profiles the experimental dataset.
    Supports CSV, Excel (.xlsx, .xls), and TSV.
    """

    SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls"}

    def __init__(self) -> None:
        self.settings = get_settings()
        self.cfg = self.settings.data

    def load(self) -> DataBundle:
        path = self._resolve_path()
        logger.info(f"Loading data from: {path}")

        df = self._read_file(path)
        logger.info(f"Raw shape: {df.shape}")

        df = self._clean(df)
        df = self._cast_types(df)

        outcome = self._resolve_outcome(df)
        entity_col = self._detect_entity_column(df)
        time_col = self._detect_time_column(df, entity_col)

        profiles = self._profile_columns(df)

        numeric_cols = [c for c, p in profiles.items() if p.is_numeric]
        categorical_cols = [c for c, p in profiles.items() if p.is_categorical]
        datetime_cols = [c for c, p in profiles.items() if p.is_datetime]

        exclude = set(self.cfg.exclude_columns or [])
        if entity_col:
            exclude.add(entity_col)
        if time_col:
            exclude.add(time_col)
        exclude.add(outcome)

        feature_columns = [
            c for c in numeric_cols + categorical_cols if c not in exclude
        ]

        bundle = DataBundle(
            df=df,
            outcome_variable=outcome,
            entity_id_column=entity_col,
            time_column=time_col,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            datetime_columns=datetime_cols,
            feature_columns=feature_columns,
            column_profiles=profiles,
            n_rows=len(df),
            n_cols=len(df.columns),
            has_missing=df.isnull().any().any(),
            source_path=str(path),
            schema_summary=self._build_schema_summary(
                df, profiles, outcome, entity_col, time_col, feature_columns
            ),
        )

        logger.info(
            f"DataBundle ready — rows: {bundle.n_rows}, "
            f"features: {len(feature_columns)}, "
            f"outcome: {outcome}"
        )
        return bundle

    # ── Private helpers ───────────────────────────────────────────────────────

    def _resolve_path(self) -> Path:
        p = Path(self.cfg.file_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            raise DataFileNotFoundError(
                f"Data file not found: {p}\n"
                f"Update 'data.file_path' in configs/data_config.yaml"
            )
        if p.suffix not in self.SUPPORTED_EXTENSIONS:
            raise IngestionError(
                f"Unsupported file type: {p.suffix}. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )
        return p

    def _read_file(self, path: Path) -> pd.DataFrame:
        try:
            if path.suffix == ".csv":
                return pd.read_csv(path)
            elif path.suffix == ".tsv":
                return pd.read_csv(path, sep="\t")
            elif path.suffix in {".xlsx", ".xls"}:
                return pd.read_excel(path, skiprows=1)
        except Exception as e:
            raise IngestionError(f"Failed to read {path}: {e}") from e

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # Strip whitespace from column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        # Drop fully empty rows/cols
        df = df.dropna(how="all").dropna(axis=1, how="all")
        # Strip whitespace from string values
        str_cols = df.select_dtypes(include="object").columns
        df[str_cols] = df[str_cols].apply(
            lambda col: col.str.strip() if col.dtype == "object" else col
        )
        return df.reset_index(drop=True)

    def _cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            # Try datetime
            if df[col].dtype == object:
                try:
                    df[col] = pd.to_datetime(df[col])
                    continue
                except (ValueError, TypeError):
                    pass
            # Try numeric
            if df[col].dtype == object:
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass
        return df

    def _resolve_outcome(self, df: pd.DataFrame) -> str:
        outcome = self.cfg.outcome_variable
        if outcome not in df.columns:
            raise IngestionError(
                f"Outcome variable '{outcome}' not found in dataset.\n"
                f"Available columns: {list(df.columns)}\n"
                f"Update 'data.outcome_variable' in configs/data_config.yaml"
            )
        return outcome

    def _detect_entity_column(self, df: pd.DataFrame) -> str | None:
        # Use config if provided
        if self.cfg.entity_id_column:
            if self.cfg.entity_id_column in df.columns:
                return self.cfg.entity_id_column

        # Auto-detect: column with "id" in name and high cardinality
        candidates = [
            c
            for c in df.columns
            if any(
                kw in c.lower()
                for kw in ["id", "entity", "sample", "unit", "experiment"]
            )
        ]
        for c in candidates:
            if df[c].nunique() > 1:
                logger.info(f"Auto-detected entity column: '{c}'")
                return str(c)
        return None

    def _detect_time_column(
        self, df: pd.DataFrame, entity_col: str | None
    ) -> str | None:
        if self.cfg.time_column:
            if self.cfg.time_column in df.columns:
                return self.cfg.time_column

        # Datetime columns first
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        if datetime_cols:
            logger.info(f"Auto-detected time column: '{datetime_cols[0]}'")
            return str(datetime_cols[0])

        # Numeric columns with time-like names (exclude entity column)
        candidates = [
            c
            for c in df.select_dtypes(include="number").columns
            if any(
                kw in c.lower() for kw in ["time", "hour", "day", "min", "second", "t_"]
            )
            and c != entity_col
        ]
        if candidates:
            logger.info(f"Auto-detected time column: '{candidates[0]}'")
            return str(candidates[0])

        return None

    def _profile_columns(self, df: pd.DataFrame) -> dict[str, ColumnProfile]:
        profiles = {}
        for col in df.columns:
            series = df[col]
            is_numeric = pd.api.types.is_numeric_dtype(series)
            is_datetime = pd.api.types.is_datetime64_any_dtype(series)
            is_categorical = (
                not is_numeric and not is_datetime
            ) or series.nunique() < 15

            profiles[col] = ColumnProfile(
                name=col,
                dtype=str(series.dtype),
                n_unique=series.nunique(),
                n_missing=series.isnull().sum(),
                missing_pct=round(series.isnull().mean() * 100, 2),
                min=series.min() if is_numeric else None,
                max=series.max() if is_numeric else None,
                mean=round(float(series.mean()), 4) if is_numeric else None,
                std=round(float(series.std()), 4) if is_numeric else None,
                is_numeric=is_numeric,
                is_categorical=is_categorical,
                is_datetime=is_datetime,
                sample_values=series.dropna().unique()[:5].tolist(),
            )
        return profiles

    def _build_schema_summary(
        self,
        df: pd.DataFrame,
        profiles: dict[str, ColumnProfile],
        outcome: str,
        entity_col: str | None,
        time_col: str | None,
        feature_cols: list[str],
    ) -> str:
        lines = [
            f"Dataset: {len(df)} rows x {len(df.columns)} columns",
            f"Outcome variable: {outcome}",
            f"Entity column: {entity_col or 'not detected'}",
            f"Time column: {time_col or 'not detected'}",
            f"Feature columns ({len(feature_cols)}): {', '.join(feature_cols)}",
            "",
            "Column profiles:",
        ]
        for col, p in profiles.items():
            tag = []
            if col == outcome:
                tag.append("OUTCOME")
            if col == entity_col:
                tag.append("ENTITY")
            if col == time_col:
                tag.append("TIME")
            tag_str = f" [{', '.join(tag)}]" if tag else ""

            if p.is_numeric:
                lines.append(
                    f"  {col}{tag_str}: numeric, "
                    f"range=[{p.min}, {p.max}], mean={p.mean}, std={p.std}, "
                    f"missing={p.missing_pct}%"
                )
            else:
                lines.append(
                    f"  {col}{tag_str}: categorical, "
                    f"unique={p.n_unique}, "
                    f"values={p.sample_values[:3]}, "
                    f"missing={p.missing_pct}%"
                )
        return "\n".join(lines)
