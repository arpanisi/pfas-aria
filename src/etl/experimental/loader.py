"""
Experimental Data ETL Loader.
Replaces the original DataLoader with a full ETL pipeline:
  validate → transform → version → persist → return DataBundle

Handles datasets from 100 rows to 100,000+ rows.
All processed data saved as parquet for fast re-loading.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pandas as pd

from src.etl.experimental.schema import ExperimentalDataValidator
from src.etl.experimental.transformer import ExperimentalTransformer
from src.ingestion.data_loader import ColumnProfile, DataBundle
from src.ingestion.unified_experimental_sheet import load_excel_bytes_with_layout
from src.utils.config import get_settings
from src.utils.exceptions import DataFileNotFoundError, IngestionError
from src.utils.logging import get_logger
from src.utils.paths import PROCESSED_DIR, PROJECT_ROOT

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls"}


class ExperimentalETL:
    """
    Full ETL pipeline for experimental PFAS data.
    Validates, transforms, versions, and returns a DataBundle.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.validator = ExperimentalDataValidator()
        self.transformer = ExperimentalTransformer()

    def run(
        self,
        file_path: Path | None = None,
        feature_columns: list[str] | None = None,
        exclude_columns: list[str] | None = None,
        strict_validation: bool = False,
    ) -> DataBundle:
        """
        Run the full ETL pipeline.

        Args:
            file_path: Override the config file path
            feature_columns: Override feature column selection
            exclude_columns: Columns to always exclude
            strict_validation: Apply domain-specific range checks

        Returns:
            DataBundle ready for the pipeline
        """
        # Step 1: Resolve and load raw file
        path = self._resolve_path(file_path)
        logger.info(f"ETL: Loading {path.name}")
        raw_df = self._read_file(path)
        content_hash = self._hash_dataframe(raw_df)

        logger.info(f"ETL: Raw shape {raw_df.shape}, hash={content_hash[:8]}")

        # Step 2: Check if already processed (parquet cache)
        parquet_path = PROCESSED_DIR / f"{content_hash[:16]}.parquet"
        if parquet_path.exists():
            logger.info("ETL: Loading from parquet cache")
            df = pd.read_parquet(parquet_path)
        else:
            # Step 3: Validate
            outcome = self.settings.data.outcome_variable
            features = feature_columns or self._infer_feature_columns(
                raw_df, outcome, exclude_columns or []
            )

            validation_report = self.validator.validate(
                raw_df, feature_columns=features, strict=strict_validation
            )

            if not validation_report.passed:
                raise IngestionError(
                    "Data validation failed:\n" + "\n".join(validation_report.errors)
                )

            if validation_report.warnings:
                for w in validation_report.warnings:
                    logger.warning(f"ETL validation: {w}")

            # Step 4: Transform
            df, transform_log = self.transformer.transform(
                df=raw_df,
                outcome_variable=outcome,
                feature_columns=features,
            )

            logger.info(
                f"ETL: Transform complete — "
                f"{transform_log.rows_before}→{transform_log.rows_after} rows"
            )

            # Step 5: Save parquet
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(parquet_path, index=False)
            logger.info(f"ETL: Saved parquet → {parquet_path.name}")

        # Step 6: Build DataBundle
        bundle = self._build_bundle(df, path, feature_columns, exclude_columns)
        logger.info(
            f"ETL: DataBundle ready — "
            f"{bundle.n_rows} rows, {len(bundle.feature_columns)} features"
        )
        return bundle

    # ── Private ───────────────────────────────────────────────────────────────

    def _resolve_path(self, override: Path | None) -> Path:
        if override:
            path = override
        else:
            p = self.settings.data.file_path
            path = Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p

        if not path.exists():
            raise DataFileNotFoundError(
                f"Data file not found: {path}\n"
                f"Set 'data.file_path' in configs/data_config.yaml"
            )
        if path.suffix not in SUPPORTED_EXTENSIONS:
            raise IngestionError(f"Unsupported file type: {path.suffix}")
        return path

    def _read_file(self, path: Path) -> pd.DataFrame:
        try:
            if path.suffix == ".csv":
                # Try to detect encoding and delimiter
                df = pd.read_csv(path, encoding="utf-8", low_memory=False)
            elif path.suffix == ".tsv":
                df = pd.read_csv(path, sep="\t", low_memory=False)
            elif path.suffix in {".xlsx", ".xls"}:
                df, _, _ = load_excel_bytes_with_layout(path.read_bytes())
            else:
                raise IngestionError(f"Unsupported: {path.suffix}")
        except Exception as e:
            raise IngestionError(f"Failed to read {path.name}: {e}") from e

        # Standardize column names immediately (idempotent if loader already normalized)
        df.columns = [
            str(c).strip().lower().replace(" ", "_").replace("-", "_")
            for c in df.columns
        ]
        return df

    def _hash_dataframe(self, df: pd.DataFrame) -> str:
        """Content hash for cache lookup."""
        buf = io.BytesIO()
        df.to_parquet(buf)
        return hashlib.sha256(buf.getvalue()).hexdigest()

    def _infer_feature_columns(
        self,
        df: pd.DataFrame,
        outcome: str,
        exclude: list[str],
    ) -> list[str]:
        """Infer usable feature columns from the dataframe."""
        exclude_set = set(exclude) | {outcome}

        # Detect entity and time columns to exclude
        for col in df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ["id", "entity", "experiment", "sample"]):
                exclude_set.add(col)
            if any(kw in col_lower for kw in ["time", "hour", "day", "date"]):
                exclude_set.add(col)

        return [c for c in df.columns if c not in exclude_set]

    def _build_bundle(
        self,
        df: pd.DataFrame,
        source_path: Path,
        feature_columns: list[str] | None,
        exclude_columns: list[str] | None,
    ) -> DataBundle:
        """Build a DataBundle from a transformed DataFrame."""
        outcome = self.settings.data.outcome_variable
        exclude = set(exclude_columns or [])

        non_unique_cols = [c for c in df.columns if int(df[c].nunique(dropna=True)) > 1]
        if non_unique_cols:
            _nu = df[non_unique_cols]
            cat_cols = [
                col
                for col in _nu.select_dtypes(include="object").columns
                if int(_nu[col].nunique(dropna=True)) <= 5
            ]
        else:
            cat_cols = []
        cat_col_set = set(cat_cols)
        datetime_cols = df.select_dtypes(include="datetime").columns.tolist()

        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        features = feature_columns or [
            c for c in numeric_cols + cat_cols if c not in exclude and c != outcome
        ]

        # Build column profiles
        profiles: dict[str, ColumnProfile] = {}
        for col in df.columns:
            series = df[col]
            is_numeric = pd.api.types.is_numeric_dtype(series)
            is_datetime = pd.api.types.is_datetime64_any_dtype(series)
            profiles[col] = ColumnProfile(
                name=col,
                dtype=str(series.dtype),
                n_unique=int(series.nunique()),
                n_missing=int(series.isnull().sum()),
                missing_pct=round(float(series.isnull().mean() * 100), 2),
                min=float(series.min()) if is_numeric else None,
                max=float(series.max()) if is_numeric else None,
                mean=round(float(series.mean()), 4) if is_numeric else None,
                std=round(float(series.std()), 4) if is_numeric else None,
                is_numeric=is_numeric,
                is_categorical=col in cat_col_set,
                is_datetime=is_datetime,
                sample_values=series.dropna().unique()[:5].tolist(),
            )

        # Detect entity and time columns
        entity_col = self.settings.data.entity_id_column
        time_col = self.settings.data.time_column

        if not entity_col:
            for col in df.columns:
                if any(kw in col.lower() for kw in ["id", "entity", "experiment"]):
                    entity_col = col
                    break

        if not time_col:
            for col in datetime_cols:
                time_col = col
                break
            if not time_col:
                for col in numeric_cols:
                    if any(kw in col.lower() for kw in ["time", "hour", "day"]):
                        if col != entity_col:
                            time_col = col
                            break

        schema_lines = [
            f"Dataset: {len(df)} rows × {len(df.columns)} columns",
            f"Outcome: {outcome}",
            f"Entity: {entity_col or 'not detected'}",
            f"Time: {time_col or 'not detected'}",
            f"Features ({len(features)}): {', '.join(features[:10])}",
        ]

        return DataBundle(
            df=df,
            outcome_variable=outcome,
            entity_id_column=entity_col,
            time_column=time_col,
            numeric_columns=numeric_cols,
            categorical_columns=cat_cols,
            datetime_columns=datetime_cols,
            feature_columns=features,
            column_profiles=profiles,
            n_rows=len(df),
            n_cols=len(df.columns),
            has_missing=df.isnull().any().any(),
            source_path=str(source_path),
            schema_summary="\n".join(schema_lines),
        )
