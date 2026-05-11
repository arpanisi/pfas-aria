"""
ORM Models — Full Schema.
All PostgreSQL tables for PFAS-ARIA.

Tables:
  runs              — pipeline run metadata
  data_versions     — experimental dataset versions
  experiment_segmentation_batches — structured regime segmentation per file/batch
  experiment_segmented_regimes    — regimes within a batch
  experiment_regime_rows          — row index membership per regime
  experiment_regime_column_stats  — per-regime column summaries
  experiment_regime_regression_specs — regression column lists per regime
  papers            — corpus paper registry
  regimes           — detected data regimes per run
  hypotheses        — generated hypotheses per round
  model_results     — model fit results per hypothesis
  validation_results— statistical validation per model
  citations         — literature citations per model result
  arxiv_cache_meta  — metadata about cached arXiv papers (content in Redis)
  s2_cache_meta     — metadata about cached S2 papers (content in Redis)
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.postgres import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.utcnow()


# ── Runs ──────────────────────────────────────────────────────────────────────


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = (
        Index("ix_runs_status", "status"),
        Index("ix_runs_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    run_name: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(50), default="initializing")
    outcome_variable: Mapped[str] = mapped_column(String(255), nullable=True)
    selected_features: Mapped[list] = mapped_column(JSON, default=list)
    excluded_features: Mapped[list] = mapped_column(JSON, default=list)
    data_version_id: Mapped[str] = mapped_column(
        ForeignKey("data_versions.id"), nullable=True
    )
    n_rounds_completed: Mapped[int] = mapped_column(Integer, default=0)
    final_match_score: Mapped[float] = mapped_column(Float, default=0.0)
    converged: Mapped[bool] = mapped_column(Boolean, default=False)
    stop_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    config_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)

    data_version: Mapped[DataVersion | None] = relationship(back_populates="runs")
    regimes: Mapped[list[Regime]] = relationship(back_populates="run")
    hypotheses: Mapped[list[Hypothesis]] = relationship(back_populates="run")
    segmentation_batches: Mapped[list[ExperimentSegmentationBatch]] = (
        relationship(back_populates="run")
    )


# ── Data versions ─────────────────────────────────────────────────────────────


class DataVersion(Base):
    __tablename__ = "data_versions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String(255))
    content_hash: Mapped[str] = mapped_column(String(64), unique=True)
    n_rows: Mapped[int] = mapped_column(Integer)
    n_cols: Mapped[int] = mapped_column(Integer)
    column_names: Mapped[list] = mapped_column(JSON, default=list)
    schema_version: Mapped[str] = mapped_column(String(50), nullable=True)
    processed_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    dvc_hash: Mapped[str] = mapped_column(String(64), nullable=True)
    validation_passed: Mapped[bool] = mapped_column(Boolean, default=False)
    validation_errors: Mapped[list] = mapped_column(JSON, default=list)
    parquet_path: Mapped[str] = mapped_column(String(512), nullable=True)

    runs: Mapped[list[Run]] = relationship(back_populates="data_version")
    segmentation_batches: Mapped[list[ExperimentSegmentationBatch]] = (
        relationship(back_populates="data_version")
    )


# ── Corpus papers ─────────────────────────────────────────────────────────────


class Paper(Base):
    __tablename__ = "papers"
    __table_args__ = (UniqueConstraint("content_hash"),)

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String(255))
    content_hash: Mapped[str] = mapped_column(String(64))
    title: Mapped[str] = mapped_column(Text, nullable=True)
    authors: Mapped[list] = mapped_column(JSON, default=list)
    parsed_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    n_chunks: Mapped[int] = mapped_column(Integer, default=0)
    n_tokens: Mapped[int] = mapped_column(Integer, default=0)
    embedding_model: Mapped[str] = mapped_column(String(100), nullable=True)
    indexed_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    chunk_size: Mapped[int] = mapped_column(Integer, nullable=True)
    source: Mapped[str] = mapped_column(String(50), default="corpus")
    # MongoDB object IDs for chunk lookup
    mongo_chunk_ids: Mapped[list] = mapped_column(JSON, default=list)


# ── Regimes ───────────────────────────────────────────────────────────────────


class Regime(Base):
    __tablename__ = "regimes"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"))
    label: Mapped[str] = mapped_column(String(100))
    method: Mapped[str] = mapped_column(String(100))
    size: Mapped[int] = mapped_column(Integer)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    summary_stats: Mapped[dict] = mapped_column(JSON, default=dict)

    run: Mapped[Run] = relationship(back_populates="regimes")


# ── Experimental segmentation (structured regimes, SQL-friendly) ─────────────


class ExperimentSegmentationBatch(Base):
    """
    One segmentation pass over a dataset file (e.g. one Excel workbook).
    Normalizes the old ``results_iter[iter]`` dict into queryable rows.
    """

    __tablename__ = "experiment_segmentation_batches"
    __table_args__ = (
        Index("ix_exp_seg_batch_data_version", "data_version_id"),
        Index("ix_exp_seg_batch_run", "run_id"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    data_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("data_versions.id"), nullable=True
    )
    run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id"), nullable=True)
    batch_index: Mapped[int] = mapped_column(Integer, default=0)
    source_filename: Mapped[str] = mapped_column(String(512))
    segmentation_method: Mapped[str] = mapped_column(String(100), default="legacy_masks")
    panel_entity_col: Mapped[str | None] = mapped_column(String(255), nullable=True)
    panel_time_col: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # Dataset-level column lists (same as legacy result_iter top-level lists)
    dataset_columns: Mapped[dict] = mapped_column(JSON, default=dict)

    data_version: Mapped[DataVersion | None] = relationship(
        back_populates="segmentation_batches"
    )
    run: Mapped[Run | None] = relationship(back_populates="segmentation_batches")
    segmented_regimes: Mapped[list[ExperimentSegmentedRegime]] = relationship(
        back_populates="batch", cascade="all, delete-orphan"
    )


class ExperimentSegmentedRegime(Base):
    """One scientific regime within a batch (e.g. r1..r5 in the legacy notebook)."""

    __tablename__ = "experiment_segmented_regimes"
    __table_args__ = (
        UniqueConstraint("batch_id", "regime_code", name="uq_exp_seg_regime_batch_code"),
        Index("ix_exp_seg_regime_batch", "batch_id"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    batch_id: Mapped[str] = mapped_column(
        ForeignKey("experiment_segmentation_batches.id", ondelete="CASCADE")
    )
    regime_code: Mapped[str] = mapped_column(String(50))
    n_rows: Mapped[int] = mapped_column(Integer, default=0)
    conditions_json: Mapped[list] = mapped_column(JSON, default=list)
    extra_metadata: Mapped[dict] = mapped_column(JSON, default=dict)

    batch: Mapped[ExperimentSegmentationBatch] = relationship(
        back_populates="segmented_regimes"
    )
    row_memberships: Mapped[list[ExperimentRegimeRow]] = relationship(
        back_populates="segmented_regime", cascade="all, delete-orphan"
    )
    column_stats: Mapped[list[ExperimentRegimeColumnStat]] = relationship(
        back_populates="segmented_regime", cascade="all, delete-orphan"
    )
    regression_spec: Mapped[ExperimentRegimeRegressionSpec | None] = relationship(
        back_populates="segmented_regime",
        uselist=False,
        cascade="all, delete-orphan",
    )


class ExperimentRegimeRow(Base):
    """Maps a regime to source dataframe row indices (provenance)."""

    __tablename__ = "experiment_regime_rows"
    __table_args__ = (
        Index("ix_exp_regime_rows_regime", "segmented_regime_id"),
        Index("ix_exp_regime_rows_source_idx", "segmented_regime_id", "source_row_index"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    segmented_regime_id: Mapped[str] = mapped_column(
        ForeignKey("experiment_segmented_regimes.id", ondelete="CASCADE")
    )
    source_row_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_row_label: Mapped[str | None] = mapped_column(String(64), nullable=True)

    segmented_regime: Mapped[ExperimentSegmentedRegime] = relationship(
        back_populates="row_memberships"
    )


class ExperimentRegimeColumnStat(Base):
    """Per-regime, per-column summary (constants vs varying inputs)."""

    __tablename__ = "experiment_regime_column_stats"
    __table_args__ = (
        UniqueConstraint(
            "segmented_regime_id",
            "column_name",
            name="uq_exp_regime_col_stat",
        ),
        Index("ix_exp_regime_col_regime", "segmented_regime_id"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    segmented_regime_id: Mapped[str] = mapped_column(
        ForeignKey("experiment_segmented_regimes.id", ondelete="CASCADE")
    )
    column_name: Mapped[str] = mapped_column(String(512))
    column_role: Mapped[str] = mapped_column(String(50), default="input")
    is_constant_in_regime: Mapped[bool] = mapped_column(Boolean, default=False)
    n_distinct: Mapped[int] = mapped_column(Integer, default=0)
    dtype: Mapped[str] = mapped_column(String(80), default="unknown")
    value_counts: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    segmented_regime: Mapped[ExperimentSegmentedRegime] = relationship(
        back_populates="column_stats"
    )


class ExperimentRegimeRegressionSpec(Base):
    """Regression-facing column grouping for one regime (legacy regression_cols)."""

    __tablename__ = "experiment_regime_regression_specs"

    segmented_regime_id: Mapped[str] = mapped_column(
        ForeignKey("experiment_segmented_regimes.id", ondelete="CASCADE"),
        primary_key=True,
    )
    candidate_inputs: Mapped[list] = mapped_column(JSON, default=list)
    categorical_inputs: Mapped[list] = mapped_column(JSON, default=list)
    numeric_inputs: Mapped[list] = mapped_column(JSON, default=list)
    outputs: Mapped[list] = mapped_column(JSON, default=list)

    segmented_regime: Mapped[ExperimentSegmentedRegime] = relationship(
        back_populates="regression_spec"
    )


# ── Hypotheses ────────────────────────────────────────────────────────────────


class Hypothesis(Base):
    __tablename__ = "hypotheses"
    __table_args__ = (
        Index("ix_hypotheses_run_id", "run_id"),
        Index("ix_hypotheses_round", "round"),
        Index("ix_hypotheses_priority", "priority_score"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"))
    hypothesis_id: Mapped[str] = mapped_column(String(50))
    round: Mapped[int] = mapped_column(Integer)
    description: Mapped[str] = mapped_column(Text)
    rationale: Mapped[str] = mapped_column(Text, nullable=True)
    primary_variables: Mapped[list] = mapped_column(JSON, default=list)
    interaction_terms: Mapped[list] = mapped_column(JSON, default=list)
    control_variables: Mapped[list] = mapped_column(JSON, default=list)
    model_family: Mapped[str] = mapped_column(String(100))
    priority_score: Mapped[float] = mapped_column(Float, default=0.0)
    is_refinement: Mapped[bool] = mapped_column(Boolean, default=False)
    rag_support: Mapped[list] = mapped_column(JSON, default=list)

    run: Mapped[Run] = relationship(back_populates="hypotheses")
    model_results: Mapped[list[ModelResult]] = relationship(back_populates="hypothesis")


# ── Model results ─────────────────────────────────────────────────────────────


class ModelResult(Base):
    __tablename__ = "model_results"
    __table_args__ = (
        Index("ix_model_results_hypothesis_id", "hypothesis_id"),
        Index("ix_model_results_r_squared", "r_squared"),
        Index("ix_model_results_match_score", "match_score"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    hypothesis_id: Mapped[str] = mapped_column(ForeignKey("hypotheses.id"))
    model_type: Mapped[str] = mapped_column(String(100))
    regime_label: Mapped[str] = mapped_column(String(100), nullable=True)
    r_squared: Mapped[float] = mapped_column(Float, default=0.0)
    adj_r_squared: Mapped[float] = mapped_column(Float, default=0.0)
    aic: Mapped[float] = mapped_column(Float, nullable=True)
    bic: Mapped[float] = mapped_column(Float, nullable=True)
    n_observations: Mapped[int] = mapped_column(Integer, default=0)
    coefficients: Mapped[dict] = mapped_column(JSON, default=dict)
    p_values: Mapped[dict] = mapped_column(JSON, default=dict)
    confidence_intervals: Mapped[dict] = mapped_column(JSON, default=dict)
    standard_errors: Mapped[dict] = mapped_column(JSON, default=dict)
    significant_variables: Mapped[list] = mapped_column(JSON, default=list)
    highly_significant: Mapped[list] = mapped_column(JSON, default=list)
    match_score: Mapped[float] = mapped_column(Float, default=0.0)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error_message: Mapped[str] = mapped_column(Text, nullable=True)

    hypothesis: Mapped[Hypothesis] = relationship(back_populates="model_results")
    validation: Mapped[ValidationResult | None] = relationship(
        back_populates="model_result", uselist=False
    )
    citations: Mapped[list[Citation]] = relationship(back_populates="model_result")


# ── Validation results ────────────────────────────────────────────────────────


class ValidationResult(Base):
    __tablename__ = "validation_results"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    model_result_id: Mapped[str] = mapped_column(
        ForeignKey("model_results.id"), unique=True
    )
    vif_passed: Mapped[bool] = mapped_column(Boolean, default=True)
    max_vif: Mapped[float] = mapped_column(Float, default=0.0)
    shapiro_p: Mapped[float] = mapped_column(Float, default=1.0)
    normality_passed: Mapped[bool] = mapped_column(Boolean, default=True)
    bp_p: Mapped[float] = mapped_column(Float, default=1.0)
    homoscedasticity_passed: Mapped[bool] = mapped_column(Boolean, default=True)
    anova_f: Mapped[float] = mapped_column(Float, default=0.0)
    anova_p: Mapped[float] = mapped_column(Float, default=1.0)
    anova_passed: Mapped[bool] = mapped_column(Boolean, default=True)
    cv_r2_mean: Mapped[float] = mapped_column(Float, default=0.0)
    cv_r2_std: Mapped[float] = mapped_column(Float, default=0.0)
    cv_passed: Mapped[bool] = mapped_column(Boolean, default=True)
    eta_squared: Mapped[float] = mapped_column(Float, default=0.0)
    effect_size_label: Mapped[str] = mapped_column(String(50), nullable=True)
    overall_passed: Mapped[bool] = mapped_column(Boolean, default=False)
    pass_rate: Mapped[float] = mapped_column(Float, default=0.0)
    n_tests_run: Mapped[int] = mapped_column(Integer, default=0)
    n_tests_passed: Mapped[int] = mapped_column(Integer, default=0)
    warnings: Mapped[list] = mapped_column(JSON, default=list)

    model_result: Mapped[ModelResult] = relationship(back_populates="validation")


# ── Citations ─────────────────────────────────────────────────────────────────


class Citation(Base):
    __tablename__ = "citations"
    __table_args__ = (
        Index("ix_citations_model_result_id", "model_result_id"),
        Index("ix_citations_similarity_score", "similarity_score"),
        Index("ix_citations_source", "source"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    model_result_id: Mapped[str] = mapped_column(ForeignKey("model_results.id"))
    source: Mapped[str] = mapped_column(String(50))  # corpus | arxiv | semantic_scholar
    title: Mapped[str] = mapped_column(Text)
    authors: Mapped[list] = mapped_column(JSON, default=list)
    url: Mapped[str] = mapped_column(Text, nullable=True)
    year: Mapped[str] = mapped_column(String(10), nullable=True)
    similarity_score: Mapped[float] = mapped_column(Float, default=0.0)
    abstract_snippet: Mapped[str] = mapped_column(Text, nullable=True)
    variable: Mapped[str] = mapped_column(String(255), nullable=True)

    model_result: Mapped[ModelResult] = relationship(back_populates="citations")


# ── Literature cache metadata ─────────────────────────────────────────────────


class ArxivCacheMeta(Base):
    """Tracks what's in Redis arXiv cache for management and TTL visibility."""

    __tablename__ = "arxiv_cache_meta"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    arxiv_id: Mapped[str] = mapped_column(String(50), unique=True)
    title: Mapped[str] = mapped_column(Text)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    hit_count: Mapped[int] = mapped_column(Integer, default=0)


class S2CacheMeta(Base):
    """Tracks what's in Redis S2 cache."""

    __tablename__ = "s2_cache_meta"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    paper_id: Mapped[str] = mapped_column(String(100), unique=True)
    title: Mapped[str] = mapped_column(Text)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    hit_count: Mapped[int] = mapped_column(Integer, default=0)
