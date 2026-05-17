"""
Pydantic request/response schemas for the pipeline API.

Kept separate from route logic so service modules can import types without
creating a circular dependency on FastAPI route decorators.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    n_unique: int
    missing_pct: float
    is_numeric: bool
    sample_values: list


class DatasetPreview(BaseModel):
    filename: str
    n_rows: int
    n_cols: int
    columns: list[ColumnInfo]
    excel_header_row: int | None = None
    input_cols: list[str] = Field(default_factory=list)
    output_cols: list[str] = Field(default_factory=list)
    preview_rows: list[dict[str, str]] = Field(default_factory=list)
    cleaning_notes: list[str] = Field(default_factory=list)
    metadata_cols: list[str] = Field(default_factory=list)
    all_columns: list[str] = Field(default_factory=list)
    label_encoding_record_id: str | None = Field(default=None)


class LegacyRegimeSummaryOut(BaseModel):
    regime_id: int
    n_rows: int
    row_indices_sample: list[str]
    condition_values_sample: list[str]
    non_constant_input_cols: list[str]
    non_constant_output_cols: list[str]


class RegimeRowCountOut(BaseModel):
    regime_id: int
    n_rows: int


class LegacySegmentationPreviewOut(BaseModel):
    filename: str
    n_rows: int
    n_cols: int
    n_regimes: int
    input_cols: list[str]
    output_cols: list[str]
    regimes: list[LegacyRegimeSummaryOut]
    regime_row_counts: list[RegimeRowCountOut]
    warnings: list[str]


class RunStatus(BaseModel):
    run_id: str
    run_name: str
    status: str
    current_round: int
    final_match_score: float
    converged: bool
    n_rounds_completed: int
    run_kind: str = "pipeline"
    regime_id: int | None = None
    hypotheses_tested: int | None = None
    dataset_filename: str | None = None
    created_at: datetime | None = None
    screening_phase: str | None = None


class AutomatedScreeningIterationIn(BaseModel):
    filename: str
    run_name: str = ""
    regime_id: int | None = None
    convergence_threshold: float | None = None


class AutomatedScreeningIterationOut(BaseModel):
    hypotheses_tested: int
    run_id: str | None = None


class ScreeningGroundedIn(BaseModel):
    filename: str
    regime_id: int
    run_name: str = ""
    run_id: str | None = None


class GroundingJobStart(BaseModel):
    job_id: str


class ScreeningHypothesisOut(BaseModel):
    id: str
    hypothesis_id: str
    round: int
    description: str
    rationale: str | None
    primary_variables: list[str]
    model_family: str
    priority_score: float
    is_refinement: bool


class ScreeningModelOut(BaseModel):
    id: str
    hypothesis_id: str
    model_type: str
    r_squared: float
    adj_r_squared: float
    n_observations: int
    coefficients: dict
    p_values: dict
    significant_variables: list[str]
    match_score: float
    validation_passed: bool
    diagnostic_score: float | None = None


class ScreeningCitationOut(BaseModel):
    id: str
    source: str
    title: str
    url: str | None = None
    year: str | None = None
    similarity_score: float
    abstract_snippet: str | None = None
    variable: str | None = None
    hypothesis_id: str | None = None


class ScreeningBundleOut(BaseModel):
    hypothesis: ScreeningHypothesisOut
    model_result: ScreeningModelOut
    citations: list[ScreeningCitationOut]


class ScreeningGroundedOut(BaseModel):
    run_name: str
    filename: str
    display_title: str
    dataset_n_rows: int
    dataset_n_cols: int
    n_corpus_papers: int
    regime_id: int
    regime_n_rows: int
    bundles: list[ScreeningBundleOut]
    warnings: list[str]
    persisted_to_run_id: str | None = None
    system_summary: str | None = None
    next_steps: str | None = None


class GroundingJobProgress(BaseModel):
    pct: int
    stage: str
    done: bool
    eta_seconds: int | None = None
    result: ScreeningGroundedOut | None = None
    error: str | None = None


class ScreeningStatsIn(BaseModel):
    filename: str
    regime_id: int
    run_name: str = ""
    run_id: str | None = None


class ScreeningStatsBundleOut(BaseModel):
    hypothesis: ScreeningHypothesisOut
    model_result: ScreeningModelOut
    diagnostics: dict | None = None


class ScreeningStatsOut(BaseModel):
    run_name: str
    filename: str
    display_title: str
    dataset_n_rows: int
    dataset_n_cols: int
    regime_id: int
    regime_n_rows: int
    bundles: list[ScreeningStatsBundleOut]
    warnings: list[str]
    persisted_to_run_id: str | None = None


class ClearScreeningRunsOut(BaseModel):
    deleted: int
