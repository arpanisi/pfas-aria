"""
Config loader.
Reads all YAML configs from the configs/ directory and merges them
into a single typed Settings object accessible anywhere in the codebase.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# ── Config root ──────────────────────────────────────────────────────────────

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _load_yaml(filename: str) -> dict[str, Any]:
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ── Sub-models ────────────────────────────────────────────────────────────────


class DataConfig(BaseModel):
    file_path: str = "data/raw/pfas_data.csv"
    outcome_variable: str = "degradation_rate"
    entity_id_column: str | None = None
    time_column: str | None = None
    exclude_columns: list[str] = Field(default_factory=list)


class CorpusConfig(BaseModel):
    directory: str = "data/corpus"
    chunk_size: int = 512
    chunk_overlap: int = 64


class DomainConfig(BaseModel):
    context: str = ""
    keywords: list[str] = Field(default_factory=list)


class SupervisorConfig(BaseModel):
    convergence_threshold: float = 0.75
    max_rounds: int = 10
    hypotheses_per_round: int = 6
    min_passing_hypotheses: int = 2


class ConvergenceWeights(BaseModel):
    rag_similarity: float = 0.50
    arxiv_citation: float = 0.30
    validation_pass_rate: float = 0.20


class ConvergenceConfig(BaseModel):
    weights: ConvergenceWeights = Field(default_factory=ConvergenceWeights)


class LLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = "llama3.1:8b"
    runpod_endpoint: str | None = None
    temperature: float = 0.1
    max_tokens: int = 4096
    request_timeout: int = 120


class EmbeddingConfig(BaseModel):
    model: str = "all-MiniLM-L6-v2"
    device: str = "cpu"


class VectorStoreConfig(BaseModel):
    provider: str = "chromadb"
    persist_directory: str = "data/outputs/chroma"
    collection_name: str = "pfas_corpus"


class MLflowConfig(BaseModel):
    tracking_uri: str = "mlflow"
    experiment_name: str = "pfas-aria"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_to_file: bool = True
    log_directory: str = "data/outputs/logs"
    rotation: str = "10 MB"
    retention: str = "30 days"


class GroundingConfig(BaseModel):
    arxiv_max_results: int = 10
    min_similarity_score: float = 0.60


class HypothesisConfig(BaseModel):
    max_variables_per_model: int = 8
    max_interaction_terms: int = 3
    allow_polynomial_terms: bool = True
    allow_log_transforms: bool = True


class PipelinePhases(BaseModel):
    ingestion: bool = True
    rag_build: bool = True
    data_intelligence: bool = True
    hypothesis_generation: bool = True
    modeling: bool = True
    validation: bool = True
    grounding: bool = True
    report: bool = True


class PipelineConfig(BaseModel):
    run_name: str | None = None
    phases: PipelinePhases = Field(default_factory=PipelinePhases)
    resume_from_run_id: str | None = None
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# ── Root settings ─────────────────────────────────────────────────────────────


class Settings(BaseModel):
    # Data
    data: DataConfig = Field(default_factory=DataConfig)
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    domain: DomainConfig = Field(default_factory=DomainConfig)

    # Agents
    supervisor: SupervisorConfig = Field(default_factory=SupervisorConfig)
    convergence: ConvergenceConfig = Field(default_factory=ConvergenceConfig)
    hypothesis: HypothesisConfig = Field(default_factory=HypothesisConfig)
    grounding: GroundingConfig = Field(default_factory=GroundingConfig)

    # Models
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)

    # Pipeline
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and merge all YAML configs into a single Settings object.
    Cached after first call — import and call freely."""
    raw: dict[str, Any] = {}

    for filename in [
        "data_config.yaml",
        "agent_config.yaml",
        "model_config.yaml",
        "pipeline_config.yaml",
    ]:
        raw.update(_load_yaml(filename))

    return Settings(**raw)
