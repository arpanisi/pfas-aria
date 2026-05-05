"""
Custom exception hierarchy for PFAS-ARIA.
All exceptions trace back to ARIAError so callers can catch broadly
or narrowly as needed.
"""


class ARIAError(Exception):
    """Base exception for all PFAS-ARIA errors."""


# ── Ingestion ─────────────────────────────────────────────────────────────────


class IngestionError(ARIAError):
    """Raised when data or corpus ingestion fails."""


class DataFileNotFoundError(IngestionError):
    """Raised when the experimental data file cannot be located."""


class CorpusEmptyError(IngestionError):
    """Raised when no PDFs are found in the corpus directory."""


# ── RAG ──────────────────────────────────────────────────────────────────────


class RAGError(ARIAError):
    """Raised when the RAG pipeline fails."""


class VectorStoreError(RAGError):
    """Raised when ChromaDB operations fail."""


# ── Agent ────────────────────────────────────────────────────────────────────


class AgentError(ARIAError):
    """Raised when an agent fails to complete its task."""


class LLMError(AgentError):
    """Raised when the LLM call fails or returns unparseable output."""


class HypothesisGenerationError(AgentError):
    """Raised when the hypothesis agent cannot produce valid hypotheses."""


# ── Modeling ─────────────────────────────────────────────────────────────────


class ModelingError(ARIAError):
    """Raised when a model fails to fit or produce results."""


class ValidationError(ARIAError):
    """Raised when model validation checks cannot be completed."""


# ── Grounding ────────────────────────────────────────────────────────────────


class GroundingError(ARIAError):
    """Raised when literature grounding fails."""


# ── Pipeline ─────────────────────────────────────────────────────────────────


class PipelineError(ARIAError):
    """Raised when the supervisor pipeline fails to orchestrate agents."""


class ConvergenceError(PipelineError):
    """Raised when convergence cannot be computed."""


# ── Config ───────────────────────────────────────────────────────────────────


class ConfigError(ARIAError):
    """Raised when configuration is invalid or missing."""
