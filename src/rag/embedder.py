"""
Embedder.
Wraps sentence-transformers to produce embeddings for RAG and similarity scoring.
Singleton pattern — model loads once and is reused across all agents.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

_instance: Embedder | None = None


class Embedder:
    """Thin wrapper around SentenceTransformer with batch encoding."""

    def __init__(self) -> None:
        cfg = get_settings().embeddings
        logger.info(f"Loading embedding model: {cfg.model} on {cfg.device}")
        self._model = SentenceTransformer(cfg.model, device=cfg.device)
        self._model_name = cfg.model
        logger.info("Embedding model ready")

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts. Returns shape (n, embedding_dim)."""
        if not texts:
            raise ValueError("Cannot embed empty list")

        raw = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,  # cosine similarity via dot product
            convert_to_numpy=True,
        )
        embeddings: np.ndarray = np.array(raw)
        return embeddings

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text. Returns shape (embedding_dim,)."""
        result: np.ndarray = self.embed([text])[0]
        return result

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two embedding vectors."""
        # Embeddings are already normalized so dot product = cosine sim
        return float(np.dot(a, b))

    def batch_similarity(self, query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
        """Cosine similarity of one query vector against a corpus matrix.
        Returns shape (n_corpus,)."""
        result: np.ndarray = corpus @ query
        return result

    @property
    def model_name(self) -> str:
        return self._model_name


def get_embedder() -> Embedder:
    """Return the singleton Embedder instance."""
    global _instance
    if _instance is None:
        _instance = Embedder()
    return _instance
