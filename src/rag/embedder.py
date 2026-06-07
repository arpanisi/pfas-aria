"""
Embedder.
Calls Jina AI Embeddings API to produce vectors for RAG and similarity scoring.
Singleton pattern — client initialises once and is reused across all agents.
"""

from __future__ import annotations

import os

import httpx
import numpy as np

from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

_JINA_URL = "https://api.jina.ai/v1/embeddings"

_instance: Embedder | None = None


class Embedder:
    """Calls Jina Embeddings API. Vectors are L2-normalised by the API."""

    def __init__(self) -> None:
        api_key = os.getenv("JINA_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "JINA_API_KEY is not set — add it to your .env or Render env vars"
            )
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._model = get_settings().embeddings.model
        logger.info("Jina embedder ready: {}", self._model)

    def _call(self, texts: list[str], task: str) -> np.ndarray:
        resp = httpx.post(
            _JINA_URL,
            headers=self._headers,
            json={"model": self._model, "input": texts, "task": task, "normalized": True},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        vecs = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
        return np.array(vecs, dtype=np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed corpus passages. Returns shape (n, embedding_dim)."""
        if not texts:
            raise ValueError("Cannot embed empty list")
        return self._call(texts, "retrieval.passage")

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single query. Returns shape (embedding_dim,)."""
        return self._call([text], "retrieval.query")[0]

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity — vectors are already normalised so dot product suffices."""
        return float(np.dot(a, b))

    def batch_similarity(self, query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
        """Cosine similarity of one query vector against a corpus matrix.
        Returns shape (n_corpus,)."""
        result: np.ndarray = corpus @ query
        return result

    @property
    def model_name(self) -> str:
        return self._model


def get_embedder() -> Embedder:
    """Return the singleton Embedder instance."""
    global _instance
    if _instance is None:
        _instance = Embedder()
    return _instance
