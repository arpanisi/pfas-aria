"""
Reranker.
Calls Jina Reranker API to re-order retrieved chunks by true relevance
to the query, correcting for cosine similarity's inability to distinguish
semantically similar but topically irrelevant chunks.
"""

from __future__ import annotations

import os

import httpx

from src.rag.vector_store import RetrievedChunk
from src.utils.logging import get_logger

logger = get_logger(__name__)

_JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
_RERANK_MODEL = "jina-reranker-v2-base-multilingual"


class Reranker:
    def __init__(self) -> None:
        api_key = os.getenv("JINA_API_KEY", "").strip()
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._enabled = bool(api_key)
        if not self._enabled:
            logger.warning("JINA_API_KEY not set — reranking disabled")

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: int | None = None,
    ) -> list[RetrievedChunk]:
        """Re-order chunks by relevance to query. Returns same list if disabled or on error."""
        if not self._enabled or not chunks:
            return chunks

        top_n = top_n or len(chunks)
        documents = [c.text[:1000] for c in chunks]

        try:
            resp = httpx.post(
                _JINA_RERANK_URL,
                headers=self._headers,
                json={
                    "model": _RERANK_MODEL,
                    "query": query,
                    "documents": documents,
                    "top_n": top_n,
                },
                timeout=30,
            )
            resp.raise_for_status()
            results = resp.json()["results"]
        except Exception as e:
            logger.warning("Reranker failed, keeping original order: {}", e)
            return chunks

        reranked = []
        for r in results:
            chunk = chunks[r["index"]]
            chunk.similarity_score = round(float(r["relevance_score"]), 4)
            reranked.append(chunk)
        return reranked
