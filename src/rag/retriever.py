"""
Retriever.
High-level interface that agents use to query the corpus.
Abstracts away ChromaDB details — agents just ask questions.
"""

from __future__ import annotations

from src.rag.vector_store import RetrievedChunk, VectorStore
from src.utils.logging import get_logger

logger = get_logger(__name__)

_instance: Retriever | None = None


class Retriever:
    """
    The single interface all agents use to access the paper corpus.
    Provides domain-aware retrieval methods tailored to each agent's needs.
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self._store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.50,
    ) -> list[RetrievedChunk]:
        """Retrieve most relevant paper chunks for a query."""
        logger.debug(f"RAG query: '{query[:80]}...' top_k={top_k}")
        results = self._store.search(query, top_k=top_k, min_similarity=min_similarity)
        logger.debug(f"Retrieved {len(results)} chunks")
        return results

    def retrieve_for_hypothesis(
        self,
        variable_names: list[str],
        domain_context: str,
        top_k: int = 8,
    ) -> list[RetrievedChunk]:
        """Retrieve context relevant to a set of variables.
        Used by the Hypothesis Agent."""
        queries = [f"{domain_context} {var}" for var in variable_names] + [
            domain_context
        ]

        return self._store.search_many(
            queries=queries,
            top_k=top_k,
            min_similarity=0.40,
        )

    def retrieve_for_grounding(
        self,
        finding: str,
        top_k: int = 6,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks to ground a specific model finding.
        Used by the Literature Grounding Agent."""
        return self._store.search(
            query=finding,
            top_k=top_k,
            min_similarity=0.45,
        )

    def format_context(
        self,
        chunks: list[RetrievedChunk],
        max_chars: int = 4000,
    ) -> str:
        """Format retrieved chunks into a context string for LLM prompts."""
        parts = []
        total_chars = 0

        for chunk in chunks:
            section = (
                f"[Source: {chunk.title[:80]} | "
                f"File: {chunk.source_file} | "
                f"Similarity: {chunk.similarity_score:.2f}]\n"
                f"{chunk.text}"
            )
            if total_chars + len(section) > max_chars:
                break
            parts.append(section)
            total_chars += len(section)

        return "\n\n---\n\n".join(parts)

    def is_ready(self) -> bool:
        return self._store.is_built()

    def stats(self) -> dict:
        return {"total_chunks": self._store.count()}


def get_retriever(vector_store: VectorStore | None = None) -> Retriever:
    """Return the singleton Retriever. Pass vector_store on first call."""
    global _instance
    if _instance is None:
        if vector_store is None:
            raise RuntimeError(
                "Retriever not initialized. Pass a VectorStore on first call."
            )
        _instance = Retriever(vector_store)
    return _instance
