"""
Retriever.
High-level interface that agents use to query the corpus.
Caches query results in Redis to avoid redundant vector searches
across pipeline rounds (same query asked multiple times).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from src.rag.vector_store import RetrievedChunk, VectorStore
from src.utils.logging import get_logger

logger = get_logger(__name__)

_instance: Retriever | None = None

RAG_CACHE_TTL = 60 * 60 * 24  # 24 hours


class Retriever:
    """
    The single interface all agents use to access the paper corpus.
    Provides domain-aware retrieval methods tailored to each agent's needs.
    Results are cached in Redis — same query in round 1 and round 3
    hits the cache instead of hitting the vector store again.
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self._store = vector_store
        self._redis_available = self._check_redis()

    def _check_redis(self) -> bool:
        try:
            import redis

            r = redis.Redis(host="localhost", port=6379, db=1, socket_timeout=1)
            r.ping()
            logger.debug("RAG cache: Redis connected")
            return True
        except Exception:
            logger.debug("RAG cache: Redis unavailable — no caching")
            return False

    def _get_redis(self) -> Any:
        import redis

        return redis.Redis(host="localhost", port=6379, db=1)

    def _cache_key(self, query: str, top_k: int, min_similarity: float) -> str:
        raw = f"rag:{query}:{top_k}:{min_similarity}:{self._store.count()}"
        return f"rag:{hashlib.md5(raw.encode()).hexdigest()[:16]}"

    def _get_cache(self, key: str) -> list[RetrievedChunk] | None:
        if not self._redis_available:
            return None
        try:
            r = self._get_redis()
            data = r.get(key)
            if not data:
                return None
            raw = json.loads(data)
            return [RetrievedChunk(**item) for item in raw]
        except Exception:
            return None

    def _set_cache(self, key: str, chunks: list[RetrievedChunk]) -> None:
        if not self._redis_available:
            return
        try:
            r = self._get_redis()
            serialized = json.dumps(
                [
                    {
                        "doc_id": c.doc_id,
                        "source_file": c.source_file,
                        "title": c.title,
                        "text": c.text,
                        "chunk_index": c.chunk_index,
                        "similarity_score": c.similarity_score,
                        "metadata": c.metadata,
                    }
                    for c in chunks
                ]
            )
            r.setex(key, RAG_CACHE_TTL, serialized)
        except Exception:
            pass

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.50,
    ) -> list[RetrievedChunk]:
        """Retrieve most relevant paper chunks for a query.
        Results are cached in Redis for 24 hours."""
        cache_key = self._cache_key(query, top_k, min_similarity)
        cached = self._get_cache(cache_key)
        if cached is not None:
            logger.debug(f"RAG cache hit: '{query[:60]}'")
            return cached

        logger.debug(f"RAG query: '{query[:80]}' top_k={top_k}")
        results = self._store.search(query, top_k=top_k, min_similarity=min_similarity)
        self._set_cache(cache_key, results)
        logger.debug(f"Retrieved {len(results)} chunks")
        return results

    def retrieve_batch(
        self,
        queries: list[str],
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[list[RetrievedChunk]]:
        """Embed all queries in a single model pass, serving cache hits individually.

        Cache hits are served per-query; only misses go through batch embedding.
        """
        if not queries:
            return []

        results: list[list[RetrievedChunk] | None] = [None] * len(queries)
        miss_indices: list[int] = []
        miss_queries: list[str] = []

        for i, q in enumerate(queries):
            cache_key = self._cache_key(q, top_k, min_similarity)
            cached = self._get_cache(cache_key)
            if cached is not None:
                results[i] = cached
            else:
                miss_indices.append(i)
                miss_queries.append(q)

        if miss_queries:
            batch = self._store.search_batch(
                miss_queries, top_k=top_k, min_similarity=min_similarity
            )
            for local_i, (idx, chunks) in enumerate(zip(miss_indices, batch)):
                q = queries[idx]
                cache_key = self._cache_key(q, top_k, min_similarity)
                self._set_cache(cache_key, chunks)
                results[idx] = chunks

        return [r if r is not None else [] for r in results]

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
        return self.retrieve(
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

    def invalidate_cache(self) -> None:
        """Clear all RAG query cache entries. Call when corpus is updated."""
        if not self._redis_available:
            return
        try:
            r = self._get_redis()
            keys = list(r.scan_iter("rag:*"))
            if keys:
                r.delete(*keys)
                logger.info(f"RAG cache cleared: {len(keys)} entries")
        except Exception:
            pass

    def is_ready(self) -> bool:
        return self._store.is_built()

    def stats(self) -> dict:
        return {
            "total_chunks": self._store.count(),
            "cache_enabled": self._redis_available,
        }


def get_retriever(vector_store: VectorStore | None = None) -> Retriever:
    """Return the Retriever. Pass ``vector_store`` whenever the index may have changed."""
    global _instance
    if vector_store is not None:
        _instance = Retriever(vector_store)
        return _instance
    if _instance is None:
        raise RuntimeError(
            "Retriever not initialized. Pass a VectorStore on first call."
        )
    return _instance
