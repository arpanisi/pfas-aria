"""
Vector store backed by MongoDB Atlas chunk documents.

Embeddings are written to each chunk's ``embedding`` field during ingest.
Semantic search delegates to the Atlas ``$vectorSearch`` aggregation stage —
no in-memory index is built and no full-collection fetch is needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pymongo import MongoClient, UpdateOne
from pymongo.errors import OperationFailure

from src.ingestion.corpus_loader import Document
from src.rag.embedder import get_embedder
from src.utils.config import get_settings
from src.utils.exceptions import VectorStoreError
from src.utils.logging import get_logger

logger = get_logger(__name__)

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "pfas_aria")


@dataclass
class RetrievedChunk:
    """A retrieved document chunk with its similarity score."""

    doc_id: str
    source_file: str
    title: str
    text: str
    chunk_index: int
    similarity_score: float
    metadata: dict


class VectorStore:
    """
    MongoDB-backed store: embeds ``chunks`` rows in place and runs similarity
    search over stored vectors.
    """

    def __init__(self, *, user_sub: str | None = None) -> None:
        cfg = get_settings().vector_store
        if (cfg.provider or "").lower() != "mongodb":
            raise VectorStoreError(
                f"Unsupported vector_store.provider={cfg.provider!r}; "
                "only 'mongodb' is supported."
            )
        self._collection_name = cfg.collection_name
        self._vector_search_index = cfg.atlas_vector_index
        self._user_sub = user_sub
        self._client: MongoClient[Any] = MongoClient(
            MONGO_URL, serverSelectionTimeoutMS=5000
        )
        self._coll = self._client[MONGO_DB][self._collection_name]
        self._embedder = get_embedder()
        self._model_name = self._embedder.model_name

        n = self.count()
        logger.info(
            "VectorStore (MongoDB) ready — collection=%s, embedded_chunks=%s",
            self._collection_name,
            n,
        )

    @property
    def scope_key(self) -> str:
        return self._user_sub or "global"

    def _tenant_filter(self) -> dict[str, Any]:
        return {"user_sub": self._user_sub} if self._user_sub else {}

    def ensure_all_chunks_embedded(self, *, force_rebuild: bool = False) -> int:
        """Embed every chunk missing vectors or on wrong model. Returns rows updated."""
        if force_rebuild:
            self._coll.update_many(
                self._tenant_filter(),
                {
                    "$set": {
                        "embedding": None,
                        "embedding_model": None,
                        "embedded_at": None,
                    }
                },
            )
            logger.info("Cleared all chunk embeddings for rebuild")

        query: dict[str, Any] = {
            **self._tenant_filter(),
            "text": {"$exists": True, "$nin": [None, ""]},
            "$or": [
                {"embedding": None},
                {"embedding": {"$exists": False}},
                {"embedding_model": {"$ne": self._model_name}},
            ],
        }
        to_process = list(self._coll.find(query))
        if not to_process:
            logger.info("No chunks need embedding")
            return 0

        logger.info(
            "Embedding %s Mongo chunks (model=%s)...", len(to_process), self._model_name
        )
        batch_size = 32
        updated = 0
        now = datetime.utcnow()
        for start in range(0, len(to_process), batch_size):
            batch = to_process[start : start + batch_size]
            texts = [str(d["text"]) for d in batch]
            vectors = self._embedder.embed(texts)
            ops: list[UpdateOne] = []
            for doc, row in zip(batch, vectors, strict=True):
                ops.append(
                    UpdateOne(
                        {"_id": doc["_id"]},
                        {
                            "$set": {
                                "embedding": row.astype(float).tolist(),
                                "embedding_model": self._model_name,
                                "embedded_at": now,
                            }
                        },
                    )
                )
            if ops:
                self._coll.bulk_write(ops, ordered=False)
                updated += len(ops)
        logger.info("Wrote embeddings for {} chunks", updated)
        return updated

    def build(
        self, documents: list[Document] | None = None, force_rebuild: bool = False
    ) -> None:
        """Ensure Mongo chunks carry embeddings. Legacy ``documents`` is ignored."""
        self.ensure_all_chunks_embedded(force_rebuild=force_rebuild)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[RetrievedChunk]:
        if not self.is_built():
            raise VectorStoreError(
                "Vector store is empty. Upload PDFs and ensure Mongo chunks are embedded."
            )
        q_vec: list[float] = self._embedder.embed_one(query).astype(float).tolist()
        vector_stage: dict[str, Any] = {
            "$vectorSearch": {
                "index": self._vector_search_index,
                "path": "embedding",
                "queryVector": q_vec,
                "numCandidates": max(top_k * 10, 100),
                "limit": top_k,
            }
        }
        if self._user_sub:
            vector_stage["$vectorSearch"]["filter"] = {"user_sub": self._user_sub}
        pipeline = self._search_pipeline(vector_stage, top_k=top_k)
        try:
            rows = list(self._coll.aggregate(pipeline))
        except OperationFailure as exc:
            if not self._is_missing_vector_filter_index_error(exc):
                raise
            logger.warning(
                "Atlas vector index does not support user_sub filtering; "
                "falling back to post-vector tenant match."
            )
            fallback_stage = {
                "$vectorSearch": {
                    "index": self._vector_search_index,
                    "path": "embedding",
                    "queryVector": q_vec,
                    "numCandidates": max(top_k * 50, 500),
                    "limit": max(top_k * 20, 100),
                }
            }
            rows = list(
                self._coll.aggregate(self._search_pipeline(fallback_stage, top_k=top_k))
            )

        chunks: list[RetrievedChunk] = []
        for r in rows:
            sim = float(r.get("score", 0.0))
            if sim < min_similarity:
                continue
            raw_meta = r.get("metadata")
            meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
            fname = r.get("filename") or meta.get("source_file", "unknown")
            title = r.get("title") or meta.get("title", "unknown")
            chunks.append(
                RetrievedChunk(
                    doc_id=str(r["_id"]),
                    source_file=str(fname),
                    title=str(title),
                    text=str(r.get("text", "")),
                    chunk_index=int(r.get("chunk_index", 0)),
                    similarity_score=round(sim, 4),
                    metadata=meta,
                )
            )
        return chunks

    def _search_pipeline(
        self, vector_stage: dict[str, Any], *, top_k: int
    ) -> list[dict[str, Any]]:
        return [
            vector_stage,
            *([{"$match": {"user_sub": self._user_sub}}] if self._user_sub else []),
            {
                "$project": {
                    "text": 1,
                    "filename": 1,
                    "title": 1,
                    "chunk_index": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
            {"$limit": top_k},
        ]

    @staticmethod
    def _is_missing_vector_filter_index_error(exc: OperationFailure) -> bool:
        return "needs to be indexed as filter" in str(exc)

    def search_batch(
        self,
        queries: list[str],
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[list[RetrievedChunk]]:
        return [
            self.search(q, top_k=top_k, min_similarity=min_similarity) for q in queries
        ]

    def search_many(
        self,
        queries: list[str],
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[RetrievedChunk]:
        seen_ids: set[str] = set()
        all_chunks: list[RetrievedChunk] = []
        for q in queries:
            for chunk in self.search(q, top_k=top_k, min_similarity=min_similarity):
                if chunk.doc_id not in seen_ids:
                    all_chunks.append(chunk)
                    seen_ids.add(chunk.doc_id)
        return sorted(all_chunks, key=lambda c: c.similarity_score, reverse=True)

    def count(self) -> int:
        return int(
            self._coll.count_documents(
                {
                    **self._tenant_filter(),
                    "embedding": {
                        "$exists": True,
                        "$type": "array",
                        "$not": {"$size": 0},
                    },
                }
            )
        )

    def is_built(self) -> bool:
        return self.count() > 0
