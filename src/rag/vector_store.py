"""
Vector store backed by MongoDB chunk documents.

Embeddings are written to each chunk's ``embedding`` field (same schema as
corpus upload). Semantic search uses cosine similarity against all embedded
chunks (suitable for modest corpora).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from pymongo import MongoClient, UpdateOne

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

    def __init__(self) -> None:
        cfg = get_settings().vector_store
        if (cfg.provider or "").lower() != "mongodb":
            raise VectorStoreError(
                f"Unsupported vector_store.provider={cfg.provider!r}; "
                "only 'mongodb' is supported."
            )
        self._collection_name = cfg.collection_name
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

    def ensure_all_chunks_embedded(self, *, force_rebuild: bool = False) -> int:
        """Embed every chunk missing vectors or on wrong model. Returns rows updated."""
        if force_rebuild:
            self._coll.update_many(
                {},
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
        logger.info("Wrote embeddings for %s chunks", updated)
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
        cursor = self._coll.find(
            {"embedding": {"$ne": None}},
            projection={
                "text": 1,
                "filename": 1,
                "title": 1,
                "chunk_index": 1,
                "metadata": 1,
                "embedding": 1,
            },
        )
        rows = list(cursor)
        rows = [
            r
            for r in rows
            if r.get("embedding") is not None
            and isinstance(r["embedding"], list)
            and len(r["embedding"]) > 0
        ]
        if not rows:
            raise VectorStoreError(
                "Vector store is empty. Upload PDFs and ensure Mongo chunks are embedded."
            )

        q = self._embedder.embed_one(query).astype(np.float64)
        q = q / (float(np.linalg.norm(q)) + 1e-12)
        matrix = np.stack([np.asarray(r["embedding"], dtype=np.float64) for r in rows])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
        matrix = matrix / norms
        sims = (matrix @ q).astype(np.float64)
        order = np.argsort(-sims)
        k = min(top_k, len(rows))
        chunks: list[RetrievedChunk] = []
        for idx in order[:k]:
            i = int(idx)
            sim = float(sims[i])
            if sim < min_similarity:
                continue
            r = rows[i]
            meta = r.get("metadata") if isinstance(r.get("metadata"), dict) else {}
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
        return sorted(chunks, key=lambda c: c.similarity_score, reverse=True)

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
                    "embedding": {
                        "$exists": True,
                        "$type": "array",
                        "$not": {"$size": 0},
                    }
                }
            )
        )

    def is_built(self) -> bool:
        return self.count() > 0
