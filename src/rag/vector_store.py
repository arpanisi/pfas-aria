"""
Vector Store.
Wraps ChromaDB for persistent storage and retrieval of paper embeddings.
Handles build, upsert, and semantic search operations.
"""

from __future__ import annotations

from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.ingestion.corpus_loader import Document
from src.rag.embedder import get_embedder
from src.utils.config import get_settings
from src.utils.exceptions import VectorStoreError
from src.utils.logging import get_logger
from src.utils.paths import PROJECT_ROOT

logger = get_logger(__name__)


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
    ChromaDB-backed persistent vector store for the paper corpus.
    Supports build from scratch, incremental upsert, and semantic search.
    """

    def __init__(self) -> None:
        cfg = get_settings().vector_store
        persist_dir = PROJECT_ROOT / cfg.persist_directory
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection_name = cfg.collection_name
        self._collection = self._get_or_create_collection()
        self._embedder = get_embedder()

        logger.info(
            f"VectorStore ready — collection: {self._collection_name}, "
            f"documents: {self._collection.count()}"
        )

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, documents: list[Document], force_rebuild: bool = False) -> None:
        """Embed all documents and store in ChromaDB.
        Skips documents that already exist unless force_rebuild=True."""

        if force_rebuild:
            self._client.delete_collection(self._collection_name)
            self._collection = self._get_or_create_collection()
            logger.info("Collection cleared for rebuild")

        existing_ids = set(self._collection.get()["ids"])
        new_docs = [d for d in documents if d.doc_id not in existing_ids]

        if not new_docs:
            logger.info("All documents already indexed — skipping build")
            return

        logger.info(f"Embedding {len(new_docs)} new documents...")

        # Batch embed
        texts = [d.text for d in new_docs]
        embeddings = self._embedder.embed(texts)

        # Upsert into ChromaDB
        self._collection.upsert(
            ids=[d.doc_id for d in new_docs],
            embeddings=embeddings.tolist(),
            documents=[d.text for d in new_docs],
            metadatas=[d.metadata for d in new_docs],
        )

        logger.info(
            f"Indexed {len(new_docs)} chunks. "
            f"Total in store: {self._collection.count()}"
        )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Semantic search against the corpus.
        Returns top_k most similar chunks above min_similarity threshold."""

        if self._collection.count() == 0:
            raise VectorStoreError(
                "Vector store is empty. Run the ingestion pipeline first."
            )

        query_embedding = self._embedder.embed_one(query)

        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for i, doc_id in enumerate(results["ids"][0]):
            # ChromaDB returns L2 distance; convert to similarity score
            distance = results["distances"][0][i]
            similarity = max(0.0, 1.0 - distance / 2.0)

            if similarity < min_similarity:
                continue

            meta = results["metadatas"][0][i]
            chunks.append(
                RetrievedChunk(
                    doc_id=doc_id,
                    source_file=meta.get("source_file", "unknown"),
                    title=meta.get("title", "unknown"),
                    text=results["documents"][0][i],
                    chunk_index=meta.get("chunk_index", 0),
                    similarity_score=round(similarity, 4),
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
        """Search with multiple queries, deduplicate, and return top results."""
        seen_ids: set[str] = set()
        all_chunks: list[RetrievedChunk] = []

        for query in queries:
            chunks = self.search(query, top_k=top_k, min_similarity=min_similarity)
            for chunk in chunks:
                if chunk.doc_id not in seen_ids:
                    all_chunks.append(chunk)
                    seen_ids.add(chunk.doc_id)

        return sorted(all_chunks, key=lambda c: c.similarity_score, reverse=True)

    def count(self) -> int:
        return int(self._collection.count())

    def is_built(self) -> bool:
        return bool(self._collection.count() > 0)

    # ── Private ───────────────────────────────────────────────────────────────

    def _get_or_create_collection(self):
        return self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "l2"},
        )
