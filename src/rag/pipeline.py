"""
RAG Pipeline.
Ensures MongoDB chunk embeddings exist, then exposes the Retriever.
"""

from __future__ import annotations

from src.ingestion.corpus_loader import CorpusBundle, CorpusLoader
from src.rag.embedder import get_embedder
from src.rag.retriever import Retriever, get_retriever
from src.rag.vector_store import VectorStore
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Module-level cache: avoids re-parsing PDFs on every request when Mongo is current.
# Cleared by force_rebuild=True (e.g. after a new PDF is uploaded to the corpus).
_cached_retriever: Retriever | None = None
_cached_corpus_bundle: CorpusBundle | None = None


def clear_rag_cache() -> None:
    """Invalidate the module-level retriever cache (call after corpus uploads)."""
    global _cached_retriever, _cached_corpus_bundle
    _cached_retriever = None
    _cached_corpus_bundle = None


class RAGPipeline:
    """
    Builds and exposes the RAG retriever.
    Entry point: call .build() once, then .retriever for all queries.
    """

    def __init__(self) -> None:
        self._vector_store: VectorStore | None = None
        self._retriever: Retriever | None = None
        self._corpus_bundle: CorpusBundle | None = None

    def build(self, force_rebuild: bool = False) -> Retriever:
        """
        1. Return cached retriever immediately if Mongo embeddings are current.
        2. Otherwise open Mongo-backed vector store, embed any missing chunks, cache result.
        force_rebuild=True clears the cache (use after uploading new corpus PDFs).
        """
        global _cached_retriever, _cached_corpus_bundle

        if force_rebuild:
            clear_rag_cache()

        if _cached_retriever is not None:
            self._retriever = _cached_retriever
            self._corpus_bundle = _cached_corpus_bundle
            logger.info("=== RAG Pipeline: Using cached retriever ===")
            return _cached_retriever

        logger.info("=== RAG Pipeline: Starting ===")

        self._vector_store = VectorStore()

        logger.info("Warming up embedding model...")
        get_embedder()

        logger.info("Syncing Mongo chunk embeddings...")
        self._vector_store.build(force_rebuild=force_rebuild)

        if not self._vector_store.is_built():
            logger.warning(
                "No embedded chunks in Mongo — upload PDFs via the corpus API first."
            )

        self._retriever = get_retriever(self._vector_store)
        self._warm_corpus_bundle_for_summary()

        logger.info(
            "=== RAG Pipeline: Ready — %s embedded chunks ===",
            self._vector_store.count(),
        )

        _cached_retriever = self._retriever
        _cached_corpus_bundle = self._corpus_bundle
        return self._retriever

    def _warm_corpus_bundle_for_summary(self) -> None:
        """Optional: load disk corpus for human-readable titles in corpus_summary."""
        try:
            self._corpus_bundle = CorpusLoader().load()
        except Exception as e:  # noqa: BLE001
            logger.debug("CorpusLoader skipped for summary: {}", e)
            self._corpus_bundle = None

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            raise RuntimeError("RAG pipeline not built. Call .build() first.")
        return self._retriever

    @property
    def corpus_bundle(self) -> CorpusBundle | None:
        return self._corpus_bundle

    def corpus_summary(self) -> str:
        """Human-readable summary for logging and dashboard."""
        if not self._corpus_bundle:
            if self._vector_store is not None:
                return f"Mongo embedded chunks: {self._vector_store.count()}"
            return "Corpus not loaded"
        b = self._corpus_bundle
        lines = [
            f"Papers: {b.n_papers}",
            f"Chunks: {b.n_chunks}",
            f"Failed: {len(b.failed_files)}",
            "Titles:",
        ]
        for title in b.paper_titles[:10]:
            lines.append(f"  - {title[:80]}")
        if len(b.paper_titles) > 10:
            lines.append(f"  ... and {len(b.paper_titles) - 10} more")
        return "\n".join(lines)
