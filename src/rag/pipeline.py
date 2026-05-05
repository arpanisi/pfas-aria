"""
RAG Pipeline.
Orchestrates the full build: load corpus → embed → store in ChromaDB.
Called once at startup. Subsequent runs skip already-indexed documents.
"""

from __future__ import annotations

from src.ingestion.corpus_loader import CorpusBundle, CorpusLoader
from src.rag.embedder import get_embedder
from src.rag.retriever import Retriever, get_retriever
from src.rag.vector_store import VectorStore
from src.utils.logging import get_logger

logger = get_logger(__name__)


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
        Full pipeline:
          1. Load and parse PDFs
          2. Load embedding model
          3. Embed and index into ChromaDB
          4. Return ready Retriever

        If the store is already built and force_rebuild=False,
        skips embedding and returns immediately.
        """
        logger.info("=== RAG Pipeline: Starting ===")

        # Step 1: Initialize vector store
        self._vector_store = VectorStore()

        if self._vector_store.is_built() and not force_rebuild:
            logger.info(
                f"Vector store already contains "
                f"{self._vector_store.count()} chunks — skipping rebuild"
            )
            self._retriever = get_retriever(self._vector_store)
            logger.info("=== RAG Pipeline: Ready (from cache) ===")
            return self._retriever

        # Step 2: Load corpus
        logger.info("Loading corpus...")
        loader = CorpusLoader()
        self._corpus_bundle = loader.load()

        logger.info(
            f"Corpus loaded: {self._corpus_bundle.n_papers} papers, "
            f"{self._corpus_bundle.n_chunks} chunks"
        )

        if self._corpus_bundle.failed_files:
            logger.warning(
                f"Failed to parse: {self._corpus_bundle.failed_files}"
            )

        # Step 3: Ensure embedder is warm
        logger.info("Warming up embedding model...")
        get_embedder()

        # Step 4: Build vector store
        logger.info("Building vector index...")
        self._vector_store.build(
            documents=self._corpus_bundle.documents,
            force_rebuild=force_rebuild,
        )

        # Step 5: Initialize retriever
        self._retriever = get_retriever(self._vector_store)

        logger.info(
            f"=== RAG Pipeline: Ready — "
            f"{self._vector_store.count()} chunks indexed ==="
        )
        return self._retriever

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
