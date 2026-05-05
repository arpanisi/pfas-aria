"""Unit tests for RAG components."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.rag.embedder import Embedder
from src.rag.retriever import Retriever
from src.rag.vector_store import RetrievedChunk


class TestEmbedder:

    @patch("src.rag.embedder.SentenceTransformer")
    @patch("src.rag.embedder.get_settings")
    def test_embed_returns_correct_shape(self, mock_settings, mock_st):
        mock_settings.return_value.embeddings.model = "all-MiniLM-L6-v2"
        mock_settings.return_value.embeddings.device = "cpu"

        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(3, 384).astype(np.float32)
        mock_st.return_value = mock_model

        embedder = Embedder()
        result = embedder.embed(["text1", "text2", "text3"])
        assert result.shape == (3, 384)

    @patch("src.rag.embedder.SentenceTransformer")
    @patch("src.rag.embedder.get_settings")
    def test_embed_one_returns_1d(self, mock_settings, mock_st):
        mock_settings.return_value.embeddings.model = "all-MiniLM-L6-v2"
        mock_settings.return_value.embeddings.device = "cpu"

        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(1, 384).astype(np.float32)
        mock_st.return_value = mock_model

        embedder = Embedder()
        result = embedder.embed_one("single text")
        assert result.ndim == 1

    @patch("src.rag.embedder.SentenceTransformer")
    @patch("src.rag.embedder.get_settings")
    def test_similarity_range(self, mock_settings, mock_st):
        mock_settings.return_value.embeddings.model = "all-MiniLM-L6-v2"
        mock_settings.return_value.embeddings.device = "cpu"
        mock_st.return_value = MagicMock()

        embedder = Embedder()
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert embedder.similarity(a, b) == pytest.approx(1.0)

        c = np.array([0.0, 1.0, 0.0])
        assert embedder.similarity(a, c) == pytest.approx(0.0)

    @patch("src.rag.embedder.SentenceTransformer")
    @patch("src.rag.embedder.get_settings")
    def test_empty_list_raises(self, mock_settings, mock_st):
        mock_settings.return_value.embeddings.model = "all-MiniLM-L6-v2"
        mock_settings.return_value.embeddings.device = "cpu"
        mock_st.return_value = MagicMock()

        embedder = Embedder()
        with pytest.raises(ValueError):
            embedder.embed([])


class TestRetriever:

    def _make_chunk(self, score: float, text: str = "sample text") -> RetrievedChunk:
        return RetrievedChunk(
            doc_id="abc123",
            source_file="paper.pdf",
            title="Test Paper",
            text=text,
            chunk_index=0,
            similarity_score=score,
            metadata={},
        )

    def test_format_context_respects_max_chars(self):
        mock_store = MagicMock()
        retriever = Retriever(mock_store)

        chunks = [self._make_chunk(0.9, "x" * 1000) for _ in range(10)]
        context = retriever.format_context(chunks, max_chars=2000)
        assert len(context) <= 2500  # some tolerance for headers

    def test_format_context_includes_source(self):
        mock_store = MagicMock()
        retriever = Retriever(mock_store)

        chunks = [self._make_chunk(0.85, "Important PFAS finding")]
        context = retriever.format_context(chunks)
        assert "Test Paper" in context
        assert "paper.pdf" in context

    def test_is_ready_delegates_to_store(self):
        mock_store = MagicMock()
        mock_store.is_built.return_value = True
        retriever = Retriever(mock_store)
        assert retriever.is_ready() is True

    def test_retrieve_calls_store_search(self):
        mock_store = MagicMock()
        mock_store.search.return_value = []
        retriever = Retriever(mock_store)

        retriever.retrieve("PFAS UV degradation", top_k=5)
        mock_store.search.assert_called_once_with(
            "PFAS UV degradation", top_k=5, min_similarity=0.50
        )
