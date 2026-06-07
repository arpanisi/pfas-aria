"""Unit tests for RAG components."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.rag.embedder import Embedder
from src.rag.reranker import Reranker
from src.rag.retriever import Retriever
from src.rag.vector_store import RetrievedChunk


def _make_jina_response(n: int, dim: int = 1024) -> MagicMock:
    """Build a fake httpx response matching Jina's embeddings API shape."""
    vecs = np.random.rand(n, dim).astype(np.float32)
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "data": [{"index": i, "embedding": vecs[i].tolist()} for i in range(n)]
    }
    return mock_resp


class TestEmbedder:
    @patch("src.rag.embedder.httpx.post")
    @patch("src.rag.embedder.get_settings")
    @patch.dict("os.environ", {"JINA_API_KEY": "test-key"})
    def test_embed_returns_correct_shape(self, mock_settings, mock_post):
        mock_settings.return_value.embeddings.model = "jina-embeddings-v3"
        mock_post.return_value = _make_jina_response(3, dim=1024)

        embedder = Embedder()
        result = embedder.embed(["text1", "text2", "text3"])
        assert result.shape == (3, 1024)

    @patch("src.rag.embedder.httpx.post")
    @patch("src.rag.embedder.get_settings")
    @patch.dict("os.environ", {"JINA_API_KEY": "test-key"})
    def test_embed_one_returns_1d(self, mock_settings, mock_post):
        mock_settings.return_value.embeddings.model = "jina-embeddings-v3"
        mock_post.return_value = _make_jina_response(1, dim=1024)

        embedder = Embedder()
        result = embedder.embed_one("single text")
        assert result.ndim == 1

    @patch("src.rag.embedder.httpx.post")
    @patch("src.rag.embedder.get_settings")
    @patch.dict("os.environ", {"JINA_API_KEY": "test-key"})
    def test_similarity_range(self, mock_settings, mock_post):
        mock_settings.return_value.embeddings.model = "jina-embeddings-v3"
        mock_post.return_value = _make_jina_response(1, dim=1024)

        embedder = Embedder()
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert embedder.similarity(a, b) == pytest.approx(1.0)

        c = np.array([0.0, 1.0, 0.0])
        assert embedder.similarity(a, c) == pytest.approx(0.0)

    @patch("src.rag.embedder.httpx.post")
    @patch("src.rag.embedder.get_settings")
    @patch.dict("os.environ", {"JINA_API_KEY": "test-key"})
    def test_empty_list_raises(self, mock_settings, mock_post):
        mock_settings.return_value.embeddings.model = "jina-embeddings-v3"
        mock_post.return_value = _make_jina_response(0)

        embedder = Embedder()
        with pytest.raises(ValueError):
            embedder.embed([])

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_raises(self):
        with pytest.raises(RuntimeError, match="JINA_API_KEY"):
            Embedder()


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


def _make_chunk(text: str, score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(
        doc_id="doc1",
        source_file="paper.pdf",
        title="PFAS Review",
        text=text,
        chunk_index=0,
        similarity_score=score,
        metadata={},
    )


class TestReranker:
    @patch("src.rag.reranker.httpx.post")
    @patch.dict("os.environ", {"JINA_API_KEY": "test-key"})
    def test_rerank_reorders_chunks(self, mock_post):
        chunks = [
            _make_chunk("UV photolysis"),
            _make_chunk("electrochemical oxidation"),
        ]
        mock_post.return_value.raise_for_status = MagicMock()
        mock_post.return_value.json.return_value = {
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.3},
            ]
        }
        reranker = Reranker()
        result = reranker.rerank("electrochemical PFAS degradation", chunks)
        assert result[0].text == "electrochemical oxidation"
        assert result[0].similarity_score == pytest.approx(0.9)
        assert result[1].text == "UV photolysis"
        assert result[1].similarity_score == pytest.approx(0.3)

    @patch("src.rag.reranker.httpx.post")
    @patch.dict("os.environ", {"JINA_API_KEY": "test-key"})
    def test_rerank_falls_back_on_error(self, mock_post):
        chunks = [_make_chunk("chunk A"), _make_chunk("chunk B")]
        mock_post.side_effect = Exception("timeout")
        reranker = Reranker()
        result = reranker.rerank("PFAS", chunks)
        assert result == chunks  # original order preserved

    @patch.dict("os.environ", {}, clear=True)
    def test_rerank_disabled_without_key(self):
        chunks = [_make_chunk("chunk A"), _make_chunk("chunk B")]
        reranker = Reranker()
        assert reranker.rerank("PFAS", chunks) == chunks
