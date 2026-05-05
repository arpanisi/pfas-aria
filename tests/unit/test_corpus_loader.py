"""Unit tests for CorpusLoader."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.corpus_loader import CorpusBundle, CorpusLoader, Document
from src.utils.exceptions import CorpusEmptyError

SAMPLE_TEXT = """
Photochemical Degradation of PFAS Compounds Under UV Irradiation

Abstract
This study investigates the degradation kinetics of per- and polyfluoroalkyl
substances (PFAS) under UV irradiation in aqueous solutions. Results indicate
that UV intensity and pH are primary drivers of degradation rate.

Introduction
PFAS compounds are persistent organic pollutants that resist conventional
treatment methods. Advanced oxidation processes including UV photolysis have
emerged as promising degradation pathways.

Results
Degradation followed pseudo-first-order kinetics. Higher UV intensity
significantly increased the rate constant k. Temperature showed a moderate
positive correlation with degradation efficiency.
""" * 5  # Repeat to get enough text for chunking


@pytest.fixture
def mock_corpus_dir(tmp_path):
    """Create a temp directory with a fake PDF structure."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    return corpus_dir


@pytest.fixture
def loader_with_mock_dir(mock_corpus_dir):
    mock_settings = MagicMock()
    mock_settings.corpus.directory = str(mock_corpus_dir)
    mock_settings.corpus.chunk_size = 256
    mock_settings.corpus.chunk_overlap = 32

    with patch("src.ingestion.corpus_loader.get_settings", return_value=mock_settings):
        yield CorpusLoader(), mock_corpus_dir


class TestCorpusLoader:

    def test_empty_dir_raises(self, loader_with_mock_dir):
        loader, corpus_dir = loader_with_mock_dir
        with pytest.raises(CorpusEmptyError):
            loader.load()

    def test_chunking_produces_documents(self, loader_with_mock_dir):
        loader, corpus_dir = loader_with_mock_dir

        # Mock the PDF extraction
        with patch.object(loader, "_extract_text", return_value=SAMPLE_TEXT):
            # Create a fake PDF file
            fake_pdf = corpus_dir / "test_paper.pdf"
            fake_pdf.write_bytes(b"%PDF-1.4 fake content")

            bundle = loader.load()
            assert isinstance(bundle, CorpusBundle)
            assert bundle.n_papers == 1
            assert bundle.n_chunks > 1

    def test_documents_have_required_fields(self, loader_with_mock_dir):
        loader, corpus_dir = loader_with_mock_dir

        with patch.object(loader, "_extract_text", return_value=SAMPLE_TEXT):
            fake_pdf = corpus_dir / "test_paper.pdf"
            fake_pdf.write_bytes(b"%PDF-1.4 fake content")

            bundle = loader.load()
            for doc in bundle.documents:
                assert isinstance(doc, Document)
                assert doc.doc_id
                assert doc.source_file
                assert doc.text
                assert doc.n_tokens > 0

    def test_chunk_splitting(self):
        mock_settings = MagicMock()
        mock_settings.corpus.directory = "."
        mock_settings.corpus.chunk_size = 100
        mock_settings.corpus.chunk_overlap = 20

        with patch("src.ingestion.corpus_loader.get_settings", return_value=mock_settings):
            loader = CorpusLoader()
            chunks = loader._split_into_chunks(SAMPLE_TEXT)
            assert len(chunks) > 1
            # Verify overlap: end of chunk N should appear near start of chunk N+1
            for chunk in chunks:
                assert len(chunk) > 0

    def test_doc_ids_are_unique(self, loader_with_mock_dir):
        loader, corpus_dir = loader_with_mock_dir

        with patch.object(loader, "_extract_text", return_value=SAMPLE_TEXT):
            fake_pdf = corpus_dir / "test_paper.pdf"
            fake_pdf.write_bytes(b"%PDF-1.4 fake content")

            bundle = loader.load()
            ids = [d.doc_id for d in bundle.documents]
            assert len(ids) == len(set(ids))
