"""
Corpus Loader.
Parses all PDFs from data/corpus/, extracts text, splits into chunks,
and returns a list of Document objects ready for the RAG pipeline.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

from src.utils.config import get_settings
from src.utils.exceptions import CorpusEmptyError, IngestionError
from src.utils.logging import get_logger
from src.utils.paths import PROJECT_ROOT

logger = get_logger(__name__)


# ── Output schema ─────────────────────────────────────────────────────────────

@dataclass
class Document:
    """A single text chunk from a paper, with full provenance."""
    doc_id: str               # Unique hash of source_file + chunk_index
    source_file: str          # Original PDF filename
    title: str                # Extracted or inferred title
    chunk_index: int          # Position within the source paper
    text: str                 # Chunk text content
    n_tokens: int             # Approximate token count
    metadata: dict = field(default_factory=dict)


@dataclass
class CorpusBundle:
    """All documents parsed from the corpus directory."""
    documents: list[Document]
    n_papers: int
    n_chunks: int
    paper_titles: list[str]
    failed_files: list[str]   # PDFs that could not be parsed


# ── Loader ────────────────────────────────────────────────────────────────────

class CorpusLoader:
    """
    Parses PDFs → extracts text → splits into overlapping chunks.
    Handles malformed PDFs gracefully (logs and skips).
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.cfg = self.settings.corpus
        self.chunk_size = self.cfg.chunk_size        # tokens (approx)
        self.chunk_overlap = self.cfg.chunk_overlap

    def load(self) -> CorpusBundle:
        corpus_dir = self._resolve_corpus_dir()
        pdf_files = sorted(corpus_dir.glob("*.pdf"))

        if not pdf_files:
            raise CorpusEmptyError(
                f"No PDF files found in: {corpus_dir}\n"
                f"Add your papers and re-run."
            )

        logger.info(f"Found {len(pdf_files)} PDFs in corpus")

        all_documents: list[Document] = []
        paper_titles: list[str] = []
        failed_files: list[str] = []

        for pdf_path in pdf_files:
            try:
                docs, title = self._process_pdf(pdf_path)
                all_documents.extend(docs)
                paper_titles.append(title)
                logger.info(
                    f"  ✓ {pdf_path.name} → {len(docs)} chunks (title: {title[:60]})"
                )
            except Exception as e:
                logger.warning(f"  ✗ Failed to parse {pdf_path.name}: {e}")
                failed_files.append(pdf_path.name)

        if not all_documents:
            raise CorpusEmptyError(
                "All PDFs failed to parse. Check your corpus directory."
            )

        bundle = CorpusBundle(
            documents=all_documents,
            n_papers=len(pdf_files) - len(failed_files),
            n_chunks=len(all_documents),
            paper_titles=paper_titles,
            failed_files=failed_files,
        )

        logger.info(
            f"Corpus loaded — papers: {bundle.n_papers}, "
            f"chunks: {bundle.n_chunks}, "
            f"failed: {len(failed_files)}"
        )
        return bundle

    # ── Private helpers ───────────────────────────────────────────────────────

    def _resolve_corpus_dir(self) -> Path:
        p = Path(self.cfg.directory)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            raise IngestionError(
                f"Corpus directory not found: {p}\n"
                f"Update 'corpus.directory' in configs/data_config.yaml"
            )
        return p

    def _process_pdf(self, path: Path) -> tuple[list[Document], str]:
        """Extract text from PDF and split into chunks."""
        raw_text = self._extract_text(path)
        title = self._extract_title(raw_text, path.stem)
        chunks = self._split_into_chunks(raw_text)

        documents = []
        for i, chunk_text in enumerate(chunks):
            doc_id = hashlib.md5(
                f"{path.name}_{i}".encode()
            ).hexdigest()[:12]

            documents.append(Document(
                doc_id=doc_id,
                source_file=path.name,
                title=title,
                chunk_index=i,
                text=chunk_text,
                n_tokens=self._count_tokens(chunk_text),
                metadata={
                    "source_file": path.name,
                    "title": title,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
            ))

        return documents, title

    def _extract_text(self, path: Path) -> str:
        """Extract full text from PDF using pdfplumber."""
        pages_text = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text.strip())

        full_text = "\n\n".join(pages_text)

        if not full_text.strip():
            raise IngestionError(f"No text extracted from {path.name} (scanned PDF?)")

        return full_text

    def _extract_title(self, text: str, fallback: str) -> str:
        """Heuristically extract paper title from first lines of text."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        # Title is usually in the first few non-empty lines
        # Take the longest line in the first 5 lines as the title candidate
        candidates = lines[:5]
        if candidates:
            title = max(candidates, key=len)
            # Truncate if too long
            return title[:200] if len(title) > 10 else fallback
        return fallback

    def _split_into_chunks(self, text: str) -> list[str]:
        """Split text into overlapping chunks by approximate token count."""
        # Approximate: 1 token ≈ 4 characters
        chars_per_chunk = self.chunk_size * 4
        overlap_chars = self.chunk_overlap * 4

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chars_per_chunk, text_len)

            # Try to break at sentence boundary
            if end < text_len:
                for boundary in [". ", ".\n", "\n\n"]:
                    pos = text.rfind(boundary, start, end)
                    if pos != -1:
                        end = pos + len(boundary)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = max(start + 1, end - overlap_chars)

        return chunks

    @staticmethod
    def _count_tokens(text: str) -> int:
        """Approximate token count (1 token ≈ 4 chars). Minimum 1 for non-empty text."""
        return max(1, len(text) // 4)
