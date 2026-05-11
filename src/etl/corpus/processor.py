"""
Corpus Processor.
Incremental PDF processing pipeline.
Only processes PDFs not already in the paper registry.
Stores chunks in MongoDB, embeddings in ChromaDB, metadata in PostgreSQL.

Handles corpora from 50 to 10,000+ papers efficiently.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

from src.utils.config import get_settings
from src.utils.exceptions import CorpusEmptyError, IngestionError
from src.utils.logging import get_logger
from src.utils.paths import CORPUS_DIR

logger = get_logger(__name__)


@dataclass
class ProcessedChunk:
    """A single text chunk ready for MongoDB + ChromaDB."""

    doc_id: str
    paper_filename: str
    paper_title: str
    chunk_index: int
    text: str
    n_tokens: int
    content_hash: str
    source_type: str = "uploaded"
    source_provider: str = "manual_upload"
    metadata: dict = field(default_factory=dict)


@dataclass
class ProcessingResult:
    """Result of processing one PDF."""

    filename: str
    title: str
    n_chunks: int
    n_tokens: int
    chunks: list[ProcessedChunk]
    mongo_ids: list[str] = field(default_factory=list)
    success: bool = True
    error: str = ""


class CorpusProcessor:
    """
    Processes PDFs into chunks and stores them across MongoDB + ChromaDB.
    Skips already-processed papers via the registry.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.chunk_size = self.settings.corpus.chunk_size
        self.chunk_overlap = self.settings.corpus.chunk_overlap

    def scan_new_papers(
        self,
        processed_filenames: set[str],
        corpus_dir: Path | None = None,
    ) -> list[Path]:
        """
        Return paths of PDFs not yet in the registry.
        Scans the corpus directory for new files.
        """
        directory = corpus_dir or CORPUS_DIR
        all_pdfs = sorted(directory.glob("*.pdf"))

        if not all_pdfs:
            raise CorpusEmptyError(f"No PDFs found in {directory}")

        new_pdfs = [p for p in all_pdfs if p.name not in processed_filenames]

        logger.info(
            f"Corpus scan: {len(all_pdfs)} total, "
            f"{len(new_pdfs)} new, "
            f"{len(all_pdfs) - len(new_pdfs)} already processed"
        )
        return new_pdfs

    def process_pdf(self, pdf_path: Path) -> ProcessingResult:
        """
        Parse, chunk, and prepare one PDF for storage.
        Does not write to any database — caller handles persistence.
        """
        try:
            raw_text = self._extract_text(pdf_path)
            title = self._extract_title(raw_text, pdf_path.stem)
            chunks = self._split_into_chunks(raw_text)
            paper_hash = self._hash_file(pdf_path)

            processed_chunks = []
            total_tokens = 0

            for i, text in enumerate(chunks):
                chunk_hash = hashlib.sha256(
                    f"{paper_hash}:{i}:{text}".encode("utf-8")
                ).hexdigest()
                doc_id = hashlib.sha256(f"{paper_hash}:{i}".encode()).hexdigest()[:16]
                n_tokens = max(1, len(text) // 4)
                total_tokens += n_tokens

                processed_chunks.append(
                    ProcessedChunk(
                        doc_id=doc_id,
                        paper_filename=pdf_path.name,
                        paper_title=title,
                        chunk_index=i,
                        text=text,
                        n_tokens=n_tokens,
                        content_hash=chunk_hash,
                        source_type="uploaded",
                        source_provider="manual_upload",
                        metadata={
                            "source_file": pdf_path.name,
                            "title": title,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "paper_content_hash": paper_hash,
                            "content_hash": chunk_hash,
                            "source_type": "uploaded",
                            "source_provider": "manual_upload",
                        },
                    )
                )

            return ProcessingResult(
                filename=pdf_path.name,
                title=title,
                n_chunks=len(processed_chunks),
                n_tokens=total_tokens,
                chunks=processed_chunks,
            )

        except Exception as e:
            logger.warning(f"Failed to process {pdf_path.name}: {e}")
            return ProcessingResult(
                filename=pdf_path.name,
                title=pdf_path.stem,
                n_chunks=0,
                n_tokens=0,
                chunks=[],
                success=False,
                error=str(e),
            )

    def process_batch(
        self, pdf_paths: list[Path], batch_size: int = 10
    ) -> list[ProcessingResult]:
        """
        Process a list of PDFs in batches.
        Returns results for all PDFs including failures.
        """
        results = []
        total = len(pdf_paths)

        for i in range(0, total, batch_size):
            batch = pdf_paths[i : i + batch_size]
            logger.info(
                f"Processing batch {i // batch_size + 1}/"
                f"{(total + batch_size - 1) // batch_size} "
                f"({len(batch)} PDFs)"
            )
            for pdf_path in batch:
                result = self.process_pdf(pdf_path)
                results.append(result)
                if result.success:
                    logger.info(f"  ✓ {result.filename} — {result.n_chunks} chunks")
                else:
                    logger.warning(f"  ✗ {result.filename} — {result.error}")

        n_success = sum(1 for r in results if r.success)
        logger.info(f"Batch complete: {n_success}/{total} papers processed")
        return results

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _hash_file(path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _extract_text(self, path: Path) -> str:
        pages = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text.strip())

        full_text = "\n\n".join(pages)
        if not full_text.strip():
            raise IngestionError(f"No text extracted from {path.name}")
        return full_text

    def _extract_title(self, text: str, fallback: str) -> str:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if lines:
            title = max(lines[:5], key=len)
            return title[:200] if len(title) > 10 else fallback
        return fallback

    def _split_into_chunks(self, text: str) -> list[str]:
        chars_per_chunk = self.chunk_size * 4
        overlap_chars = self.chunk_overlap * 4
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chars_per_chunk, text_len)
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
