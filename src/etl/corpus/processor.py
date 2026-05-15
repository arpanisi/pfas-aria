"""
Corpus Processor.
Incremental PDF processing pipeline.
Only processes PDFs not already in the paper registry.
Stores chunks in MongoDB (embeddings added on first RAG sync), metadata in PostgreSQL.

Handles corpora from 50 to 10,000+ papers efficiently.
"""

from __future__ import annotations

import hashlib
import re
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
    """A single text chunk ready for MongoDB (vectors added during RAG indexing)."""

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


_REF_HEADER = re.compile(
    r"\n(References|Bibliography|REFERENCES|BIBLIOGRAPHY|Works\s+Cited)\s*\n",
    re.IGNORECASE,
)
_ET_AL = re.compile(r"\bet\s+al\.", re.IGNORECASE)
_BRACKET_NUM = re.compile(r"\[\d+\]")
_AUTHOR_YEAR = re.compile(r"\([A-Z][a-z]+(?:\s+et\s+al\.)?,\s*\d{4}\)")
_DOI = re.compile(r"(doi:|https?://doi\.org/)", re.IGNORECASE)


class CorpusProcessor:
    """
    Processes PDFs into chunks and stores them in MongoDB + PostgreSQL registry.
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
            truncated_text, ref_found = self._truncate_at_references(raw_text)
            if ref_found:
                removed_pct = (len(raw_text) - len(truncated_text)) / len(raw_text)
                logger.info(
                    f"  {pdf_path.name}: truncated references section ({removed_pct:.0%} removed)"
                )
            raw_chunks = self._split_into_chunks(truncated_text)
            chunks, n_dropped = self._filter_reference_chunks(raw_chunks)
            if n_dropped:
                logger.info(
                    f"  {pdf_path.name}: dropped {n_dropped} reference-like chunks"
                )
            paper_hash = self._hash_file(pdf_path)

            processed_chunks = []
            total_tokens = 0

            for i, text in enumerate(chunks):
                chunk_hash = hashlib.sha256(
                    f"{paper_hash}:{i}:{text}".encode()
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

    @staticmethod
    def _truncate_at_references(text: str) -> tuple[str, bool]:
        m = _REF_HEADER.search(text)
        if m:
            return text[: m.start()], True
        return text, False

    @staticmethod
    def _filter_reference_chunks(chunks: list[str]) -> tuple[list[str], int]:
        kept, dropped = [], 0
        for chunk in chunks:
            lines = [ln for ln in chunk.splitlines() if ln.strip()]
            if not lines:
                kept.append(chunk)
                continue

            et_al_count = len(_ET_AL.findall(chunk))
            bracket_nums = _BRACKET_NUM.findall(chunk)
            author_year = len(_AUTHOR_YEAR.findall(chunk))
            doi_count = len(_DOI.findall(chunk))
            short_lines = sum(1 for ln in lines if len(ln.strip()) < 80)
            short_line_ratio = short_lines / len(lines)

            nums = [int(m[1:-1]) for m in bracket_nums]
            sequential = len(nums) >= 3 and all(
                nums[i + 1] - nums[i] == 1 for i in range(len(nums) - 1)
            )

            is_ref = (
                et_al_count > 6
                or (sequential and len(bracket_nums) >= 5)
                or doi_count >= 3
                or (
                    short_line_ratio > 0.60
                    and (et_al_count > 3 or author_year > 3 or doi_count > 1)
                )
            )
            if is_ref:
                dropped += 1
            else:
                kept.append(chunk)
        return kept, dropped

    def _split_into_chunks(self, text: str) -> list[str]:
        chars_per_chunk = self.chunk_size * 4
        overlap_chars = self.chunk_overlap * 4
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            hard_end = min(start + chars_per_chunk, text_len)
            end = hard_end
            if hard_end < text_len:
                min_boundary = start + int(chars_per_chunk * 0.70)
                for boundary in ["\n\n", ".\n", ". "]:
                    pos = text.rfind(boundary, min_boundary, hard_end)
                    if pos != -1:
                        end = pos + len(boundary)
                        break
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= text_len:
                break
            start = max(start + 1, end - overlap_chars)

        return chunks
