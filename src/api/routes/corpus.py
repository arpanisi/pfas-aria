"""
Corpus Routes.
Endpoints for managing the scientific paper corpus.
Upload PDFs, index them immediately into MongoDB/PostgreSQL, and expose corpus stats.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from pymongo import UpdateOne
from sqlalchemy import delete, select

from src.api.deps import CurrentUser, DBSession
from src.db.mongodb import get_mongo_db
from src.db.orm import Paper
from src.etl.corpus.processor import CorpusProcessor
from src.rag.pipeline import clear_rag_cache
from src.utils.logging import get_logger
from src.utils.paths import CORPUS_DIR

logger = get_logger(__name__)
router = APIRouter(prefix="/corpus", tags=["Corpus"])


class PaperOut(BaseModel):
    id: str
    filename: str
    title: str | None
    n_chunks: int
    n_tokens: int
    embedding_model: str | None
    source: str


class CorpusStats(BaseModel):
    n_papers: int
    n_chunks_total: int
    n_tokens_total: int
    papers: list[PaperOut]


class UploadPaperOut(BaseModel):
    filename: str
    size_bytes: int
    status: str
    paper_id: str
    n_chunks: int
    n_tokens: int
    source: str = "uploaded"
    message: str


class DeleteCorpusOut(BaseModel):
    status: str
    deleted_papers: int
    deleted_chunks: int


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._()\- ]+", "_", name).strip()
    return name or f"paper_{uuid.uuid4().hex}.pdf"


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _chunk_content_hash(paper_hash: str, chunk_index: int, text: str) -> str:
    payload = f"{paper_hash}:{chunk_index}:{text}".encode()
    return hashlib.sha256(payload).hexdigest()


class DomainContextOut(BaseModel):
    domain_context: str
    n_papers: int
    corpus_fingerprint: str


# Process-level cache — survives repeated calls until corpus changes.
_domain_cache: dict[str, str] = {"fingerprint": "", "context": ""}


def _corpus_fingerprint(paper_ids: list[str]) -> str:
    payload = "|".join(sorted(paper_ids)).encode()
    return hashlib.md5(payload, usedforsecurity=False).hexdigest()


@router.post("/domain-context", response_model=DomainContextOut)
async def infer_domain_context(
    payload: dict,
    user: CurrentUser,
    db: DBSession,
) -> DomainContextOut:
    """Infer a one-line domain description from dataset columns + corpus paper titles."""
    col_names: list[str] = payload.get("col_names", [])
    result = await db.execute(select(Paper).order_by(Paper.parsed_at.desc()))
    papers = result.scalars().all()

    fingerprint = _corpus_fingerprint([p.id for p in papers])

    if len(papers) < 3:
        _domain_cache["fingerprint"] = ""
        _domain_cache["content"] = ""
        return DomainContextOut(
            domain_context="", n_papers=len(papers), corpus_fingerprint=fingerprint
        )

    if fingerprint == _domain_cache.get("fingerprint") and _domain_cache.get("context"):
        return DomainContextOut(
            domain_context=_domain_cache["context"],
            n_papers=len(papers),
            corpus_fingerprint=fingerprint,
        )

    paper_titles = [p.title for p in papers if p.title]

    def _infer() -> str:
        from src.utils.domain_context import infer_domain_context

        return infer_domain_context(col_names, paper_titles)

    context = await asyncio.to_thread(_infer)
    _domain_cache["fingerprint"] = fingerprint
    _domain_cache["context"] = context
    return DomainContextOut(
        domain_context=context, n_papers=len(papers), corpus_fingerprint=fingerprint
    )


@router.get("/stats", response_model=CorpusStats)
async def get_corpus_stats(user: CurrentUser, db: DBSession) -> CorpusStats:
    result = await db.execute(select(Paper).order_by(Paper.parsed_at.desc()))
    papers = result.scalars().all()

    return CorpusStats(
        n_papers=len(papers),
        n_chunks_total=sum(p.n_chunks for p in papers),
        n_tokens_total=sum(p.n_tokens for p in papers),
        papers=[
            PaperOut(
                id=p.id,
                filename=p.filename,
                title=p.title,
                n_chunks=p.n_chunks,
                n_tokens=p.n_tokens,
                embedding_model=p.embedding_model,
                source=p.source,
            )
            for p in papers
        ],
    )


@router.post("/upload", response_model=UploadPaperOut)
async def upload_paper(
    user: CurrentUser,
    db: DBSession,
    file: UploadFile = File(...),
) -> UploadPaperOut:
    """
    Upload and ingest a PDF immediately.

    Current scope:
      PDF -> text extraction -> chunks -> MongoDB papers/chunks (``embedding`` filled on first RAG build) -> PostgreSQL registry.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Only PDF files are accepted")

    content = await file.read()
    if not content:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded PDF is empty")

    content_hash = _sha256_bytes(content)

    existing_result = await db.execute(
        select(Paper).where(Paper.content_hash == content_hash)
    )
    existing = existing_result.scalar_one_or_none()
    if existing:
        logger.info(f"PDF already indexed: {file.filename} ({content_hash[:8]})")
        return UploadPaperOut(
            filename=existing.filename,
            size_bytes=len(content),
            status="already_indexed",
            paper_id=existing.id,
            n_chunks=existing.n_chunks,
            n_tokens=existing.n_tokens,
            source=existing.source,
            message="Paper already exists in corpus",
        )

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = _safe_filename(file.filename)
    dest = CORPUS_DIR / safe_name
    if dest.exists():
        dest = CORPUS_DIR / f"{Path(safe_name).stem}_{content_hash[:8]}.pdf"

    dest.write_bytes(content)

    logger.info(f"PDF ingestion started: {file.filename} ({len(content)} bytes)")

    processor = CorpusProcessor()
    processed = processor.process_pdf(dest)

    if not processed.success or not processed.chunks:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Could not extract usable text from PDF: {processed.error or 'no chunks'}",
        )

    paper_id = str(uuid.uuid4())
    now = datetime.utcnow()
    mongo_db = get_mongo_db()

    chunk_docs: list[dict] = []
    mongo_chunk_ids: list[str] = []

    for chunk in processed.chunks:
        chunk_hash = getattr(chunk, "content_hash", None) or _chunk_content_hash(
            content_hash, chunk.chunk_index, chunk.text
        )
        chunk_id = f"{paper_id}:{chunk.chunk_index}"
        mongo_chunk_ids.append(chunk_id)

        chunk_docs.append(
            {
                "_id": chunk_id,
                "chunk_id": chunk_id,
                "paper_id": paper_id,
                "paper_content_hash": content_hash,
                "content_hash": chunk_hash,
                "filename": processed.filename,
                "title": processed.title,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "n_tokens": chunk.n_tokens,
                "source": "uploaded",
                "source_type": "uploaded",
                "source_provider": "manual_upload",
                "metadata": {
                    **chunk.metadata,
                    "paper_id": paper_id,
                    "content_hash": chunk_hash,
                    "paper_content_hash": content_hash,
                    "source_type": "uploaded",
                    "source_provider": "manual_upload",
                },
                "created_at": now,
                "embedding": None,
                "embedding_model": None,
                "embedded_at": None,
            }
        )

    await mongo_db["papers"].update_one(
        {"content_hash": content_hash},
        {
            "$setOnInsert": {"_id": paper_id},
            "$set": {
                "paper_id": paper_id,
                "filename": processed.filename,
                "title": processed.title,
                "content_hash": content_hash,
                "n_chunks": processed.n_chunks,
                "n_tokens": processed.n_tokens,
                "source": "uploaded",
                "source_type": "uploaded",
                "source_provider": "manual_upload",
                "created_at": now,
                "indexed_at": now,
                "embedding_model": None,
                "local_path": str(dest),
            },
        },
        upsert=True,
    )

    if chunk_docs:
        await mongo_db["chunks"].bulk_write(
            [
                UpdateOne(
                    {"content_hash": doc["content_hash"]},
                    {
                        "$setOnInsert": {"_id": doc["_id"]},
                        "$set": {k: v for k, v in doc.items() if k != "_id"},
                    },
                    upsert=True,
                )
                for doc in chunk_docs
            ],
            ordered=False,
        )

    paper = Paper(
        id=paper_id,
        filename=dest.name,
        content_hash=content_hash,
        title=processed.title,
        parsed_at=now,
        n_chunks=processed.n_chunks,
        n_tokens=processed.n_tokens,
        embedding_model="none",
        indexed_at=now,
        chunk_size=processor.chunk_size,
        source="uploaded",
        mongo_chunk_ids=mongo_chunk_ids,
    )
    db.add(paper)
    await db.flush()

    logger.info(
        f"PDF ingested into corpus: {processed.filename} "
        f"({processed.n_chunks} chunks, {processed.n_tokens} tokens)"
    )

    clear_rag_cache()
    return UploadPaperOut(
        filename=paper.filename,
        size_bytes=len(content),
        status="indexed",
        paper_id=paper.id,
        n_chunks=paper.n_chunks,
        n_tokens=paper.n_tokens,
        source=paper.source,
        message="Paper indexed into corpus",
    )


@router.delete("", response_model=DeleteCorpusOut)
@router.delete("/", response_model=DeleteCorpusOut)
async def clear_corpus(user: CurrentUser, db: DBSession) -> DeleteCorpusOut:
    result = await db.execute(select(Paper))
    papers = result.scalars().all()

    mongo_db = get_mongo_db()

    # Wipe entire MongoDB collections — not just PostgreSQL-tracked papers,
    # so orphaned chunks (uploaded but never recorded in PostgreSQL) are also removed.
    chunks_result = await mongo_db["chunks"].delete_many({})
    deleted_chunks = int(chunks_result.deleted_count)
    await mongo_db["papers"].delete_many({})

    await db.execute(delete(Paper))

    if CORPUS_DIR.exists():
        for path in CORPUS_DIR.glob("*.pdf"):
            try:
                path.unlink()
            except OSError:
                logger.warning(f"Could not delete local corpus PDF: {path}")

    clear_rag_cache()
    logger.info(f"Cleared corpus: {len(papers)} papers, {deleted_chunks} chunks")

    return DeleteCorpusOut(
        status="cleared",
        deleted_papers=len(papers),
        deleted_chunks=deleted_chunks,
    )


@router.delete(
    "/{paper_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def delete_paper(paper_id: str, user: CurrentUser, db: DBSession) -> None:
    paper = await db.get(Paper, paper_id)
    if not paper:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Paper {paper_id} not found")

    mongo_db = get_mongo_db()
    await mongo_db["chunks"].delete_many({"paper_id": paper_id})
    await mongo_db["papers"].delete_many({"paper_id": paper_id})

    pdf_path = CORPUS_DIR / paper.filename
    if pdf_path.exists():
        pdf_path.unlink()

    await db.delete(paper)
    clear_rag_cache()

    logger.info(f"Deleted paper from corpus: {paper.filename}")
