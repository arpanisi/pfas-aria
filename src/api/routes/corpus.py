"""
Corpus Routes.
Endpoints for managing the scientific paper corpus.
Upload PDFs, check indexing status, view paper registry.
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import select

from src.api.deps import CurrentUser, DBSession
from src.db.orm import Paper
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


@router.get("/stats", response_model=CorpusStats)
async def get_corpus_stats(user: CurrentUser, db: DBSession) -> CorpusStats:
    """Return corpus statistics and paper list."""
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


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_paper(
    user: CurrentUser,
    file: UploadFile = File(...),
) -> dict:
    """
    Upload a PDF to the corpus directory.
    The paper will be processed and indexed on the next pipeline run.
    """
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Only PDF files are accepted",
        )

    content = await file.read()
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    dest = CORPUS_DIR / file.filename
    dest.write_bytes(content)

    logger.info(f"PDF uploaded to corpus: {file.filename} ({len(content)} bytes)")

    return {
        "filename": file.filename,
        "size_bytes": len(content),
        "status": "queued_for_indexing",
        "message": "Paper will be indexed on the next pipeline run",
    }


@router.delete(
    "/{paper_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def delete_paper(paper_id: str, user: CurrentUser, db: DBSession) -> None:
    """Remove a paper from the corpus registry and delete the PDF."""
    paper = await db.get(Paper, paper_id)
    if not paper:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Paper {paper_id} not found")

    # Delete PDF file
    pdf_path = CORPUS_DIR / paper.filename
    if pdf_path.exists():
        pdf_path.unlink()

    await db.delete(paper)
    logger.info(f"Deleted paper: {paper.filename}")
