"""
Corpus Paper Registry.
Tracks every processed PDF by content hash in PostgreSQL.
Prevents re-processing papers already indexed in ChromaDB.
Handles deduplication by content hash — same paper uploaded twice is safe.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.orm import Paper
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PaperRegistry:
    """
    PostgreSQL-backed registry of all processed corpus papers.
    The single source of truth for what has been indexed.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def is_processed(self, pdf_path: Path) -> bool:
        """Check if a PDF has already been processed by content hash."""
        content_hash = self._hash_file(pdf_path)
        result = await self._session.execute(
            select(Paper).where(Paper.content_hash == content_hash)
        )
        return result.scalar_one_or_none() is not None

    async def register(
        self,
        pdf_path: Path,
        title: str,
        n_chunks: int,
        n_tokens: int,
        embedding_model: str,
        chunk_size: int,
        mongo_chunk_ids: list[str],
        paper_id: str | None = None,
        source: str = "corpus",
    ) -> Paper:
        """Register a newly processed paper."""
        content_hash = self._hash_file(pdf_path)

        paper_kwargs = {
            "filename": pdf_path.name,
            "content_hash": content_hash,
            "title": title,
            "parsed_at": datetime.utcnow(),
            "n_chunks": n_chunks,
            "n_tokens": n_tokens,
            "embedding_model": embedding_model,
            "indexed_at": datetime.utcnow(),
            "chunk_size": chunk_size,
            "source": source,
            "mongo_chunk_ids": mongo_chunk_ids,
        }
        if paper_id is not None:
            paper_kwargs["id"] = paper_id

        paper = Paper(**paper_kwargs)
        self._session.add(paper)
        await self._session.flush()

        logger.info(
            f"Registered paper: {pdf_path.name} "
            f"({n_chunks} chunks, hash={content_hash[:8]})"
        )
        return paper

    async def get_all(self) -> list[Paper]:
        """Return all registered papers."""
        result = await self._session.execute(select(Paper))
        return list(result.scalars().all())

    async def get_unembedded(self, embedding_model: str) -> list[Paper]:
        """Return papers not yet embedded with the current model."""
        result = await self._session.execute(
            select(Paper).where(
                (Paper.embedding_model != embedding_model)
                | (Paper.indexed_at.is_(None))
            )
        )
        return list(result.scalars().all())

    async def count(self) -> int:
        """Total number of registered papers."""
        result = await self._session.execute(select(Paper))
        return len(result.scalars().all())

    @staticmethod
    def _hash_file(path: Path) -> str:
        """SHA-256 content hash of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
