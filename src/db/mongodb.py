"""
MongoDB client.
Used for corpus chunk storage — variable-length text, metadata, no fixed schema.
Motor (async MongoDB driver) is used throughout.

Collections:
  chunks      — parsed text chunks from PDFs with full metadata
  papers      — paper-level metadata mirror (primary in PostgreSQL)
"""

from __future__ import annotations

import os

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "pfas_aria")

_client: AsyncIOMotorClient | None = None


def get_mongo_client() -> AsyncIOMotorClient:
    """Return the singleton Motor client."""
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(MONGO_URL)
    return _client


def get_mongo_db() -> AsyncIOMotorDatabase:
    """Return the pfas_aria MongoDB database."""
    return get_mongo_client()[MONGO_DB]


async def get_chunks_collection():  # type: ignore[return]
    """Return the chunks collection."""
    return get_mongo_db()["chunks"]


async def get_papers_collection():  # type: ignore[return]
    """Return the papers metadata collection."""
    return get_mongo_db()["papers"]


async def ensure_indexes() -> None:
    """Create MongoDB indexes on startup."""
    db = get_mongo_db()

    # Chunks: fast lookup by paper_id and chunk_index
    await db["chunks"].create_index([("paper_id", 1), ("chunk_index", 1)])
    await db["chunks"].create_index([("content_hash", 1)], unique=True)

    # Full-text search index on chunk text
    await db["chunks"].create_index([("text", "text")])

    # Papers: lookup by content hash and filename
    await db["papers"].create_index([("content_hash", 1)], unique=True)
    await db["papers"].create_index([("filename", 1)])
