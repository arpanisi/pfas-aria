# Data engineering layer

This document describes where PFAS-ARIA stores data, how it moves through ingestion and ETL, and how the API and pipeline read it back. Paths refer to the repository unless noted otherwise.

For **which ML models run and how they are trained or scored**, see [ML_DESIGN.md](./ML_DESIGN.md).

## Architecture overview

The system uses a **hybrid storage model**:

| Layer | Technology | Primary purpose |
|--------|------------|-----------------|
| **Relational** | PostgreSQL (async SQLAlchemy) | Run metadata, structured pipeline outputs, paper registry pointers, experimental segmentation, upload encodings |
| **Document / vectors** | MongoDB (Motor async for API; PyMongo sync for batch embedding / `$vectorSearch`) | PDF text chunks, per-chunk embeddings and metadata for RAG |
| **Cache** | Redis (async) | External literature API responses, ephemeral run status, short-lived DB aggregation cache |
| **Filesystem** | Project `data/` tree | Raw uploads, corpus PDFs, processed artifacts, logs, reports (see `src/utils/paths.py`) |

Configuration for default data locations lives in `configs/data_config.yaml` (e.g. `data.file_path`, `corpus.directory`). Runtime layout constants are centralized in `src/utils/paths.py` (`RAW_DIR`, `CORPUS_DIR`, `PROCESSED_DIR`, `OUTPUTS_DIR`, etc.).

---

## PostgreSQL

**Client:** `src/db/postgres.py` — async engine from `DATABASE_URL`, `AsyncSessionFactory`, `get_session()` generator used as FastAPI dependencies (`src/api/deps.py`).

**Schema:** `src/db/orm.py` — SQLAlchemy models. High-level groupings:

- **Runs and datasets**
  - `runs` — pipeline run identity, status, convergence, feature selection snapshot, link to `data_versions`.
  - `data_versions` — fingerprinted dataset metadata (filename, `content_hash`, row/column counts, column list, optional `parquet_path`, validation flags).

- **Uploads and layout**
  - `dataset_upload_encodings` — per-user (`user_sub`), per-filename JSON maps to invert `LabelEncoder` and store variable layout for Excel uploads (see `src/api/routes/pipeline.py`, `src/ingestion/upload_data_cleaning.py`).

- **Corpus registry (not the full text)**
  - `papers` — title, authors, hashes, chunk counts, embedding model, `mongo_chunk_ids` linking to MongoDB chunk documents.

- **Pipeline analytics**
  - `regimes` — regime labels per run from data intelligence.
  - `hypotheses`, `model_results`, `validation_results`, `citations` — hypothesis rounds, fits, checks, literature links.

- **Structured experimental segmentation**
  - `experiment_segmentation_batches`, `experiment_segmented_regimes`, `experiment_regime_rows`, `experiment_regime_column_stats`, `experiment_regime_regression_specs` — normalized segmentation for screening and SQL-friendly queries; populated via `src/db/experimental_segmentation_persist.py`.

- **Literature cache bookkeeping**
  - `arxiv_cache_meta`, `s2_cache_meta` — metadata rows; **payload bodies** for those APIs are cached in Redis (see below).

**Materialized views:** `src/db/postgres.py` (`create_materialized_views`) defines views such as `mv_run_convergence`, `mv_top_variables`, `mv_top_citations` for dashboard-style aggregations (refreshed as part of DB lifecycle / pipeline usage).

**Writes after pipeline work:** `src/etl/outputs/persister.py` (`OutputPersister`) commits hypotheses, models, validation, citations, and run status into PostgreSQL and triggers Redis DB cache invalidation where appropriate.

**Reads:** `src/api/routes/results.py` loads runs, hypotheses, model results, validation, citations via SQLAlchemy `select` / `db.get`; expensive aggregates can use Redis helpers `get_db_query` / `set_db_query` (`src/db/redis_client.py`). Screening and pipeline routes query segmentation and run tables in `src/api/routes/pipeline.py`.

---

## MongoDB

**Async access (API / Motor):** `src/db/mongodb.py` — `MONGO_URL`, `MONGO_DB`, `get_mongo_db()`, `get_chunks_collection()`, `get_papers_collection()`, indexes on startup (`ensure_indexes`).

**Collections (as used in code):**

- **`chunks`** — One document per text segment: `text`, optional `embedding` / `embedding_model` / `embedded_at`, `content_hash` (unique), linkage fields such as `paper_id`, `chunk_index`, and flexible `metadata`. PDF upload flow writes here; embeddings are filled by the RAG layer.

- **`papers`** — Optional mirror collection for paper-level metadata; the **canonical paper list** for the UI and domain context is still primarily the PostgreSQL `papers` table.

**Vector search:** `src/rag/vector_store.py` uses **PyMongo** against the same `MONGO_URL` / `MONGO_DB`, targeting the collection from `configs` (`vector_store.collection_name`) and MongoDB Atlas **`$vectorSearch`** on the `embedding` path (`vector_store.atlas_vector_index`). `ensure_all_chunks_embedded` batch-embeds missing or stale vectors via `src/rag/embedder.py`.

**How chunks get there:** `src/etl/corpus/processor.py` (`CorpusProcessor`) parses PDFs (e.g. with pdfplumber), splits into `ProcessedChunk` records, and bulk-writes to Mongo; `src/api/routes/corpus.py` orchestrates upload, hashing, PostgreSQL `Paper` rows, and chunk inserts (`UpdateOne` / bulk operations).

---

## Redis

**Client:** `src/db/redis_client.py` — `REDIS_URL`, singleton async client.

**Key patterns:**

| Pattern | Purpose |
|---------|---------|
| `arxiv:{id}`, `arxiv_query:{hash}` | Cached arXiv paper JSON and search results (TTL ~30d / 7d) |
| `s2:{id}`, `s2_query:{hash}` | Semantic Scholar paper and query cache |
| `run:{run_id}:status` | Live pipeline status JSON for polling / WebSockets |
| `db:{cache_key}` | Short TTL cache (~5 min) for heavy PostgreSQL read aggregates |

Invalidation helpers include `invalidate_db_cache(run_id)` and `delete_run_status`. Literature bulk clear: `flush_literature_cache`.

---

## Filesystem and ingestion

**Raw experimental files**

- UI uploads land under **`data/raw/`** (`RAW_DIR`); `src/api/routes/pipeline.py` writes bytes there and later reads by filename for preview, segmentation, and screening.
- CLI / config-driven loads use `configs/data_config.yaml` `data.file_path` (often under `data/raw/`).

**Excel / unified layout**

- `src/ingestion/unified_experimental_sheet.py` — `load_excel_bytes_with_layout` enforces the two-row “role + name” header layout and returns a normalized `DataFrame` plus `UnifiedSheetMeta` (input/output column lists, experiment id / time columns).
- CSV/TSV paths use pandas directly in the same API helpers where applicable.

**Cleaning and layout**

- `src/ingestion/upload_data_cleaning.py` — label encoding for categorical columns; `infer_layout_column_lists` uses `UnifiedSheetMeta` when present.

**Schema validation (ETL)**

- `src/etl/experimental/schema.py` (Pandera) and related loaders under `src/etl/experimental/` validate and transform tabular data for modeling.

**Corpus PDFs**

- **`data/corpus/`** (`CORPUS_DIR`) — on-disk PDFs referenced by corpus upload and processing.

**Outputs**

- **`data/outputs/`** — logs, generated reports, etc. (`LOGS_DIR`, `REPORTS_DIR` under `src/utils/paths.py`).

**Offline pipeline**

- `src/ingestion/data_loader.py` — `DataLoader` reads configured paths, supports CSV/TSV/Excel via `load_excel_bytes_with_layout` for Excel, normalizes columns, builds `DataBundle` for agents.

---

## How data is fetched (by component)

| Consumer | Storage | Mechanism |
|----------|---------|-----------|
| **FastAPI `/results/*`** | PostgreSQL (+ Redis optional) | `DBSession` → SQLAlchemy queries; some responses use `get_db_query` / `set_db_query` |
| **FastAPI `/pipeline/*`** | PostgreSQL, `RAW_DIR`, Redis | SQL for runs/segmentation; `path.read_bytes()` for files; `set_run_status` / `get_run_status` for jobs |
| **FastAPI `/corpus/*`** | PostgreSQL `Paper`, Mongo `chunks`, `CORPUS_DIR` | Upload stream → disk + hash; processor → Mongo chunks; SQL registry update |
| **RAG / screening** | Mongo chunks + embeddings | `VectorStore` `$vectorSearch`; embedder fills vectors; retriever wired from `src/rag/` |
| **Grounding (arXiv / S2)** | Redis (+ PG meta tables) | `get_arxiv_paper` / `set_arxiv_paper`, S2 analogs in `redis_client.py` |
| **Supervisor / worker** | PostgreSQL, config, files | `OutputPersister`; `DataLoader` from paths; queue worker shares DB env |

---

## Environment variables (data-related)

Typical values for deployment (see also `render.yaml` and `.env.example`):

- `DATABASE_URL` — PostgreSQL async URL (e.g. `postgresql+asyncpg://...`)
- `MONGO_URL`, `MONGO_DB` — MongoDB connection and database name
- `REDIS_URL` — Redis for caches and run status

---

## Further reading in-repo

- `src/db/orm.py` — full table list and relationships
- `src/db/postgres.py` — engine, sessions, materialized view definitions
- `src/db/mongodb.py`, `src/db/redis_client.py` — non-SQL stores
- `src/utils/paths.py` — directory contract
- `src/etl/outputs/persister.py` — pipeline → PostgreSQL write path
- `src/api/routes/pipeline.py`, `results.py`, `corpus.py` — HTTP read/write boundaries
- [ML_DESIGN.md](./ML_DESIGN.md) — algorithms used for embeddings, regression, clustering, and screening
