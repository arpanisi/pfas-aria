# PFAS-ARIA Autonomous Research Intelligence Agent for PFAS Degradation Analysis 

[![Continuous Integration](https://github.com/arpanisi/pfas-aria/actions/workflows/ci.yml/badge.svg)](https://github.com/arpanisi/pfas-aria/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/arpanisi/pfas-aria/branch/main/graph/badge.svg)](https://codecov.io/gh/arpanisi/pfas-aria)
[![Security](https://img.shields.io/badge/security-Bandit%20%7C%20pip--audit%20%7C%20CodeQL%20%7C%20Semgrep-purple.svg)](.github/workflows/ci.yml)

PFAS-ARIA is a research tool for discovering which experimental conditions make hard-to-break chemicals, such as PFAS or forever chemicals degrade faster, slower, or through different pathways. PFAS are a large family of synthetic chemicals used in many everyday products and industrial processes and hence can persist in water, soil, food systems, and the human body. Thus, understanding how to break them down matters for public health and environmental cleanup.

PFAS-ARIA takes structured experimental datasets, searches for statistically meaningful predictor-outcome relationships, checks those relationships against diagnostics, and grounds the strongest candidates against uploaded and retrieved literature. The goal is to create an evidence map: which experimental conditions, treatment settings, time variables, or chemical descriptors deserve closer statistical and mechanistic review.

![PFAS-ARIA System Overview](docs/aria.png)

---

## What It Does

```text
dataset + literature
  -> data framing
  -> hypothesis screening
  -> statistical modeling
  -> robustness checks
  -> literature grounding
  -> ranked evidence for review
```

PFAS-ARIA supports two related workflows:

- **Screening workflow:** upload a dataset, identify input/output columns, test many candidate relationships, rank the strongest results, and view evidence in the UI.
- **Grounding workflow:** compare statistically strong candidates with uploaded PDFs and external literature sources, then use guarded LLM summaries to make the evidence readable.

The system is designed for expert review. It should be used to prioritize follow-up analysis, not to replace domain judgment.

---

## Data Framing

The active upload and screening path is conservative about missing data:

- blank cells remain missing
- placeholder values such as `-`, `N/A`, and `ND` become missing values
- missing values are not silently converted to zero
- the upload preview displays missing values as blank cells
- categorical columns are encoded only when they are genuinely categorical
- numeric-looking columns with placeholder strings are coerced to numeric with missing cells preserved
- input/output roles from unified Excel sheets are preserved for screening

This matters because zero and missing are different scientific claims. A zero may mean a measured absence; a blank may mean not measured, not applicable, below detection, or unavailable.

Some preview columns may look empty because the first sampled rows are missing even though values exist later in the file. The preview is a cleaned sample, not a full missingness report.

The repository also contains a stricter experimental ETL path that can validate data, drop high-missingness columns, median-impute numeric features, create log transforms, and cache processed parquet files. That path exists as groundwork, but it is not the main upload/screening behavior today. Any imputation should be treated as a modeling assumption and tested through sensitivity analysis.

See [docs/ML_DESIGN.md](docs/ML_DESIGN.md) for the statistical framing.

---

## Statistical Modeling

The screening system fits many small, interpretable models rather than one large black-box model.

Primary models include:

- ordinary least squares for interpretable association screening
- ridge regression fallback when OLS is singular or numerically unstable
- panel/fixed-effect style modeling when repeated observations and time/entity structure are available

The broader modeling code also includes additional model families for robustness and comparison, including LASSO, elastic net, random forest, gradient boosting, XGBoost, random effects, two-stage kinetics/treatment models, grouped fixed effects, and mixture-of-regressions research modules.

Candidates are judged using statistical fit, diagnostics, and literature resemblance. A high R2 alone is not enough; strong candidates should also survive basic robustness checks and have relevant literature support.

---

## Literature Grounding and LLMs

The LLM layer is used for explanation, not statistical inference.

The system computes model results, diagnostics, retrieval scores, and citation matches first. LLMs are then asked to turn structured evidence into concise scientific language. Guardrails reject malformed, generic, prompt-like, or unsupported text and fall back to deterministic wording when needed.

The project favors free or open-source model routes where possible, mainly through OpenRouter free models, with optional local Ollama support and an unused-but-supported OpenAI-compatible endpoint path for hosted models such as RunPod.

See [docs/LLM_GROUNDING.md](docs/LLM_GROUNDING.md) for grounding details, model-provider notes, and guardrails.

---

## Main Components

| Area | Purpose |
|---|---|
| `frontend/` | React UI for upload, configuration, screening results, corpus management, and run history. |
| `src/api/` | FastAPI routes for dataset upload, screening, grounding jobs, results, and corpus APIs. |
| `src/ingestion/` | Active upload parsing, unified Excel layout handling, column normalization, and categorical encoding. |
| `src/pipeline/` | Screening and grounded candidate generation. |
| `src/modeling/` | Statistical model implementations and diagnostics. |
| `src/rag/` | Embeddings, retrieval, vector search, and literature matching. |
| `src/reporting/` | Narrative cleanup, guarded summaries, charts, and reports. |
| `src/etl/experimental/` | Stricter ETL groundwork; useful but not the current upload/screening default. |
| `docs/` | Design notes for ML/statistics, data engineering, and LLM grounding. |

---

## Local Setup

Prerequisites:

- Python 3.11+
- Node.js 18+
- Docker, if using the local Postgres/Mongo/Redis stack

```bash
git clone https://github.com/arpanisi/pfas-aria.git
cd pfas-aria

make setup
cp .env.example .env
```

Edit `.env` for local database URLs, auth settings, model-provider keys, and any optional API credentials.

Start local infrastructure:

```bash
make db-up
```

Run the backend:

```bash
make run-api
```

Run the frontend:

```bash
make run-frontend
```

The frontend dev server is started by Vite, usually at `http://localhost:5173`.

---

## Typical UI Workflow

1. Upload an experimental dataset.
2. Confirm the inferred input and output columns.
3. Review the cleaned preview and missing-value behavior.
4. Upload literature PDFs to the corpus.
5. Run screening and grounding.
6. Review ranked hypotheses, model diagnostics, citations, and generated summaries.

Unified Excel workbooks can declare variable roles using the two-row role/name layout. CSV and TSV uploads are also supported.

---

## Useful Commands

```bash
make install        # Install Python dependencies
make install-dev    # Install Python dev tools and pre-commit hooks
make db-up          # Start Postgres, MongoDB, and Redis
make db-down        # Stop local database services
make run-api        # Start FastAPI backend
make run-frontend   # Start React frontend
make test           # Run backend tests
make lint           # Run Ruff on Python code
make frontend-check # Install, lint, build, and audit frontend
```

Frontend-only checks:

```bash
cd frontend
npm run lint
npm run build
```

---

## Configuration

Most behavior is controlled through `configs/` and `.env`.

| File | Controls |
|---|---|
| `configs/data_config.yaml` | Config-driven data path, outcome variable, and data defaults. |
| `configs/agent_config.yaml` | Agent loop and screening settings. |
| `configs/model_config.yaml` | LLM provider, embeddings, and model routing. |
| `configs/pipeline_config.yaml` | Pipeline phases and logging behavior. |

The UI upload path stores raw uploaded datasets in `data/raw/`. Uploaded papers live under `data/corpus/` and are indexed for retrieval.

---

## Documentation

- [ML design](docs/ML_DESIGN.md): data framing, models, robustness checks, and statistical interpretation.
- [LLM grounding](docs/LLM_GROUNDING.md): free/open model strategy, guardrails, and literature scoring.
- [Data engineering](docs/DATA_ENGINEERING.md): storage, databases, ingestion paths, and API data flow.

---

## Current Status

Implemented:

- dataset upload and preview
- unified Excel layout parsing
- conservative missing-value handling
- categorical encoding for genuine categorical variables
- screening over input/output candidates
- OLS/ridge/panel-style modeling paths
- diagnostics and ranking
- PDF corpus upload and vector retrieval
- external literature retrieval hooks
- guarded LLM summaries and fallbacks
- React/FastAPI application shell

Available but not fully promoted as the default path:

- stricter experimental ETL with validation, high-missingness dropping, median imputation, log transforms, and parquet caching
- broader model-comparison modules used for robustness or future expansion
- optional hosted open-weight inference through OpenAI-compatible endpoints

---

## License

MIT.
