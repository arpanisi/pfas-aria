# ML design: what runs and how

This document is about **machine learning and statistical learning as implemented in code**: which libraries and algorithms are used, where `fit` / `encode` / inference happen, and how vectors enter retrieval. It is not about databases or file paths; see [DATA_ENGINEERING.md](./DATA_ENGINEERING.md) for storage and ingestion.

**Out of scope here:** Large language models (OpenRouter, etc.) are used as **remote APIs** for text generation; their weights are not trained or fine-tuned in this repository.

---

## 1. Dense text embeddings (SentenceTransformers)

**Library:** `sentence_transformers.SentenceTransformer` (`src/rag/embedder.py`).

**Model:** `configs/model_config.yaml` → `embeddings.model` (default `all-MiniLM-L6-v2`), `embeddings.device` (e.g. `cpu`).

**How encoding works**

- `Embedder.embed(texts)` calls `SentenceTransformer.encode(..., normalize_embeddings=True, batch_size=32)` and returns a NumPy matrix `(n_texts, dim)`. Normalization makes **dot product equal to cosine similarity** between query and chunk vectors.
- `embed_one` is a single-row wrapper.

**Where embeddings are produced**

| Step | Code | What happens |
|------|------|----------------|
| Warm / singleton | `get_embedder()` | Loads the transformer once and reuses it. |
| Backfill chunk vectors | `VectorStore.ensure_all_chunks_embedded` (`src/rag/vector_store.py`) | Finds Mongo `chunks` missing `embedding` or wrong `embedding_model`; batches texts through `embedder.embed`; writes float vectors + model name via `bulk_write`. |
| RAG pipeline sync | `src/rag/pipeline.py` | Triggers the above so the retriever has vectors. |

**How retrieval uses them**

| Step | Code | What happens |
|------|------|----------------|
| Vector search | `VectorStore.search` | `embed_one(query)` → query vector; MongoDB aggregation **`$vectorSearch`** on field `embedding` (Atlas index from config `vector_store.atlas_vector_index`); returns top chunks with scores. |
| Agent-facing retrieval | `Retriever` (`src/rag/retriever.py`) | Delegates to `VectorStore`; optional Redis serialization cache for repeated identical queries. |
| Screening literature score | `screening_grounded.py` | Builds short strings from column names; `embed_one` / batched `embed` for queries; similarity against corpus or external hits (see that file for arXiv/S2 scoring paths). |
| Grounding score vs corpus | `GroundingScorer` (`src/grounding/scorer.py`) | Embeds a “finding” sentence and candidate paper snippets; **`batch_similarity`** is a single matrix multiply `corpus @ query` because vectors are normalized. |

So the **only neural “model” trained offline and shipped with the app** in the usual sense is the **pretrained sentence-transformer** used in inference mode (weights loaded from HuggingFace / cache, not updated by PFAS-ARIA training loops).

---

## 2. Supervised regression (main agent pipeline)

**Orchestration:** `ModelingRunner` (`src/modeling/runner.py`) loops hypotheses → **`ModelingEngine.run`** → **`ValidationEngine.validate`** → picks **`best_model`** by R² among fits that pass validation.

**Design matrix construction:** `ModelingEngine._prepare_data` (`src/modeling/engine.py`)

- Subsets columns from the run’s DataFrame, `dropna`, casts outcome to float.
- **`pandas.get_dummies(..., drop_first=True)`** for categoricals.
- Applies **log / sqrt** transforms when the hypothesis requests them and values are in domain.
- Adds **multiplicative interaction** columns for `(var1, var2)` pairs.

**Fitted estimators (each is `.fit` on numeric design, then coefficients / R² extracted into `ModelResult`):**

| `hypothesis.model_family` | Underlying call | Notes |
|---------------------------|-----------------|--------|
| `ols` | `statsmodels.api.OLS(y, add_constant(X)).fit()` | Classical linear regression with inference. |
| `fixed_effects` | `linearmodels.panel.PanelOLS(..., entity_effects=True).fit(cov_type="clustered", ...)` | Needs entity + time columns heuristically detected on the frame; **on failure → OLS**. |
| `random_effects` | `linearmodels.panel.RandomEffects(...).fit()` | Same panel index build; **on failure → OLS**. |
| `lasso` | `StandardScaler` + **`sklearn.linear_model.LassoCV(cv=5)`** `.fit` / `.predict` | L1 path with CV for `alpha`; approximate p-values for reporting layer. |
| `gradient_boosting` | **`sklearn.ensemble.GradientBoostingRegressor`** | Nonlinear tree ensemble on the same design matrix. |

**Per-regime:** If the frame has a `regime` column and `per_regime=True`, the engine fits the same family **once globally** and **once per regime label** (skips tiny groups).

Allowed family names for the hypothesis agent come from **`configs/agent_config.yaml`** → `modeling.allowed_models`.

---

## 3. Post-fit checks (statistical tests on residuals / design)

**Module:** `src/validation/validator.py` — not “training” new models, but **scoring the fitted linear structure**:

- **VIF** via auxiliary OLS loops on columns of the design matrix.
- **Shapiro–Wilk** (`scipy.stats`) on residuals.
- **Breusch–Pagan**-style homoscedasticity using regression on squared residuals.
- **ANOVA** across regimes when a regime column exists.
- **K-fold cross-validation** using **`sklearn.linear_model.LinearRegression`** and **`sklearn.model_selection.cross_val_score`** on a prediction-focused check.
- Effect-size style summaries from variance decomposition.

These gates decide **`ValidationReport.overall_passed`** and therefore whether a fit can become **`best_model`**.

---

## 4. Unsupervised structure on the experimental table (regimes)

**Module:** `src/agents/data_intelligence.py` — labels rows for downstream `regime` column. Methods are configured under **`configs/agent_config.yaml`** → `data_intelligence.regime_methods`.

| Method | Library / API | What it does |
|--------|----------------|--------------|
| PELT changepoints | **`ruptures`** (`rpt.Pelt`) | 1D signal segmentation (ordered subsamples). |
| HDBSCAN | **`hdbscan.HDBSCAN`** | Density-based clustering on scaled numeric features (`StandardScaler` then clusterer). |
| K-means style regimes | **`sklearn.cluster.KMeans`**, **`sklearn.metrics.silhouette_score`** | Partition scaled space; silhouette used in logic for quality / choice. |
| Label encoding for non-numeric keys | **`sklearn.preprocessing.LabelEncoder`** | Where categorical experiment keys need integer codes for clustering. |

These routines **assign labels** to rows; they do not replace the supervised models in section 2.

---

## 5. Upload-time categorical encoding (not hypothesis models)

**Module:** `src/ingestion/upload_data_cleaning.py` — **`sklearn.preprocessing.LabelEncoder`** per eligible string column so numeric pipelines see integers; maps are persisted for inversion (see data engineering doc).

---

## 6. Automated screening (many shallow fits for exploration / UI)

**Module:** `src/pipeline/automated_screening.py`.

**Pattern:** `sklearn.pipeline.Pipeline` with **`StandardScaler`** then a regressor, over a combinatorial search of small predictor subsets and numeric outputs (and regime segments from layout helpers).

**Regressors used (via `Pipeline(...).fit`):**

- `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`
- `GradientBoostingRegressor`, `RandomForestRegressor`

Guards enforce minimum `n`, multicollinearity caps (`_corr_ok`), and panel-style design helpers where applicable. Counts of fits feed “hypotheses tested” style metrics in the UI.

---

## 7. Grounded screening (OLS + Ridge fallback + retrieval scoring)

**Module:** `src/pipeline/screening_grounded.py`.

**Primary fit:** `statsmodels.OLS` with constant on prepared `(X, y)`; on numerical failure, **`sklearn.linear_model.Ridge`** for point estimates and a coarse R².

**Literature side:** Retrieval and embedding calls (section 1) score how well each candidate’s implied query matches the corpus (and optional arXiv/S2 branches in the same file). Ranking combines **in-sample R²** and **similarity-style literature scores**.

---

## 8. Standalone research modules (not wired into `ModelingEngine`)

These implement additional ML/stats ideas but are **not** selected by `hypothesis.model_family` today:

| Module | Technique |
|--------|-----------|
| `src/modeling/mixture_of_regressions.py` | **EM algorithm**: softmax gating (`scipy.special.softmax`) over regimes, weighted least-squares style M-steps, BIC for `K`. |
| `src/modeling/grouped_fixed_effects.py` | Iterative **KMeans** on residual trajectories + augmented design OLS for group–time effects (Bonhomme–Manresa–style loop). |

Use them from notebooks or future pipeline hooks if you wire them in.

---

## Quick file index

| Concern | File |
|---------|------|
| Transformer inference | `src/rag/embedder.py` |
| Chunk vectors + `$vectorSearch` | `src/rag/vector_store.py` |
| Retriever + cache | `src/rag/retriever.py` |
| Hypothesis-driven fits | `src/modeling/engine.py` |
| Fit → validate → best model | `src/modeling/runner.py` |
| Residual / VIF / CV tests | `src/validation/validator.py` |
| Regime clustering / changepoints | `src/agents/data_intelligence.py` |
| Screening grids | `src/pipeline/automated_screening.py` |
| OLS/Ridge + lit ranking | `src/pipeline/screening_grounded.py` |
| MoR / grouped FE (standalone) | `src/modeling/mixture_of_regressions.py`, `src/modeling/grouped_fixed_effects.py` |
