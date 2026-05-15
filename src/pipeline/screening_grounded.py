"""
Rank automated screening fits by statistical fit and corpus similarity.

Collects OLS-style results for input subsets × numeric outputs in one screening
segment (``regime_id`` in the API), scores each candidate against the RAG
vector store, and returns bundles for the static UI.
"""

from __future__ import annotations

import secrets
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge

from src.grounding.arxiv_client import ArxivClient
from src.grounding.crossref_client import CrossrefClient
from src.grounding.europe_pmc_client import EuropePMCClient
from src.grounding.openalex_client import OpenAlexClient
from src.grounding.semantic_scholar import SemanticScholarClient
from src.ingestion.dataset_screening_layout import (
    assign_screening_layout_from_column_lists,
    assign_screening_layout_from_legacy_column_slices,
)
from src.ingestion.unified_experimental_sheet import UnifiedSheetMeta
from src.pipeline.automated_screening import (
    _build_panel_design,
    _corr_ok,
    _detect_panel_columns,
    _numeric_output_names,
    _prepare_xy,
    _subset_iterator,
    screening_numeric_input_pool,
)
from src.rag.embedder import get_embedder
from src.rag.retriever import Retriever
from src.utils.logging import get_logger

logger = get_logger(__name__)

TOP_POOL = 72
FINAL_N = 6
STATS_TOP_N = 12
MIN_R2 = 0.06
MIN_LIT = 0.30
LIT_QUERY_MIN_SIM = 0.18
EXT_CIT_MIN_SCORE = 0.45  # discard arXiv/S2 hits below this cosine score
MAX_EXTERNAL_CITATIONS = 2


@dataclass
class _Candidate:
    y_col: str
    x_cols: list[str]
    r_squared: float
    adj_r_squared: float
    n_obs: int
    coefficients: dict[str, float]
    p_values: dict[str, float]
    significant_variables: list[str]
    lit_score: float


def _fit_ols_or_ridge(
    x_mat: np.ndarray, y_vec: np.ndarray, names: list[str]
) -> tuple[float, float, dict[str, float], dict[str, float], list[str]]:
    """Return r2, adj_r2, coefs (no const), pvals, significant names."""
    if np.std(y_vec) == 0:
        return 0.0, 0.0, {n: 0.0 for n in names}, {n: 1.0 for n in names}, []
    df_x = pd.DataFrame(x_mat, columns=names)
    try:
        xc = sm.add_constant(df_x, has_constant="add")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            res = sm.OLS(y_vec, xc).fit()
        r2 = float(res.rsquared)
        ar2 = float(res.rsquared_adj)
        coefs = {k: float(v) for k, v in res.params.items() if k != "const"}
        pvals = {k: float(v) for k, v in res.pvalues.items() if k != "const"}
        sig = [k for k, p in pvals.items() if p < 0.05]
        return r2, ar2, coefs, pvals, sig
    except Exception as e:  # noqa: BLE001
        logger.debug("OLS failed ({}), using Ridge for point estimates", e)
        ridge = Ridge(alpha=1.0, random_state=0).fit(x_mat, y_vec)
        r2 = float(ridge.score(x_mat, y_vec))
        coefs = {n: float(c) for n, c in zip(names, ridge.coef_, strict=True)}
        pvals = {n: 0.5 for n in names}
        sig = [n for n, c in coefs.items() if abs(c) > 1e-8][: max(1, len(names))]
        return r2, r2, coefs, pvals, sig


def _humanize(col: str) -> str:
    """Minimal column name humanization for readable queries without LLM."""
    s = col.replace("_", " ").strip()
    for unit in ("mg l", "mg kg", "ug l", "ng l", "pct", "percent"):
        s = s.replace(unit, "").strip()
    return s


def _llm_grounding_query(y_col: str, x_cols: list[str]) -> str:
    """Build a natural-language retrieval query from column names without an LLM call.

    Running LLM here was blocking the sync executor for up to 6 × 15 s. The
    deterministic query is indistinguishable for cosine-similarity retrieval.
    """
    predictors = ", ".join(_humanize(c) for c in x_cols[:5])
    outcome = _humanize(y_col)
    return f"{outcome} as a function of {predictors}"


def _external_search_query(y_col: str, x_cols: list[str]) -> str:
    """Build a keyword-optimised query for arXiv / Semantic Scholar.

    The embedding query is too verbose for keyword APIs — "concentration c4 as a
    function of applied voltage kv, gas flowrate" sends arXiv into physics papers.
    This function extracts chemical species and process type from column names so
    the search stays in the right scientific domain.
    """
    cols = " ".join([y_col] + list(x_cols)).lower()

    terms: list[str] = []

    # ── Chemical species ──────────────────────────────────────────────────────
    species: list[str] = []
    if "pfoa" in cols or ("c8" in cols and "concentration" in cols):
        species.append("PFOA")
    if "pfos" in cols:
        species.append("PFOS")
    if "pfhxa" in cols or ("c6" in cols and "concentration" in cols):
        species.append("PFHxA")
    if "pfna" in cols or ("c9" in cols and "concentration" in cols):
        species.append("PFNA")
    if "pfbs" in cols or ("c4" in cols and "concentration" in cols):
        species.append("PFBS")
    if "pfhxs" in cols or "c6s" in cols:
        species.append("PFHxS")
    # Always include generic PFAS alongside any specific compound —
    # improves recall on arXiv/S2 where specific short-chain compounds are less indexed.
    if any(t in cols for t in ("pfas", "pfaa", "perfluoro", "fluoride", "fluoriode")) or species:
        species.append("PFAS")
    terms.extend(dict.fromkeys(species))  # deduplicate, preserve order

    # ── Treatment / process ───────────────────────────────────────────────────
    if any(t in cols for t in ("voltage", "discharge", "plasma", "gas_flow", "gas_used", "flowrate", "kv")):
        terms.append("cold atmospheric plasma")
    if any(t in cols for t in ("h2o2", "peroxide")):
        terms.append("hydrogen peroxide")
    if any(t in cols for t in ("uv", "lamp", "fluence")):
        terms.append("UV photolysis")
    if any(t in cols for t in ("ozone", " o3 ")):
        terms.append("ozonation")
    if any(t in cols for t in ("sonication", "ultrasound", "acoustic")):
        terms.append("sonochemical")
    if any(t in cols for t in ("electrochemical", "electrolysis", "current", "electrode")):
        terms.append("electrochemical oxidation")

    # ── Outcome type ──────────────────────────────────────────────────────────
    if any(t in cols for t in ("fluoride", "fluoriode", "defluor")):
        terms.append("defluorination")
    elif any(t in cols for t in ("removal", "degradation", "concentration")):
        terms.append("degradation")

    # ── Matrix — always include for PFAS (always water context) ──────────────
    if species or any(t in cols for t in ("water", "solution", "aqueous", "ph", "conductivity")):
        terms.append("water treatment")

    if not terms:
        # Last resort: fall back to the compact humanized form
        return _llm_grounding_query(y_col, x_cols)

    return " ".join(terms)


def _literature_score(retriever: Retriever, y_col: str, x_cols: list[str]) -> float:
    # Use humanized (no LLM) query here — this is called for every candidate so must be fast.
    xs = ", ".join(_humanize(x) for x in x_cols[:6])
    outcome = _humanize(y_col)
    query = f"Effect of {xs} on {outcome}."
    try:
        chunks = retriever.retrieve(
            query=query, top_k=8, min_similarity=LIT_QUERY_MIN_SIM
        )
        if not chunks:
            return 0.0
        return float(max(c.similarity_score for c in chunks))
    except Exception as e:  # noqa: BLE001
        logger.warning("Literature retrieval failed: {}", e)
        return 0.0


def run_screening_grounded_for_regime(
    df: pd.DataFrame,
    *,
    unified_meta: UnifiedSheetMeta | None,
    regime_id: int,
    retriever: Retriever,
    input_start: int = 2,
    input_end: int = 37,
    output_start: int = 37,
) -> dict:
    """
    Build ranked hypothesis bundles for one screening segment (API ``regime_id``).

    Returns a dict with keys matching the static screening API schema.
    """
    if unified_meta is not None:
        res = assign_screening_layout_from_column_lists(
            df,
            unified_meta.input_cols,
            unified_meta.output_cols,
            replace_no_in_inputs=True,
        )
    else:
        res = assign_screening_layout_from_legacy_column_slices(
            df,
            input_start=input_start,
            input_end=input_end,
            output_start=output_start,
            replace_no_in_inputs=True,
        )

    rid = int(regime_id)
    regimes = res.regimes
    if rid not in regimes:
        return {
            "regime_id": rid,
            "regime_n_rows": 0,
            "bundles": [],
            "warnings": [f"Regime {rid} has no rows in this dataset."],
        }

    sub_df = regimes[rid]
    exp_col, time_col = _detect_panel_columns(df)

    candidates: list[_Candidate] = []
    sizes = (2, 3, 5)
    max_subsets_per_k = 36

    if len(sub_df) < 12:
        return {
            "regime_id": rid,
            "regime_n_rows": len(sub_df),
            "bundles": [],
            "warnings": ["Not enough rows in this regime for screening (minimum 12)."],
        }

    pool = screening_numeric_input_pool(sub_df, res.input_cols)
    if len(pool) < 2:
        return {
            "regime_id": rid,
            "regime_n_rows": len(sub_df),
            "bundles": [],
            "warnings": [
                "This segment has fewer than two numeric input columns with enough "
                "varying observations for automated screening."
            ],
        }

    num_block = sub_df[pool].apply(pd.to_numeric, errors="coerce")
    corr = num_block.corr(numeric_only=True).abs()
    outs = _numeric_output_names(sub_df, res.output_cols)

    for y_col in outs:
        is_panel = (
            exp_col is not None
            and time_col is not None
            and exp_col in sub_df.columns
            and time_col in sub_df.columns
        )
        for x_subset in _subset_iterator(pool, sizes, max_per_size=max_subsets_per_k):
            if not _corr_ok(x_subset, corr):
                continue
            prep = _prepare_xy(sub_df, x_subset, y_col)
            if prep is None:
                continue
            x_mat, y_vec = prep
            r2, ar2, coefs, pvals, sig = _fit_ols_or_ridge(x_mat, y_vec, x_subset)
            if r2 < MIN_R2:
                continue
            # Optional panel augmentation: require panel design only for tagging
            _ = is_panel and _build_panel_design(
                sub_df, x_subset, y_col, exp_col or "", time_col or ""
            )
            lit = _literature_score(retriever, y_col, x_subset)
            candidates.append(
                _Candidate(
                    y_col=y_col,
                    x_cols=list(x_subset),
                    r_squared=r2,
                    adj_r_squared=ar2,
                    n_obs=int(len(y_vec)),
                    coefficients=coefs,
                    p_values=pvals,
                    significant_variables=sig,
                    lit_score=lit,
                )
            )

    candidates.sort(key=lambda c: c.r_squared, reverse=True)
    pool_c = candidates[:TOP_POOL]

    ranked = sorted(
        pool_c,
        key=lambda c: (c.lit_score * c.r_squared, c.lit_score, c.r_squared),
        reverse=True,
    )
    strong = [c for c in ranked if c.lit_score >= MIN_LIT]
    chosen = strong[:FINAL_N] if len(strong) >= 2 else ranked[:FINAL_N]

    bundles: list[dict] = []
    for i, c in enumerate(chosen, start=1):
        hid = f"H{i}"
        oid = secrets.token_hex(4)
        mid = secrets.token_hex(4)
        desc = (
            f"{' · '.join(c.x_cols[:6])}{' …' if len(c.x_cols) > 6 else ''} "
            f"jointly associate with {c.y_col} in regime {rid} "
            f"(n={c.n_obs} observations)."
        )
        rationale = (
            f"Automated screening fit (OLS or ridge fallback) with in-sample R² "
            f"{c.r_squared:.3f}. Corpus embedding resemblance peak "
            f"{c.lit_score:.0%} for a query built from this outcome and predictors."
        )
        bundles.append(
            {
                "hypothesis": {
                    "id": oid,
                    "hypothesis_id": hid,
                    "round": 1,
                    "description": desc,
                    "rationale": rationale,
                    "primary_variables": c.x_cols[:10],
                    "model_family": "screening_ols",
                    "priority_score": round(min(1.0, c.lit_score), 4),
                    "is_refinement": False,
                },
                "model_result": {
                    "id": mid,
                    "hypothesis_id": hid,
                    "model_type": "screening_linear",
                    "r_squared": round(c.r_squared, 4),
                    "adj_r_squared": round(c.adj_r_squared, 4),
                    "n_observations": c.n_obs,
                    "coefficients": {k: round(v, 6) for k, v in c.coefficients.items()},
                    "p_values": {k: round(v, 6) for k, v in c.p_values.items()},
                    "significant_variables": c.significant_variables,
                    "match_score": round(min(1.0, c.lit_score), 4),
                    "validation_passed": c.r_squared >= MIN_R2
                    and c.lit_score >= MIN_LIT,
                },
                "citations": _citations_for_candidate(retriever, c, hid),
            }
        )

    warnings: list[str] = []
    if not candidates:
        warnings.append(
            "No passing screening fits for this segment (check numeric inputs and outputs)."
        )
    elif not strong and chosen:
        warnings.append(
            "Corpus similarity was weak for most fits; showing top statistical fits anyway."
        )

    return {
        "regime_id": rid,
        "regime_n_rows": len(sub_df),
        "bundles": bundles,
        "warnings": warnings,
    }


def run_screening_stats_for_regime(
    df: pd.DataFrame,
    *,
    unified_meta: UnifiedSheetMeta | None,
    regime_id: int,
    input_start: int = 2,
    input_end: int = 37,
    output_start: int = 37,
) -> dict:
    """
    Build hypothesis bundles ranked by R² only — no RAG or corpus required.

    Returns up to ``STATS_TOP_N`` bundles for the static stats UI.
    """
    if unified_meta is not None:
        res = assign_screening_layout_from_column_lists(
            df,
            unified_meta.input_cols,
            unified_meta.output_cols,
            replace_no_in_inputs=True,
        )
    else:
        res = assign_screening_layout_from_legacy_column_slices(
            df,
            input_start=input_start,
            input_end=input_end,
            output_start=output_start,
            replace_no_in_inputs=True,
        )

    rid = int(regime_id)
    regimes = res.regimes
    if rid not in regimes:
        return {
            "regime_id": rid,
            "regime_n_rows": 0,
            "bundles": [],
            "warnings": [f"Regime {rid} has no rows in this dataset."],
        }

    sub_df = regimes[rid]

    if len(sub_df) < 12:
        return {
            "regime_id": rid,
            "regime_n_rows": len(sub_df),
            "bundles": [],
            "warnings": ["Not enough rows in this regime for screening (minimum 12)."],
        }

    pool = screening_numeric_input_pool(sub_df, res.input_cols)
    if len(pool) < 2:
        return {
            "regime_id": rid,
            "regime_n_rows": len(sub_df),
            "bundles": [],
            "warnings": [
                "This segment has fewer than two numeric input columns with enough "
                "varying observations for automated screening."
            ],
        }

    num_block = sub_df[pool].apply(pd.to_numeric, errors="coerce")
    corr = num_block.corr(numeric_only=True).abs()
    outs = _numeric_output_names(sub_df, res.output_cols)

    candidates: list[_Candidate] = []
    sizes = (2, 3, 5)
    max_subsets_per_k = 36

    for y_col in outs:
        for x_subset in _subset_iterator(pool, sizes, max_per_size=max_subsets_per_k):
            if not _corr_ok(x_subset, corr):
                continue
            prep = _prepare_xy(sub_df, x_subset, y_col)
            if prep is None:
                continue
            x_mat, y_vec = prep
            r2, ar2, coefs, pvals, sig = _fit_ols_or_ridge(x_mat, y_vec, x_subset)
            if r2 < MIN_R2:
                continue
            candidates.append(
                _Candidate(
                    y_col=y_col,
                    x_cols=list(x_subset),
                    r_squared=r2,
                    adj_r_squared=ar2,
                    n_obs=int(len(y_vec)),
                    coefficients=coefs,
                    p_values=pvals,
                    significant_variables=sig,
                    lit_score=0.0,
                )
            )

    candidates.sort(key=lambda c: c.r_squared, reverse=True)
    chosen = candidates[:STATS_TOP_N]

    bundles: list[dict] = []
    for i, c in enumerate(chosen, start=1):
        hid = f"H{i}"
        oid = secrets.token_hex(4)
        mid = secrets.token_hex(4)
        desc = (
            f"{' · '.join(c.x_cols[:6])}{' …' if len(c.x_cols) > 6 else ''} "
            f"jointly associate with {c.y_col} in regime {rid} "
            f"(n={c.n_obs} observations)."
        )
        bundles.append(
            {
                "y_col": c.y_col,
                "hypothesis": {
                    "id": oid,
                    "hypothesis_id": hid,
                    "round": 1,
                    "description": desc,
                    "rationale": None,
                    "primary_variables": c.x_cols[:10],
                    "model_family": "screening_ols",
                    "priority_score": round(c.r_squared, 4),
                    "is_refinement": False,
                },
                "model_result": {
                    "id": mid,
                    "hypothesis_id": hid,
                    "model_type": "screening_linear",
                    "r_squared": round(c.r_squared, 4),
                    "adj_r_squared": round(c.adj_r_squared, 4),
                    "n_observations": c.n_obs,
                    "coefficients": {k: round(v, 6) for k, v in c.coefficients.items()},
                    "p_values": {k: round(v, 6) for k, v in c.p_values.items()},
                    "significant_variables": c.significant_variables,
                    "match_score": 0.0,
                    "validation_passed": c.r_squared >= MIN_R2,
                },
            }
        )

    warnings: list[str] = []
    if not candidates:
        warnings.append(
            "No passing screening fits for this segment (check numeric inputs and outputs)."
        )

    return {
        "regime_id": rid,
        "regime_n_rows": len(sub_df),
        "bundles": bundles,
        "warnings": warnings,
    }


def run_grounding_from_precomputed(
    candidate_rows: list[dict],
    retriever: Retriever,
    *,
    regime_id: int,
    regime_n_rows: int,
) -> dict:
    """Grounding pass using pre-computed OLS candidates (skips fitting entirely).

    ``candidate_rows`` is a list of dicts with keys matching ``_Candidate`` fields
    plus ``y_col`` and ``x_cols`` (as stored in DB rag_support metadata).
    """
    candidates: list[_Candidate] = []
    for row in candidate_rows:
        candidates.append(
            _Candidate(
                y_col=row["y_col"],
                x_cols=list(row["x_cols"]),
                r_squared=float(row["r_squared"]),
                adj_r_squared=float(row["adj_r_squared"]),
                n_obs=int(row["n_obs"]),
                coefficients={k: float(v) for k, v in row["coefficients"].items()},
                p_values={k: float(v) for k, v in row["p_values"].items()},
                significant_variables=list(row["significant_variables"]),
                lit_score=0.0,
            )
        )

    # Batch-embed all lit-score queries in a single model pass (was 12 serial calls).
    lit_queries = [
        f"Effect of {', '.join(_humanize(x) for x in c.x_cols[:6])} on "
        f"{_humanize(c.y_col)}."
        for c in candidates
    ]
    try:
        lit_chunk_lists = retriever.retrieve_batch(
            lit_queries, top_k=8, min_similarity=LIT_QUERY_MIN_SIM
        )
        for c, chunks in zip(candidates, lit_chunk_lists):
            c.lit_score = float(
                max((ch.similarity_score for ch in chunks), default=0.0)
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("Batch lit retrieval failed, using zero scores: {}", e)

    ranked = sorted(
        candidates,
        key=lambda c: (c.lit_score * c.r_squared, c.lit_score, c.r_squared),
        reverse=True,
    )
    strong = [c for c in ranked if c.lit_score >= MIN_LIT]
    chosen = strong[:FINAL_N] if len(strong) >= 2 else ranked[:FINAL_N]

    # Batch-embed all citation queries in a single model pass.
    cit_queries = [_llm_grounding_query(c.y_col, c.x_cols) for c in chosen]
    cit_search_queries = [_external_search_query(c.y_col, c.x_cols) for c in chosen]
    try:
        cit_chunk_lists = retriever.retrieve_batch(
            cit_queries, top_k=1, min_similarity=LIT_QUERY_MIN_SIM
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Batch citation retrieval failed: {}", e)
        cit_chunk_lists = [[] for _ in chosen]

    # Embed queries: q_vecs for corpus retrieval scoring; sq_vecs for arXiv/S2 scoring.
    embedder = get_embedder()
    cit_q_vecs: list[np.ndarray] = []
    cit_sq_vecs: list[np.ndarray] = []
    for q, sq in zip(cit_queries, cit_search_queries):
        cit_q_vecs.append(_norm_vec(embedder.embed_one(q).astype(np.float64)))
        cit_sq_vecs.append(_norm_vec(embedder.embed_one(sq).astype(np.float64)))

    bundles: list[dict] = []
    for i, (c, cit_chunks, q, sq, q_vec, sq_vec) in enumerate(
        zip(chosen, cit_chunk_lists, cit_queries, cit_search_queries, cit_q_vecs, cit_sq_vecs), start=1
    ):
        hid = f"H{i}"
        oid = secrets.token_hex(4)
        mid = secrets.token_hex(4)
        desc = (
            f"{' · '.join(c.x_cols[:6])}{' …' if len(c.x_cols) > 6 else ''} "
            f"jointly associate with {c.y_col} in regime {regime_id} "
            f"(n={c.n_obs} observations)."
        )
        rationale = (
            f"Automated screening fit (OLS or ridge fallback) with in-sample R² "
            f"{c.r_squared:.3f}. Corpus embedding resemblance peak "
            f"{c.lit_score:.0%} for a query built from this outcome and predictors."
        )
        citations: list[dict] = []
        for ch in cit_chunks[:1]:
            citations.append({
                "id": f"{c.y_col}:0:{ch.doc_id[:12]}",
                "source": "corpus",
                "title": ch.title or ch.source_file,
                "url": ch.source_file,
                "year": None,
                "similarity_score": round(float(ch.similarity_score), 4),
                "abstract_snippet": ch.text[:500].strip() if ch.text else None,
                "variable": c.y_col,
                "hypothesis_id": hid,
            })
        citations.extend(_external_citations(sq, sq_vec, hid, c.y_col))

        bundles.append(
            {
                "hypothesis": {
                    "id": oid,
                    "hypothesis_id": hid,
                    "round": 1,
                    "description": desc,
                    "rationale": rationale,
                    "primary_variables": c.x_cols[:10],
                    "model_family": "screening_ols",
                    "priority_score": round(min(1.0, c.lit_score), 4),
                    "is_refinement": False,
                },
                "model_result": {
                    "id": mid,
                    "hypothesis_id": hid,
                    "model_type": "screening_linear",
                    "r_squared": round(c.r_squared, 4),
                    "adj_r_squared": round(c.adj_r_squared, 4),
                    "n_observations": c.n_obs,
                    "coefficients": {k: round(v, 6) for k, v in c.coefficients.items()},
                    "p_values": {k: round(v, 6) for k, v in c.p_values.items()},
                    "significant_variables": c.significant_variables,
                    "match_score": round(min(1.0, c.lit_score), 4),
                    "validation_passed": c.r_squared >= MIN_R2
                    and c.lit_score >= MIN_LIT,
                },
                "citations": citations,
            }
        )

    warnings: list[str] = []
    if not chosen:
        warnings.append("No candidates loaded from prior screening results.")
    elif not strong and chosen:
        warnings.append(
            "Corpus similarity was weak for most fits; showing top statistical fits anyway."
        )

    return {
        "regime_id": regime_id,
        "regime_n_rows": regime_n_rows,
        "bundles": bundles,
        "warnings": warnings,
    }


def _score_against_query(query_vec: np.ndarray, text: str) -> float:
    v = get_embedder().embed_one(text).astype(np.float64)
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        return 0.0
    return float(np.dot(query_vec, v / norm))


def _arxiv_citation(
    search_query: str, sq_vec: np.ndarray, hid: str, variable: str
) -> dict | None:
    """Search arXiv with domain keyword query; score results against the search-query embedding.

    sq_vec is embedded from search_query (scientific domain language), not from the
    column-name query — both sq_vec and the paper abstract live in the same semantic
    register, giving cosine scores in the 40–70% range for truly relevant papers.
    """
    try:
        papers = ArxivClient().search(search_query)
        if not papers:
            return None
        best = max(papers, key=lambda p: _score_against_query(sq_vec, f"{p.title}. {p.abstract}"[:1000]))
        score = _score_against_query(sq_vec, f"{best.title}. {best.abstract}"[:1000])
        if score < EXT_CIT_MIN_SCORE:
            logger.debug("arXiv best score {:.1%} below threshold — skipping", score)
            return None
        return {
            "id": f"arxiv:{best.arxiv_id}",
            "source": "arxiv",
            "title": best.title,
            "url": best.url,
            "year": best.published[:4] if best.published and best.published != "unknown" else None,
            "similarity_score": round(score, 4),
            "abstract_snippet": best.abstract[:500].strip(),
            "variable": variable,
            "hypothesis_id": hid,
        }
    except Exception as e:
        logger.warning("arXiv citation failed: {}", e)
        return None


def _s2_citation(
    search_query: str, sq_vec: np.ndarray, hid: str, variable: str
) -> dict | None:
    """Search Semantic Scholar with domain keyword query; score results against the search-query embedding."""
    try:
        papers = SemanticScholarClient().search(search_query, max_results=10)
        if not papers:
            return None
        best = max(papers, key=lambda p: _score_against_query(sq_vec, f"{p.title}. {p.abstract}"[:1000]))
        score = _score_against_query(sq_vec, f"{best.title}. {best.abstract}"[:1000])
        if score < EXT_CIT_MIN_SCORE:
            logger.debug("S2 best score {:.1%} below threshold — skipping", score)
            return None
        return {
            "id": f"s2:{best.paper_id}",
            "source": "semantic_scholar",
            "title": best.title,
            "url": best.url,
            "year": str(best.year) if best.year else None,
            "similarity_score": round(score, 4),
            "abstract_snippet": best.abstract[:500].strip(),
            "variable": variable,
            "hypothesis_id": hid,
        }
    except Exception as e:
        logger.warning("Semantic Scholar citation failed: {}", e)
        return None


def _openalex_citation(
    search_query: str, sq_vec: np.ndarray, hid: str, variable: str
) -> dict | None:
    try:
        return _best_external_citation(
            OpenAlexClient().search(search_query, max_results=10),
            sq_vec,
            hid,
            variable,
            source="openalex",
            id_prefix="openalex",
            get_id=lambda p: p.doi or p.work_id,
            get_title=lambda p: p.title,
            get_abstract=lambda p: p.abstract,
            get_url=lambda p: p.open_access_url or p.url,
            get_year=lambda p: str(p.year) if p.year else None,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("OpenAlex citation failed: {}", e)
        return None


def _europe_pmc_citation(
    search_query: str, sq_vec: np.ndarray, hid: str, variable: str
) -> dict | None:
    try:
        return _best_external_citation(
            EuropePMCClient().search(search_query, max_results=10),
            sq_vec,
            hid,
            variable,
            source="europe_pmc",
            id_prefix="epmc",
            get_id=lambda p: p.doi or p.paper_id,
            get_title=lambda p: p.title,
            get_abstract=lambda p: p.abstract,
            get_url=lambda p: p.url,
            get_year=lambda p: str(p.year) if p.year else None,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Europe PMC citation failed: {}", e)
        return None


def _crossref_citation(
    search_query: str, sq_vec: np.ndarray, hid: str, variable: str
) -> dict | None:
    try:
        return _best_external_citation(
            CrossrefClient().search(search_query, max_results=10),
            sq_vec,
            hid,
            variable,
            source="crossref",
            id_prefix="crossref",
            get_id=lambda p: p.doi,
            get_title=lambda p: p.title,
            get_abstract=lambda p: p.abstract,
            get_url=lambda p: p.url,
            get_year=lambda p: str(p.year) if p.year else None,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Crossref citation failed: {}", e)
        return None


def _best_external_citation(
    papers: list,
    sq_vec: np.ndarray,
    hid: str,
    variable: str,
    *,
    source: str,
    id_prefix: str,
    get_id,
    get_title,
    get_abstract,
    get_url,
    get_year,
) -> dict | None:
    if not papers:
        return None

    def text_for(p: object) -> str:
        return f"{get_title(p) or ''}. {get_abstract(p) or ''}"[:1200]

    scored = [
        (p, _score_against_query(sq_vec, text_for(p)))
        for p in papers
        if str(get_title(p) or "")
    ]
    if not scored:
        return None
    best, score = max(scored, key=lambda x: x[1])
    if score < EXT_CIT_MIN_SCORE:
        logger.debug("{} best score {:.1%} below threshold — skipping", source, score)
        return None
    raw_id = str(get_id(best) or get_url(best) or get_title(best) or "")
    return {
        "id": f"{id_prefix}:{raw_id[:160]}",
        "source": source,
        "title": str(get_title(best) or ""),
        "url": get_url(best),
        "year": get_year(best),
        "similarity_score": round(float(score), 4),
        "abstract_snippet": str(get_abstract(best) or "")[:500].strip() or None,
        "variable": variable,
        "hypothesis_id": hid,
    }


def _dedupe_citations(citations: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for c in citations:
        key = str(c.get("url") or c.get("id") or c.get("title") or "").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _external_citations(
    search_query: str,
    sq_vec: np.ndarray,
    hid: str,
    variable: str,
    *,
    max_citations: int = MAX_EXTERNAL_CITATIONS,
) -> list[dict]:
    """Collect broad external literature references before slower fallbacks."""
    citations: list[dict] = []
    for provider in (
        _openalex_citation,
        _europe_pmc_citation,
        _crossref_citation,
        _s2_citation,
        _arxiv_citation,
    ):
        cit = provider(search_query, sq_vec, hid, variable)
        if cit:
            citations = _dedupe_citations([*citations, cit])
        if len(citations) >= max_citations:
            break
    return citations[:max_citations]


def _norm_vec(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _citations_for_candidate(
    retriever: Retriever, c: _Candidate, hypothesis_label: str
) -> list[dict]:
    embedder = get_embedder()
    query = _llm_grounding_query(c.y_col, c.x_cols)           # natural-language → corpus retrieval
    search_query = _external_search_query(c.y_col, c.x_cols)   # domain keywords → arXiv/S2 search + scoring

    # sq_vec: used to score arXiv/S2 results (domain-keyword language ↔ paper abstract language)
    sq_vec = _norm_vec(embedder.embed_one(search_query).astype(np.float64))

    logger.debug("Citation query: {} | search: {}", query, search_query)

    out: list[dict] = []

    # 1 — uploaded corpus (scored with q_vec — same register as retrieval query)
    try:
        chunks = retriever.retrieve(query=query, top_k=1, min_similarity=LIT_QUERY_MIN_SIM)
        for ch in chunks[:1]:
            out.append({
                "id": f"{c.y_col}:0:{ch.doc_id[:12]}",
                "source": "corpus",
                "title": ch.title or ch.source_file,
                "url": ch.source_file,
                "year": None,
                "similarity_score": round(float(ch.similarity_score), 4),
                "abstract_snippet": ch.text[:500].strip() if ch.text else None,
                "variable": c.y_col,
                "hypothesis_id": hypothesis_label,
            })
    except Exception as e:
        logger.debug("Corpus citation retrieve failed: {}", e)

    out.extend(_external_citations(search_query, sq_vec, hypothesis_label, c.y_col))

    return out
