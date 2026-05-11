"""
Rank automated screening fits by statistical fit and corpus similarity.

Used after legacy regime screening: collects OLS-style results for
input subsets × numeric outputs in one regime, scores each candidate
against the RAG vector store, and returns bundles for the static UI.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge

from src.ingestion.legacy_pfca_regime_masks import (
    assign_legacy_pfca_regime_masks,
    assign_pfca_regime_masks_from_column_lists,
)
from src.ingestion.unified_experimental_sheet import UnifiedSheetMeta
from src.pipeline.automated_screening import (
    _build_panel_design,
    _corr_ok,
    _detect_panel_columns,
    _mg_l_input_names,
    _numeric_output_names,
    _prepare_xy,
    _subset_iterator,
)
from src.rag.retriever import Retriever
from src.utils.logging import get_logger

logger = get_logger(__name__)

TOP_POOL = 72
FINAL_N = 6
MIN_R2 = 0.06
MIN_LIT = 0.30
LIT_QUERY_MIN_SIM = 0.18


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
    df_x = pd.DataFrame(x_mat, columns=names)
    try:
        xc = sm.add_constant(df_x, has_constant="add")
        res = sm.OLS(y_vec, xc).fit()
        r2 = float(res.rsquared)
        ar2 = float(res.rsquared_adj)
        coefs = {k: float(v) for k, v in res.params.items() if k != "const"}
        pvals = {k: float(v) for k, v in res.pvalues.items() if k != "const"}
        sig = [k for k, p in pvals.items() if p < 0.05]
        return r2, ar2, coefs, pvals, sig
    except Exception as e:  # noqa: BLE001
        logger.debug("OLS failed (%s), using Ridge for point estimates", e)
        ridge = Ridge(alpha=1.0, random_state=0).fit(x_mat, y_vec)
        r2 = float(ridge.score(x_mat, y_vec))
        coefs = {n: float(c) for n, c in zip(names, ridge.coef_, strict=True)}
        pvals = {n: 0.5 for n in names}
        sig = [n for n, c in coefs.items() if abs(c) > 1e-8][: max(1, len(names))]
        return r2, r2, coefs, pvals, sig


def _literature_score(retriever: Retriever, y_col: str, x_cols: list[str]) -> float:
    xs = ", ".join(x_cols[:6])
    query = (
        f"PFAS degradation and treatment: statistical relationships between "
        f"predictors ({xs}) and outcome {y_col} in experimental water chemistry."
    )
    try:
        chunks = retriever.retrieve(
            query=query, top_k=8, min_similarity=LIT_QUERY_MIN_SIM
        )
        if not chunks:
            return 0.0
        return float(max(c.similarity_score for c in chunks))
    except Exception as e:  # noqa: BLE001
        logger.warning("Literature retrieval failed: %s", e)
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
    Build ranked hypothesis bundles for one regime.

    Returns a dict with keys matching the static screening API schema.
    """
    if unified_meta is not None:
        res = assign_pfca_regime_masks_from_column_lists(
            df,
            unified_meta.input_cols,
            unified_meta.output_cols,
            replace_no_in_inputs=True,
        )
    else:
        res = assign_legacy_pfca_regime_masks(
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

    mg_inputs = _mg_l_input_names(res.input_cols)
    pool = [c for c in mg_inputs if c in sub_df.columns]
    if len(pool) < 2:
        return {
            "regime_id": rid,
            "regime_n_rows": len(sub_df),
            "bundles": [],
            "warnings": [
                "This regime slice has fewer than two mg/L-style numeric inputs; "
                "automated screening cannot form predictor subsets."
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
                    "validation_passed": c.r_squared >= MIN_R2 and c.lit_score >= MIN_LIT,
                },
                "citations": _citations_for_candidate(retriever, c, hid),
            }
        )

    warnings: list[str] = []
    if not candidates:
        warnings.append(
            "No passing screening fits for this regime (check mg/L inputs and numeric outputs)."
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


def _citations_for_candidate(
    retriever: Retriever, c: _Candidate, hypothesis_label: str
) -> list[dict]:
    query = (
        f"PFAS {c.y_col} degradation mechanisms involving "
        f"{', '.join(c.x_cols[:5])} in experimental studies."
    )
    try:
        chunks = retriever.retrieve(
            query=query, top_k=4, min_similarity=LIT_QUERY_MIN_SIM
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("Citation retrieve failed: %s", e)
        return []

    out: list[dict] = []
    for j, ch in enumerate(chunks):
        out.append(
            {
                "id": f"{c.y_col}:{j}:{ch.doc_id[:12]}",
                "source": "corpus",
                "title": ch.title or ch.source_file,
                "url": ch.source_file,
                "year": None,
                "similarity_score": round(float(ch.similarity_score), 4),
                "variable": c.y_col,
                "hypothesis_id": hypothesis_label,
            }
        )
    return out
