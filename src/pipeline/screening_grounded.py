"""
Rank automated screening fits by statistical fit and corpus similarity.

Collects OLS-style results for input subsets × numeric outputs in one screening
segment (``regime_id`` in the API), scores each candidate against the RAG
vector store, and returns bundles for the static UI.
"""

from __future__ import annotations

import re
import secrets
import warnings
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge

from src.etl.experimental.detector import add_derived_entity_column
from src.grounding.arxiv_client import ArxivClient
from src.grounding.crossref_client import CrossrefClient
from src.grounding.europe_pmc_client import EuropePMCClient
from src.grounding.jina_client import JinaSearchClient
from src.grounding.openalex_client import OpenAlexClient
from src.grounding.semantic_scholar import SemanticScholarClient
from src.ingestion.dataset_screening_layout import (
    assign_screening_layout_from_column_lists,
    assign_screening_layout_from_legacy_column_slices,
)
from src.ingestion.unified_experimental_sheet import UnifiedSheetMeta
from src.modeling.diagnostics import (
    DiagnosticResult,
    diagnose_ols,
    diagnose_panel,
    diagnose_ridge,
)
from src.pipeline.automated_screening import (
    _build_panel_design,
    _corr_ok,
    _detect_panel_columns,
    _numeric_output_names,
    _prepare_xy,
    _subset_iterator,
    _within_entity_pool,
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
    diagnostic: DiagnosticResult | None = None
    panel_diagnostic: DiagnosticResult | None = None


def _candidate_description(c: _Candidate) -> str:
    predictors = f"{' · '.join(c.x_cols[:6])}{' …' if len(c.x_cols) > 6 else ''}"
    verb = "associates with" if len(c.x_cols) == 1 else "jointly associate with"
    return f"{predictors} {verb} {c.y_col} (n={c.n_obs} observations)."


def _fit_ols_or_ridge(
    x_mat: np.ndarray, y_vec: np.ndarray, names: list[str]
) -> tuple[
    float, float, dict[str, float], dict[str, float], list[str], DiagnosticResult
]:
    """Return r2, adj_r2, coefs (no const), pvals, significant names, diagnostic."""
    if np.std(y_vec) == 0:
        d = DiagnosticResult(model_family="ols", effective_n=len(y_vec))
        d.diagnostic_warnings.append("Zero variance in outcome — no valid fit")
        d.diagnostic_score = 0.0
        return 0.0, 0.0, {n: 0.0 for n in names}, {n: 1.0 for n in names}, [], d
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
        diag = diagnose_ols(res, x_mat, names)
        return r2, ar2, coefs, pvals, sig, diag
    except Exception as e:  # noqa: BLE001
        logger.debug("OLS failed ({}), using Ridge for point estimates", e)
        ridge = Ridge(alpha=1.0, random_state=0).fit(x_mat, y_vec)
        r2 = float(ridge.score(x_mat, y_vec))
        coefs = {n: float(c) for n, c in zip(names, ridge.coef_, strict=True)}
        pvals = {n: 0.5 for n in names}
        sig = [n for n, c in coefs.items() if abs(c) > 1e-8][: max(1, len(names))]
        return r2, r2, coefs, pvals, sig, diagnose_ridge()


def _fit_panel_ols(
    sub_df: pd.DataFrame,
    x_cols: list[str],
    y_col: str,
    entity_col: str,
    time_col: str,
) -> (
    tuple[float, float, dict[str, float], dict[str, float], list[str], DiagnosticResult]
    | None
):
    """Fit entity fixed-effects PanelOLS using the pattern from scripts/ols_vs_panel.py.

    Only called for pre-selected top candidates. Returns None if panel conditions
    are not met (too few entities, no within-entity variance) or on any fit error.
    """
    try:
        from linearmodels.panel import PanelOLS as _PanelOLS
    except ImportError:
        logger.debug("linearmodels not installed — panel OLS skipped")
        return None

    cols = list(dict.fromkeys([entity_col, time_col, y_col] + x_cols))
    avail = [c for c in cols if c in sub_df.columns]
    mdf = sub_df[avail].apply(pd.to_numeric, errors="coerce").dropna()

    if len(mdf) < 12 or mdf[entity_col].nunique() < 3:
        return None

    # Within-entity variance fraction filter (same criterion: MIN_WITHIN_FRACTION=0.05)
    def _within_frac(col: str) -> float:
        s = mdf[col]
        total_var = float(s.var())
        if total_var < 1e-9:
            return 0.0
        entity_means = mdf.groupby(entity_col)[col].transform("mean")
        return float((s - entity_means).var()) / total_var

    fe_cols = [
        c
        for c in x_cols
        if c != time_col and c in mdf.columns and _within_frac(c) >= 0.05
    ]
    # Time is always a valid within-entity predictor (kinetics regressor); a time-only
    # panel model is the kinetics baseline — don't require other within-varying features.

    try:
        panel_df = mdf.set_index([entity_col, time_col])
        y_panel = panel_df[y_col]

        # Feature matrix: within-varying features + time as kinetic trend regressor
        X_panel = (
            panel_df[fe_cols].copy() if fe_cols else pd.DataFrame(index=panel_df.index)
        )
        X_panel[time_col] = mdf[time_col].values
        X_panel = sm.add_constant(X_panel)

        fe_model = _PanelOLS(
            y_panel,
            X_panel,
            entity_effects=True,
            time_effects=False,
            drop_absorbed=True,
        ).fit(cov_type="clustered", cluster_entity=True)

        r2_overall = float(fe_model.rsquared_overall)
        r2_within = float(fe_model.rsquared)
        coefs = {k: float(v) for k, v in fe_model.params.items() if k != "const"}
        pvals = {k: float(v) for k, v in fe_model.pvalues.items() if k != "const"}
        sig = [k for k, p in pvals.items() if p < 0.05]

        # Include time_col in ADF cols so stationarity is tested even for
        # time-only models where fe_cols=[].
        adf_cols = ([time_col] if time_col in mdf.columns else []) + fe_cols
        panel_d = diagnose_panel(
            mdf,
            y_col,
            entity_col,
            adf_cols,
            within_r2=r2_within,
            between_r2=float(fe_model.rsquared_between),
        )
        panel_d.model_family = "panel_fe"
        n_ents = int(fe_model.entity_info.total)
        panel_d.n_entities = n_ents
        panel_d.effective_n = int(fe_model.nobs)

        # F-test for joint significance
        try:
            panel_d.f_statistic = round(float(fe_model.f_statistic.stat), 4)
            panel_d.f_pvalue = round(float(fe_model.f_statistic.pval), 6)
            (
                panel_d.passed_tests
                if panel_d.f_pvalue < 0.05
                else panel_d.failed_tests
            ).append("fe_joint_significance")
        except Exception:
            pass

        # Entity count adequacy
        (panel_d.passed_tests if n_ents >= 10 else panel_d.failed_tests).append(
            "entity_count_adequate"
        )

        # Fallback ICC computation — diagnose_panel may fail silently if entity
        # col dtype or index alignment causes a silent exception there.
        if panel_d.icc is None:
            try:
                y_s = pd.to_numeric(mdf[y_col], errors="coerce")
                ent_s = mdf[entity_col]
                gm = y_s.groupby(ent_s).transform("mean")
                bv = float(gm.var())
                tv = float(y_s.var())
                if tv > 1e-9:
                    panel_d.icc = round(bv / tv, 4)
                    (
                        panel_d.passed_tests
                        if panel_d.icc >= 0.10
                        else panel_d.failed_tests
                    ).append("icc_adequate")
            except Exception as icc_exc:
                logger.debug("Panel ICC fallback failed: {}", icc_exc)

        # Recompute composite score now that f_pvalue + n_entities + icc are set
        from src.modeling.diagnostics import _panel_composite  # noqa: PLC0415

        panel_d.diagnostic_score = _panel_composite(panel_d)

        return r2_overall, r2_within, coefs, pvals, sig, panel_d

    except Exception as e:
        logger.debug("PanelOLS failed ({}), caller will fall back to OLS", e)
        return None


def _effective_diagnostic_score(c: _Candidate) -> float:
    """Combined diagnostic score for a candidate.

    When panel diagnostics are present they act as a floor: if the panel
    structure is non-stationary or the ICC is too low the score is dampened,
    even if OLS assumptions hold. Uses geometric mean so both components must
    be healthy for a top score.
    """
    ols_s = float(c.diagnostic.diagnostic_score) if c.diagnostic else 0.5
    if c.panel_diagnostic is None:
        return ols_s
    panel_s = float(c.panel_diagnostic.diagnostic_score)
    return float(round((ols_s * panel_s) ** 0.5, 4))


def _safe_float(value: object, digits: int = 4) -> float | None:
    try:
        v = float(value)  # type: ignore[arg-type]
        if not np.isfinite(v):
            return None
        return round(v, digits)
    except Exception:
        return None


def _ols_extended_diagnostics(
    df: pd.DataFrame,
    x_cols: list[str],
    y_col: str,
) -> dict[str, object]:
    """Extra OLS diagnostics mirrored from scripts/ols_vs_panel.py."""
    cols = [c for c in x_cols if c in df.columns and c != y_col]
    if not cols:
        return {"error": "no usable columns"}
    block = df[cols + [y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    n, k = len(block), len(cols)
    if n < k + 3:
        return {"error": f"only {n} rows, need > {k + 3}"}

    y = block[y_col]
    X = block[cols]
    Xc = sm.add_constant(X)
    res = sm.OLS(y, Xc).fit()
    resid = np.asarray(res.resid)
    out: dict[str, object] = {
        "n_obs": int(n),
        "n_predictors": int(k),
        "obs_per_predictor": _safe_float(n / max(k, 1), 1),
        "rmse": _safe_float(np.sqrt(np.mean(resid**2))),
        "mae": _safe_float(np.mean(np.abs(resid))),
        "y_std": _safe_float(y.std()),
        "aic": _safe_float(res.aic, 2),
        "bic": _safe_float(res.bic, 2),
    }
    y_arr = y.to_numpy(dtype=float)
    out["mape"] = (
        _safe_float(np.mean(np.abs(resid / y_arr)) * 100, 2)
        if not np.any(y_arr == 0)
        else None
    )

    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score

        n_splits = min(5, n // 5)
        if n_splits >= 2:
            scores = cross_val_score(
                LinearRegression(),
                X.to_numpy(),
                y.to_numpy(),
                cv=n_splits,
                scoring="r2",
            )
            out["cv_r2"] = _safe_float(np.mean(scores))
            out["cv_r2_std"] = _safe_float(np.std(scores))
    except Exception as exc:
        out["cv_error"] = str(exc)[:120]

    try:
        from statsmodels.stats.outliers_influence import OLSInfluence

        infl = OLSInfluence(res)
        cooks = infl.cooks_distance[0]
        hat = np.asarray(infl.hat_matrix_diag)
        out["n_influential"] = int(np.sum(cooks > 4.0 / n))
        out["n_high_leverage"] = int(np.sum(hat > 2.0 * k / n))
        out["max_studentized"] = _safe_float(
            np.nanmax(np.abs(res.outlier_test()["student_resid"]))
        )
    except Exception as exc:
        out["influence_error"] = str(exc)[:120]

    try:
        from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_white

        _, bg_p, _, _ = acorr_breusch_godfrey(res, nlags=2)
        _, white_p, _, _ = het_white(resid, Xc.to_numpy())
        out["breusch_godfrey_p"] = _safe_float(bg_p, 6)
        out["white_p"] = _safe_float(white_p, 6)
    except Exception as exc:
        out["extra_test_error"] = str(exc)[:120]

    try:
        Xz = (X - X.mean()) / X.std().replace(0, 1)
        yz = (y - y.mean()) / max(float(y.std()), 1e-9)
        res_z = sm.OLS(yz, sm.add_constant(Xz)).fit()
        out["standardized_betas"] = {
            k_: _safe_float(v) for k_, v in res_z.params.items() if k_ != "const"
        }
    except Exception:
        out["standardized_betas"] = {}
    return out


def _panel_extended_diagnostics(
    df: pd.DataFrame,
    x_cols: list[str],
    y_col: str,
    entity_col: str | None,
    time_col: str | None,
) -> dict[str, object] | None:
    """Extra PanelOLS diagnostics mirrored from scripts/ols_vs_panel.py."""
    if (
        not entity_col
        or not time_col
        or entity_col not in df.columns
        or time_col not in df.columns
    ):
        return None
    try:
        from linearmodels.panel import PanelOLS, RandomEffects
        from scipy import stats as scipy_stats
    except Exception as exc:
        return {"error": f"panel dependencies unavailable: {exc}"}

    cols = [c for c in x_cols if c in df.columns and c != y_col]
    if time_col not in cols and time_col in df.columns:
        cols = [time_col] + cols
    cols = list(dict.fromkeys(cols))
    all_cols = list(dict.fromkeys([entity_col, time_col, y_col] + cols))
    mdf = df[all_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(mdf) < 10 or mdf[entity_col].nunique() < 3:
        return {"error": "too few rows or entities for panel diagnostics"}

    obs_per_entity = mdf.groupby(entity_col).size()
    out: dict[str, object] = {
        "n_obs": int(len(mdf)),
        "n_entities": int(mdf[entity_col].nunique()),
        "min_obs_per_entity": int(obs_per_entity.min()),
        "max_obs_per_entity": int(obs_per_entity.max()),
        "mean_obs_per_entity": _safe_float(obs_per_entity.mean(), 1),
        "balanced": bool(obs_per_entity.nunique() == 1),
        "y_std": _safe_float(mdf[y_col].std()),
    }

    try:
        pnl = mdf.set_index([entity_col, time_col])
        y_panel = pnl[y_col]
        panel_features = [c for c in cols if c != time_col]
        Xp = (
            pnl[panel_features].copy()
            if panel_features
            else pd.DataFrame(index=pnl.index)
        )
        Xp[time_col] = mdf[time_col].values
        Xp = sm.add_constant(Xp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fe_m = PanelOLS(y_panel, Xp, entity_effects=True, drop_absorbed=True).fit(
                cov_type="clustered", cluster_entity=True
            )
        resid = np.asarray(fe_m.resids)
        out.update(
            {
                "rmse": _safe_float(np.sqrt(np.mean(resid**2))),
                "mae": _safe_float(np.mean(np.abs(resid))),
                "within_r2": _safe_float(fe_m.rsquared),
                "between_r2": _safe_float(fe_m.rsquared_between),
                "overall_r2": _safe_float(fe_m.rsquared_overall),
                "f_statistic": _safe_float(fe_m.f_statistic.stat),
                "f_pvalue": _safe_float(fe_m.f_statistic.pval, 6),
            }
        )
        try:
            k = len(fe_m.params)
            ll = float(fe_m.loglik)
            out["aic"] = _safe_float(-2 * ll + 2 * k, 2)
            out["bic"] = _safe_float(-2 * ll + np.log(len(mdf)) * k, 2)
        except Exception:
            out["aic"] = out["bic"] = None

        try:
            re_m = RandomEffects(y_panel, Xp).fit()
            common = [
                v for v in fe_m.params.index if v in re_m.params.index and v != "const"
            ]
            if common:
                b_diff = fe_m.params[common].values - re_m.params[common].values
                Vfe = fe_m.cov.loc[common, common].values
                Vre = re_m.cov.loc[common, common].values
                h_stat = float(b_diff @ np.linalg.pinv(Vfe - Vre) @ b_diff)
                if h_stat >= 0:
                    h_p = float(1.0 - scipy_stats.chi2.cdf(h_stat, df=len(common)))
                    out["hausman_stat"] = _safe_float(h_stat)
                    out["hausman_p"] = _safe_float(h_p, 6)
                    out["fe_necessary"] = bool(h_p < 0.05)
        except Exception as exc:
            out["hausman_error"] = str(exc)[:120]
    except Exception as exc:
        out["fit_error"] = str(exc)[:120]
    return out


def _additional_family_diagnostics(
    df: pd.DataFrame,
    x_cols: list[str],
    y_col: str,
    entity_col: str | None,
    time_col: str | None,
) -> dict[str, object]:
    """Fit non-OLS families for the selected screening model's variables."""
    cols = [c for c in x_cols if c in df.columns and c != y_col]
    if not cols:
        return {}
    block = df[cols + [y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(block) < max(12, len(cols) + 4):
        return {"error": "too few complete rows for additional model families"}
    X_df = block[cols]
    X = X_df.to_numpy(dtype=float)
    y = block[y_col].to_numpy(dtype=float)
    out: dict[str, object] = {}

    try:
        from sklearn.model_selection import KFold, cross_val_score
        from xgboost import XGBRegressor

        n_splits = min(5, max(2, len(y) // 5))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        model = XGBRegressor(
            n_estimators=160,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            tree_method="hist",
            verbosity=0,
        )
        model.fit(X, y)
        pred = model.predict(X)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
        in_r2 = float(model.score(X, y))
        cv_r2 = float(np.mean(cv_scores))
        out["xgboost"] = {
            "in_sample_r2": _safe_float(in_r2),
            "cv_r2": _safe_float(cv_r2),
            "cv_r2_std": _safe_float(np.std(cv_scores)),
            "generalization_gap": _safe_float(in_r2 - cv_r2),
            "rmse": _safe_float(np.sqrt(np.mean((y - pred) ** 2))),
            "mae": _safe_float(np.mean(np.abs(y - pred))),
            "n_features_used": int(np.sum(model.feature_importances_ > 0)),
            "n_features_total": len(cols),
            "feature_importances": {
                f: _safe_float(v)
                for f, v in zip(cols, model.feature_importances_, strict=True)
            },
        }
    except Exception as exc:
        out["xgboost"] = {"error": str(exc)[:120]}

    try:
        from sklearn.linear_model import LassoCV, RidgeCV
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        n_splits = min(5, max(2, len(y) // 5))
        Xz = StandardScaler().fit_transform(X)
        lasso = LassoCV(cv=n_splits, max_iter=10000, random_state=42).fit(Xz, y)
        l_pred = lasso.predict(Xz)
        l_cv = cross_val_score(lasso, Xz, y, cv=n_splits, scoring="r2")
        n_nonzero = int(np.sum(np.abs(lasso.coef_) > 1e-10))
        out["lasso"] = {
            "selected_alpha": _safe_float(lasso.alpha_, 6),
            "n_nonzero_coefs": n_nonzero,
            "n_total_features": len(cols),
            "sparsity": _safe_float(1.0 - n_nonzero / max(1, len(cols))),
            "in_sample_r2": _safe_float(lasso.score(Xz, y)),
            "cv_r2": _safe_float(np.mean(l_cv)),
            "rmse": _safe_float(np.sqrt(np.mean((y - l_pred) ** 2))),
            "mae": _safe_float(np.mean(np.abs(y - l_pred))),
            "coefficients": {
                f: _safe_float(c) for f, c in zip(cols, lasso.coef_, strict=True)
            },
        }

        ridge = RidgeCV(alphas=np.logspace(-3, 4, 50), cv=n_splits, scoring="r2").fit(
            Xz, y
        )
        r_pred = ridge.predict(Xz)
        r_cv = cross_val_score(ridge, Xz, y, cv=n_splits, scoring="r2")
        sv = np.linalg.svd(Xz, compute_uv=False)
        eff_df = float(np.sum(sv**2 / (sv**2 + float(ridge.alpha_))))
        try:
            ols = sm.OLS(y, sm.add_constant(X_df)).fit()
            ols_norm = float(np.linalg.norm(ols.params.values[1:]))
        except Exception:
            ols_norm = np.nan
        ridge_norm = float(np.linalg.norm(ridge.coef_))
        shrinkage = (
            1.0 - ridge_norm / max(ols_norm, 1e-9) if np.isfinite(ols_norm) else None
        )
        out["ridge"] = {
            "selected_alpha": _safe_float(ridge.alpha_, 6),
            "effective_df": _safe_float(eff_df, 2),
            "n_predictors": len(cols),
            "shrinkage": _safe_float(shrinkage) if shrinkage is not None else None,
            "in_sample_r2": _safe_float(ridge.score(Xz, y)),
            "cv_r2": _safe_float(np.mean(r_cv)),
            "generalization_gap": _safe_float(
                float(ridge.score(Xz, y)) - float(np.mean(r_cv))
            ),
            "rmse": _safe_float(np.sqrt(np.mean((y - r_pred) ** 2))),
            "mae": _safe_float(np.mean(np.abs(y - r_pred))),
            "condition_number": _safe_float(np.linalg.cond(X)),
        }
    except Exception as exc:
        out.setdefault("lasso", {"error": str(exc)[:120]})
        out.setdefault("ridge", {"error": str(exc)[:120]})

    panel = _panel_extended_diagnostics(df, cols, y_col, entity_col, time_col)
    if panel and "error" not in panel and panel.get("within_r2") is not None:
        two_stage: dict[str, object] = {
            "stage1_within_r2": panel.get("within_r2"),
            "stage1_f_stat": panel.get("f_statistic"),
            "stage1_f_pvalue": panel.get("f_pvalue"),
            "n_entities": panel.get("n_entities"),
            "stage2_r2": None,
            "stage2_rmse": None,
            "stage2_mae": None,
            "stage2_f_stat": None,
            "stage2_f_pvalue": None,
            "stage2_n_obs": None,
            "stage2_obs_per_predictor": None,
        }
        if (
            entity_col
            and time_col
            and entity_col in df.columns
            and time_col in df.columns
        ):
            try:
                treatment_cols = [c for c in cols if c != time_col]
                stage_rows = []
                for entity, grp in df.groupby(entity_col):
                    g = (
                        grp[[time_col, y_col]]
                        .apply(pd.to_numeric, errors="coerce")
                        .dropna()
                    )
                    if g[time_col].nunique() < 2:
                        continue
                    m1 = sm.OLS(g[y_col], sm.add_constant(g[time_col])).fit()
                    row: dict[str, object] = {
                        "entity": entity,
                        "entity_fe": float(m1.params.iloc[0]),
                    }
                    for col in treatment_cols:
                        vals = pd.to_numeric(grp[col], errors="coerce").dropna()
                        row[col] = float(vals.iloc[0]) if len(vals) else np.nan
                    stage_rows.append(row)
                stage_df = pd.DataFrame(stage_rows)
                stage_df = stage_df.apply(pd.to_numeric, errors="coerce").dropna()
                if treatment_cols and len(stage_df) > len(treatment_cols) + 2:
                    y2 = stage_df["entity_fe"]
                    X2 = stage_df[treatment_cols]
                    res2 = sm.OLS(y2, sm.add_constant(X2)).fit()
                    pred2 = np.asarray(res2.fittedvalues)
                    resid2 = y2.to_numpy(dtype=float) - pred2
                    two_stage.update(
                        {
                            "stage2_r2": _safe_float(res2.rsquared),
                            "stage2_rmse": _safe_float(np.sqrt(np.mean(resid2**2))),
                            "stage2_mae": _safe_float(np.mean(np.abs(resid2))),
                            "stage2_f_stat": _safe_float(res2.fvalue),
                            "stage2_f_pvalue": _safe_float(res2.f_pvalue, 6),
                            "stage2_n_obs": int(len(stage_df)),
                            "stage2_obs_per_predictor": _safe_float(
                                len(stage_df) / max(1, len(treatment_cols)), 1
                            ),
                        }
                    )
                else:
                    two_stage["stage2_error"] = (
                        "too few entity-level observations for stage 2"
                    )
            except Exception as exc:
                two_stage["stage2_error"] = str(exc)[:120]
        out["two_stage"] = two_stage
    return out


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


_SEARCH_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "avg",
    "average",
    "by",
    "col",
    "column",
    "conc",
    "concentration",
    "data",
    "dose",
    "final",
    "for",
    "from",
    "id",
    "in",
    "initial",
    "input",
    "level",
    "mean",
    "mg",
    "ml",
    "ng",
    "of",
    "output",
    "pct",
    "percent",
    "rate",
    "result",
    "sample",
    "sec",
    "time",
    "to",
    "total",
    "ug",
    "value",
    "with",
}


def _keyword_terms_from_columns(
    y_col: str, x_cols: list[str], *, limit: int = 10
) -> list[str]:
    """Extract domain-neutral search terms from actual column names."""
    raw = " ".join([y_col, *x_cols]).lower()
    tokens = re.findall(r"[a-z][a-z0-9]{2,}", raw.replace("_", " "))
    terms: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in _SEARCH_STOPWORDS:
            continue
        if token not in seen:
            terms.append(token)
            seen.add(token)
        if len(terms) >= limit:
            break
    return terms


def _process_terms_from_columns(cols: str) -> list[str]:
    """Generic process hints; intentionally not tied to a specific chemical family."""
    process_rules = (
        (("plasma", "discharge", "voltage", "kv"), "plasma treatment"),
        (("uv", "lamp", "fluence", "photolysis"), "photolysis"),
        (("ozone", "o3"), "ozonation"),
        (("sonication", "ultrasound", "acoustic"), "sonochemical treatment"),
        (
            ("electrochemical", "electrolysis", "current", "electrode"),
            "electrochemical treatment",
        ),
        (("thermal", "temperature", "pyrolysis", "incineration"), "thermal treatment"),
        (("catalyst", "photocatalyst", "catalytic"), "catalysis"),
        (("adsorption", "sorbent", "carbon", "resin"), "adsorption"),
        (("peroxide", "h2o2", "oxidation", "oxidant"), "oxidation"),
    )
    out: list[str] = []
    for needles, term in process_rules:
        if any(n in cols for n in needles):
            out.append(term)
    return out


def _external_search_query(y_col: str, x_cols: list[str]) -> str:
    """Build a keyword-optimised query for external literature APIs.

    This intentionally avoids hardcoded domain species. It derives compact search
    terms from the selected outcome/predictor names and adds only generic process
    hints such as photolysis, electrochemical treatment, or adsorption.
    """
    cols = " ".join([y_col] + list(x_cols)).lower()

    terms = [
        *_keyword_terms_from_columns(y_col, x_cols),
        *_process_terms_from_columns(cols),
    ]
    if any(t in cols for t in ("removal", "degradation", "decay", "yield")):
        terms.append("degradation")
    if any(t in cols for t in ("water", "solution", "aqueous", "ph", "conductivity")):
        terms.append("water treatment")

    if not terms:
        # Last resort: fall back to the compact humanized form
        return _llm_grounding_query(y_col, x_cols)

    return " ".join(dict.fromkeys(terms))


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
    # Derive entity from input combinations BEFORE slicing regimes so that
    # sub_df views automatically include the _derived_entity_id column.
    if unified_meta is not None and unified_meta.input_cols:
        time_col = str(unified_meta.time_col) if unified_meta.time_col else None
        exp_col = add_derived_entity_column(df, list(unified_meta.input_cols), time_col)
    else:
        exp_col, time_col = _detect_panel_columns(df)

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

    is_panel = (
        exp_col is not None
        and time_col is not None
        and exp_col in sub_df.columns
        and time_col in sub_df.columns
    )
    within_pool: frozenset[str] = frozenset()
    if is_panel:
        assert exp_col is not None
        within_pool = _within_entity_pool(sub_df, pool, exp_col)

    for y_col in outs:
        for x_subset in _subset_iterator(pool, sizes, max_per_size=max_subsets_per_k):
            if not _corr_ok(x_subset, corr):
                continue
            prep = _prepare_xy(sub_df, x_subset, y_col)
            if prep is None:
                continue
            x_mat, y_vec = prep
            r2, ar2, coefs, pvals, sig, ols_diag = _fit_ols_or_ridge(
                x_mat, y_vec, x_subset
            )
            if r2 < MIN_R2:
                continue
            # Panel augmentation: run panel diagnostics on within-entity features
            panel_diag: DiagnosticResult | None = None
            if is_panel:
                assert exp_col is not None and time_col is not None
                within_subset = [f for f in x_subset if f in within_pool]
                if within_subset:
                    _build_panel_design(sub_df, within_subset, y_col, exp_col, time_col)
                    panel_diag = diagnose_panel(sub_df, y_col, exp_col, within_subset)
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
                    diagnostic=ols_diag,
                    panel_diagnostic=panel_diag,
                )
            )

    # Panel screening: one PanelOLS fit per output; time_col always included.
    if is_panel:
        assert exp_col is not None and time_col is not None
        within_features = [f for f in pool if f in within_pool]
        if time_col in sub_df.columns and time_col not in within_features:
            within_features = [time_col] + within_features
        for y_col in outs:
            panel_res = _fit_panel_ols(
                sub_df, within_features, y_col, exp_col, time_col
            )
            if panel_res is not None:
                r2, ar2, coefs, pvals, sig, panel_diag = panel_res
                lit = _literature_score(retriever, y_col, within_features)
                candidates.append(
                    _Candidate(
                        y_col=y_col,
                        x_cols=within_features,
                        r_squared=r2,
                        adj_r_squared=ar2,
                        n_obs=panel_diag.effective_n,
                        coefficients=coefs,
                        p_values=pvals,
                        significant_variables=sig,
                        lit_score=lit,
                        diagnostic=panel_diag,
                        panel_diagnostic=None,
                    )
                )

    candidates.sort(key=lambda c: c.r_squared, reverse=True)
    pool_c = candidates[:TOP_POOL]

    ranked = sorted(
        pool_c,
        key=lambda c: (
            c.lit_score * c.r_squared * _effective_diagnostic_score(c),
            c.lit_score,
            c.r_squared,
        ),
        reverse=True,
    )
    strong = [c for c in ranked if c.lit_score >= MIN_LIT]
    chosen = strong[:FINAL_N] if len(strong) >= 2 else ranked[:FINAL_N]

    bundles: list[dict] = []
    for i, c in enumerate(chosen, start=1):
        hid = f"H{i}"
        oid = secrets.token_hex(4)
        mid = secrets.token_hex(4)
        is_panel_fe = c.diagnostic and c.diagnostic.model_family == "panel_fe"
        model_type = "panel_fixed_effects" if is_panel_fe else "screening_linear"
        desc = _candidate_description(c)
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
                    "model_family": "screening_panel_fe"
                    if is_panel_fe
                    else "screening_ols",
                    "priority_score": round(min(1.0, c.lit_score), 4),
                    "is_refinement": False,
                },
                "model_result": {
                    "id": mid,
                    "hypothesis_id": hid,
                    "model_type": model_type,
                    "r_squared": round(c.r_squared, 4),
                    "adj_r_squared": round(c.adj_r_squared, 4),
                    "n_observations": c.n_obs,
                    "coefficients": {k: round(v, 6) for k, v in c.coefficients.items()},
                    "p_values": {k: round(v, 6) for k, v in c.p_values.items()},
                    "significant_variables": c.significant_variables,
                    "match_score": round(min(1.0, c.lit_score), 4),
                    "validation_passed": c.r_squared >= MIN_R2
                    and c.lit_score >= MIN_LIT,
                    "diagnostic_score": _effective_diagnostic_score(c),
                },
                "diagnostics": {
                    "ols": c.diagnostic.to_dict() if c.diagnostic else None,
                    "panel": c.panel_diagnostic.to_dict()
                    if c.panel_diagnostic
                    else None,
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
    # Derive entity from input combinations BEFORE slicing regimes
    if unified_meta is not None and unified_meta.input_cols:
        time_col = str(unified_meta.time_col) if unified_meta.time_col else None
        exp_col = add_derived_entity_column(df, list(unified_meta.input_cols), time_col)
    else:
        exp_col, time_col = _detect_panel_columns(df)

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

    is_panel = (
        exp_col is not None
        and time_col is not None
        and exp_col in sub_df.columns
        and time_col in sub_df.columns
    )
    within_pool: frozenset[str] = frozenset()
    if is_panel:
        assert exp_col is not None
        within_pool = _within_entity_pool(sub_df, pool, exp_col)

    candidates: list[_Candidate] = []
    param_sizes = (1, 2, 3, 4)
    time_param_sizes = (1, 2, 3)
    max_subsets_per_k = 36
    param_pool = [c for c in pool if c != time_col]

    subset_specs: list[list[str]] = []
    seen_subsets: set[tuple[str, ...]] = set()

    def _add_subset(cols: list[str]) -> None:
        key = tuple(cols)
        if not cols or key in seen_subsets:
            return
        seen_subsets.add(key)
        subset_specs.append(cols)

    # Parameter-only treatment/input models: 1, 2, 3, and 4 parameter subsets.
    for subset in _subset_iterator(
        param_pool, param_sizes, max_per_size=max_subsets_per_k
    ):
        _add_subset(subset)

    # Time + treatment/input models: bounded combinations with time forced in.
    if time_col and time_col in sub_df.columns:
        for subset in _subset_iterator(
            param_pool, time_param_sizes, max_per_size=max_subsets_per_k
        ):
            _add_subset([time_col, *subset])

    for y_col in outs:
        for x_subset in subset_specs:
            if not _corr_ok(x_subset, corr):
                continue
            prep = _prepare_xy(sub_df, x_subset, y_col)
            if prep is None:
                continue
            x_mat, y_vec = prep
            r2, ar2, coefs, pvals, sig, ols_diag = _fit_ols_or_ridge(
                x_mat, y_vec, x_subset
            )
            if r2 < MIN_R2:
                continue
            panel_diag: DiagnosticResult | None = None
            if is_panel:
                assert exp_col is not None and time_col is not None
                within_subset = [f for f in x_subset if f in within_pool]
                if within_subset:
                    _build_panel_design(sub_df, within_subset, y_col, exp_col, time_col)
                    panel_diag = diagnose_panel(sub_df, y_col, exp_col, within_subset)
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
                    diagnostic=ols_diag,
                    panel_diagnostic=panel_diag,
                )
            )

    candidates.sort(
        key=lambda c: c.r_squared * _effective_diagnostic_score(c), reverse=True
    )

    # Panel screening: one PanelOLS fit per output using within-entity features.
    # Treatment-level inputs (constant within a run) are absorbed by entity dummies —
    # time_col is explicitly included as the kinetics regressor even when within_pool
    # is otherwise empty (all inputs are between-entity in many experimental layouts).
    panel_candidates: list[_Candidate] = []
    if is_panel:
        assert exp_col is not None and time_col is not None
        within_features = [f for f in pool if f in within_pool]
        # Always include time as a within-entity predictor (script pattern)
        if time_col in sub_df.columns and time_col not in within_features:
            within_features = [time_col] + within_features
        for y_col in outs:
            panel_res = _fit_panel_ols(
                sub_df, within_features, y_col, exp_col, time_col
            )
            if panel_res is not None:
                r2, ar2, coefs, pvals, sig, panel_diag = panel_res
                panel_candidates.append(
                    _Candidate(
                        y_col=y_col,
                        x_cols=within_features,
                        r_squared=r2,
                        adj_r_squared=ar2,
                        n_obs=panel_diag.effective_n,
                        coefficients=coefs,
                        p_values=pvals,
                        significant_variables=sig,
                        lit_score=0.0,
                        diagnostic=panel_diag,
                        panel_diagnostic=None,
                    )
                )
    panel_candidates.sort(key=lambda c: c.adj_r_squared, reverse=True)

    def _model_class(c: _Candidate) -> str:
        has_time = bool(time_col and time_col in c.x_cols)
        if has_time and len(c.x_cols) == 1:
            return "time_only"
        if has_time:
            return "time_plus_parameter"
        return "parameter_only"

    def _rank_key(c: _Candidate) -> float:
        score = (
            c.adj_r_squared
            if c.diagnostic and c.diagnostic.model_family == "panel_fe"
            else c.r_squared
        )
        return float(score * _effective_diagnostic_score(c))

    all_candidates = [*panel_candidates, *candidates]
    buckets = {
        "time_only": sorted(
            [c for c in all_candidates if _model_class(c) == "time_only"],
            key=_rank_key,
            reverse=True,
        ),
        "time_plus_parameter": sorted(
            [c for c in all_candidates if _model_class(c) == "time_plus_parameter"],
            key=_rank_key,
            reverse=True,
        ),
        "parameter_only": sorted(
            [c for c in all_candidates if _model_class(c) == "parameter_only"],
            key=_rank_key,
            reverse=True,
        ),
    }

    # Keep the final page balanced across the three scientific questions:
    # kinetics over time, treatment/input effects, and time-adjusted treatment effects.
    quotas = {"time_only": 2, "time_plus_parameter": 5, "parameter_only": 5}
    chosen: list[_Candidate] = []
    chosen_keys: set[tuple[str, tuple[str, ...]]] = set()

    def _choose(c: _Candidate) -> None:
        key = (c.y_col, tuple(c.x_cols))
        if key in chosen_keys or len(chosen) >= STATS_TOP_N:
            return
        chosen_keys.add(key)
        chosen.append(c)

    for name, quota in quotas.items():
        for c in buckets[name][:quota]:
            _choose(c)

    if len(chosen) < STATS_TOP_N:
        leftovers = sorted(all_candidates, key=_rank_key, reverse=True)
        for c in leftovers:
            _choose(c)
            if len(chosen) >= STATS_TOP_N:
                break

    bundles: list[dict] = []
    for i, c in enumerate(chosen, start=1):
        hid = f"H{i}"
        oid = secrets.token_hex(4)
        mid = secrets.token_hex(4)
        is_panel_fe = c.diagnostic and c.diagnostic.model_family == "panel_fe"
        model_type = "panel_fixed_effects" if is_panel_fe else "screening_linear"
        desc = _candidate_description(c)
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
                    "model_family": "screening_panel_fe"
                    if is_panel_fe
                    else "screening_ols",
                    "priority_score": round(c.r_squared, 4),
                    "is_refinement": False,
                },
                "model_result": {
                    "id": mid,
                    "hypothesis_id": hid,
                    "model_type": model_type,
                    "r_squared": round(c.r_squared, 4),
                    "adj_r_squared": round(c.adj_r_squared, 4),
                    "n_observations": c.n_obs,
                    "coefficients": {k: round(v, 6) for k, v in c.coefficients.items()},
                    "p_values": {k: round(v, 6) for k, v in c.p_values.items()},
                    "significant_variables": c.significant_variables,
                    "match_score": 0.0,
                    "validation_passed": c.r_squared >= MIN_R2,
                    "diagnostic_score": _effective_diagnostic_score(c),
                },
                "diagnostics": {
                    "ols": c.diagnostic.to_dict() if c.diagnostic else None,
                    "panel": c.panel_diagnostic.to_dict()
                    if c.panel_diagnostic
                    else None,
                    "screening_model_class": _model_class(c),
                    "extended": {
                        "ols": _ols_extended_diagnostics(sub_df, c.x_cols, c.y_col),
                        "panel": _panel_extended_diagnostics(
                            sub_df, c.x_cols, c.y_col, exp_col, time_col
                        ),
                    },
                    "additional_families": _additional_family_diagnostics(
                        sub_df, c.x_cols, c.y_col, exp_col, time_col
                    ),
                    "overall_robustness_score": _effective_diagnostic_score(c),
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
        zip(
            chosen,
            cit_chunk_lists,
            cit_queries,
            cit_search_queries,
            cit_q_vecs,
            cit_sq_vecs,
        ),
        start=1,
    ):
        hid = f"H{i}"
        oid = secrets.token_hex(4)
        mid = secrets.token_hex(4)
        desc = _candidate_description(c)
        rationale = (
            f"Automated screening fit (OLS or ridge fallback) with in-sample R² "
            f"{c.r_squared:.3f}. Corpus embedding resemblance peak "
            f"{c.lit_score:.0%} for a query built from this outcome and predictors."
        )
        citations: list[dict] = []
        for ch in cit_chunks[:1]:
            citations.append(
                {
                    "id": f"{c.y_col}:0:{ch.doc_id[:12]}",
                    "source": "corpus",
                    "title": ch.title or ch.source_file,
                    "url": ch.source_file,
                    "year": None,
                    "similarity_score": round(float(ch.similarity_score), 4),
                    "abstract_snippet": ch.text[:500].strip() if ch.text else None,
                    "variable": c.y_col,
                    "hypothesis_id": hid,
                }
            )
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
    v: np.ndarray = get_embedder().embed_one(text).astype(np.float64)
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
        best = max(
            papers,
            key=lambda p: _score_against_query(
                sq_vec, f"{p.title}. {p.abstract}"[:1000]
            ),
        )
        score = _score_against_query(sq_vec, f"{best.title}. {best.abstract}"[:1000])
        if score < EXT_CIT_MIN_SCORE:
            logger.debug("arXiv best score {:.1%} below threshold — skipping", score)
            return None
        return {
            "id": f"arxiv:{best.arxiv_id}",
            "source": "arxiv",
            "title": best.title,
            "url": best.url,
            "year": best.published[:4]
            if best.published and best.published != "unknown"
            else None,
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
        best = max(
            papers,
            key=lambda p: _score_against_query(
                sq_vec, f"{p.title}. {p.abstract}"[:1000]
            ),
        )
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


def _jina_citation(
    search_query: str, sq_vec: np.ndarray, hid: str, variable: str
) -> dict | None:
    try:
        return _best_external_citation(
            JinaSearchClient().search(search_query),
            sq_vec,
            hid,
            variable,
            source="jina",
            id_prefix="jina",
            get_id=lambda p: p.url,
            get_title=lambda p: p.title,
            get_abstract=lambda p: p.abstract,
            get_url=lambda p: p.url,
            get_year=lambda p: p.published[:4] if p.published else None,
        )
    except Exception as e:
        logger.warning("Jina citation failed: {}", e)
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
        _jina_citation,
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
    return cast(np.ndarray, v / n if n > 1e-12 else v)


def _citations_for_candidate(
    retriever: Retriever, c: _Candidate, hypothesis_label: str
) -> list[dict]:
    embedder = get_embedder()
    query = _llm_grounding_query(
        c.y_col, c.x_cols
    )  # natural-language → corpus retrieval
    search_query = _external_search_query(
        c.y_col, c.x_cols
    )  # domain keywords → arXiv/S2 search + scoring

    # sq_vec: used to score arXiv/S2 results (domain-keyword language ↔ paper abstract language)
    sq_vec = _norm_vec(embedder.embed_one(search_query).astype(np.float64))

    logger.debug("Citation query: {} | search: {}", query, search_query)

    out: list[dict] = []

    # 1 — uploaded corpus (scored with q_vec — same register as retrieval query)
    try:
        chunks = retriever.retrieve(
            query=query, top_k=1, min_similarity=LIT_QUERY_MIN_SIM
        )
        for ch in chunks[:1]:
            out.append(
                {
                    "id": f"{c.y_col}:0:{ch.doc_id[:12]}",
                    "source": "corpus",
                    "title": ch.title or ch.source_file,
                    "url": ch.source_file,
                    "year": None,
                    "similarity_score": round(float(ch.similarity_score), 4),
                    "abstract_snippet": ch.text[:500].strip() if ch.text else None,
                    "variable": c.y_col,
                    "hypothesis_id": hypothesis_label,
                }
            )
    except Exception as e:
        logger.debug("Corpus citation retrieve failed: {}", e)

    out.extend(_external_citations(search_query, sq_vec, hypothesis_label, c.y_col))

    return out
