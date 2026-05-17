"""
Model interpretability diagnostics.

Each model family gets a family-specific diagnostic suite that produces a
DiagnosticResult with raw test statistics, pass/fail flags per test, and a
weighted composite diagnostic_score (0–1). Higher score = more interpretable.

Families covered:
  ols        — Breusch-Pagan, Durbin-Watson, Jarque-Bera, RESET, VIF, Cook's D
  ridge      — penalised placeholder (assumption tests invalid for Ridge)
  panel      — ADF stationarity, ICC, within/between R²
  xgboost    — cross-val R², generalization gap
  two_stage  — stage-1 within-R², stage-2 R², entity count adequacy
  lasso      — CV R², sparsity ratio
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor as _vif
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.stattools import adfuller

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class DiagnosticResult:
    """Interpretability diagnostics for one fitted model.

    Numeric fields are None when not applicable to the model family or when the
    test could not be computed (too few obs, singular matrix, etc.).
    """

    model_family: str  # "ols" | "ridge" | "panel" | "xgboost" | "two_stage" | "lasso"

    # ── Common ────────────────────────────────────────────────────────────────
    aic: float | None = None
    bic: float | None = None
    effective_n: int = 0
    n_predictors: int = 0

    # ── OLS overall model tests ───────────────────────────────────────────────
    f_statistic: float | None = None         # F-test for joint significance
    f_pvalue: float | None = None            # p-value for F-test
    df_model: int | None = None              # degrees of freedom (model)
    df_resid: int | None = None              # degrees of freedom (residual)

    # ── OLS assumption tests ──────────────────────────────────────────────────
    breusch_pagan_stat: float | None = None  # LM statistic
    breusch_pagan_p: float | None = None     # >0.05 → homoscedastic
    durbin_watson: float | None = None       # ~2 → no autocorrelation
    jarque_bera_stat: float | None = None
    jarque_bera_p: float | None = None       # >0.05 → normal residuals
    reset_stat: float | None = None
    reset_p: float | None = None             # >0.05 → correct functional form
    max_vif: float | None = None             # <5 → low multicollinearity
    condition_number: float | None = None    # <30 → acceptable conditioning
    max_cooks_d: float | None = None         # <4/n → no influential outliers

    # ── Panel-specific ────────────────────────────────────────────────────────
    within_r2: float | None = None
    between_r2: float | None = None
    icc: float | None = None                 # intraclass correlation coefficient
    adf_stationary_fraction: float | None = None  # fraction of vars passing ADF
    n_entities: int | None = None            # number of unique entities in panel

    # ── XGBoost-specific ──────────────────────────────────────────────────────
    cv_r2: float | None = None               # mean cross-validation R²
    generalization_gap: float | None = None  # in_sample_r2 − cv_r2; lower = better

    # ── Two-stage-specific ────────────────────────────────────────────────────
    stage1_within_r2: float | None = None
    stage2_r2: float | None = None

    # ── LASSO-specific ────────────────────────────────────────────────────────
    n_nonzero_coefs: int | None = None
    cv_r2_lasso: float | None = None

    # ── Composite ─────────────────────────────────────────────────────────────
    diagnostic_score: float = 0.5            # 0–1; higher → more interpretable
    passed_tests: list[str] = field(default_factory=list)
    failed_tests: list[str] = field(default_factory=list)
    diagnostic_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to a plain dict for API / DB storage."""
        return {
            "model_family": self.model_family,
            "diagnostic_score": self.diagnostic_score,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "warnings": self.diagnostic_warnings,
            # OLS overall
            "f_statistic": self.f_statistic,
            "f_pvalue": self.f_pvalue,
            "df_model": self.df_model,
            "df_resid": self.df_resid,
            # OLS assumptions
            "breusch_pagan_p": self.breusch_pagan_p,
            "durbin_watson": self.durbin_watson,
            "jarque_bera_p": self.jarque_bera_p,
            "reset_p": self.reset_p,
            "max_vif": self.max_vif,
            "condition_number": self.condition_number,
            "max_cooks_d": self.max_cooks_d,
            "aic": self.aic,
            "bic": self.bic,
            # Panel
            "within_r2": self.within_r2,
            "between_r2": self.between_r2,
            "icc": self.icc,
            "adf_stationary_fraction": self.adf_stationary_fraction,
            "n_entities": self.n_entities,
            # XGBoost
            "cv_r2": self.cv_r2,
            "generalization_gap": self.generalization_gap,
            # Two-stage
            "stage1_within_r2": self.stage1_within_r2,
            "stage2_r2": self.stage2_r2,
            # LASSO
            "n_nonzero_coefs": self.n_nonzero_coefs,
            "cv_r2_lasso": self.cv_r2_lasso,
        }


# ── OLS diagnostics ───────────────────────────────────────────────────────────


def diagnose_ols(
    fitted_res,        # statsmodels OLSResults object
    x_mat: np.ndarray,  # design matrix WITHOUT constant, shape (n, k)
    names: list[str],
) -> DiagnosticResult:
    """Full OLS assumption diagnostic suite on a fitted statsmodels OLS result."""
    exog = fitted_res.model.exog  # (n, k+1) — includes constant at column 0
    resid = np.asarray(fitted_res.resid)

    diag = DiagnosticResult(
        model_family="ols",
        aic=float(fitted_res.aic),
        bic=float(fitted_res.bic),
        effective_n=int(fitted_res.nobs),
        n_predictors=int(fitted_res.df_model),
        df_model=int(fitted_res.df_model),
        df_resid=int(fitted_res.df_resid),
    )
    try:
        diag.f_statistic = round(float(fitted_res.fvalue), 4)
        diag.f_pvalue = round(float(fitted_res.f_pvalue), 6)
    except Exception:
        pass

    # ── Breusch-Pagan (heteroscedasticity) ────────────────────────────────────
    try:
        lm, lm_p, _, _ = het_breuschpagan(resid, exog)
        diag.breusch_pagan_stat = round(float(lm), 6)
        diag.breusch_pagan_p = round(float(lm_p), 6)
        _record(diag, "homoscedasticity", lm_p > 0.05)
    except Exception as exc:
        diag.diagnostic_warnings.append(f"BP test skipped: {exc}")

    # ── Durbin-Watson (autocorrelation) ───────────────────────────────────────
    try:
        dw = float(durbin_watson(resid))
        diag.durbin_watson = round(dw, 4)
        _record(diag, "no_autocorrelation", 1.5 < dw < 2.5)
    except Exception as exc:
        diag.diagnostic_warnings.append(f"DW test skipped: {exc}")

    # ── Jarque-Bera (normality of residuals) ──────────────────────────────────
    try:
        jb_stat, jb_p, _, _ = jarque_bera(resid)
        diag.jarque_bera_stat = round(float(jb_stat), 6)
        diag.jarque_bera_p = round(float(jb_p), 6)
        _record(diag, "residual_normality", jb_p > 0.05)
    except Exception as exc:
        diag.diagnostic_warnings.append(f"JB test skipped: {exc}")

    # ── RESET test (functional form misspecification) ─────────────────────────
    try:
        from statsmodels.stats.diagnostic import linear_reset

        reset_res = linear_reset(fitted_res, power=2, use_f=True)
        diag.reset_stat = round(float(reset_res.statistic), 6)
        diag.reset_p = round(float(reset_res.pvalue), 6)
        _record(diag, "functional_form", reset_res.pvalue > 0.05)
    except Exception as exc:
        diag.diagnostic_warnings.append(f"RESET test skipped: {exc}")

    # ── VIF (multicollinearity) ───────────────────────────────────────────────
    try:
        k = x_mat.shape[1]
        if k >= 2:
            # Column 0 of exog is const; predictors start at index 1
            vifs = [float(_vif(exog, i + 1)) for i in range(k)]
            max_vif = max(vifs)
        else:
            max_vif = 1.0
        diag.max_vif = round(max_vif, 4)
        _record(diag, "low_multicollinearity", max_vif < 5.0)
    except Exception as exc:
        diag.diagnostic_warnings.append(f"VIF skipped: {exc}")

    # ── Condition number ──────────────────────────────────────────────────────
    try:
        diag.condition_number = round(float(np.linalg.cond(exog)), 2)
    except Exception:
        pass

    # ── Cook's distance (influential observations) ────────────────────────────
    try:
        from statsmodels.stats.outliers_influence import OLSInfluence

        cooks_d = OLSInfluence(fitted_res).cooks_distance[0]
        diag.max_cooks_d = round(float(np.nanmax(cooks_d)), 6)
        threshold = 4.0 / max(1, diag.effective_n)
        _record(diag, "no_influential_outliers", diag.max_cooks_d < threshold)
    except Exception as exc:
        diag.diagnostic_warnings.append(f"Cook's D skipped: {exc}")

    diag.diagnostic_score = _ols_composite(diag)
    return diag


def _ols_composite(diag: DiagnosticResult) -> float:
    """Weighted composite of OLS assumption tests. Missing tests are excluded."""
    checks = [
        (diag.breusch_pagan_p,  lambda p: p > 0.05,          0.28),  # invalid p-values = critical
        (diag.durbin_watson,    lambda dw: 1.5 < dw < 2.5,   0.22),  # serial corr in residuals
        (diag.max_vif,          lambda v: v < 5.0,            0.22),  # inflated std errors
        (diag.jarque_bera_p,    lambda p: p > 0.05,           0.15),  # OLS inference validity
        (diag.reset_p,          lambda p: p > 0.05,           0.13),  # functional form
    ]
    total_w = weighted_pass = 0.0
    for value, test, weight in checks:
        if value is None:
            continue
        total_w += weight
        if test(value):
            weighted_pass += weight
    return round(weighted_pass / total_w, 4) if total_w > 0 else 0.5


def diagnose_ridge() -> DiagnosticResult:
    """Placeholder diagnostic for Ridge fallback — assumption tests are invalid."""
    d = DiagnosticResult(model_family="ridge")
    d.diagnostic_warnings.append(
        "OLS failed; Ridge used — p-values are invalid and assumption tests unavailable"
    )
    d.diagnostic_score = 0.40  # structural penalty: can't validate assumptions
    return d


# ── Panel diagnostics ─────────────────────────────────────────────────────────


def diagnose_panel(
    df: pd.DataFrame,
    y_col: str,
    entity_col: str,
    x_cols: list[str],
    *,
    within_r2: float | None = None,
    between_r2: float | None = None,
) -> DiagnosticResult:
    """Panel-specific diagnostics: ICC, ADF stationarity on within-entity series."""
    diag = DiagnosticResult(
        model_family="panel",
        effective_n=len(df),
        n_predictors=len(x_cols),
        within_r2=within_r2,
        between_r2=between_r2,
    )

    # ── ICC (intraclass correlation coefficient) ───────────────────────────────
    try:
        y = pd.to_numeric(df[y_col], errors="coerce")
        entity = df[entity_col]
        group_means = y.groupby(entity).transform("mean")
        between_var = float(group_means.var())
        total_var = float(y.var())
        if total_var > 1e-9:
            diag.icc = round(between_var / total_var, 4)
            _record(diag, "icc_adequate", diag.icc >= 0.10)
    except Exception as exc:
        diag.diagnostic_warnings.append(f"ICC skipped: {exc}")

    # ── ADF stationarity on within-entity demeaned series ────────────────────
    try:
        n_tested = n_stationary = 0
        for col in x_cols:
            s = pd.to_numeric(df[col], errors="coerce")
            entity_means = s.groupby(df[entity_col]).transform("mean")
            s_within = (s - entity_means).dropna()
            if len(s_within) < 10:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, adf_p, *_ = adfuller(s_within, autolag="AIC")
                n_tested += 1
                if adf_p < 0.05:  # reject unit root → stationary
                    n_stationary += 1
            except Exception:
                continue
        if n_tested > 0:
            frac = n_stationary / n_tested
            diag.adf_stationary_fraction = round(frac, 4)
            _record(diag, "stationarity", frac >= 0.80)
    except Exception as exc:
        diag.diagnostic_warnings.append(f"ADF skipped: {exc}")

    diag.diagnostic_score = _panel_composite(diag)
    return diag


def _panel_composite(diag: DiagnosticResult) -> float:
    score = 0.0
    total_w = 0.0

    # Stationarity — critical: spurious regression if non-stationary
    if diag.adf_stationary_fraction is not None:
        score += 0.35 * diag.adf_stationary_fraction
        total_w += 0.35

    # ICC — whether FE is warranted (high ICC = entity effects matter)
    if diag.icc is not None:
        icc_reward = min(1.0, diag.icc / 0.5)  # full weight at ICC ≥ 0.5
        score += 0.20 * icc_reward
        total_w += 0.20

    # Within R² — does the model explain within-entity variation
    if diag.within_r2 is not None:
        score += 0.25 * min(1.0, diag.within_r2 / 0.5)
        total_w += 0.25

    # F-test joint significance of FE + predictors (set after diagnose_panel returns)
    if diag.f_pvalue is not None:
        score += 0.15 * (1.0 if diag.f_pvalue < 0.05 else 0.0)
        total_w += 0.15

    # Entity count adequacy: ≥10 entities for reliable clustered SEs
    if diag.n_entities is not None:
        entity_score = min(1.0, diag.n_entities / 10)
        score += 0.05 * entity_score
        total_w += 0.05

    return round(score / total_w, 4) if total_w > 0 else 0.5


# ── XGBoost diagnostics ───────────────────────────────────────────────────────


def diagnose_xgboost(
    model,
    x_mat: np.ndarray,
    y_vec: np.ndarray,
    in_sample_r2: float,
    *,
    cv_folds: int = 5,
) -> DiagnosticResult:
    """XGBoost generalization diagnostics via cross-validation."""
    from sklearn.model_selection import cross_val_score

    diag = DiagnosticResult(
        model_family="xgboost",
        effective_n=len(y_vec),
        n_predictors=x_mat.shape[1],
    )

    try:
        n_splits = min(cv_folds, len(y_vec) // 5)
        if n_splits < 2:
            diag.diagnostic_warnings.append("Too few observations for cross-validation")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_scores = cross_val_score(model, x_mat, y_vec, cv=n_splits, scoring="r2")
            cv_r2 = float(np.mean(cv_scores))
            diag.cv_r2 = round(cv_r2, 4)
            diag.generalization_gap = round(in_sample_r2 - cv_r2, 4)
            _record(diag, "generalization", diag.generalization_gap < 0.15)
            if cv_r2 < 0:
                diag.diagnostic_warnings.append(
                    f"Negative CV R² ({cv_r2:.3f}) — model may not generalise"
                )
    except Exception as exc:
        diag.diagnostic_warnings.append(f"CV skipped: {exc}")

    diag.diagnostic_score = _xgb_composite(diag)
    return diag


def _xgb_composite(diag: DiagnosticResult) -> float:
    if diag.generalization_gap is None:
        return 0.5
    # Gap <0.05 → full score; >0.30 → zero; linear between
    gap_score = 1.0 - min(1.0, max(0.0, (diag.generalization_gap - 0.05) / 0.25))
    if diag.cv_r2 is not None and diag.cv_r2 < 0:
        gap_score *= 0.30
    return round(gap_score, 4)


# ── Two-stage diagnostics ─────────────────────────────────────────────────────


def diagnose_two_stage(
    stage1_within_r2: float,
    stage2_r2: float,
    n_entities: int,
) -> DiagnosticResult:
    """Two-stage model interpretability: stage quality and entity adequacy."""
    diag = DiagnosticResult(
        model_family="two_stage",
        stage1_within_r2=round(stage1_within_r2, 4),
        stage2_r2=round(stage2_r2, 4),
        effective_n=n_entities,
    )

    _record(diag, "kinetics_explained", stage1_within_r2 >= 0.30)
    _record(diag, "treatment_effects_explained", stage2_r2 >= 0.30)

    if n_entities < 10:
        diag.diagnostic_warnings.append(
            f"Only {n_entities} entities — stage-2 estimates may be underpowered"
        )

    diag.diagnostic_score = _two_stage_composite(diag)
    return diag


def _two_stage_composite(diag: DiagnosticResult) -> float:
    s1 = min(1.0, (diag.stage1_within_r2 or 0.0) / 0.60)
    s2 = min(1.0, (diag.stage2_r2 or 0.0) / 0.60)
    score = 0.50 * s1 + 0.50 * s2
    if (diag.effective_n or 0) < 10:
        score *= 0.80
    return round(score, 4)


# ── LASSO diagnostics ─────────────────────────────────────────────────────────


def diagnose_lasso(
    model,
    x_mat: np.ndarray,
    y_vec: np.ndarray,
    *,
    cv_r2: float | None = None,
) -> DiagnosticResult:
    """LASSO sparsity and cross-validation diagnostics."""
    diag = DiagnosticResult(
        model_family="lasso",
        effective_n=len(y_vec),
        n_predictors=x_mat.shape[1],
    )

    try:
        n_nonzero = int(np.sum(np.abs(model.coef_) > 1e-10))
        diag.n_nonzero_coefs = n_nonzero
        sparsity = 1.0 - n_nonzero / max(1, x_mat.shape[1])
        _record(diag, "sparsity", sparsity > 0.20)
    except Exception as exc:
        diag.diagnostic_warnings.append(f"Sparsity check skipped: {exc}")

    if cv_r2 is not None:
        diag.cv_r2_lasso = round(cv_r2, 4)
        _record(diag, "cv_fit", cv_r2 > 0.10)

    diag.diagnostic_score = _lasso_composite(diag)
    return diag


def _lasso_composite(diag: DiagnosticResult) -> float:
    score = 0.0
    total_w = 0.0
    if diag.cv_r2_lasso is not None:
        score += 0.60 * min(1.0, max(0.0, diag.cv_r2_lasso))
        total_w += 0.60
    if diag.n_nonzero_coefs is not None and diag.n_predictors > 0:
        sparsity = 1.0 - diag.n_nonzero_coefs / diag.n_predictors
        score += 0.40 * sparsity
        total_w += 0.40
    return round(score / total_w, 4) if total_w > 0 else 0.5


# ── Helpers ───────────────────────────────────────────────────────────────────


def _record(diag: DiagnosticResult, name: str, passed: bool) -> None:
    (diag.passed_tests if passed else diag.failed_tests).append(name)
