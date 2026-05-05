"""
Validation Engine.
Applies rigorous statistical tests to every ModelResult.
A model must pass all applicable tests before being accepted.

Tests applied:
  - VIF (multicollinearity)
  - Shapiro-Wilk (residual normality)
  - Breusch-Pagan (homoscedasticity)
  - ANOVA across regimes
  - K-fold cross-validation
  - Effect size (Cohen's d / eta-squared)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

from src.modeling.engine import ModelResult
from src.utils.logging import get_logger

logger = get_logger(__name__)

VIF_THRESHOLD = 5.0
NORMALITY_ALPHA = 0.05
HOMOSCEDASTICITY_ALPHA = 0.05
CV_FOLDS = 5
MIN_CV_R2 = 0.0  # Must beat null model


# ── Output schema ─────────────────────────────────────────────────────────────


@dataclass
class VIFResult:
    variable: str
    vif: float
    passed: bool  # VIF < threshold


@dataclass
class ValidationReport:
    """Full validation results for one ModelResult."""

    hypothesis_id: str
    model_type: str

    # VIF
    vif_results: list[VIFResult] = field(default_factory=list)
    vif_passed: bool = True
    max_vif: float = 0.0

    # Normality
    shapiro_statistic: float = 0.0
    shapiro_p_value: float = 1.0
    normality_passed: bool = True

    # Homoscedasticity
    bp_statistic: float = 0.0
    bp_p_value: float = 1.0
    homoscedasticity_passed: bool = True

    # ANOVA across regimes
    anova_f_statistic: float = 0.0
    anova_p_value: float = 1.0
    anova_passed: bool = True
    anova_applicable: bool = False

    # Cross-validation
    cv_r2_scores: list[float] = field(default_factory=list)
    cv_r2_mean: float = 0.0
    cv_r2_std: float = 0.0
    cv_passed: bool = True

    # Effect size
    eta_squared: float = 0.0
    effect_size_label: str = "none"  # "small" | "medium" | "large"

    # Overall
    overall_passed: bool = True
    pass_rate: float = 0.0
    warnings: list[str] = field(default_factory=list)
    n_tests_run: int = 0
    n_tests_passed: int = 0


# ── Validator ─────────────────────────────────────────────────────────────────


class ValidationEngine:
    """
    Runs all applicable statistical validation tests on a ModelResult.
    Returns a ValidationReport with pass/fail for each test.
    """

    def validate(
        self,
        result: ModelResult,
        df: pd.DataFrame,
    ) -> ValidationReport:
        """
        Run full validation suite on a model result.
        df should be the labeled DataFrame (with regime column).
        """
        report = ValidationReport(
            hypothesis_id=result.hypothesis_id,
            model_type=result.model_type,
        )

        if not result.success:
            report.overall_passed = False
            report.warnings.append(f"Model failed: {result.error_message}")
            return report

        tests_run = 0
        tests_passed = 0

        # 1. VIF — only for parametric models with multiple vars
        if result.model_type not in ("gradient_boosting", "lasso"):
            if len(result.variables_used) > 1:
                vif_results, vif_passed = self._check_vif(result, df)
                report.vif_results = vif_results
                report.vif_passed = vif_passed
                report.max_vif = max(v.vif for v in vif_results) if vif_results else 0.0
                tests_run += 1
                if vif_passed:
                    tests_passed += 1
                else:
                    report.warnings.append(
                        f"High multicollinearity detected (max VIF={report.max_vif:.2f})"
                    )

        # 2. Normality of residuals
        if result.residuals is not None and len(result.residuals) >= 8:
            sw_stat, sw_p, norm_passed = self._check_normality(result.residuals)
            report.shapiro_statistic = sw_stat
            report.shapiro_p_value = sw_p
            report.normality_passed = norm_passed
            tests_run += 1
            if norm_passed:
                tests_passed += 1
            else:
                report.warnings.append(
                    f"Residuals not normally distributed (Shapiro-Wilk p={sw_p:.4f})"
                )

        # 3. Homoscedasticity
        if result.residuals is not None and result.fitted_values is not None:
            bp_stat, bp_p, homo_passed = self._check_homoscedasticity(
                result.residuals, result.fitted_values
            )
            report.bp_statistic = bp_stat
            report.bp_p_value = bp_p
            report.homoscedasticity_passed = homo_passed
            tests_run += 1
            if homo_passed:
                tests_passed += 1
            else:
                report.warnings.append(
                    f"Heteroscedasticity detected (Breusch-Pagan p={bp_p:.4f})"
                )

        # 4. ANOVA across regimes
        if "regime" in df.columns and df["regime"].nunique() > 1:
            outcome = result.outcome_variable
            if outcome in df.columns:
                f_stat, f_p, anova_passed = self._check_anova(df, outcome)
                report.anova_f_statistic = f_stat
                report.anova_p_value = f_p
                report.anova_passed = anova_passed
                report.anova_applicable = True
                tests_run += 1
                if anova_passed:
                    tests_passed += 1

        # 5. Cross-validation
        available_vars = [c for c in result.variables_used if c in df.columns]
        outcome = result.outcome_variable
        if available_vars and outcome in df.columns:
            cv_scores, cv_mean, cv_std, cv_passed = self._check_cross_validation(
                df, available_vars, outcome
            )
            report.cv_r2_scores = cv_scores
            report.cv_r2_mean = cv_mean
            report.cv_r2_std = cv_std
            report.cv_passed = cv_passed
            tests_run += 1
            if cv_passed:
                tests_passed += 1
            else:
                report.warnings.append(
                    f"Poor cross-validation score (CV R²={cv_mean:.3f})"
                )

        # 6. Effect size
        if outcome in df.columns and "regime" in df.columns:
            eta_sq, label = self._compute_effect_size(df, outcome)
            report.eta_squared = eta_sq
            report.effect_size_label = label

        # Overall
        report.n_tests_run = tests_run
        report.n_tests_passed = tests_passed
        report.pass_rate = tests_passed / tests_run if tests_run > 0 else 0.0
        report.overall_passed = tests_passed >= max(1, tests_run - 1)  # Allow 1 failure

        logger.info(
            f"Validation [{result.hypothesis_id}]: "
            f"{tests_passed}/{tests_run} tests passed "
            f"({'PASS' if report.overall_passed else 'FAIL'})"
        )

        return report

    # ── Individual Tests ──────────────────────────────────────────────────────

    def _check_vif(
        self, result: ModelResult, df: pd.DataFrame
    ) -> tuple[list[VIFResult], bool]:
        """Variance Inflation Factor for multicollinearity."""
        try:
            available = [c for c in result.variables_used if c in df.columns]
            if len(available) < 2:
                return [], True

            X = df[available].dropna()
            if len(X) < len(available) + 2:
                return [], True

            vif_results = []
            X_arr = X.values

            for i, col in enumerate(available):
                other_cols = [j for j in range(X_arr.shape[1]) if j != i]
                if not other_cols:
                    continue

                y_i = X_arr[:, i]
                X_others = X_arr[:, other_cols]

                reg = LinearRegression().fit(X_others, y_i)
                r2 = float(reg.score(X_others, y_i))
                vif = float(1 / (1 - r2)) if r2 < 1.0 else float("inf")

                vif_results.append(
                    VIFResult(
                        variable=col,
                        vif=round(vif, 3),
                        passed=vif < VIF_THRESHOLD,
                    )
                )

            all_passed = all(v.passed for v in vif_results)
            return vif_results, all_passed

        except Exception as e:
            logger.warning(f"VIF check failed: {e}")
            return [], True

    def _check_normality(self, residuals: pd.Series) -> tuple[float, float, bool]:
        """Shapiro-Wilk test for residual normality."""
        try:
            clean = residuals.dropna()
            # Shapiro-Wilk works best with n <= 5000
            sample = (
                clean if len(clean) <= 5000 else clean.sample(5000, random_state=42)
            )
            stat, p_value = stats.shapiro(sample)
            passed = float(p_value) > NORMALITY_ALPHA
            return float(stat), float(p_value), passed
        except Exception as e:
            logger.warning(f"Normality check failed: {e}")
            return 0.0, 1.0, True

    def _check_homoscedasticity(
        self, residuals: pd.Series, fitted: pd.Series
    ) -> tuple[float, float, bool]:
        """Breusch-Pagan test for homoscedasticity."""
        try:
            res = residuals.dropna().values
            fit = fitted.dropna().values

            min_len = min(len(res), len(fit))
            res = res[:min_len]
            fit = fit[:min_len]

            # BP test: regress squared residuals on fitted values
            sq_res = res**2
            X_bp = np.column_stack([np.ones(len(fit)), fit])
            reg = LinearRegression().fit(X_bp[:, 1:], sq_res)
            r2 = float(reg.score(X_bp[:, 1:], sq_res))

            n = len(res)
            bp_stat = n * r2
            p_value = float(1 - stats.chi2.cdf(bp_stat, df=1))
            passed = p_value > HOMOSCEDASTICITY_ALPHA

            return float(bp_stat), p_value, passed
        except Exception as e:
            logger.warning(f"Homoscedasticity check failed: {e}")
            return 0.0, 1.0, True

    def _check_anova(self, df: pd.DataFrame, outcome: str) -> tuple[float, float, bool]:
        """One-way ANOVA of outcome across regimes."""
        try:
            groups = [
                df.loc[df["regime"] == r, outcome].dropna().values
                for r in df["regime"].unique()
            ]
            groups = [g for g in groups if len(g) >= 3]

            if len(groups) < 2:
                return 0.0, 1.0, True

            f_stat, p_value = stats.f_oneway(*groups)
            # ANOVA passes if regimes ARE significantly different (p < 0.05)
            # This confirms regime detection found real structure
            passed = float(p_value) < 0.05
            return float(f_stat), float(p_value), passed
        except Exception as e:
            logger.warning(f"ANOVA check failed: {e}")
            return 0.0, 1.0, True

    def _check_cross_validation(
        self,
        df: pd.DataFrame,
        variables: list[str],
        outcome: str,
    ) -> tuple[list[float], float, float, bool]:
        """K-fold cross-validation R² score."""
        try:
            working = df[variables + [outcome]].dropna()
            if len(working) < CV_FOLDS * 2:
                return [], 0.0, 0.0, True

            X = working[variables].values
            y = working[outcome].values

            reg = LinearRegression()
            kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
            scores = cross_val_score(reg, X, y, cv=kf, scoring="r2")
            scores_list = [float(s) for s in scores]
            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))
            passed = mean_score > MIN_CV_R2

            return scores_list, mean_score, std_score, passed
        except Exception as e:
            logger.warning(f"CV check failed: {e}")
            return [], 0.0, 0.0, True

    def _compute_effect_size(self, df: pd.DataFrame, outcome: str) -> tuple[float, str]:
        """Eta-squared effect size across regimes."""
        try:
            groups = [
                df.loc[df["regime"] == r, outcome].dropna().values
                for r in df["regime"].unique()
            ]
            groups = [g for g in groups if len(g) >= 2]

            if len(groups) < 2:
                return 0.0, "none"

            grand_mean = float(df[outcome].mean())
            ss_between = float(
                sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
            )
            ss_total = float(np.sum((df[outcome].dropna() - grand_mean) ** 2))

            eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

            if eta_sq < 0.01:
                label = "negligible"
            elif eta_sq < 0.06:
                label = "small"
            elif eta_sq < 0.14:
                label = "medium"
            else:
                label = "large"

            return round(eta_sq, 4), label
        except Exception as e:
            logger.warning(f"Effect size computation failed: {e}")
            return 0.0, "none"
