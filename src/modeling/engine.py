"""
Modeling Engine.
Executes statistical and ML models for each hypothesis.
Supports: OLS, Fixed Effects, Random Effects, LASSO, Gradient Boosting.
All models return a standardized ModelResult object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from src.agents.hypothesis import Hypothesis
from src.utils.exceptions import ModelingError
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ── Output schema ─────────────────────────────────────────────────────────────


@dataclass
class ModelResult:
    """Standardized output from any model family."""

    hypothesis_id: str
    model_type: str  # "ols" | "fixed_effects" | "random_effects" | "lasso" | "gradient_boosting"
    outcome_variable: str
    variables_used: list[str]
    interaction_terms: list[tuple[str, str]]

    # Core statistics
    coefficients: dict[str, float]
    p_values: dict[str, float]
    confidence_intervals: dict[str, tuple[float, float]]
    standard_errors: dict[str, float]

    # Fit metrics
    r_squared: float
    adj_r_squared: float
    aic: float | None
    bic: float | None
    n_observations: int

    # Significance
    significant_variables: list[str]  # p < 0.05
    highly_significant: list[str]  # p < 0.01

    # Raw model object for validation
    model_object: Any = field(default=None, repr=False)
    residuals: pd.Series | None = field(default=None, repr=False)
    fitted_values: pd.Series | None = field(default=None, repr=False)

    # Status
    success: bool = True
    error_message: str = ""
    warnings: list[str] = field(default_factory=list)


# ── Engine ────────────────────────────────────────────────────────────────────


class ModelingEngine:
    """
    Executes models for a given hypothesis against a labeled DataFrame.
    Handles data preparation, model fitting, and result extraction.
    """

    ALPHA = 0.05  # Significance threshold

    def run(
        self,
        hypothesis: Hypothesis,
        df: pd.DataFrame,
        per_regime: bool = False,
    ) -> list[ModelResult]:
        """
        Fit model(s) for a hypothesis.
        If per_regime=True and regime column exists, fits one model per regime.
        Always fits one model on the full dataset.

        Returns list of ModelResult (one global + one per regime if requested).
        """
        results = []

        # Global model
        logger.info(
            f"Fitting [{hypothesis.model_family}] for {hypothesis.id} "
            f"on full dataset (n={len(df)})"
        )
        result = self._fit(hypothesis, df, label="global")
        results.append(result)

        # Per-regime models
        if per_regime and "regime" in df.columns:
            for regime_label in df["regime"].unique():
                regime_df = df[df["regime"] == regime_label].copy()
                if len(regime_df) < 10:
                    logger.warning(
                        f"Regime {regime_label} too small (n={len(regime_df)}) — skipping"
                    )
                    continue
                logger.info(
                    f"Fitting [{hypothesis.model_family}] for {hypothesis.id} "
                    f"on {regime_label} (n={len(regime_df)})"
                )
                r = self._fit(hypothesis, regime_df, label=regime_label)
                results.append(r)

        return results

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def _fit(self, hypothesis: Hypothesis, df: pd.DataFrame, label: str) -> ModelResult:
        """Prepare data and dispatch to the correct model family."""
        try:
            x_design, y, col_names = self._prepare_data(hypothesis, df)

            if x_design.shape[0] < x_design.shape[1] + 2:
                return self._error_result(
                    hypothesis,
                    f"Insufficient observations (n={x_design.shape[0]}) for {x_design.shape[1]} predictors",
                )

            model_type = hypothesis.model_family

            if model_type == "ols":
                return self._fit_ols(hypothesis, x_design, y, col_names, df, label)
            elif model_type in ("fixed_effects", "random_effects"):
                return self._fit_panel(hypothesis, x_design, y, col_names, df, label)
            elif model_type == "lasso":
                return self._fit_lasso(hypothesis, x_design, y, col_names, df, label)
            elif model_type == "gradient_boosting":
                return self._fit_gradient_boosting(
                    hypothesis, x_design, y, col_names, df, label
                )
            else:
                return self._fit_ols(hypothesis, x_design, y, col_names, df, label)

        except Exception as e:
            logger.warning(f"Model failed for {hypothesis.id}: {e}")
            return self._error_result(hypothesis, str(e))

    # ── Data Preparation ──────────────────────────────────────────────────────

    def _prepare_data(
        self, hypothesis: Hypothesis, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        """Build design matrix X and outcome y from hypothesis spec."""
        outcome = hypothesis.outcome_variable
        primary_vars = list(dict.fromkeys(hypothesis.primary_variables))
        control_vars = list(dict.fromkeys(hypothesis.control_variables))
        all_vars = list(dict.fromkeys(primary_vars + control_vars))

        # Require at least one valid primary predictor for hypothesis fit.
        primary_available = [c for c in primary_vars if c in df.columns]
        if not primary_available:
            raise ModelingError(
                f"No valid primary variables available for {hypothesis.id}"
            )

        # Filter to available columns
        available = [c for c in all_vars if c in df.columns]
        if not available:
            raise ModelingError(f"No valid columns available for {hypothesis.id}")

        working = df[available + [outcome]].dropna()
        if len(working) < 5:
            raise ModelingError(f"Too few observations after dropna: {len(working)}")

        y = working[outcome].astype(float)
        x_design = working[available].copy()

        # Encode categoricals
        x_design = pd.get_dummies(x_design, drop_first=True)

        # Apply suggested transforms
        for col, transform in hypothesis.suggested_transforms.items():
            if col in x_design.columns:
                if transform == "log" and (x_design[col] > 0).all():
                    x_design[col] = np.log(x_design[col])
                elif transform == "sqrt" and (x_design[col] >= 0).all():
                    x_design[col] = np.sqrt(x_design[col])

        # Add interaction terms
        for var1, var2 in hypothesis.interaction_terms:
            col1 = var1 if var1 in x_design.columns else None
            col2 = var2 if var2 in x_design.columns else None
            if col1 and col2:
                interaction_name = f"{var1}_x_{var2}"
                x_design[interaction_name] = x_design[col1] * x_design[col2]

        col_names = list(x_design.columns)
        return x_design, y, col_names

    # ── OLS ───────────────────────────────────────────────────────────────────

    def _fit_ols(
        self,
        hypothesis: Hypothesis,
        x_design: pd.DataFrame,
        y: pd.Series,
        col_names: list[str],
        df: pd.DataFrame,
        label: str,
    ) -> ModelResult:
        import statsmodels.api as sm

        x_const = sm.add_constant(x_design)
        model = sm.OLS(y, x_const).fit()

        return self._extract_statsmodels_result(
            hypothesis=hypothesis,
            model=model,
            col_names=col_names,
            model_type="ols",
            label=label,
        )

    # ── Panel (Fixed / Random Effects) ────────────────────────────────────────

    def _fit_panel(
        self,
        hypothesis: Hypothesis,
        x_design: pd.DataFrame,
        y: pd.Series,
        col_names: list[str],
        df: pd.DataFrame,
        label: str,
    ) -> ModelResult:
        entity_col = None
        time_col = None

        # Try to find entity and time columns in df
        for col in df.columns:
            if any(
                kw in col.lower() for kw in ["id", "entity", "experiment", "sample"]
            ):
                entity_col = col
                break
        for col in df.columns:
            if any(kw in col.lower() for kw in ["time", "hour", "day", "t_"]):
                time_col = col
                break

        if entity_col is None or time_col is None:
            logger.warning(
                f"{hypothesis.id}: Panel columns not found — falling back to OLS"
            )
            return self._fit_ols(hypothesis, x_design, y, col_names, df, label)

        try:
            from linearmodels.panel import PanelOLS, RandomEffects

            # Build panel dataframe
            panel_df = x_design.copy()
            panel_df[hypothesis.outcome_variable] = y.values
            panel_df["__entity__"] = df.loc[x_design.index, entity_col].values
            panel_df["__time__"] = df.loc[x_design.index, time_col].values
            panel_df = panel_df.set_index(["__entity__", "__time__"])

            y_panel = panel_df[hypothesis.outcome_variable]
            x_panel = panel_df.drop(columns=[hypothesis.outcome_variable])

            from typing import Any as AnyModel

            panel_model: AnyModel
            if hypothesis.model_family == "fixed_effects":
                panel_model = PanelOLS(
                    y_panel,
                    x_panel,
                    entity_effects=True,
                    drop_absorbed=True,
                ).fit(cov_type="clustered", cluster_entity=True)
            else:
                panel_model = RandomEffects(y_panel, x_panel).fit()

            return self._extract_linearmodels_result(
                hypothesis=hypothesis,
                model=panel_model,
                col_names=col_names,
                model_type=hypothesis.model_family,
                label=label,
            )

        except Exception as e:
            logger.warning(f"Panel model failed: {e} — falling back to OLS")
            return self._fit_ols(hypothesis, x_design, y, col_names, df, label)

    # ── LASSO ─────────────────────────────────────────────────────────────────

    def _fit_lasso(
        self,
        hypothesis: Hypothesis,
        x_design: pd.DataFrame,
        y: pd.Series,
        col_names: list[str],
        df: pd.DataFrame,
        label: str,
    ) -> ModelResult:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_design)

        lasso = LassoCV(cv=5, random_state=42, max_iter=5000)
        lasso.fit(x_scaled, y)

        y_pred = lasso.predict(x_scaled)
        residuals = pd.Series(y.values - y_pred, index=y.index)
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        n, p = x_design.shape
        adj_r2 = float(1 - (1 - r2) * (n - 1) / (n - p - 1)) if n > p + 1 else r2

        coefficients = {col: float(c) for col, c in zip(col_names, lasso.coef_)}
        significant = [k for k, v in coefficients.items() if abs(v) > 0.01]

        return ModelResult(
            hypothesis_id=f"{hypothesis.id}_{label}",
            model_type="lasso",
            outcome_variable=hypothesis.outcome_variable,
            variables_used=col_names,
            interaction_terms=hypothesis.interaction_terms,
            coefficients=coefficients,
            p_values={k: 0.0 for k in col_names},  # LASSO has no p-values
            confidence_intervals={k: (0.0, 0.0) for k in col_names},
            standard_errors={k: 0.0 for k in col_names},
            r_squared=r2,
            adj_r_squared=adj_r2,
            aic=None,
            bic=None,
            n_observations=n,
            significant_variables=significant,
            highly_significant=significant,
            model_object=lasso,
            residuals=residuals,
            fitted_values=pd.Series(y_pred, index=y.index),
        )

    # ── Gradient Boosting ─────────────────────────────────────────────────────

    def _fit_gradient_boosting(
        self,
        hypothesis: Hypothesis,
        x_design: pd.DataFrame,
        y: pd.Series,
        col_names: list[str],
        df: pd.DataFrame,
        label: str,
    ) -> ModelResult:
        from sklearn.ensemble import GradientBoostingRegressor

        gb = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
        )
        gb.fit(x_design, y)

        y_pred = gb.predict(x_design)
        residuals = pd.Series(y.values - y_pred, index=y.index)
        r2 = float(gb.score(x_design, y))
        n, p = x_design.shape
        adj_r2 = float(1 - (1 - r2) * (n - 1) / (n - p - 1)) if n > p + 1 else r2

        # Feature importance as proxy for coefficients
        importance = {
            col: float(imp) for col, imp in zip(col_names, gb.feature_importances_)
        }
        significant = sorted(importance, key=lambda k: importance[k], reverse=True)[
            : max(1, len(col_names) // 3)
        ]

        return ModelResult(
            hypothesis_id=f"{hypothesis.id}_{label}",
            model_type="gradient_boosting",
            outcome_variable=hypothesis.outcome_variable,
            variables_used=col_names,
            interaction_terms=hypothesis.interaction_terms,
            coefficients=importance,
            p_values={k: 0.0 for k in col_names},
            confidence_intervals={k: (0.0, 0.0) for k in col_names},
            standard_errors={k: 0.0 for k in col_names},
            r_squared=r2,
            adj_r_squared=adj_r2,
            aic=None,
            bic=None,
            n_observations=n,
            significant_variables=significant,
            highly_significant=significant[: max(1, len(significant) // 2)],
            model_object=gb,
            residuals=residuals,
            fitted_values=pd.Series(y_pred, index=y.index),
        )

    # ── Result Extraction ─────────────────────────────────────────────────────

    def _extract_statsmodels_result(
        self,
        hypothesis: Hypothesis,
        model: Any,
        col_names: list[str],
        model_type: str,
        label: str,
    ) -> ModelResult:
        params = model.params.to_dict()
        pvals = model.pvalues.to_dict()
        bse = model.bse.to_dict()
        conf = model.conf_int()

        # Remove constant from named vars
        coefficients = {k: float(v) for k, v in params.items() if k != "const"}
        p_values = {k: float(v) for k, v in pvals.items() if k != "const"}
        std_errors = {k: float(v) for k, v in bse.items() if k != "const"}
        ci = {
            k: (float(conf.loc[k, 0]), float(conf.loc[k, 1]))
            for k in coefficients
            if k in conf.index
        }

        significant = [k for k, v in p_values.items() if v < self.ALPHA]
        highly_sig = [k for k, v in p_values.items() if v < 0.01]

        residuals = pd.Series(model.resid)
        fitted = pd.Series(model.fittedvalues)

        return ModelResult(
            hypothesis_id=f"{hypothesis.id}_{label}",
            model_type=model_type,
            outcome_variable=hypothesis.outcome_variable,
            variables_used=col_names,
            interaction_terms=hypothesis.interaction_terms,
            coefficients=coefficients,
            p_values=p_values,
            confidence_intervals=ci,
            standard_errors=std_errors,
            r_squared=float(model.rsquared),
            adj_r_squared=float(model.rsquared_adj),
            aic=float(model.aic),
            bic=float(model.bic),
            n_observations=int(model.nobs),
            significant_variables=significant,
            highly_significant=highly_sig,
            model_object=model,
            residuals=residuals,
            fitted_values=fitted,
        )

    def _extract_linearmodels_result(
        self,
        hypothesis: Hypothesis,
        model: Any,
        col_names: list[str],
        model_type: str,
        label: str,
    ) -> ModelResult:
        params = model.params.to_dict()
        pvals = model.pvalues.to_dict()
        bse = model.std_errors.to_dict()

        coefficients = {k: float(v) for k, v in params.items()}
        p_values = {k: float(v) for k, v in pvals.items()}
        std_errors = {k: float(v) for k, v in bse.items()}

        ci_df = model.conf_int()
        ci = {
            k: (float(ci_df.loc[k, "lower"]), float(ci_df.loc[k, "upper"]))
            for k in coefficients
            if k in ci_df.index
        }

        significant = [k for k, v in p_values.items() if v < self.ALPHA]
        highly_sig = [k for k, v in p_values.items() if v < 0.01]

        r2 = float(getattr(model, "rsquared", 0.0))
        r2_adj = float(getattr(model, "rsquared_adj", r2))

        return ModelResult(
            hypothesis_id=f"{hypothesis.id}_{label}",
            model_type=model_type,
            outcome_variable=hypothesis.outcome_variable,
            variables_used=col_names,
            interaction_terms=hypothesis.interaction_terms,
            coefficients=coefficients,
            p_values=p_values,
            confidence_intervals=ci,
            standard_errors=std_errors,
            r_squared=r2,
            adj_r_squared=r2_adj,
            aic=None,
            bic=None,
            n_observations=int(model.nobs),
            significant_variables=significant,
            highly_significant=highly_sig,
            model_object=model,
            residuals=None,
            fitted_values=None,
        )

    def _error_result(self, hypothesis: Hypothesis, message: str) -> ModelResult:
        return ModelResult(
            hypothesis_id=hypothesis.id,
            model_type=hypothesis.model_family,
            outcome_variable=hypothesis.outcome_variable,
            variables_used=[],
            interaction_terms=[],
            coefficients={},
            p_values={},
            confidence_intervals={},
            standard_errors={},
            r_squared=0.0,
            adj_r_squared=0.0,
            aic=None,
            bic=None,
            n_observations=0,
            significant_variables=[],
            highly_significant=[],
            success=False,
            error_message=message,
        )
