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
        For two_stage, returns two results per fit (kinetics + treatment).
        """
        results = []

        # Global model
        logger.info(
            f"Fitting [{hypothesis.model_family}] for {hypothesis.id} "
            f"on full dataset (n={len(df)})"
        )
        if hypothesis.model_family == "two_stage":
            results.extend(self._fit_two_stage(hypothesis, df, label="global"))
        else:
            results.append(self._fit(hypothesis, df, label="global"))

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
                if hypothesis.model_family == "two_stage":
                    results.extend(
                        self._fit_two_stage(hypothesis, regime_df, label=regime_label)
                    )
                else:
                    results.append(self._fit(hypothesis, regime_df, label=regime_label))

        return results

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def _fit(self, hypothesis: Hypothesis, df: pd.DataFrame, label: str) -> ModelResult:
        """Prepare data and dispatch to the correct model family."""
        try:
            # XGBoost handles NaN natively — bypass the standard dropna-based prepare_data
            # so that sparse features don't eliminate all rows before reaching the model.
            if hypothesis.model_family == "xgboost":
                return self._fit_xgboost(hypothesis, df, label)

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
            elif model_type == "xgboost":
                return self._fit_xgboost(hypothesis, df, label)  # should not reach here
            elif model_type == "two_stage":
                # Should be intercepted in run() before reaching here,
                # but fall back to OLS for safety.
                return self._fit_ols(hypothesis, x_design, y, col_names, df, label)
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

    # ── XGBoost ───────────────────────────────────────────────────────────────

    def _fit_xgboost(
        self,
        hypothesis: Hypothesis,
        df: pd.DataFrame,
        label: str,
    ) -> ModelResult:
        """
        XGBoost regressor with NaN-native handling.

        Unlike OLS/LASSO, XGBoost accepts NaN values in X natively (tree splitting
        learns the optimal direction for missing values). This means the feature
        matrix is built from ALL hypothesis-specified columns without global dropna,
        accessing more features than OLS on sparse datasets.

        Outcome y must be complete (rows with missing y are dropped).
        """
        import xgboost as xgb

        outcome = hypothesis.outcome_variable
        primary_vars = list(dict.fromkeys(hypothesis.primary_variables))
        control_vars = list(dict.fromkeys(hypothesis.control_variables))
        all_vars = list(dict.fromkeys(primary_vars + control_vars))
        available = [c for c in all_vars if c in df.columns]

        if not available:
            return self._error_result(hypothesis, "No valid columns for XGBoost")

        # Require complete y; keep NaN in X (XGBoost handles them natively)
        work = df[available + [outcome]].copy()
        for col in available:
            work[col] = pd.to_numeric(work[col], errors="coerce")
        work[outcome] = pd.to_numeric(work[outcome], errors="coerce")
        work = work[work[outcome].notna()].reset_index(drop=True)

        if len(work) < 10:
            return self._error_result(
                hypothesis, f"Too few complete outcome rows: {len(work)}"
            )

        # Encode categoricals (label-encode; XGBoost needs numeric)
        x_design = work[available].copy()
        for col in x_design.select_dtypes(include="object").columns:
            x_design[col] = pd.factorize(x_design[col])[0].astype(float)
            x_design[col] = x_design[col].replace(-1, np.nan)

        y = work[outcome].astype(float)
        used_cols = list(x_design.columns)

        model = xgb.XGBRegressor(
            n_estimators=300,
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
        model.fit(x_design, y)

        y_pred = model.predict(x_design)
        residuals = pd.Series(y.values - y_pred, index=y.index)
        r2 = float(model.score(x_design, y))
        n, p = x_design.shape
        adj_r2 = float(1 - (1 - r2) * (n - 1) / (n - p - 1)) if n > p + 1 else r2

        importance = {
            col: float(imp) for col, imp in zip(used_cols, model.feature_importances_)
        }
        # Top third by importance treated as "significant" (no p-values from XGBoost)
        top_k = max(1, len(used_cols) // 3)
        significant = sorted(importance, key=lambda k: importance[k], reverse=True)[
            :top_k
        ]

        return ModelResult(
            hypothesis_id=f"{hypothesis.id}_{label}",
            model_type="xgboost",
            outcome_variable=outcome,
            variables_used=used_cols,
            interaction_terms=hypothesis.interaction_terms,
            coefficients=importance,
            p_values={k: 0.0 for k in used_cols},
            confidence_intervals={k: (0.0, 0.0) for k in used_cols},
            standard_errors={k: 0.0 for k in used_cols},
            r_squared=r2,
            adj_r_squared=adj_r2,
            aic=None,
            bic=None,
            n_observations=n,
            significant_variables=significant,
            highly_significant=significant[: max(1, top_k // 2)],
            model_object=model,
            residuals=residuals,
            fitted_values=pd.Series(y_pred, index=y.index),
        )

    # ── Two-Stage Modeling ────────────────────────────────────────────────────

    def _detect_time_col(self, df: pd.DataFrame) -> str | None:
        """Find the time column by name pattern."""
        from src.etl.experimental.detector import DERIVED_ENTITY_COL

        for col in df.columns:
            if col == DERIVED_ENTITY_COL:
                continue
            if any(kw in col.lower() for kw in ["time", "hour", "min", "day", "t_"]):
                if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() >= 3:
                    return str(col)
        return None

    def _fit_two_stage(
        self,
        hypothesis: Hypothesis,
        df: pd.DataFrame,
        label: str,
    ) -> list[ModelResult]:
        """
        Two-stage modeling for panel experimental data.

        Stage 1 — Kinetics: PanelOLS fixed effects, entity=unique input combination,
            time predictor only. The time coefficient is the pooled kinetic rate
            (% yield per time unit). Within-R² measures how much time alone explains
            within-entity variation.

        Stage 2 — Treatment effects: Cross-sectional OLS on run-level AUC.
            One observation per experimental run. Between-entity design variables
            explain which conditions produce higher integrated fluoride yield.
        """
        from src.etl.experimental.detector import add_derived_entity_column

        time_col = self._detect_time_col(df)
        if time_col is None:
            return [self._error_result(hypothesis, "two_stage requires a time column")]

        outcome = hypothesis.outcome_variable
        if outcome not in df.columns:
            return [self._error_result(hypothesis, f"Outcome {outcome!r} not in df")]

        # Derive entity column from all hypothesis-specified features minus time
        all_vars = list(
            dict.fromkeys(hypothesis.primary_variables + hypothesis.control_variables)
        )
        input_cols = [c for c in all_vars if c in df.columns and c != time_col]
        if not input_cols:
            input_cols = [c for c in df.columns if c not in {time_col, outcome}]

        df = df.copy()
        entity_col = add_derived_entity_column(df, input_cols, time_col)
        if entity_col is None:
            return [
                self._error_result(
                    hypothesis, "Could not derive entity column for two_stage"
                )
            ]

        results: list[ModelResult] = []

        try:
            results.append(
                self._fit_kinetics_stage(
                    hypothesis, df, entity_col, time_col, outcome, label
                )
            )
        except Exception as e:
            logger.warning(f"Kinetics stage failed for {hypothesis.id}: {e}")
            results.append(self._error_result(hypothesis, f"Kinetics stage error: {e}"))

        try:
            results.append(
                self._fit_treatment_stage(
                    hypothesis, df, entity_col, time_col, outcome, input_cols, label
                )
            )
        except Exception as e:
            logger.warning(f"Treatment stage failed for {hypothesis.id}: {e}")
            results.append(
                self._error_result(hypothesis, f"Treatment stage error: {e}")
            )

        return results

    def _fit_kinetics_stage(
        self,
        hypothesis: Hypothesis,
        df: pd.DataFrame,
        entity_col: str,
        time_col: str,
        outcome: str,
        label: str,
    ) -> ModelResult:
        """PanelOLS FE with time as the only predictor → pooled kinetic rate."""
        import statsmodels.api as sm
        from linearmodels.panel import PanelOLS

        work = df[[entity_col, time_col, outcome]].copy()
        work[time_col] = pd.to_numeric(work[time_col], errors="coerce")
        work[outcome] = pd.to_numeric(work[outcome], errors="coerce")
        work = work.dropna()

        if len(work) < 20:
            raise ModelingError(f"Too few rows for kinetics stage: {len(work)}")

        panel_df = work.set_index([entity_col, time_col])
        y_panel = panel_df[outcome]
        x_panel = sm.add_constant(
            pd.DataFrame({"time": work[time_col].values}, index=panel_df.index)
        )

        model = PanelOLS(y_panel, x_panel, entity_effects=True, drop_absorbed=True).fit(
            cov_type="clustered", cluster_entity=True
        )

        params = model.params.to_dict()
        pvals = model.pvalues.to_dict()
        bse = model.std_errors.to_dict()
        ci_df = model.conf_int()
        ci = {
            k: (float(ci_df.loc[k, "lower"]), float(ci_df.loc[k, "upper"]))
            for k in params
            if k in ci_df.index
        }

        return ModelResult(
            hypothesis_id=f"{hypothesis.id}_{label}_kinetics",
            model_type="two_stage_kinetics",
            outcome_variable=outcome,
            variables_used=[time_col],
            interaction_terms=[],
            coefficients={k: float(v) for k, v in params.items()},
            p_values={k: float(v) for k, v in pvals.items()},
            confidence_intervals=ci,
            standard_errors={k: float(v) for k, v in bse.items()},
            r_squared=float(model.rsquared),  # within R²
            adj_r_squared=float(getattr(model, "rsquared_adj", model.rsquared)),
            aic=None,
            bic=None,
            n_observations=int(model.nobs),
            significant_variables=[k for k, v in pvals.items() if v < self.ALPHA],
            highly_significant=[k for k, v in pvals.items() if v < 0.01],
            model_object=model,
        )

    def _fit_treatment_stage(
        self,
        hypothesis: Hypothesis,
        df: pd.DataFrame,
        entity_col: str,
        time_col: str,
        outcome: str,
        input_cols: list[str],
        label: str,
    ) -> ModelResult:
        """Cross-sectional OLS on run-level AUC using between-entity design variables."""
        import statsmodels.api as sm

        min_within_fraction = 0.05

        # Identify between-entity features (within-fraction < 5% → not absorbed by FE)
        between_cols: list[str] = []
        for col in input_cols:
            if col not in df.columns:
                continue
            s = pd.to_numeric(df[col], errors="coerce")
            total_var = float(s.var())
            if total_var < 1e-9:
                continue
            # Group on the numeric-coerced series to avoid object-dtype transform errors
            entity_means = s.groupby(df[entity_col]).transform("mean")
            within_var = float((s - entity_means).var())
            if within_var / total_var < min_within_fraction:
                between_cols.append(col)

        if not between_cols:
            raise ModelingError(
                "No between-entity features available for treatment stage"
            )

        # Aggregate to run level: AUC (trapezoidal) per entity
        run_records: list[dict] = []
        for entity_id, grp in df.groupby(entity_col):
            t_arr = pd.to_numeric(grp[time_col], errors="coerce").values
            y_arr = pd.to_numeric(grp[outcome], errors="coerce").values
            mask = np.isfinite(t_arr) & np.isfinite(y_arr)
            if mask.sum() < 2:
                continue
            pairs = sorted(zip(t_arr[mask], y_arr[mask]))
            t_s, y_s = zip(*pairs)
            auc = float(np.trapz(y_s, t_s))

            row: dict = {"_entity": entity_id, "_auc": auc}
            for col in between_cols:
                vals = pd.to_numeric(grp[col], errors="coerce").dropna()
                row[col] = float(vals.iloc[0]) if len(vals) > 0 else np.nan
            run_records.append(row)

        run_df = pd.DataFrame(run_records)
        n_runs = len(run_df)

        # Keep 100%-dense features at run level; relax to 80% if none survive
        dense_cols = [c for c in between_cols if int(run_df[c].notna().sum()) == n_runs]
        if not dense_cols:
            dense_cols = [c for c in between_cols if run_df[c].notna().mean() >= 0.8]
        if not dense_cols:
            raise ModelingError("No dense between-entity features at run level")

        work = run_df[dense_cols + ["_auc"]].dropna()
        if len(work) < len(dense_cols) + 3:
            raise ModelingError(f"Too few runs for treatment stage: {len(work)}")

        x_design = sm.add_constant(work[dense_cols].astype(float))
        y = work["_auc"].astype(float)
        model = sm.OLS(y, x_design).fit()

        params = {k: float(v) for k, v in model.params.items() if k != "const"}
        pvals = {k: float(v) for k, v in model.pvalues.items() if k != "const"}
        bse = {k: float(v) for k, v in model.bse.items() if k != "const"}
        ci_raw = model.conf_int()
        ci = {
            k: (float(ci_raw.loc[k, 0]), float(ci_raw.loc[k, 1]))
            for k in params
            if k in ci_raw.index
        }

        return ModelResult(
            hypothesis_id=f"{hypothesis.id}_{label}_treatment",
            model_type="two_stage_treatment",
            outcome_variable="_auc",
            variables_used=dense_cols,
            interaction_terms=[],
            coefficients=params,
            p_values=pvals,
            confidence_intervals=ci,
            standard_errors=bse,
            r_squared=float(model.rsquared),
            adj_r_squared=float(model.rsquared_adj),
            aic=float(model.aic),
            bic=float(model.bic),
            n_observations=int(model.nobs),
            significant_variables=[k for k, v in pvals.items() if v < self.ALPHA],
            highly_significant=[k for k, v in pvals.items() if v < 0.01],
            model_object=model,
            residuals=pd.Series(model.resid),
            fitted_values=pd.Series(model.fittedvalues),
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
