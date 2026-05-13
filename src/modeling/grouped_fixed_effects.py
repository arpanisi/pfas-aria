"""
Grouped Fixed Effects Panel Model (Bonhomme & Manresa, 2015).

Discovers latent experiment groups with shared temporal behavior by clustering
on residual temporal structure.

Model:
    y_it = x_it'β + α_{g_i,t} + ε_it

where:
    i = experiment
    t = time
    g_i = latent group assignment for experiment i
    α_{g_i,t} = group-specific time fixed effect
    β = global coefficients (shared across groups)

Algorithm:
    1. Fit global panel OLS: y_it = x_it'β + ε_it
    2. Extract residuals: e_it = y_it - x_it'β
    3. Cluster experiments by residual time-series using k-means
    4. Re-estimate β with group-specific time dummies
    5. Iterate until convergence

Number of groups K selected via BIC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GroupedFixedEffectsResult:
    """Result from grouped fixed effects estimation."""

    k: int                                  # number of groups
    coefficients: dict[str, float]          # global coefficients
    group_assignments: dict               # experiment_id → group_id
    group_time_effects: dict[int, dict[float, float]]  # group_id → {time: effect}
    bic: float
    aic: float
    r_squared: float
    converged: bool
    n_iterations: int

    predictor_names: list[str]
    time_values: list[float]


class GroupedFixedEffects:
    """
    Grouped fixed effects panel model.

    Usage:
        gfe = GroupedFixedEffects(max_k=5, max_iter=20)
        result = gfe.fit(df, meta, outcome_col='fluoride_yield')
    """

    def __init__(
        self,
        max_k: int = 5,
        min_k: int = 2,
        max_iter: int = 20,
        tol: float = 1e-3,
        random_state: int | None = None,
    ):
        self.max_k = max_k
        self.min_k = min_k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(
        self,
        df: pd.DataFrame,
        meta,  # UnifiedSheetMeta
        outcome_col: str,
        predictor_cols: list[str] | None = None,
    ) -> GroupedFixedEffectsResult:
        """
        Fit grouped FE model, selecting optimal K via BIC.
        """
        if predictor_cols is None:
            predictor_cols = meta.candidate_predictor_cols(df)

        X, y, times, exp_ids, valid_idx = self._prepare_data(
            df, meta, outcome_col, predictor_cols
        )

        if len(np.unique(exp_ids)) < self.min_k:
            raise ValueError(
                f"Too few experiments ({len(np.unique(exp_ids))}) for grouping. "
                f"Need at least {self.min_k}."
            )

        logger.info(
            f"Fitting GFE: {len(X)} obs, {len(np.unique(exp_ids))} experiments, "
            f"{X.shape[1]} predictors, outcome={outcome_col}"
        )

        # Fit for each K
        results = []
        for k in range(self.min_k, min(self.max_k + 1, len(np.unique(exp_ids)))):
            try:
                res = self._fit_single_k(
                    X, y, times, exp_ids, k, predictor_cols, meta
                )
                results.append(res)
                logger.info(f"  K={k}: BIC={res.bic:.2f}, R²={res.r_squared:.3f}")
            except Exception as e:
                logger.warning(f"  K={k} failed: {e}")

        if not results:
            raise ValueError("All K values failed")

        best = min(results, key=lambda r: r.bic)
        logger.info(f"Selected K={best.k} (BIC={best.bic:.2f})")
        return best

    def _prepare_data(
        self, df, meta, outcome_col: str, predictor_cols: list[str]
    ):
        """Extract numeric arrays."""
        y_raw = pd.to_numeric(df[outcome_col], errors='coerce')
        X_raw = df[predictor_cols].apply(pd.to_numeric, errors='coerce')
        times_raw = pd.to_numeric(df[meta.time_col], errors='coerce')
        exp_ids_raw = df[meta.experiment_id_col]

        valid = (
            y_raw.notna()
            & times_raw.notna()
            & X_raw.notna().all(axis=1)
        )
        valid_idx = np.where(valid)[0]

        y = y_raw[valid].values
        times = times_raw[valid].values
        exp_ids = exp_ids_raw[valid].values
        X = X_raw[valid].values

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X, y, times, exp_ids, valid_idx

    def _fit_single_k(
        self,
        X: np.ndarray,
        y: np.ndarray,
        times: np.ndarray,
        exp_ids: np.ndarray,
        k: int,
        predictor_names: list[str],
        meta,
    ) -> GroupedFixedEffectsResult:
        """Fit for a single K via iterative clustering."""
        n, p = X.shape
        unique_times = np.unique(times)
        unique_exps = np.unique(exp_ids)
        n_exps = len(unique_exps)

        # Step 1: Global OLS (ignoring groups)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

        # Step 2: Extract residuals per experiment
        residuals = y - X @ beta

        # Build residual matrix: rows = experiments, cols = timepoints
        residual_matrix = np.full((n_exps, len(unique_times)), np.nan)
        for i, exp in enumerate(unique_exps):
            mask = exp_ids == exp
            for j, t in enumerate(unique_times):
                t_mask = mask & (times == t)
                if t_mask.any():
                    residual_matrix[i, j] = residuals[t_mask].mean()

        # Fill NaNs with column mean (some experiments may not have all timepoints)
        col_means = np.nanmean(residual_matrix, axis=0)
        for j in range(residual_matrix.shape[1]):
            mask = np.isnan(residual_matrix[:, j])
            residual_matrix[mask, j] = col_means[j]

        # Step 3: K-means clustering on residual trajectories
        kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        group_labels = kmeans.fit_predict(residual_matrix)

        # Map experiments to groups
        exp_to_group = {exp: group_labels[i] for i, exp in enumerate(unique_exps)}

        # Iterate: re-estimate with group-specific time effects
        beta_prev = beta.copy()
        for iteration in range(self.max_iter):
            # Build design matrix with group × time dummies
            X_aug, group_time_map = self._build_augmented_design(
                X, times, exp_ids, exp_to_group, unique_times, k
            )

            # OLS
            params = np.linalg.lstsq(X_aug, y, rcond=None)[0]
            beta = params[:p]
            group_time_params = params[p:]

            # Re-cluster on new residuals
            residuals_new = y - X @ beta
            residual_matrix_new = np.full((n_exps, len(unique_times)), np.nan)
            for i, exp in enumerate(unique_exps):
                mask = exp_ids == exp
                for j, t in enumerate(unique_times):
                    t_mask = mask & (times == t)
                    if t_mask.any():
                        residual_matrix_new[i, j] = residuals_new[t_mask].mean()

            for j in range(residual_matrix_new.shape[1]):
                mask = np.isnan(residual_matrix_new[:, j])
                if mask.any():
                    residual_matrix_new[mask, j] = np.nanmean(residual_matrix_new[:, j])

            group_labels_new = kmeans.fit_predict(residual_matrix_new)
            exp_to_group_new = {
                exp: group_labels_new[i] for i, exp in enumerate(unique_exps)
            }

            # Check convergence
            if np.allclose(beta, beta_prev, atol=self.tol) and exp_to_group == exp_to_group_new:
                converged = True
                break

            beta_prev = beta.copy()
            exp_to_group = exp_to_group_new
        else:
            converged = False

        # Extract group-time effects
        group_time_effects = {}
        for g in range(k):
            group_time_effects[g + 1] = {}
            for t in unique_times:
                idx = group_time_map.get((g, t))
                if idx is not None:
                    group_time_effects[g + 1][float(t)] = float(group_time_params[idx])

        # Compute fit statistics
        y_pred = X @ beta
        for i, (exp, t) in enumerate(zip(exp_ids, times)):
            g = exp_to_group[exp]
            idx = group_time_map.get((g, t))
            if idx is not None:
                y_pred[i] += group_time_params[idx]

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot

        # BIC/AIC
        n_params = p + k * len(unique_times)
        log_lik = -0.5 * n * np.log(2 * np.pi * ss_res / n) - 0.5 * n
        bic = -2 * log_lik + n_params * np.log(n)
        aic = -2 * log_lik + 2 * n_params

        # Package
        coefficients = {name: float(beta[j]) for j, name in enumerate(predictor_names)}
        group_assignments = {str(exp): int(exp_to_group[exp]) + 1 for exp in unique_exps}

        return GroupedFixedEffectsResult(
            k=k,
            coefficients=coefficients,
            group_assignments=group_assignments,
            group_time_effects=group_time_effects,
            bic=bic,
            aic=aic,
            r_squared=r2,
            converged=converged,
            n_iterations=iteration + 1,
            predictor_names=predictor_names,
            time_values=unique_times.tolist(),
        )

    def _build_augmented_design(
        self,
        X: np.ndarray,
        times: np.ndarray,
        exp_ids: np.ndarray,
        exp_to_group: dict,
        unique_times: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, dict]:
        """Build design matrix with group × time dummies."""
        n, p = X.shape
        n_times = len(unique_times)

        # group × time dummies
        group_time_dummies = np.zeros((n, k * n_times))
        group_time_map = {}

        for i in range(n):
            g = exp_to_group[exp_ids[i]]
            t = times[i]
            t_idx = np.where(unique_times == t)[0][0]
            col_idx = g * n_times + t_idx
            group_time_dummies[i, col_idx] = 1.0
            group_time_map[(g, t)] = col_idx

        X_aug = np.hstack([X, group_time_dummies])
        return X_aug, group_time_map
