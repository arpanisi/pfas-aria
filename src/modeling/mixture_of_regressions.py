"""
Mixture of Regressions for panel data.

Discovers K local linear models where the gating function π_k(x) determines
which model activates for each observation based on input features.

Model:
    p(y | x, t) = Σ_k π_k(x) · N(y | x'β_k + α_k(t), σ_k²)

where:
    π_k(x) = softmax gating function (input-dependent regime probabilities)
    β_k    = regime-specific coefficients
    α_k(t) = regime-specific time effects (fixed effects per timepoint)
    σ_k²   = regime-specific noise variance

Fitted via EM algorithm. Number of regimes K selected via BIC.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MixtureOfRegressionsResult:
    """Result from fitting a mixture of regressions model."""
    
    k: int                                    # number of regimes
    coefficients: dict[int, dict[str, float]] # regime_id → {var: coef}
    time_effects: dict[int, dict[float, float]] # regime_id → {time: effect}
    gating_weights: np.ndarray                # shape (K, n_features) — gating fn params
    regime_probs: np.ndarray                  # shape (n_obs, K) — posterior probabilities
    regime_assignments: np.ndarray            # shape (n_obs,) — hard assignments (argmax)
    bic: float
    aic: float
    log_likelihood: float
    converged: bool
    n_iterations: int
    
    predictor_names: list[str] = field(default_factory=list)
    time_values: list[float] = field(default_factory=list)


class MixtureOfRegressions:
    """
    Mixture of regressions for panel data with input-dependent gating.
    
    Usage:
        mor = MixtureOfRegressions(max_k=5, max_iter=100)
        result = mor.fit(df, meta, outcome_col='fluoride_yield')
    """
    
    def __init__(
        self,
        max_k: int = 5,
        min_k: int = 2,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: int | None = None,
    ):
        self.max_k = max_k
        self.min_k = min_k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
    
    def fit(
        self,
        df: pd.DataFrame,
        meta,  # UnifiedSheetMeta
        outcome_col: str,
        predictor_cols: list[str] | None = None,
    ) -> MixtureOfRegressionsResult:
        """
        Fit mixture of regressions, selecting optimal K via BIC.
        
        Args:
            df: Full dataset
            meta: UnifiedSheetMeta with experiment_id_col, time_col
            outcome_col: Name of the output column to model
            predictor_cols: Subset of input columns to use (if None, uses all non-constant)
        
        Returns:
            MixtureOfRegressionsResult with optimal K
        """
        # Prepare data
        if predictor_cols is None:
            predictor_cols = meta.candidate_predictor_cols(df)
        
        X, y, times, valid_idx = self._prepare_data(
            df, meta, outcome_col, predictor_cols
        )
        
        if len(X) < 20:
            raise ValueError(f"Insufficient data: only {len(X)} valid observations")
        
        logger.info(
            f"Fitting MoR: {len(X)} obs, {X.shape[1]} predictors, outcome={outcome_col}"
        )
        
        # Fit for each K and select best by BIC
        results = []
        for k in range(self.min_k, self.max_k + 1):
            try:
                res = self._fit_single_k(X, y, times, k, predictor_cols)
                results.append(res)
                logger.info(f"  K={k}: BIC={res.bic:.2f}, LL={res.log_likelihood:.2f}")
            except Exception as e:
                logger.warning(f"  K={k} failed: {e}")
        
        if not results:
            raise ValueError("All K values failed to converge")
        
        # Select by BIC (lower is better)
        best = min(results, key=lambda r: r.bic)
        logger.info(f"Selected K={best.k} (BIC={best.bic:.2f})")
        return best
    
    def _prepare_data(
        self, df, meta, outcome_col: str, predictor_cols: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract and clean numeric arrays."""
        # Numeric conversion
        y_raw = pd.to_numeric(df[outcome_col], errors='coerce')
        X_raw = df[predictor_cols].apply(pd.to_numeric, errors='coerce')
        times_raw = pd.to_numeric(df[meta.time_col], errors='coerce')
        
        # Valid rows (no NaN in y, times, or any predictor)
        valid = (
            y_raw.notna()
            & times_raw.notna()
            & X_raw.notna().all(axis=1)
        )
        valid_idx = np.where(valid)[0]
        
        y = y_raw[valid].values
        times = times_raw[valid].values
        X = X_raw[valid].values
        
        # Standardize predictors
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        return X, y, times, valid_idx
    
    def _fit_single_k(
        self,
        X: np.ndarray,
        y: np.ndarray,
        times: np.ndarray,
        k: int,
        predictor_names: list[str],
    ) -> MixtureOfRegressionsResult:
        """Fit MoR for a single value of K via EM."""
        n, p = X.shape
        unique_times = np.unique(times)
        n_times = len(unique_times)
        time_to_idx = {t: i for i, t in enumerate(unique_times)}
        
        # Initialize parameters
        # Gating weights: shape (K, p)
        W = self.rng.normal(0, 0.1, size=(k, p))
        
        # Coefficients: shape (K, p)
        beta = self.rng.normal(0, 0.1, size=(k, p))
        
        # Time effects: shape (K, n_times)
        alpha = np.zeros((k, n_times))
        
        # Variances: shape (K,)
        sigma2 = np.ones(k)
        
        # EM loop
        ll_prev = -np.inf
        for iteration in range(self.max_iter):
            # E-step: compute responsibilities
            gamma = self._e_step(X, y, times, W, beta, alpha, sigma2, time_to_idx)
            
            # M-step: update parameters
            W, beta, alpha, sigma2 = self._m_step(
                X, y, times, gamma, time_to_idx, n_times
            )
            
            # Compute log-likelihood
            ll = self._log_likelihood(X, y, times, W, beta, alpha, sigma2, time_to_idx)
            
            # Check convergence
            if abs(ll - ll_prev) < self.tol:
                converged = True
                break
            ll_prev = ll
        else:
            converged = False
        
        # Compute BIC/AIC
        n_params = k * p + k * p + k * n_times + k  # W + beta + alpha + sigma2
        bic = -2 * ll + n_params * np.log(n)
        aic = -2 * ll + 2 * n_params
        
        # Hard assignments
        assignments = np.argmax(gamma, axis=1)
        
        # Package coefficients
        coefficients = {}
        for regime_id in range(k):
            coefficients[regime_id + 1] = {
                name: float(beta[regime_id, j])
                for j, name in enumerate(predictor_names)
            }
        
        # Time effects
        time_effects = {}
        for regime_id in range(k):
            time_effects[regime_id + 1] = {
                float(t): float(alpha[regime_id, time_to_idx[t]])
                for t in unique_times
            }
        
        return MixtureOfRegressionsResult(
            k=k,
            coefficients=coefficients,
            time_effects=time_effects,
            gating_weights=W,
            regime_probs=gamma,
            regime_assignments=assignments,
            bic=bic,
            aic=aic,
            log_likelihood=ll,
            converged=converged,
            n_iterations=iteration + 1,
            predictor_names=predictor_names,
            time_values=unique_times.tolist(),
        )
    
    def _e_step(
        self,
        X: np.ndarray,
        y: np.ndarray,
        times: np.ndarray,
        W: np.ndarray,
        beta: np.ndarray,
        alpha: np.ndarray,
        sigma2: np.ndarray,
        time_to_idx: dict,
    ) -> np.ndarray:
        """Compute posterior probabilities γ_ik = p(z_i=k | x_i, y_i)."""
        n, p = X.shape
        k = len(W)
        
        # Gating probabilities: π_k(x_i) = softmax(W_k · x_i)
        logits = X @ W.T  # shape (n, k)
        pi = softmax(logits, axis=1)  # shape (n, k)
        
        # Likelihood: p(y_i | x_i, z_i=k)
        log_lik = np.zeros((n, k))
        for i in range(n):
            t_idx = time_to_idx[times[i]]
            for j in range(k):
                mu = X[i] @ beta[j] + alpha[j, t_idx]
                log_lik[i, j] = -0.5 * np.log(2 * np.pi * sigma2[j]) - 0.5 * (y[i] - mu) ** 2 / sigma2[j]
        
        # Posterior: γ_ik ∝ π_k(x_i) · p(y_i | x_i, z_i=k)
        log_gamma = np.log(pi + 1e-10) + log_lik
        log_gamma -= log_gamma.max(axis=1, keepdims=True)  # numerical stability
        gamma = np.exp(log_gamma)
        gamma /= gamma.sum(axis=1, keepdims=True)
        
        return gamma
    
    def _m_step(
        self,
        X: np.ndarray,
        y: np.ndarray,
        times: np.ndarray,
        gamma: np.ndarray,
        time_to_idx: dict,
        n_times: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Update W, beta, alpha, sigma2 to maximize expected complete log-likelihood."""
        n, p = X.shape
        k = gamma.shape[1]
        
        # Update gating weights W via weighted softmax regression
        W_new = np.zeros((k, p))
        for j in range(k):
            # Weighted least squares: minimize Σ_i (z_ij - logit_j(x_i))²
            # Simplified: use current responsibilities as targets
            # Proper update would use Newton-Raphson; here we use a one-step approximation
            gamma_j = gamma[:, j]
            XtX = X.T @ (X * gamma_j[:, None])
            Xty = X.T @ (gamma_j * np.log(gamma_j + 1e-10))
            W_new[j] = np.linalg.solve(XtX + 1e-6 * np.eye(p), Xty)
        
        # Update beta, alpha, sigma2 via weighted regression
        beta_new = np.zeros((k, p))
        alpha_new = np.zeros((k, n_times))
        sigma2_new = np.zeros(k)
        
        for j in range(k):
            gamma_j = gamma[:, j]
            
            # For each time, solve weighted regression
            for t_idx in range(n_times):
                mask = np.array([time_to_idx[times[i]] == t_idx for i in range(n)])
                if not mask.any():
                    continue
                
                X_t = X[mask]
                y_t = y[mask]
                gamma_t = gamma_j[mask]
                
                if len(X_t) == 0:
                    continue
                
                # Weighted least squares
                XtX = X_t.T @ (X_t * gamma_t[:, None])
                Xty = X_t.T @ (gamma_t * y_t)
                
                try:
                    beta_t = np.linalg.solve(XtX + 1e-6 * np.eye(p), Xty)
                except np.linalg.LinAlgError:
                    beta_t = beta_new[j]  # keep previous
                
                # Extract time effect as residual mean
                residuals = y_t - X_t @ beta_t
                alpha_new[j, t_idx] = np.average(residuals, weights=gamma_t)
                
                # Accumulate beta (average across timepoints weighted by gamma)
                beta_new[j] += beta_t * gamma_t.sum()
            
            beta_new[j] /= (gamma_j.sum() + 1e-10)
            
            # Update variance
            mu = X @ beta_new[j] + np.array([alpha_new[j, time_to_idx[times[i]]] for i in range(n)])
            sigma2_new[j] = np.average((y - mu) ** 2, weights=gamma_j)
        
        return W_new, beta_new, alpha_new, sigma2_new
    
    def _log_likelihood(
        self,
        X: np.ndarray,
        y: np.ndarray,
        times: np.ndarray,
        W: np.ndarray,
        beta: np.ndarray,
        alpha: np.ndarray,
        sigma2: np.ndarray,
        time_to_idx: dict,
    ) -> float:
        """Compute complete data log-likelihood."""
        n, p = X.shape
        k = len(W)
        
        logits = X @ W.T
        pi = softmax(logits, axis=1)
        
        ll = 0.0
        for i in range(n):
            t_idx = time_to_idx[times[i]]
            p_mix = 0.0
            for j in range(k):
                mu = X[i] @ beta[j] + alpha[j, t_idx]
                p_k = pi[i, j] * np.exp(-0.5 * (y[i] - mu) ** 2 / sigma2[j]) / np.sqrt(2 * np.pi * sigma2[j])
                p_mix += p_k
            ll += np.log(p_mix + 1e-10)
        
        return ll
