"""
Derived Metrics Computation.
Computes Shannon entropy, KL divergence, and first-order rate constant k
from raw PFCA species concentration data.

These are computed in the ETL layer before modeling —
they become additional columns in the dataset that the modeling engine treats
like any other output variable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logging import get_logger

logger = get_logger(__name__)

EPS = 1e-12  # numerical stability


def compute_shannon_entropy(
    df: pd.DataFrame,
    pfca_columns: list[str],
) -> pd.Series:
    """
    Compute Shannon entropy of PFCA species distribution at each row (timepoint).

    H = -sum(p_i * log(p_i))

    High entropy = PFCA mass broadly distributed across chain lengths (intermediate-rich state)
    Low entropy  = PFCA mass concentrated in one or few species (early parent or late short-chain)
    """
    conc = df[pfca_columns].clip(lower=0.0)
    total = conc.sum(axis=1)

    # Normalize to probability distribution
    p = conc.div(total.replace(0, np.nan), axis=0).fillna(0.0)

    # Shannon entropy: -sum(p * log(p)), treating 0*log(0) = 0
    log_p = np.log(p.replace(0, np.nan)).fillna(0.0)
    entropy = -(p * log_p).sum(axis=1)

    logger.debug(
        f"Shannon entropy: min={entropy.min():.3f}, "
        f"max={entropy.max():.3f}, "
        f"mean={entropy.mean():.3f}"
    )
    return entropy.rename("shannon_entropy")


def compute_kl_divergence(
    df: pd.DataFrame,
    pfca_columns: list[str],
    entity_column: str,
    time_column: str,
) -> pd.Series:
    """
    Compute KL divergence from initial PFCA composition at each timepoint.

    KL(P || Q) = sum(p_i * log(p_i / q_i))

    Where Q is the initial composition (earliest timepoint for each experiment)
    and P is the current composition.

    High KL = distribution has shifted far from initial state
    Low KL  = still close to initial composition
    """
    result = pd.Series(index=df.index, dtype=float, name="kl_divergence")
    result[:] = 0.0

    for entity_id, group in df.groupby(entity_column):
        group_sorted = group.sort_values(time_column)
        conc = group_sorted[pfca_columns].clip(lower=0.0)
        total = conc.sum(axis=1)

        p_all = conc.div(total.replace(0, np.nan), axis=0).fillna(0.0)

        # Initial distribution Q = first timepoint
        q = p_all.iloc[0].values + EPS
        q = q / q.sum()

        for idx, row in p_all.iterrows():
            p = row.values + EPS
            p = p / p.sum()
            # KL divergence
            kl = float(np.sum(p * np.log(p / q)))
            result.loc[idx] = max(0.0, kl)  # KL is non-negative

    logger.debug(
        f"KL divergence: min={result.min():.3f}, "
        f"max={result.max():.3f}, "
        f"mean={result.mean():.3f}"
    )
    return result


def compute_rate_constants(
    df: pd.DataFrame,
    parent_column: str,
    entity_column: str,
    time_column: str,
) -> pd.DataFrame:
    """
    Fit first-order decay rate constant k for each experiment.

    Ct = C0 * exp(-k*t)
    → log(Ct/C0) = -k*t
    → OLS on log-normalized concentration vs time

    Returns a DataFrame with one row per entity:
      entity_id, k, r_squared, half_life, dt99
    """
    records = []

    for entity_id, group in df.groupby(entity_column):
        group_sorted = group.sort_values(time_column).dropna(
            subset=[parent_column, time_column]
        )
        if len(group_sorted) < 3:
            continue

        c0 = group_sorted[parent_column].iloc[0]
        if c0 < EPS:
            continue

        t = group_sorted[time_column].values
        c = group_sorted[parent_column].values

        # Normalize and log-transform
        c_norm = c / c0
        c_norm = np.clip(c_norm, EPS, None)
        log_c = np.log(c_norm)

        # Remove infinite or NaN values
        mask = np.isfinite(log_c) & np.isfinite(t)
        if mask.sum() < 3:
            continue

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                t[mask], log_c[mask]
            )
            k = -slope  # k is positive for decay
            r_squared = r_value**2

            if k <= 0:
                continue

            half_life = np.log(2) / k
            dt99 = np.log(100) / k

            records.append(
                {
                    entity_column: entity_id,
                    "k": round(k, 6),
                    "k_r_squared": round(r_squared, 4),
                    "k_p_value": round(p_value, 6),
                    "half_life_min": round(half_life, 2),
                    "dt99_min": round(dt99, 2),
                    "k_std_err": round(std_err, 6),
                }
            )

        except Exception as e:
            logger.debug(f"Rate constant fit failed for {entity_id}: {e}")
            continue

    result = pd.DataFrame(records)
    if not result.empty:
        logger.info(
            f"Rate constants fitted: {len(result)} experiments, "
            f"median k={result['k'].median():.4f}, "
            f"median R²={result['k_r_squared'].median():.3f}"
        )
    return result


def augment_dataframe(
    df: pd.DataFrame,
    pfca_columns: list[str],
    entity_column: str | None,
    time_column: str | None,
    parent_column: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add derived metric columns to the main DataFrame.
    Returns (augmented_df, rate_constants_df).

    The augmented_df has new columns:
      shannon_entropy, kl_divergence (per row/timepoint)

    The rate_constants_df has one row per experiment:
      entity_id, k, r_squared, half_life, dt99
    """
    df = df.copy()
    rate_df = pd.DataFrame()

    if len(pfca_columns) >= 3:
        df["shannon_entropy"] = compute_shannon_entropy(df, pfca_columns)
        logger.info("Shannon entropy added to dataset")

        if entity_column and time_column:
            df["kl_divergence"] = compute_kl_divergence(
                df, pfca_columns, entity_column, time_column
            )
            logger.info("KL divergence added to dataset")

    if parent_column and entity_column and time_column:
        rate_df = compute_rate_constants(df, parent_column, entity_column, time_column)
        logger.info(f"Rate constants computed for {len(rate_df)} experiments")

    return df, rate_df
