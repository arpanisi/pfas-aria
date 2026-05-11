"""
One-shot automated hypothesis screening: regime slices × output targets ×
input subsets × model families. Counts every fit attempt as one hypothesis tested.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ingestion.legacy_pfca_regime_masks import (
    assign_legacy_pfca_regime_masks,
    assign_pfca_regime_masks_from_column_lists,
)
from src.ingestion.unified_experimental_sheet import UnifiedSheetMeta

logger = logging.getLogger(__name__)


def _mg_l_input_names(input_cols: list[str]) -> list[str]:
    return [
        c
        for c in input_cols
        if "mg/l" in c.lower() or "mg_l" in c.lower().replace(" ", "")
    ]


def _numeric_output_names(df: pd.DataFrame, output_cols: list[str]) -> list[str]:
    out: list[str] = []
    for c in output_cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= 5 and float(s.notna().mean()) >= 0.5:
            out.append(c)
    return out


def _detect_panel_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    lower = {str(c).lower(): c for c in df.columns}
    exp = None
    for k in ("experiment_id", "experiment", "batch_id", "run_id"):
        if k in lower:
            exp = str(lower[k])
            break
    time_c = None
    for k in ("time", "time_h", "time_d", "day", "sample_time", "t_hours"):
        if k in lower:
            time_c = str(lower[k])
            break
    return exp, time_c


def _corr_ok(subset: list[str], corr: pd.DataFrame, *, max_abs: float = 0.95) -> bool:
    for i, a in enumerate(subset):
        for b in subset[i + 1 :]:
            try:
                if abs(float(corr.loc[a, b])) >= max_abs:
                    return False
            except (KeyError, TypeError, ValueError):
                continue
    return True


def _subset_iterator(
    cols: list[str], sizes: tuple[int, ...], *, max_per_size: int
) -> Iterator[list[str]]:
    for k in sizes:
        if k > len(cols) or k < 1:
            continue
        for i, tup in enumerate(itertools.combinations(cols, k)):
            if i >= max_per_size:
                break
            yield list(tup)


def _prepare_xy(
    sub_df: pd.DataFrame, x_cols: list[str], y_col: str
) -> tuple[np.ndarray, np.ndarray] | None:
    use = [c for c in x_cols if c in sub_df.columns]
    if not use or y_col not in sub_df.columns:
        return None
    block = sub_df[use + [y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(block) < max(8, len(use) + 3):
        return None
    x_mat = block[use].to_numpy(dtype=float)
    y_vec = block[y_col].to_numpy(dtype=float)
    if not np.all(np.isfinite(x_mat)) or not np.all(np.isfinite(y_vec)):
        return None
    return x_mat, y_vec


def _fit_and_count(
    x_mat: np.ndarray,
    y_vec: np.ndarray,
    *,
    panel_x: np.ndarray | None = None,
    panel_y: np.ndarray | None = None,
) -> int:
    """Return number of model fits attempted (including failures)."""
    n = 0
    estimators: list[tuple[str, object]] = [
        ("ols", LinearRegression()),
        ("ridge", Ridge(alpha=1.0, random_state=0)),
        (
            "lasso",
            Pipeline(
                [("sc", StandardScaler()), ("m", Lasso(alpha=0.01, random_state=0))]
            ),
        ),
        (
            "elasticnet",
            Pipeline(
                [
                    ("sc", StandardScaler()),
                    (
                        "m",
                        ElasticNet(
                            alpha=0.01, l1_ratio=0.5, random_state=0, max_iter=5000
                        ),
                    ),
                ]
            ),
        ),
        (
            "rf",
            RandomForestRegressor(
                n_estimators=24, max_depth=6, random_state=0, n_jobs=-1
            ),
        ),
        (
            "gbm",
            GradientBoostingRegressor(
                random_state=0, max_depth=3, n_estimators=40, learning_rate=0.1
            ),
        ),
    ]
    for _, est in estimators:
        n += 1
        try:
            est.fit(x_mat, y_vec)
        except Exception as e:  # noqa: BLE001
            logger.debug("Screening fit failed: %s", e)

    if panel_x is not None and panel_y is not None and len(panel_y) >= 8:
        n += 1
        try:
            Ridge(alpha=1.0, random_state=0).fit(panel_x, panel_y)
        except Exception as e:  # noqa: BLE001
            logger.debug("Panel ridge fit failed: %s", e)

    return n


def _build_panel_design(
    sub_df: pd.DataFrame,
    x_cols: list[str],
    y_col: str,
    exp_col: str,
    time_col: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    use = [c for c in x_cols if c in sub_df.columns]
    need = use + [y_col, exp_col, time_col]
    block = sub_df[need].copy()
    for c in use + [y_col, time_col]:
        block[c] = pd.to_numeric(block[c], errors="coerce")
    block = block.dropna()
    if len(block) < 10:
        return None
    # Integer codes for experiment (fixed-effect style without full linearmodels dependency)
    exp_codes, _ = pd.factorize(block[exp_col])
    t = block[time_col].to_numpy(dtype=float)
    x_num = block[use].to_numpy(dtype=float)
    x_panel = np.column_stack([x_num, t, exp_codes.astype(float)])
    y_panel = block[y_col].to_numpy(dtype=float)
    if not np.all(np.isfinite(x_panel)) or not np.all(np.isfinite(y_panel)):
        return None
    return x_panel, y_panel


def run_automated_screening_iteration(
    df: pd.DataFrame,
    *,
    unified_meta: UnifiedSheetMeta | None,
    input_start: int = 2,
    input_end: int = 37,
    output_start: int = 37,
    regime_id: int | None = None,
) -> int:
    """
    Run one screening pass over numeric outputs, input subsets, and model families.

    If ``regime_id`` is set, only that regime's rows are used; otherwise all nonempty
    regimes from the legacy mask assignment are screened.
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

    exp_col, time_col = _detect_panel_columns(df)
    total = 0
    sizes = (2, 3, 5)
    max_subsets_per_k = 36

    regimes = res.regimes
    if regime_id is not None:
        rid = int(regime_id)
        if rid not in regimes:
            logger.warning(
                "Screening regime_id=%s not in nonempty regimes %s — no fits",
                rid,
                list(regimes.keys()),
            )
            return 0
        regimes = {rid: regimes[rid]}

    for _rid, sub_df in regimes.items():
        if len(sub_df) < 12:
            continue
        mg_inputs = _mg_l_input_names(res.input_cols)
        pool = [c for c in mg_inputs if c in sub_df.columns]
        if len(pool) < 2:
            continue
        # Correlation on regime slice for multicollinearity screening
        num_block = sub_df[pool].apply(pd.to_numeric, errors="coerce")
        if num_block.shape[1] < 2:
            continue
        corr = num_block.corr(numeric_only=True).abs()
        outs = _numeric_output_names(sub_df, res.output_cols)
        for y_col in outs:
            is_panel = (
                exp_col is not None
                and time_col is not None
                and exp_col in sub_df.columns
                and time_col in sub_df.columns
            )
            for x_subset in _subset_iterator(
                pool, sizes, max_per_size=max_subsets_per_k
            ):
                if not _corr_ok(x_subset, corr):
                    continue
                prep = _prepare_xy(sub_df, x_subset, y_col)
                if prep is None:
                    continue
                x_mat, y_vec = prep
                panel_pair: tuple[np.ndarray, np.ndarray] | None = None
                if is_panel:
                    assert exp_col is not None and time_col is not None
                    panel_pair = _build_panel_design(
                        sub_df, x_subset, y_col, exp_col, time_col
                    )
                px, py = panel_pair if panel_pair else (None, None)
                total += _fit_and_count(x_mat, y_vec, panel_x=px, panel_y=py)

    return total
