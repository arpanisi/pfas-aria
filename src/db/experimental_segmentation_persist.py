"""
Persist legacy experimental ``result_iter`` structures into PostgreSQL.

Converts nested dicts / per-regime DataFrames into the normalized ORM models in
``src.db.orm`` (ExperimentSegmentationBatch, ExperimentSegmentedRegime, …).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.orm import (
    ExperimentRegimeColumnStat,
    ExperimentRegimeRegressionSpec,
    ExperimentRegimeRow,
    ExperimentSegmentationBatch,
    ExperimentSegmentedRegime,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

_ROW_CHUNK = 5000


def _coalesce_key(d: dict[Any, Any] | None, v_raw: Any) -> Any:
    if not d:
        return None
    if v_raw in d:
        return d[v_raw]
    if str(v_raw) in d:
        return d[str(v_raw)]
    try:
        ik = int(v_raw)
        if ik in d:
            return d[ik]
    except (TypeError, ValueError):
        pass
    return None


def _json_safe(obj: Any) -> Any:
    """Best-effort conversion for JSON columns."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            d = obj.to_dict()
            out: dict[str, Any] = {}
            for k, v in d.items():
                out[str(k)] = _json_safe(v)
            return out
        except Exception:
            return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json_safe(x) for x in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def _value_counts_lookup(
    input_covariates: list[dict[str, Any]] | None, col: str
) -> dict[str, Any] | None:
    if not input_covariates:
        return None
    for item in input_covariates:
        if not isinstance(item, dict) or col not in item:
            continue
        vc = item[col]
        if hasattr(vc, "head") and hasattr(vc, "to_dict"):
            d = vc.head(80).to_dict()
            return {str(k): _json_safe(v) for k, v in d.items()}
        if isinstance(vc, dict):
            return {str(k): _json_safe(v) for k, v in list(vc.items())[:80]}
    return None


def _index_to_row_fields(idx: Any) -> tuple[int | None, str | None]:
    if isinstance(idx, bool | np.bool_):
        return None, str(idx)
    if isinstance(idx, int | np.integer):
        return int(idx), None
    try:
        return int(idx), None
    except (TypeError, ValueError):
        return None, str(idx)[:64]


async def persist_legacy_result_iter(
    session: AsyncSession,
    *,
    data_version_id: str | None,
    run_id: str | None,
    batch_index: int,
    source_filename: str,
    result_iter: dict[str, Any],
    segmentation_method: str = "legacy_masks",
) -> str:
    """
    Persist one file's ``result_iter`` dict (legacy notebook shape).

    Expected keys (best-effort; missing keys skipped):
      regimes, regime_frames, input_cols, output_cols,
      composition_input_cols, composition_cols, composition_output_cols,
      primary_component_out_col, regime_row_counts, cat_cols, const_cols,
      non_unique_cols, input_covariates, conditions, regression_cols, panel_keys.

    Older payloads may still provide the former species-specific column keys;
    those are read as a fallback until stored segmentation payloads are migrated.

    At least one of ``data_version_id`` or ``run_id`` should be set.

    Returns the new ``ExperimentSegmentationBatch`` id.
    """
    if not data_version_id and not run_id:
        raise ValueError(
            "persist_legacy_result_iter requires data_version_id and/or run_id"
        )

    regime_frames: dict[Any, Any] = result_iter.get("regime_frames") or (
        result_iter.get("regimes") or {}
    )

    input_cols = list(result_iter.get("input_cols") or [])
    output_cols = list(result_iter.get("output_cols") or [])

    dataset_columns = {
        "input_cols": input_cols,
        "output_cols": output_cols,
        "composition_input_cols": list(
            result_iter.get("composition_input_cols")
            or result_iter.get("pfas_input_mg_l_cols")
            or []
        ),
        "composition_cols": list(
            result_iter.get("composition_cols") or result_iter.get("pfas_cols") or []
        ),
        "composition_output_cols": list(
            result_iter.get("composition_output_cols")
            or result_iter.get("pfas_out_cols")
            or []
        ),
        "primary_component_out_col": result_iter.get("primary_component_out_col")
        or result_iter.get("pfbs_out_col"),
        "regime_row_counts": dict(result_iter.get("regime_row_counts") or {}),
    }

    panel_entity: str | None = None
    panel_time: str | None = None
    panel_keys = result_iter.get("panel_keys") or {}
    if panel_keys:
        first = next(iter(panel_keys.values()), None)
        if isinstance(first, dict):
            panel_entity = first.get("entity_col")
            panel_time = first.get("time_col")

    batch = ExperimentSegmentationBatch(
        data_version_id=data_version_id,
        run_id=run_id,
        batch_index=batch_index,
        source_filename=source_filename[:512],
        segmentation_method=segmentation_method[:100],
        panel_entity_col=panel_entity,
        panel_time_col=panel_time,
        dataset_columns=dataset_columns,
    )
    session.add(batch)
    await session.flush()

    cat_cols_regimes = result_iter.get("cat_cols") or {}
    const_cols_regimes = result_iter.get("const_cols") or {}
    non_unique_cols_regimes = result_iter.get("non_unique_cols") or {}
    input_covariates = result_iter.get("input_covariates") or {}
    conditions_map = result_iter.get("conditions") or {}
    regression_cols = result_iter.get("regression_cols") or {}

    for v_raw, regime_df in regime_frames.items():
        if regime_df is None or len(regime_df) == 0:
            continue
        if not isinstance(regime_df, pd.DataFrame):
            continue

        v_key: str = str(v_raw)
        regime_code = str(v_raw) if str(v_raw).isdigit() else v_key

        covs_raw = _coalesce_key(conditions_map, v_raw)
        if covs_raw is None:
            cov_list: list[Any] = []
        elif isinstance(covs_raw, list):
            cov_list = [_json_safe(x) for x in covs_raw]
        else:
            cov_list = [_json_safe(covs_raw)]

        def _listify(x: Any) -> list[Any]:
            if x is None:
                return []
            return list(x) if isinstance(x, list | tuple) else [x]

        seg = ExperimentSegmentedRegime(
            batch_id=batch.id,
            regime_code=regime_code[:50],
            n_rows=int(len(regime_df)),
            conditions_json=cov_list,
            extra_metadata={
                "cat_cols": _listify(_coalesce_key(cat_cols_regimes, v_raw)),
                "const_cols": _listify(_coalesce_key(const_cols_regimes, v_raw)),
                "non_unique_cols": _listify(
                    _coalesce_key(non_unique_cols_regimes, v_raw)
                ),
            },
        )
        session.add(seg)
        await session.flush()

        icov = _coalesce_key(input_covariates, v_raw)
        if not isinstance(icov, list):
            icov = []

        for col in input_cols:
            if col not in regime_df.columns:
                continue
            ser = regime_df[col]
            nd = int(ser.nunique(dropna=True))
            is_const = nd <= 1
            stat = ExperimentRegimeColumnStat(
                segmented_regime_id=seg.id,
                column_name=str(col)[:512],
                column_role="input",
                is_constant_in_regime=is_const,
                n_distinct=nd,
                dtype=str(ser.dtype)[:80],
                value_counts=_value_counts_lookup(icov, str(col)),
            )
            session.add(stat)

        rc = _coalesce_key(regression_cols, v_raw)
        if isinstance(rc, dict):
            spec = ExperimentRegimeRegressionSpec(
                segmented_regime_id=seg.id,
                candidate_inputs=list(rc.get("candidate_inputs") or []),
                categorical_inputs=list(rc.get("categorical_inputs") or []),
                numeric_inputs=list(rc.get("numeric_inputs") or []),
                outputs=list(rc.get("outputs") or []),
            )
            session.add(spec)

        rows_to_add: list[ExperimentRegimeRow] = []
        for idx in regime_df.index:
            ri, rl = _index_to_row_fields(idx)
            rows_to_add.append(
                ExperimentRegimeRow(
                    segmented_regime_id=seg.id,
                    source_row_index=ri,
                    source_row_label=rl,
                )
            )
        for i in range(0, len(rows_to_add), _ROW_CHUNK):
            session.add_all(rows_to_add[i : i + _ROW_CHUNK])

    await session.flush()
    logger.info(
        "Persisted experimental segmentation batch %s (%s regimes) for %s",
        batch.id,
        len(regime_frames),
        source_filename,
    )
    return str(batch.id)


async def persist_legacy_results_iter_indexed(
    session: AsyncSession,
    *,
    data_version_id: str | None,
    run_id: str | None,
    results_iter: dict[int, dict[str, Any]] | dict[str, dict[str, Any]],
    filenames: list[str] | None = None,
    segmentation_method: str = "legacy_masks",
) -> list[str]:
    """
    Persist ``results_iter`` keyed by file index (legacy loop variable ``iter``).

    ``filenames[i]`` supplies the source filename when present; otherwise
    ``batch_{i}`` is used.
    """
    ids: list[str] = []
    for iter_key, result_iter in sorted(
        results_iter.items(), key=lambda kv: int(str(kv[0]))
    ):
        idx = int(str(iter_key))
        fn = filenames[idx] if filenames and idx < len(filenames) else f"batch_{idx}"
        bid = await persist_legacy_result_iter(
            session,
            data_version_id=data_version_id,
            run_id=run_id,
            batch_index=idx,
            source_filename=fn,
            result_iter=result_iter,
            segmentation_method=segmentation_method,
        )
        ids.append(bid)
    return ids
