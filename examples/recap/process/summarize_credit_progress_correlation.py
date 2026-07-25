#!/usr/bin/env python
"""Evaluate held-out credit alignment, boundary stability, and uncertainty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEY_COLUMNS = ["episode_index", "frame_index"]


def _resolve_artifact(raw_path: str, comparison_path: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = comparison_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Comparison artifact not found: {path}")
    return path


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    source: Path,
) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")


def _validate_unique_keys(frame: pd.DataFrame, source: Path) -> None:
    duplicated = frame.duplicated(KEY_COLUMNS, keep=False)
    if duplicated.any():
        preview = frame.loc[duplicated, KEY_COLUMNS].head(5).to_dict("records")
        raise ValueError(f"Duplicate episode/frame keys in {source}: {preview}")


def _spearman_rank_correlation(left: pd.Series, right: pd.Series) -> float:
    left_rank = left.rank(method="average").to_numpy(dtype=np.float64)
    right_rank = right.rank(method="average").to_numpy(dtype=np.float64)
    if len(left_rank) < 2:
        raise ValueError("Spearman correlation requires at least two rows.")
    if np.ptp(left_rank) == 0.0 or np.ptp(right_rank) == 0.0:
        raise ValueError("Spearman correlation is undefined for constant inputs.")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _optional_spearman(left: pd.Series, right: pd.Series) -> float:
    try:
        return _spearman_rank_correlation(left, right)
    except ValueError:
        return float("nan")


def _map_value_to_return(
    values: pd.Series,
    comparison: dict[str, Any],
) -> np.ndarray:
    return_min = float(comparison.get("return_min", -900.0))
    return_max = float(comparison.get("return_max", 0.0))
    value_min = float(comparison.get("value_min", -1.0))
    value_max = float(comparison.get("value_max", 0.0))
    value_range = value_max - value_min
    if return_max <= return_min or value_range <= 0.0:
        raise ValueError("Invalid value/return scale in comparison report.")
    values_array = values.to_numpy(dtype=np.float64)
    return (
        (values_array - value_min)
        / value_range
        * (return_max - return_min)
        + return_min
    )


def _normalized_distribution_entropy(
    values: pd.Series,
    *,
    values_are_logits: bool,
) -> np.ndarray:
    entropies = []
    for row_index, raw_values in enumerate(values):
        array = np.asarray(raw_values, dtype=np.float64)
        if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
            raise ValueError(f"Invalid value distribution at row {row_index}.")
        if values_are_logits:
            shifted = array - float(array.max())
            probabilities = np.exp(shifted)
        else:
            if float(array.min()) < -1e-8:
                raise ValueError(f"Negative value probability at row {row_index}.")
            probabilities = np.clip(array, 0.0, None)
        denominator = float(probabilities.sum())
        if denominator <= 0.0:
            raise ValueError(f"Zero-mass value distribution at row {row_index}.")
        probabilities = probabilities / denominator
        nonzero = probabilities > 0.0
        entropy = -float(
            np.sum(probabilities[nonzero] * np.log(probabilities[nonzero]))
        )
        entropies.append(entropy / float(np.log(len(probabilities))))
    return np.asarray(entropies, dtype=np.float64)


def _value_mae_from_report(
    comparison: dict[str, Any],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for split, payload in comparison.get("frame_level", {}).items():
        raw = payload.get("base", {}).get("mae")
        fused = payload.get("shared_mlp_fusion", {}).get("mae")
        if raw is None or fused is None:
            continue
        result[str(split)] = {
            "raw": float(raw),
            "fused": float(fused),
        }
    return result


def build_validation_diagnostics_frame(
    comparison_path: Path,
    *,
    split: str,
    num_tasks: int,
    episodes_per_task: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build held-out values, residuals, phases, and distribution entropy."""
    comparison_path = comparison_path.expanduser().resolve()
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    advantages_path = _resolve_artifact(
        str(comparison["advantages_path"]), comparison_path
    )
    predictions_path = _resolve_artifact(
        str(comparison["predictions_path"]), comparison_path
    )
    advantages = pd.read_parquet(advantages_path)
    predictions = pd.read_parquet(predictions_path)
    _require_columns(
        advantages,
        {*KEY_COLUMNS, "return", "value_current"},
        advantages_path,
    )
    _require_columns(
        predictions,
        {*KEY_COLUMNS, "split", "value_fused", "phase_true"},
        predictions_path,
    )
    probability_column = None
    values_are_logits = False
    if "value_probs_current" in advantages.columns:
        probability_column = "value_probs_current"
    elif "value_logits_current" in advantages.columns:
        probability_column = "value_logits_current"
        values_are_logits = True
    else:
        raise ValueError(
            f"{advantages_path} has neither value_probs_current nor "
            "value_logits_current."
        )

    selected = predictions[predictions["split"].astype(str) == split].copy()
    if selected.empty:
        raise ValueError(f"No prediction rows for split={split!r}.")
    frame = selected[
        [*KEY_COLUMNS, "split", "value_fused", "phase_true"]
    ].merge(
        advantages[
            [
                *KEY_COLUMNS,
                "return",
                "value_current",
                probability_column,
            ]
        ],
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    if frame.empty:
        raise ValueError("No held-out value rows overlap source advantages.")
    frame = frame.sort_values(KEY_COLUMNS).reset_index(drop=True)
    frame["task_index"] = (
        frame["episode_index"].astype(int) // int(episodes_per_task)
    )
    invalid_task = (frame["task_index"] < 0) | (
        frame["task_index"] >= int(num_tasks)
    )
    if invalid_task.any():
        raise ValueError("Held-out values contain episodes outside task ranges.")
    frame["task"] = frame["task_index"].map(lambda index: f"task{index}")
    frame["raw_prediction_return"] = _map_value_to_return(
        frame["value_current"], comparison
    )
    frame["fused_prediction_return"] = _map_value_to_return(
        frame["value_fused"], comparison
    )
    frame["raw_residual"] = (
        frame["raw_prediction_return"] - frame["return"].astype(float)
    )
    frame["fused_residual"] = (
        frame["fused_prediction_return"] - frame["return"].astype(float)
    )
    frame["raw_value_entropy"] = _normalized_distribution_entropy(
        frame[probability_column],
        values_are_logits=values_are_logits,
    )
    metadata = {
        "value_distribution_column": probability_column,
        "value_distribution_input": (
            "logits" if values_are_logits else "probabilities"
        ),
        "value_entropy_normalization": "entropy / log(number_of_return_bins)",
        "held_out_value_rows": int(len(frame)),
    }
    return frame, metadata


def build_credit_progress_frame(
    comparison_path: Path,
    *,
    split: str,
    lookahead_step: int,
    gamma: float,
    num_tasks: int,
    episodes_per_task: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Align code-native Raw/Fused credits with privileged progress deltas."""
    comparison_path = comparison_path.expanduser().resolve()
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    advantages_raw = comparison.get("advantages_path")
    predictions_raw = comparison.get("predictions_path")
    if not advantages_raw or not predictions_raw:
        raise ValueError(
            "Comparison report must contain advantages_path and predictions_path."
        )
    advantages_path = _resolve_artifact(str(advantages_raw), comparison_path)
    predictions_path = _resolve_artifact(str(predictions_raw), comparison_path)
    advantages = pd.read_parquet(advantages_path)
    predictions = pd.read_parquet(predictions_path)
    _require_columns(
        advantages,
        {
            *KEY_COLUMNS,
            "reward_sum",
            "num_valid_rewards",
            "value_current",
            "advantage_continuous",
        },
        advantages_path,
    )
    _require_columns(
        predictions,
        {
            *KEY_COLUMNS,
            "split",
            "value_fused",
            "global_progress_true",
            "phase_true",
        },
        predictions_path,
    )
    _validate_unique_keys(advantages, advantages_path)
    _validate_unique_keys(predictions, predictions_path)

    available_splits = sorted(predictions["split"].dropna().astype(str).unique())
    selected_predictions = predictions[
        predictions["split"].astype(str) == split
    ].copy()
    if selected_predictions.empty:
        raise ValueError(
            f"No prediction rows for split={split!r}; available={available_splits}"
        )

    advantage_columns = [
        *KEY_COLUMNS,
        "reward_sum",
        "num_valid_rewards",
        "value_current",
        "advantage_continuous",
    ]
    prediction_columns = [
        *KEY_COLUMNS,
        "split",
        "value_fused",
        "global_progress_true",
        "phase_true",
    ]
    current = selected_predictions[prediction_columns].merge(
        advantages[advantage_columns],
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    if current.empty:
        raise ValueError("Selected predictions do not overlap source advantages.")

    current["num_valid_rewards"] = current["num_valid_rewards"].astype(int)
    full_horizon = current["num_valid_rewards"] == int(lookahead_step)
    current = current[full_horizon].copy()
    if current.empty:
        raise ValueError(
            f"No {split!r} rows have a complete {lookahead_step}-step horizon."
        )

    future = selected_predictions[
        [*KEY_COLUMNS, "value_fused", "global_progress_true"]
    ].merge(
        advantages[[*KEY_COLUMNS, "value_current"]],
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    future["frame_index"] = future["frame_index"].astype(int) - int(
        lookahead_step
    )
    future = future.rename(
        columns={
            "value_current": "raw_value_next",
            "value_fused": "fused_value_next",
            "global_progress_true": "global_progress_next",
        }
    )
    aligned = current.merge(
        future,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    if aligned.empty:
        raise ValueError(
            f"No {split!r} rows have an aligned t+{lookahead_step} frame."
        )

    episode_index = aligned["episode_index"].astype(int)
    task_index = episode_index // int(episodes_per_task)
    invalid_task = (task_index < 0) | (task_index >= int(num_tasks))
    if invalid_task.any():
        invalid_episodes = sorted(episode_index[invalid_task].unique().tolist())
        raise ValueError(
            "Episode indices exceed configured task ranges: "
            f"{invalid_episodes[:10]}"
        )
    aligned["task_index"] = task_index
    aligned["task"] = task_index.map(lambda index: f"task{index}")
    aligned["future_frame_index"] = (
        aligned["frame_index"].astype(int) + int(lookahead_step)
    )

    gamma_factor = float(gamma) ** int(lookahead_step)
    reward_sum = aligned["reward_sum"].to_numpy(dtype=np.float64)
    raw_value = aligned["value_current"].to_numpy(dtype=np.float64)
    fused_value = aligned["value_fused"].to_numpy(dtype=np.float64)
    aligned["raw_credit"] = (
        reward_sum
        + gamma_factor
        * aligned["raw_value_next"].to_numpy(dtype=np.float64)
        - raw_value
    )
    raw_credit_difference = np.abs(
        aligned["raw_credit"].to_numpy(dtype=np.float64)
        - aligned["advantage_continuous"].to_numpy(dtype=np.float64)
    )
    raw_credit_max_abs_diff = float(raw_credit_difference.max())
    if raw_credit_max_abs_diff > 1e-5:
        raise ValueError(
            "Recomputed Raw credit does not match advantage_continuous: "
            f"max_abs_diff={raw_credit_max_abs_diff:.6g}. Check H and gamma."
        )
    aligned["fused_credit"] = (
        reward_sum
        + gamma_factor
        * aligned["fused_value_next"].to_numpy(dtype=np.float64)
        - fused_value
    )
    aligned["global_progress_delta"] = (
        aligned["global_progress_next"].to_numpy(dtype=np.float64)
        - aligned["global_progress_true"].to_numpy(dtype=np.float64)
    )
    numeric_columns = [
        "raw_credit",
        "fused_credit",
        "global_progress_true",
        "phase_true",
        "global_progress_next",
        "global_progress_delta",
    ]
    finite = np.isfinite(aligned[numeric_columns].to_numpy(dtype=np.float64)).all(
        axis=1
    )
    dropped_nonfinite = int((~finite).sum())
    aligned = aligned[finite].copy()
    if aligned.empty:
        raise ValueError("All aligned credit/progress rows are non-finite.")

    metadata = {
        "comparison_path": str(comparison_path),
        "advantages_path": str(advantages_path),
        "predictions_path": str(predictions_path),
        "split": split,
        "lookahead_step": int(lookahead_step),
        "gamma": float(gamma),
        "credit_definition": (
            "reward_sum + gamma**H * value[t+H] - value[t]"
        ),
        "progress_definition": (
            "global_progress_true[t+H] - global_progress_true[t]"
        ),
        "progress_source": (
            "privileged simulator semantic trace copied into predictions as "
            "global_progress_true"
        ),
        "horizon_policy": "complete exact-H pairs only",
        "prediction_rows_in_split": int(len(selected_predictions)),
        "full_horizon_rows_before_future_join": int(len(current)),
        "aligned_rows": int(len(aligned)),
        "dropped_nonfinite_rows": dropped_nonfinite,
        "raw_credit_source_max_abs_diff": raw_credit_max_abs_diff,
        "value_mae_from_comparison": _value_mae_from_report(comparison),
    }
    keep_columns = [
        "task",
        "task_index",
        "split",
        "episode_index",
        "frame_index",
        "future_frame_index",
        "num_valid_rewards",
        "reward_sum",
        "value_current",
        "raw_value_next",
        "value_fused",
        "fused_value_next",
        "global_progress_true",
        "phase_true",
        "global_progress_next",
        "global_progress_delta",
        "raw_credit",
        "advantage_continuous",
        "fused_credit",
    ]
    aligned = aligned[keep_columns].sort_values(KEY_COLUMNS).reset_index(drop=True)
    return aligned, metadata


def _boundary_indices(phases: np.ndarray) -> np.ndarray:
    if len(phases) < 2:
        return np.asarray([], dtype=np.int64)
    return np.flatnonzero(phases[1:] != phases[:-1]).astype(np.int64) + 1


def _near_boundary_mask(
    length: int,
    boundaries: np.ndarray,
    window: int,
) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for boundary in boundaries:
        start = max(0, int(boundary) - window)
        end = min(length, int(boundary) + window + 1)
        mask[start:end] = True
    return mask


def _mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if len(finite) else float("nan")


def _lower_is_better_improvement(raw: float, fused: float) -> float:
    if not np.isfinite(raw) or not np.isfinite(fused) or raw <= 0.0:
        return float("nan")
    return 100.0 * (1.0 - fused / raw)


def summarize_boundary_value_diagnostics(
    validation: pd.DataFrame,
    *,
    num_tasks: int,
    boundary_window: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Measure boundary-local residual jitter, MAE, and value entropy."""
    rows = []
    for task_index in range(num_tasks):
        task = f"task{task_index}"
        task_frame = validation[validation["task"] == task]
        if task_frame.empty:
            raise ValueError(f"No held-out value diagnostics for {task}.")
        raw_boundary_jitter: list[float] = []
        fused_boundary_jitter: list[float] = []
        raw_boundary_error: list[float] = []
        fused_boundary_error: list[float] = []
        boundary_entropy: list[float] = []
        interior_entropy: list[float] = []
        boundary_events = 0
        for _, episode in task_frame.groupby("episode_index", sort=True):
            episode = episode.sort_values("frame_index")
            phases = episode["phase_true"].to_numpy(dtype=np.int64)
            boundaries = _boundary_indices(phases)
            boundary_events += int(len(boundaries))
            entropy = episode["raw_value_entropy"].to_numpy(dtype=np.float64)
            if not len(boundaries):
                interior_entropy.extend(entropy.tolist())
                continue
            near_boundary = _near_boundary_mask(
                len(episode), boundaries, boundary_window
            )
            raw_residual = episode["raw_residual"].to_numpy(dtype=np.float64)
            fused_residual = episode["fused_residual"].to_numpy(
                dtype=np.float64
            )
            raw_boundary_error.extend(np.abs(raw_residual[near_boundary]).tolist())
            fused_boundary_error.extend(
                np.abs(fused_residual[near_boundary]).tolist()
            )
            boundary_entropy.extend(entropy[near_boundary].tolist())
            interior_entropy.extend(entropy[~near_boundary].tolist())
            if len(episode) >= 3:
                raw_curvature = np.abs(np.diff(raw_residual, n=2))
                fused_curvature = np.abs(np.diff(fused_residual, n=2))
                curvature_near_boundary = near_boundary[1:-1]
                raw_boundary_jitter.extend(
                    raw_curvature[curvature_near_boundary].tolist()
                )
                fused_boundary_jitter.extend(
                    fused_curvature[curvature_near_boundary].tolist()
                )

        raw_jitter = _mean_or_nan(raw_boundary_jitter)
        fused_jitter = _mean_or_nan(fused_boundary_jitter)
        raw_mae = _mean_or_nan(raw_boundary_error)
        fused_mae = _mean_or_nan(fused_boundary_error)
        entropy_error_rho = _optional_spearman(
            task_frame["raw_value_entropy"], task_frame["raw_residual"].abs()
        )
        entropy_boundary = _mean_or_nan(boundary_entropy)
        entropy_interior = _mean_or_nan(interior_entropy)
        rows.append(
            {
                "task": task,
                "task_index": task_index,
                "episodes": int(task_frame["episode_index"].nunique()),
                "frames": int(len(task_frame)),
                "phase_boundaries": boundary_events,
                "raw_boundary_residual_jitter": raw_jitter,
                "fused_boundary_residual_jitter": fused_jitter,
                "boundary_residual_jitter_improvement_pct": (
                    _lower_is_better_improvement(raw_jitter, fused_jitter)
                ),
                "raw_boundary_mae": raw_mae,
                "fused_boundary_mae": fused_mae,
                "boundary_mae_improvement_pct": (
                    _lower_is_better_improvement(raw_mae, fused_mae)
                ),
                "raw_entropy_error_spearman": entropy_error_rho,
                "raw_boundary_value_entropy": entropy_boundary,
                "raw_interior_value_entropy": entropy_interior,
                "raw_boundary_entropy_lift": entropy_boundary - entropy_interior,
            }
        )
    by_task = pd.DataFrame(rows)
    raw_jitter_macro = float(by_task["raw_boundary_residual_jitter"].mean())
    fused_jitter_macro = float(
        by_task["fused_boundary_residual_jitter"].mean()
    )
    raw_mae_macro = float(by_task["raw_boundary_mae"].mean())
    fused_mae_macro = float(by_task["fused_boundary_mae"].mean())
    residual_jitter_available = by_task[
        ["raw_boundary_residual_jitter", "fused_boundary_residual_jitter"]
    ].notna().all(axis=1)
    boundary_mae_available = by_task[
        ["raw_boundary_mae", "fused_boundary_mae"]
    ].notna().all(axis=1)
    aggregate = {
        "boundary_window_frames": int(boundary_window),
        "jitter_definition": (
            "mean absolute second temporal difference of prediction residual "
            "within +/- boundary_window of a phase transition"
        ),
        "tasks": int(len(by_task)),
        "tasks_with_phase_boundaries": int(
            (by_task["phase_boundaries"] > 0).sum()
        ),
        "tasks_with_boundary_residual_jitter": int(
            residual_jitter_available.sum()
        ),
        "tasks_with_boundary_residual_jitter_improved": int(
            (
                residual_jitter_available
                & (
                    by_task["fused_boundary_residual_jitter"]
                    < by_task["raw_boundary_residual_jitter"]
                )
            ).sum()
        ),
        "tasks_with_boundary_mae": int(boundary_mae_available.sum()),
        "tasks_with_boundary_mae_improved": int(
            (
                boundary_mae_available
                & (
                    by_task["fused_boundary_mae"]
                    < by_task["raw_boundary_mae"]
                )
            ).sum()
        ),
        "phase_boundaries": int(by_task["phase_boundaries"].sum()),
        "raw_boundary_residual_jitter": raw_jitter_macro,
        "fused_boundary_residual_jitter": fused_jitter_macro,
        "boundary_residual_jitter_improvement_pct": (
            _lower_is_better_improvement(raw_jitter_macro, fused_jitter_macro)
        ),
        "raw_boundary_mae": raw_mae_macro,
        "fused_boundary_mae": fused_mae_macro,
        "boundary_mae_improvement_pct": (
            _lower_is_better_improvement(raw_mae_macro, fused_mae_macro)
        ),
        "raw_entropy_error_spearman": float(
            by_task["raw_entropy_error_spearman"].mean()
        ),
        "raw_boundary_value_entropy": float(
            by_task["raw_boundary_value_entropy"].mean()
        ),
        "raw_interior_value_entropy": float(
            by_task["raw_interior_value_entropy"].mean()
        ),
        "raw_boundary_entropy_lift": float(
            by_task["raw_boundary_entropy_lift"].mean()
        ),
    }
    return by_task, aggregate


def summarize_boundary_credit_jitter(
    aligned: pd.DataFrame,
    *,
    num_tasks: int,
    boundary_window: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Measure boundary-local temporal variation of Raw and Fused credits."""
    rows = []
    for task_index in range(num_tasks):
        task = f"task{task_index}"
        task_frame = aligned[aligned["task"] == task]
        raw_variation: list[float] = []
        fused_variation: list[float] = []
        boundary_events = 0
        for _, episode in task_frame.groupby("episode_index", sort=True):
            episode = episode.sort_values("frame_index")
            phases = episode["phase_true"].to_numpy(dtype=np.int64)
            boundaries = _boundary_indices(phases)
            boundary_events += int(len(boundaries))
            if not len(boundaries) or len(episode) < 2:
                continue
            near_boundary = _near_boundary_mask(
                len(episode), boundaries, boundary_window
            )
            variation_near_boundary = near_boundary[1:]
            raw_difference = np.abs(
                np.diff(episode["raw_credit"].to_numpy(dtype=np.float64))
            )
            fused_difference = np.abs(
                np.diff(episode["fused_credit"].to_numpy(dtype=np.float64))
            )
            raw_variation.extend(
                raw_difference[variation_near_boundary].tolist()
            )
            fused_variation.extend(
                fused_difference[variation_near_boundary].tolist()
            )
        raw_jitter = _mean_or_nan(raw_variation)
        fused_jitter = _mean_or_nan(fused_variation)
        rows.append(
            {
                "task": task,
                "task_index": task_index,
                "phase_boundaries": boundary_events,
                "raw_boundary_credit_jitter": raw_jitter,
                "fused_boundary_credit_jitter": fused_jitter,
                "boundary_credit_jitter_improvement_pct": (
                    _lower_is_better_improvement(raw_jitter, fused_jitter)
                ),
            }
        )
    by_task = pd.DataFrame(rows)
    raw_macro = float(by_task["raw_boundary_credit_jitter"].mean())
    fused_macro = float(by_task["fused_boundary_credit_jitter"].mean())
    credit_jitter_available = by_task[
        ["raw_boundary_credit_jitter", "fused_boundary_credit_jitter"]
    ].notna().all(axis=1)
    aggregate = {
        "definition": (
            "mean absolute first temporal difference of continuous credit "
            "within +/- boundary_window of a phase transition"
        ),
        "boundary_window_frames": int(boundary_window),
        "tasks": int(len(by_task)),
        "tasks_with_phase_boundaries": int(
            (by_task["phase_boundaries"] > 0).sum()
        ),
        "tasks_with_boundary_credit_jitter": int(
            credit_jitter_available.sum()
        ),
        "tasks_with_boundary_credit_jitter_improved": int(
            (
                credit_jitter_available
                & (
                    by_task["fused_boundary_credit_jitter"]
                    < by_task["raw_boundary_credit_jitter"]
                )
            ).sum()
        ),
        "raw_boundary_credit_jitter": raw_macro,
        "fused_boundary_credit_jitter": fused_macro,
        "boundary_credit_jitter_improvement_pct": (
            _lower_is_better_improvement(raw_macro, fused_macro)
        ),
    }
    return by_task, aggregate


def summarize_credit_progress(
    aligned: pd.DataFrame,
    *,
    num_tasks: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute per-task correlations and the primary task-macro average."""
    task_rows = []
    for task_index in range(num_tasks):
        task = f"task{task_index}"
        part = aligned[aligned["task"] == task]
        if part.empty:
            raise ValueError(f"No aligned rows for {task}.")
        raw_rho = _spearman_rank_correlation(
            part["raw_credit"], part["global_progress_delta"]
        )
        fused_rho = _spearman_rank_correlation(
            part["fused_credit"], part["global_progress_delta"]
        )
        task_rows.append(
            {
                "task": task,
                "task_index": task_index,
                "episodes": int(part["episode_index"].nunique()),
                "frames": int(len(part)),
                "raw_credit_progress_spearman": raw_rho,
                "fused_credit_progress_spearman": fused_rho,
                "spearman_gain": fused_rho - raw_rho,
                "progress_delta_mean": float(
                    part["global_progress_delta"].mean()
                ),
                "progress_delta_positive_ratio": float(
                    (part["global_progress_delta"] > 0.0).mean()
                ),
            }
        )
    by_task = pd.DataFrame(task_rows)
    raw_macro = float(by_task["raw_credit_progress_spearman"].mean())
    fused_macro = float(by_task["fused_credit_progress_spearman"].mean())
    frame_micro_raw = _spearman_rank_correlation(
        aligned["raw_credit"], aligned["global_progress_delta"]
    )
    frame_micro_fused = _spearman_rank_correlation(
        aligned["fused_credit"], aligned["global_progress_delta"]
    )
    aggregate = {
        "primary_metric": "task_macro_credit_progress_spearman",
        "task_macro": {
            "tasks": int(len(by_task)),
            "raw": raw_macro,
            "fused": fused_macro,
            "gain": fused_macro - raw_macro,
            "tasks_improved": int((by_task["spearman_gain"] > 0.0).sum()),
        },
        "frame_micro_supporting": {
            "frames": int(len(aligned)),
            "raw": frame_micro_raw,
            "fused": frame_micro_fused,
            "gain": frame_micro_fused - frame_micro_raw,
        },
        "std_note": (
            "No seed standard deviation is reported; this is one fixed offline "
            "held-out split with task-macro aggregation."
        ),
    }
    return by_task, aggregate


def _build_paper_table(
    metadata: dict[str, Any],
    aggregate: dict[str, Any],
    *,
    raw_policy_sr: float | None,
    fused_policy_sr: float | None,
) -> pd.DataFrame:
    split = str(metadata.get("split", "all"))
    mae_by_split = metadata.get("value_mae_from_comparison", {})
    selected_mae = mae_by_split.get(split, mae_by_split.get("all", {}))
    task_macro = aggregate["task_macro"]
    return pd.DataFrame(
        [
            {
                "credit_source": "Raw Critic",
                "value_mae_split": split,
                "value_mae": selected_mae.get("raw"),
                "credit_progress_rank_correlation": task_macro["raw"],
                "policy_sr_percent": raw_policy_sr,
            },
            {
                "credit_source": "Full GLC-Critic",
                "value_mae_split": split,
                "value_mae": selected_mae.get("fused"),
                "credit_progress_rank_correlation": task_macro["fused"],
                "policy_sr_percent": fused_policy_sr,
            },
        ]
    )


def _build_candidate_metric_table(
    credit_aggregate: dict[str, Any],
    boundary_aggregate: dict[str, Any],
    credit_jitter_aggregate: dict[str, Any],
) -> pd.DataFrame:
    task_macro = credit_aggregate["task_macro"]
    return pd.DataFrame(
        [
            {
                "metric": "Credit-progress task-macro Spearman",
                "claim": "credit ranking follows semantic task advancement",
                "direction": "higher",
                "raw": task_macro["raw"],
                "fused": task_macro["fused"],
                "absolute_gain": task_macro["gain"],
                "improvement_pct": float("nan"),
                "tasks_covered": task_macro["tasks"],
                "tasks_improved": task_macro["tasks_improved"],
            },
            {
                "metric": "Boundary residual jitter",
                "claim": "remaining-cost prediction is stable at boundaries",
                "direction": "lower",
                "raw": boundary_aggregate[
                    "raw_boundary_residual_jitter"
                ],
                "fused": boundary_aggregate[
                    "fused_boundary_residual_jitter"
                ],
                "absolute_gain": (
                    boundary_aggregate["raw_boundary_residual_jitter"]
                    - boundary_aggregate["fused_boundary_residual_jitter"]
                ),
                "improvement_pct": boundary_aggregate[
                    "boundary_residual_jitter_improvement_pct"
                ],
                "tasks_covered": boundary_aggregate[
                    "tasks_with_boundary_residual_jitter"
                ],
                "tasks_improved": boundary_aggregate[
                    "tasks_with_boundary_residual_jitter_improved"
                ],
            },
            {
                "metric": "Boundary continuous-credit jitter",
                "claim": "step-level credit is stable at phase boundaries",
                "direction": "lower",
                "raw": credit_jitter_aggregate[
                    "raw_boundary_credit_jitter"
                ],
                "fused": credit_jitter_aggregate[
                    "fused_boundary_credit_jitter"
                ],
                "absolute_gain": (
                    credit_jitter_aggregate["raw_boundary_credit_jitter"]
                    - credit_jitter_aggregate["fused_boundary_credit_jitter"]
                ),
                "improvement_pct": credit_jitter_aggregate[
                    "boundary_credit_jitter_improvement_pct"
                ],
                "tasks_covered": credit_jitter_aggregate[
                    "tasks_with_boundary_credit_jitter"
                ],
                "tasks_improved": credit_jitter_aggregate[
                    "tasks_with_boundary_credit_jitter_improved"
                ],
            },
            {
                "metric": "Boundary-local value MAE",
                "claim": "remaining-cost prediction is accurate at boundaries",
                "direction": "lower",
                "raw": boundary_aggregate["raw_boundary_mae"],
                "fused": boundary_aggregate["fused_boundary_mae"],
                "absolute_gain": (
                    boundary_aggregate["raw_boundary_mae"]
                    - boundary_aggregate["fused_boundary_mae"]
                ),
                "improvement_pct": boundary_aggregate[
                    "boundary_mae_improvement_pct"
                ],
                "tasks_covered": boundary_aggregate[
                    "tasks_with_boundary_mae"
                ],
                "tasks_improved": boundary_aggregate[
                    "tasks_with_boundary_mae_improved"
                ],
            },
        ]
    )


def main() -> None:
    """Run the fixed held-out Critic diagnostic experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--lookahead-step", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument("--episodes-per-task", type=int, default=30)
    parser.add_argument("--boundary-window", type=int, default=10)
    parser.add_argument("--raw-policy-sr", type=float)
    parser.add_argument("--fused-policy-sr", type=float)
    args = parser.parse_args()
    if args.lookahead_step <= 0:
        raise ValueError("--lookahead-step must be positive.")
    if args.gamma < 0.0:
        raise ValueError("--gamma must be non-negative.")
    if args.num_tasks <= 0 or args.episodes_per_task <= 0:
        raise ValueError("Task and episode counts must be positive.")
    if args.boundary_window < 0:
        raise ValueError("--boundary-window must be non-negative.")
    policy_rates = {
        "--raw-policy-sr": args.raw_policy_sr,
        "--fused-policy-sr": args.fused_policy_sr,
    }
    for option, value in policy_rates.items():
        if value is not None and not 0.0 <= value <= 100.0:
            raise ValueError(f"{option} must be a percentage in [0, 100].")

    aligned, metadata = build_credit_progress_frame(
        args.comparison,
        split=args.split,
        lookahead_step=args.lookahead_step,
        gamma=args.gamma,
        num_tasks=args.num_tasks,
        episodes_per_task=args.episodes_per_task,
    )
    by_task, aggregate = summarize_credit_progress(
        aligned,
        num_tasks=args.num_tasks,
    )
    validation, validation_metadata = build_validation_diagnostics_frame(
        args.comparison,
        split=args.split,
        num_tasks=args.num_tasks,
        episodes_per_task=args.episodes_per_task,
    )
    boundary_by_task, boundary_aggregate = (
        summarize_boundary_value_diagnostics(
            validation,
            num_tasks=args.num_tasks,
            boundary_window=args.boundary_window,
        )
    )
    credit_jitter_by_task, credit_jitter_aggregate = (
        summarize_boundary_credit_jitter(
            aligned,
            num_tasks=args.num_tasks,
            boundary_window=args.boundary_window,
        )
    )
    candidates = _build_candidate_metric_table(
        aggregate,
        boundary_aggregate,
        credit_jitter_aggregate,
    )
    report = {
        **metadata,
        **validation_metadata,
        **aggregate,
        "boundary_value_diagnostics": boundary_aggregate,
        "boundary_credit_diagnostics": credit_jitter_aggregate,
        "uncertainty_diagnostics": {
            "raw_entropy_error_spearman": boundary_aggregate[
                "raw_entropy_error_spearman"
            ],
            "raw_boundary_value_entropy": boundary_aggregate[
                "raw_boundary_value_entropy"
            ],
            "raw_interior_value_entropy": boundary_aggregate[
                "raw_interior_value_entropy"
            ],
            "raw_boundary_entropy_lift": boundary_aggregate[
                "raw_boundary_entropy_lift"
            ],
            "scope_note": (
                "Current artifacts preserve the Raw categorical value "
                "distribution but not Fused logits; uncertainty calibration "
                "is therefore a Raw-Critic diagnostic."
            ),
        },
    }
    paper_table = _build_paper_table(
        metadata,
        aggregate,
        raw_policy_sr=args.raw_policy_sr,
        fused_policy_sr=args.fused_policy_sr,
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    aligned_path = output_dir / "credit_progress_aligned_frames.parquet"
    task_path = output_dir / "credit_progress_by_task.csv"
    boundary_task_path = output_dir / "boundary_value_by_task.csv"
    credit_jitter_task_path = output_dir / "boundary_credit_by_task.csv"
    candidate_path = output_dir / "candidate_metrics.csv"
    table_path = output_dir / "credit_progress_paper_table.csv"
    summary_path = output_dir / "credit_progress_summary.json"
    text_path = output_dir / "credit_progress_summary.txt"
    aligned.to_parquet(aligned_path, index=False)
    by_task.to_csv(task_path, index=False)
    boundary_by_task.to_csv(boundary_task_path, index=False)
    credit_jitter_by_task.to_csv(credit_jitter_task_path, index=False)
    candidates.to_csv(candidate_path, index=False)
    paper_table.to_csv(table_path, index=False)
    summary_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    text = "\n".join(
        [
            "Credit = reward_sum + gamma**H * value[t+H] - value[t]",
            "Progress = privileged global_progress_true[t+H] - "
            "global_progress_true[t]",
            f"Split = {args.split}; H = {args.lookahead_step}; gamma = {args.gamma}",
            "",
            by_task.to_string(
                index=False,
                float_format=lambda value: f"{value:.6f}",
            ),
            "",
            "CANDIDATE METRICS",
            candidates.to_string(
                index=False,
                float_format=lambda value: f"{value:.6f}",
            ),
            "",
            "UNCERTAINTY DIAGNOSTICS",
            json.dumps(report["uncertainty_diagnostics"], indent=2),
            "",
            "PAPER TABLE",
            paper_table.to_string(
                index=False,
                float_format=lambda value: f"{value:.6f}",
            ),
            "",
            "PRIMARY TASK-MACRO RESULT",
            json.dumps(aggregate["task_macro"], indent=2),
            "",
            "SUPPORTING FRAME-MICRO RESULT",
            json.dumps(aggregate["frame_micro_supporting"], indent=2),
            "",
        ]
    )
    text_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"Aligned frames: {aligned_path}")
    print(f"Per-task CSV: {task_path}")
    print(f"Boundary value CSV: {boundary_task_path}")
    print(f"Boundary credit CSV: {credit_jitter_task_path}")
    print(f"Candidate metrics CSV: {candidate_path}")
    print(f"Paper table CSV: {table_path}")
    print(f"JSON summary: {summary_path}")
    print(f"Text summary: {text_path}")


if __name__ == "__main__":
    main()
