#!/usr/bin/env python
"""Measure whether ReCap credit rankings agree with semantic progress."""

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
        "global_progress_next",
        "global_progress_delta",
        "raw_credit",
        "advantage_continuous",
        "fused_credit",
    ]
    aligned = aligned[keep_columns].sort_values(KEY_COLUMNS).reset_index(drop=True)
    return aligned, metadata


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
    all_mae = metadata.get("value_mae_from_comparison", {}).get("all", {})
    task_macro = aggregate["task_macro"]
    return pd.DataFrame(
        [
            {
                "credit_source": "Raw Critic",
                "value_mae": all_mae.get("raw"),
                "credit_progress_rank_correlation": task_macro["raw"],
                "policy_sr_percent": raw_policy_sr,
            },
            {
                "credit_source": "Full GLC-Critic",
                "value_mae": all_mae.get("fused"),
                "credit_progress_rank_correlation": task_macro["fused"],
                "policy_sr_percent": fused_policy_sr,
            },
        ]
    )


def main() -> None:
    """Run the fixed held-out credit-progress rank-correlation experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--lookahead-step", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument("--episodes-per-task", type=int, default=30)
    parser.add_argument("--raw-policy-sr", type=float)
    parser.add_argument("--fused-policy-sr", type=float)
    args = parser.parse_args()
    if args.lookahead_step <= 0:
        raise ValueError("--lookahead-step must be positive.")
    if args.gamma < 0.0:
        raise ValueError("--gamma must be non-negative.")
    if args.num_tasks <= 0 or args.episodes_per_task <= 0:
        raise ValueError("Task and episode counts must be positive.")
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
    report = {**metadata, **aggregate}
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
    table_path = output_dir / "credit_progress_paper_table.csv"
    summary_path = output_dir / "credit_progress_summary.json"
    text_path = output_dir / "credit_progress_summary.txt"
    aligned.to_parquet(aligned_path, index=False)
    by_task.to_csv(task_path, index=False)
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
    print(f"Paper table CSV: {table_path}")
    print(f"JSON summary: {summary_path}")
    print(f"Text summary: {text_path}")


if __name__ == "__main__":
    main()
