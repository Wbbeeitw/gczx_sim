"""Summarize ReCAP raw-Critic and Ours fused-Critic bias on v3 data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _resolve_report_path(raw_path: str, report_path: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = report_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Comparison artifact not found: {path}")
    return path


def _map_values_to_returns(values: pd.Series, report: dict[str, Any]) -> np.ndarray:
    return_min = float(report.get("return_min", -900.0))
    return_max = float(report.get("return_max", 0.0))
    value_min = float(report.get("value_min", -1.0))
    value_max = float(report.get("value_max", 0.0))
    return_range = return_max - return_min
    value_range = value_max - value_min
    if return_range <= 0.0 or value_range <= 0.0:
        raise ValueError(
            "Invalid comparison scales: "
            f"return=[{return_min}, {return_max}], "
            f"value=[{value_min}, {value_max}]"
        )
    array = values.to_numpy(dtype=np.float64)
    return (array - value_min) / value_range * return_range + return_min


def _load_method_frame(report_path: Path, prediction: str) -> pd.DataFrame:
    report_path = report_path.expanduser().resolve()
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    advantages_raw = report.get("advantages_path")
    predictions_raw = report.get("predictions_path")
    if not advantages_raw or not predictions_raw:
        raise ValueError(
            f"Comparison report lacks artifact paths: {report_path}"
        )
    advantages_path = _resolve_report_path(str(advantages_raw), report_path)
    predictions_path = _resolve_report_path(str(predictions_raw), report_path)
    advantages = pd.read_parquet(advantages_path)
    predictions = pd.read_parquet(predictions_path)
    required_advantages = {
        "episode_index",
        "frame_index",
        "return",
        "value_current",
    }
    required_predictions = {"episode_index", "frame_index", "value_fused"}
    missing_advantages = required_advantages - set(advantages.columns)
    missing_predictions = required_predictions - set(predictions.columns)
    if missing_advantages:
        raise ValueError(
            f"Advantages missing columns {sorted(missing_advantages)}: "
            f"{advantages_path}"
        )
    if missing_predictions:
        raise ValueError(
            f"Predictions missing columns {sorted(missing_predictions)}: "
            f"{predictions_path}"
        )

    prediction_columns = ["episode_index", "frame_index", "value_fused"]
    if "split" in predictions.columns:
        prediction_columns.append("split")
    frame = predictions[prediction_columns].merge(
        advantages[
            ["episode_index", "frame_index", "return", "value_current"]
        ],
        on=["episode_index", "frame_index"],
        how="inner",
        validate="one_to_one",
    )
    if frame.empty:
        raise ValueError(
            f"No overlapping rows for comparison report: {report_path}"
        )
    if "split" not in frame.columns:
        frame["split"] = "all"
    source_column = "value_current" if prediction == "base" else "value_fused"
    frame["prediction"] = _map_values_to_returns(frame[source_column], report)
    frame["target"] = frame["return"].astype(float)
    return frame[
        ["episode_index", "frame_index", "split", "target", "prediction"]
    ].copy()


def _load_aligned_frames(
    recap_comparison: Path,
    ours_comparison: Path,
) -> tuple[pd.DataFrame, float]:
    recap = _load_method_frame(recap_comparison, prediction="base").rename(
        columns={
            "split": "recap_split",
            "target": "recap_target",
            "prediction": "recap_prediction",
        }
    )
    ours_report = json.loads(ours_comparison.read_text(encoding="utf-8"))
    ours = _load_method_frame(ours_comparison, prediction="fused").rename(
        columns={
            "split": "ours_split",
            "target": "ours_target",
            "prediction": "ours_prediction",
        }
    )
    frame = recap.merge(
        ours,
        on=["episode_index", "frame_index"],
        how="inner",
        validate="one_to_one",
    )
    if frame.empty:
        raise ValueError("ReCAP and Ours comparisons have no overlapping frames.")
    if not np.allclose(frame["recap_target"], frame["ours_target"], atol=1e-6):
        raise ValueError("ReCAP and Ours target returns differ on aligned frames.")
    frame["split"] = frame["ours_split"].fillna(frame["recap_split"])
    frame["target"] = frame["ours_target"]
    frame["recap_error"] = frame["recap_prediction"] - frame["target"]
    frame["ours_error"] = frame["ours_prediction"] - frame["target"]
    return_range = float(ours_report.get("return_max", 0.0)) - float(
        ours_report.get("return_min", -900.0)
    )
    if return_range <= 0.0:
        raise ValueError(f"Invalid Ours return range: {return_range}")
    return frame, return_range


def _metric_row(part: pd.DataFrame, return_range: float) -> dict[str, Any]:
    recap_error = part["recap_error"].to_numpy(dtype=np.float64)
    ours_error = part["ours_error"].to_numpy(dtype=np.float64)
    recap_bias = float(np.mean(recap_error))
    ours_bias = float(np.mean(ours_error))
    recap_mae = float(np.mean(np.abs(recap_error)))
    ours_mae = float(np.mean(np.abs(ours_error)))
    recap_rmse = float(np.sqrt(np.mean(recap_error**2)))
    ours_rmse = float(np.sqrt(np.mean(ours_error**2)))
    bias_abs_improvement = 0.0
    if abs(recap_bias) > 0.0:
        bias_abs_improvement = 100.0 * (
            1.0 - abs(ours_bias) / abs(recap_bias)
        )
    mae_improvement = 0.0
    if recap_mae > 0.0:
        mae_improvement = 100.0 * (1.0 - ours_mae / recap_mae)
    return {
        "frames": int(len(part)),
        "episodes": int(part["episode_index"].nunique()),
        "recap_bias": recap_bias,
        "ours_bias": ours_bias,
        "recap_abs_bias": abs(recap_bias),
        "ours_abs_bias": abs(ours_bias),
        "bias_abs_improvement_pct": bias_abs_improvement,
        "recap_normalized_bias": recap_bias / return_range,
        "ours_normalized_bias": ours_bias / return_range,
        "recap_mae": recap_mae,
        "ours_mae": ours_mae,
        "mae_improvement_pct": mae_improvement,
        "recap_normalized_mae": recap_mae / return_range,
        "ours_normalized_mae": ours_mae / return_range,
        "recap_rmse": recap_rmse,
        "ours_rmse": ours_rmse,
    }


def _assign_tasks(
    frame: pd.DataFrame,
    num_tasks: int,
    episodes_per_task: int,
) -> pd.DataFrame:
    frame = frame.copy()
    task_index = frame["episode_index"].astype(int) // episodes_per_task
    invalid = (task_index < 0) | (task_index >= num_tasks)
    if invalid.any():
        invalid_episodes = sorted(
            frame.loc[invalid, "episode_index"].astype(int).unique().tolist()
        )
        raise ValueError(
            "Episode indices fall outside the configured task ranges: "
            f"{invalid_episodes[:10]}"
        )
    frame["task_index"] = task_index
    frame["task"] = frame["task_index"].map(lambda index: f"task{index}")
    frame["local_episode"] = (
        frame["episode_index"].astype(int) % episodes_per_task
    )
    return frame


def summarize_bias(
    recap_comparison: Path,
    ours_comparison: Path,
    num_tasks: int,
    episodes_per_task: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return task, task-split, episode, and aggregate bias summaries."""
    frame, return_range = _load_aligned_frames(
        recap_comparison.resolve(), ours_comparison.resolve()
    )
    frame = _assign_tasks(frame, num_tasks, episodes_per_task)

    task_rows = []
    task_split_rows = []
    for task_index in range(num_tasks):
        task = f"task{task_index}"
        task_part = frame[frame["task"] == task]
        if task_part.empty:
            continue
        task_rows.append({"task": task, **_metric_row(task_part, return_range)})
        task_split_rows.append(
            {"task": task, "split": "all", **_metric_row(task_part, return_range)}
        )
        for split in sorted(task_part["split"].dropna().astype(str).unique()):
            if split == "all":
                continue
            split_part = task_part[task_part["split"].astype(str) == split]
            task_split_rows.append(
                {
                    "task": task,
                    "split": split,
                    **_metric_row(split_part, return_range),
                }
            )

    episode_rows = []
    for (task, local_episode), part in frame.groupby(
        ["task", "local_episode"], sort=True
    ):
        episode_rows.append(
            {
                "task": task,
                "local_episode": int(local_episode),
                "global_episode": int(part["episode_index"].iloc[0]),
                "split": str(part["split"].iloc[0]),
                **_metric_row(part, return_range),
            }
        )

    task_frame = pd.DataFrame(task_rows)
    task_split_frame = pd.DataFrame(task_split_rows)
    episode_frame = pd.DataFrame(episode_rows)
    micro = _metric_row(frame, return_range)
    aggregate = {
        "bias_definition": "mean(predicted_return - realized_return_to_go)",
        "bias_interpretation": {
            "positive": "optimistic; predicted return is too high",
            "negative": "pessimistic; predicted return is too low",
        },
        "recap_prediction": "Raw/Base Critic",
        "ours_prediction": "Fused Critic",
        "return_range": return_range,
        "frame_micro": micro,
        "task_macro": {
            "tasks": int(len(task_frame)),
            "recap_mean_signed_bias": float(task_frame["recap_bias"].mean()),
            "ours_mean_signed_bias": float(task_frame["ours_bias"].mean()),
            "recap_mean_abs_task_bias": float(
                task_frame["recap_abs_bias"].mean()
            ),
            "ours_mean_abs_task_bias": float(task_frame["ours_abs_bias"].mean()),
            "recap_mean_mae": float(task_frame["recap_mae"].mean()),
            "ours_mean_mae": float(task_frame["ours_mae"].mean()),
        },
        "episode_macro": {
            "episodes": int(len(episode_frame)),
            "recap_mean_signed_bias": float(episode_frame["recap_bias"].mean()),
            "ours_mean_signed_bias": float(episode_frame["ours_bias"].mean()),
            "recap_mean_abs_episode_bias": float(
                episode_frame["recap_abs_bias"].mean()
            ),
            "ours_mean_abs_episode_bias": float(
                episode_frame["ours_abs_bias"].mean()
            ),
            "recap_mean_mae": float(episode_frame["recap_mae"].mean()),
            "ours_mean_mae": float(episode_frame["ours_mae"].mean()),
        },
    }
    return task_frame, task_split_frame, episode_frame, aggregate


def _render_task_table(task_frame: pd.DataFrame) -> str:
    columns = [
        "task",
        "frames",
        "episodes",
        "recap_bias",
        "ours_bias",
        "bias_abs_improvement_pct",
        "recap_mae",
        "ours_mae",
        "mae_improvement_pct",
        "recap_normalized_bias",
        "ours_normalized_bias",
    ]
    return task_frame[columns].to_string(
        index=False,
        formatters={
            column: (lambda value: f"{value:.4f}")
            for column in columns
            if column not in {"task", "frames", "episodes"}
        },
    )


def main() -> None:
    """Write detailed ReCAP-vs-Ours bias tables from comparison artifacts."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare ReCAP Raw/Base Critic bias with Ours Fused Critic bias."
        )
    )
    parser.add_argument("--ours-comparison", required=True)
    parser.add_argument(
        "--recap-comparison",
        help="Defaults to --ours-comparison because both share the raw Critic.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument("--episodes-per-task", type=int, default=30)
    args = parser.parse_args()
    if args.num_tasks <= 0 or args.episodes_per_task <= 0:
        raise ValueError("Task and episode counts must be positive.")

    ours_comparison = Path(args.ours_comparison).expanduser().resolve()
    recap_comparison = Path(
        args.recap_comparison or args.ours_comparison
    ).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    task_frame, task_split_frame, episode_frame, aggregate = summarize_bias(
        recap_comparison,
        ours_comparison,
        args.num_tasks,
        args.episodes_per_task,
    )

    task_path = output_dir / "critic_bias_by_task.csv"
    task_split_path = output_dir / "critic_bias_by_task_split.csv"
    episode_path = output_dir / "critic_bias_by_episode.csv"
    summary_path = output_dir / "critic_bias_summary.json"
    text_path = output_dir / "critic_bias_summary.txt"
    task_frame.to_csv(task_path, index=False)
    task_split_frame.to_csv(task_split_path, index=False)
    episode_frame.to_csv(episode_path, index=False)
    summary_path.write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    table = _render_task_table(task_frame)
    text = "\n".join(
        [
            "Bias = prediction - realized return-to-go",
            "Positive is optimistic; negative is pessimistic.",
            "ReCAP = Raw/Base Critic; Ours = Fused Critic.",
            "",
            table,
            "",
            "AGGREGATE",
            json.dumps(aggregate, indent=2, ensure_ascii=False),
            "",
        ]
    )
    text_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"Task CSV: {task_path}")
    print(f"Task/split CSV: {task_split_path}")
    print(f"Episode CSV: {episode_path}")
    print(f"JSON summary: {summary_path}")
    print(f"Text summary: {text_path}")


if __name__ == "__main__":
    main()
