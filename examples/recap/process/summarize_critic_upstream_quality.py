#!/usr/bin/env python
"""Summarize held-out upstream quality of Raw and Fused critics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from summarize_v3_critic_bias import _assign_tasks, _load_aligned_frames


KEY_COLUMNS = ["episode_index", "frame_index"]


def _resolve_artifact(raw_path: str, comparison_path: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = comparison_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Comparison artifact not found: {path}")
    return path


def _spearman(left: pd.Series, right: pd.Series) -> float:
    left_rank = left.rank(method="average").to_numpy(dtype=np.float64)
    right_rank = right.rank(method="average").to_numpy(dtype=np.float64)
    if len(left_rank) < 2:
        return float("nan")
    if np.ptp(left_rank) == 0.0 or np.ptp(right_rank) == 0.0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _improvement(raw: float, fused: float) -> float:
    if raw == 0.0:
        return 0.0
    return float(100.0 * (1.0 - fused / raw))


def _infer_episode_success(
    frame: pd.DataFrame,
    failure_reward: float,
) -> pd.Series:
    ordered = frame.sort_values(KEY_COLUMNS)
    grouped = ordered.groupby("episode_index", sort=False)
    episode_length = grouped["frame_index"].size()
    first_return = grouped["target"].first()
    return first_return > -(
        episode_length + abs(float(failure_reward)) / 2.0
    )


def _mark_phase_boundaries(
    frame: pd.DataFrame,
    boundary_window: int,
) -> pd.DataFrame:
    frame = frame.sort_values(KEY_COLUMNS).copy()
    frame["is_phase_boundary"] = False
    for _, indices in frame.groupby("episode_index", sort=False).indices.items():
        positions = np.asarray(indices, dtype=np.int64)
        episode = frame.iloc[positions]
        phases = episode["phase_true"].to_numpy()
        episode_frames = episode["frame_index"].to_numpy(dtype=np.int64)
        boundary_positions = np.flatnonzero(phases[1:] != phases[:-1]) + 1
        if len(boundary_positions) == 0:
            continue
        boundary_frames = episode_frames[boundary_positions]
        distance = np.min(
            np.abs(episode_frames[:, None] - boundary_frames[None, :]),
            axis=1,
        )
        frame.iloc[
            positions,
            frame.columns.get_loc("is_phase_boundary"),
        ] = distance <= int(boundary_window)
    return frame


def load_evaluation_frame(
    comparison_path: Path,
    *,
    split: str,
    num_tasks: int,
    episodes_per_task: int,
    failure_reward: float,
    boundary_window: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load one held-out frame table with predictions and semantic labels."""
    comparison_path = comparison_path.expanduser().resolve()
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    predictions_raw = comparison.get("predictions_path")
    if not predictions_raw:
        raise ValueError("Comparison report does not contain predictions_path.")
    predictions_path = _resolve_artifact(
        str(predictions_raw), comparison_path
    )

    frame, return_range = _load_aligned_frames(
        comparison_path,
        comparison_path,
    )
    frame = frame[frame["split"].astype(str) == split].copy()
    if frame.empty:
        available = sorted(
            _load_aligned_frames(comparison_path, comparison_path)[0]["split"]
            .dropna()
            .astype(str)
            .unique()
        )
        raise ValueError(f"No rows for split={split!r}; available={available}")

    predictions = pd.read_parquet(predictions_path)
    semantic_columns = [*KEY_COLUMNS, "phase_true", "global_progress_true"]
    missing = set(semantic_columns) - set(predictions.columns)
    if missing:
        raise ValueError(
            f"Predictions missing semantic columns: {sorted(missing)}"
        )
    if predictions.duplicated(KEY_COLUMNS).any():
        raise ValueError("Predictions contain duplicate episode/frame keys.")
    frame = frame.merge(
        predictions[semantic_columns],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    if frame[["phase_true", "global_progress_true"]].isna().any().any():
        raise ValueError("Some held-out frames lack semantic labels.")

    frame = _assign_tasks(frame, num_tasks, episodes_per_task)
    success_by_episode = _infer_episode_success(frame, failure_reward)
    frame["is_success"] = frame["episode_index"].map(success_by_episode).astype(bool)
    frame["raw_abs_error"] = frame["recap_error"].abs()
    frame["fused_abs_error"] = frame["ours_error"].abs()
    frame["fused_frame_win"] = (
        frame["fused_abs_error"] < frame["raw_abs_error"]
    )
    frame["frame_error_tie"] = np.isclose(
        frame["fused_abs_error"],
        frame["raw_abs_error"],
        atol=1e-9,
        rtol=0.0,
    )
    frame = _mark_phase_boundaries(frame, boundary_window)

    metadata = {
        "comparison_path": str(comparison_path),
        "predictions_path": str(predictions_path),
        "split": split,
        "return_range": float(return_range),
        "failure_reward": float(failure_reward),
        "boundary_window": int(boundary_window),
        "frames": int(len(frame)),
        "episodes": int(frame["episode_index"].nunique()),
        "tasks": int(frame["task"].nunique()),
        "success_episodes": int(
            frame.groupby("episode_index")["is_success"].first().sum()
        ),
    }
    metadata["failure_episodes"] = (
        metadata["episodes"] - metadata["success_episodes"]
    )
    return frame, metadata


def _frame_metrics(part: pd.DataFrame) -> dict[str, Any]:
    if part.empty:
        return {
            "frames": 0,
            "episodes": 0,
        }
    raw_error = part["recap_error"].to_numpy(dtype=np.float64)
    fused_error = part["ours_error"].to_numpy(dtype=np.float64)
    raw_abs = np.abs(raw_error)
    fused_abs = np.abs(fused_error)
    raw_mae = float(raw_abs.mean())
    fused_mae = float(fused_abs.mean())
    raw_rmse = float(np.sqrt(np.mean(raw_error**2)))
    fused_rmse = float(np.sqrt(np.mean(fused_error**2)))
    raw_bias = float(raw_error.mean())
    fused_bias = float(fused_error.mean())
    raw_p90 = float(np.quantile(raw_abs, 0.90))
    fused_p90 = float(np.quantile(fused_abs, 0.90))
    raw_p95 = float(np.quantile(raw_abs, 0.95))
    fused_p95 = float(np.quantile(fused_abs, 0.95))
    return {
        "frames": int(len(part)),
        "episodes": int(part["episode_index"].nunique()),
        "raw_mae": raw_mae,
        "fused_mae": fused_mae,
        "mae_improvement_pct": _improvement(raw_mae, fused_mae),
        "raw_rmse": raw_rmse,
        "fused_rmse": fused_rmse,
        "rmse_improvement_pct": _improvement(raw_rmse, fused_rmse),
        "raw_bias": raw_bias,
        "fused_bias": fused_bias,
        "raw_abs_bias": abs(raw_bias),
        "fused_abs_bias": abs(fused_bias),
        "abs_bias_improvement_pct": _improvement(
            abs(raw_bias), abs(fused_bias)
        ),
        "raw_p90_abs_error": raw_p90,
        "fused_p90_abs_error": fused_p90,
        "p90_improvement_pct": _improvement(raw_p90, fused_p90),
        "raw_p95_abs_error": raw_p95,
        "fused_p95_abs_error": fused_p95,
        "p95_improvement_pct": _improvement(raw_p95, fused_p95),
        "fused_frame_win_rate": float(part["fused_frame_win"].mean()),
        "frame_tie_rate": float(part["frame_error_tie"].mean()),
        "raw_return_spearman": _spearman(
            part["recap_prediction"], part["target"]
        ),
        "fused_return_spearman": _spearman(
            part["ours_prediction"], part["target"]
        ),
    }


def _episode_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, episode_index), part in frame.groupby(
        ["task", "episode_index"], sort=True
    ):
        raw_mae = float(part["raw_abs_error"].mean())
        fused_mae = float(part["fused_abs_error"].mean())
        rows.append(
            {
                "task": task,
                "task_index": int(part["task_index"].iloc[0]),
                "episode_index": int(episode_index),
                "is_success": bool(part["is_success"].iloc[0]),
                "frames": int(len(part)),
                "raw_mae": raw_mae,
                "fused_mae": fused_mae,
                "mae_gain": raw_mae - fused_mae,
                "mae_improvement_pct": _improvement(raw_mae, fused_mae),
                "fused_wins": fused_mae < raw_mae,
                "raw_return_spearman": _spearman(
                    part["recap_prediction"], part["target"]
                ),
                "fused_return_spearman": _spearman(
                    part["ours_prediction"], part["target"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _task_frame(
    frame: pd.DataFrame,
    episodes: pd.DataFrame,
    num_tasks: int,
) -> pd.DataFrame:
    rows = []
    for task_index in range(num_tasks):
        task = f"task{task_index}"
        part = frame[frame["task"] == task]
        if part.empty:
            raise ValueError(f"No held-out rows for {task}.")
        task_episodes = episodes[episodes["task"] == task]
        metrics = _frame_metrics(part)
        metrics.update(
            {
                "task": task,
                "task_index": task_index,
                "episode_macro_raw_mae": float(
                    task_episodes["raw_mae"].mean()
                ),
                "episode_macro_fused_mae": float(
                    task_episodes["fused_mae"].mean()
                ),
                "episode_win_rate": float(
                    task_episodes["fused_wins"].mean()
                ),
            }
        )
        rows.append(metrics)
    columns = ["task", "task_index"] + [
        column for column in rows[0] if column not in {"task", "task_index"}
    ]
    return pd.DataFrame(rows)[columns]


def _slice_frame(frame: pd.DataFrame) -> pd.DataFrame:
    slices = {
        "all": np.ones(len(frame), dtype=bool),
        "success": frame["is_success"].to_numpy(dtype=bool),
        "failure": ~frame["is_success"].to_numpy(dtype=bool),
        "phase_boundary": frame["is_phase_boundary"].to_numpy(dtype=bool),
        "non_boundary": ~frame["is_phase_boundary"].to_numpy(dtype=bool),
    }
    rows = []
    for name, mask in slices.items():
        rows.append({"slice": name, **_frame_metrics(frame[mask])})
    return pd.DataFrame(rows)


def _phase_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for phase, part in frame.groupby("phase_true", sort=True):
        rows.append({"phase": int(phase), **_frame_metrics(part)})
    return pd.DataFrame(rows)


def _candidate_row(
    metric: str,
    scope: str,
    direction: str,
    raw: float | None,
    fused: float | None,
    note: str,
) -> dict[str, Any]:
    if raw is None or fused is None:
        absolute_gain = None
        relative_gain = None
    elif direction == "lower":
        absolute_gain = raw - fused
        relative_gain = _improvement(raw, fused)
    else:
        absolute_gain = fused - raw
        relative_gain = None
    return {
        "metric": metric,
        "scope": scope,
        "direction": direction,
        "raw": raw,
        "fused": fused,
        "absolute_gain": absolute_gain,
        "relative_gain_pct": relative_gain,
        "note": note,
    }


def build_candidate_table(
    slices: pd.DataFrame,
    tasks: pd.DataFrame,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact table of predeclared paper-candidate metrics."""
    by_slice = slices.set_index("slice")
    overall = by_slice.loc["all"]
    boundary = by_slice.loc["phase_boundary"]
    failure = by_slice.loc["failure"]
    success = by_slice.loc["success"]
    episode_raw = float(episodes["raw_mae"].mean())
    episode_fused = float(episodes["fused_mae"].mean())
    rows = [
        _candidate_row(
            "Frame MAE",
            "held-out val",
            "lower",
            float(overall["raw_mae"]),
            float(overall["fused_mae"]),
            "Proper point-prediction error",
        ),
        _candidate_row(
            "RMSE",
            "held-out val",
            "lower",
            float(overall["raw_rmse"]),
            float(overall["fused_rmse"]),
            "Penalizes large errors",
        ),
        _candidate_row(
            "Absolute signed bias",
            "held-out val",
            "lower",
            float(overall["raw_abs_bias"]),
            float(overall["fused_abs_bias"]),
            "Systematic optimism/pessimism magnitude",
        ),
        _candidate_row(
            "P90 absolute error",
            "held-out val",
            "lower",
            float(overall["raw_p90_abs_error"]),
            float(overall["fused_p90_abs_error"]),
            "Tail robustness",
        ),
        _candidate_row(
            "P95 absolute error",
            "held-out val",
            "lower",
            float(overall["raw_p95_abs_error"]),
            float(overall["fused_p95_abs_error"]),
            "More extreme tail diagnostic",
        ),
        _candidate_row(
            "Episode-macro MAE",
            "held-out val",
            "lower",
            episode_raw,
            episode_fused,
            "Every episode receives equal weight",
        ),
        _candidate_row(
            "Task-macro return Spearman",
            "held-out val",
            "higher",
            float(tasks["raw_return_spearman"].mean()),
            float(tasks["fused_return_spearman"].mean()),
            "Ranks realized return-to-go, not semantic progress",
        ),
        _candidate_row(
            "Boundary MAE",
            "phase boundary window",
            "lower",
            float(boundary["raw_mae"]),
            float(boundary["fused_mae"]),
            "Architecture-specific transition accuracy",
        ),
        _candidate_row(
            "Failure-frame MAE",
            "failed held-out episodes",
            "lower",
            float(failure["raw_mae"]),
            float(failure["fused_mae"]),
            "Hard unsuccessful trajectories",
        ),
        _candidate_row(
            "Success-frame MAE",
            "successful held-out episodes",
            "lower",
            float(success["raw_mae"]),
            float(success["fused_mae"]),
            "Successful trajectory accuracy",
        ),
        _candidate_row(
            "Frame win rate",
            "held-out val",
            "higher",
            None,
            float(overall["fused_frame_win_rate"]),
            "Fraction of frames where Fused absolute error is lower",
        ),
        _candidate_row(
            "Episode win rate",
            "held-out val",
            "higher",
            None,
            float(episodes["fused_wins"].mean()),
            "Fraction of episodes where Fused MAE is lower",
        ),
        _candidate_row(
            "Task win rate",
            "held-out val",
            "higher",
            None,
            float((tasks["fused_mae"] < tasks["raw_mae"]).mean()),
            "Fraction of tasks where Fused frame MAE is lower",
        ),
    ]
    return pd.DataFrame(rows)


def summarize_upstream_quality(
    frame: pd.DataFrame,
    *,
    num_tasks: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return slice, task, episode, phase, and candidate metric tables."""
    episodes = _episode_frame(frame)
    tasks = _task_frame(frame, episodes, num_tasks)
    slices = _slice_frame(frame)
    phases = _phase_frame(frame)
    candidates = build_candidate_table(slices, tasks, episodes)
    return slices, tasks, episodes, phases, candidates


def main() -> None:
    """Run the held-out upstream metric sweep and write reproducible tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument("--episodes-per-task", type=int, default=30)
    parser.add_argument("--failure-reward", type=float, default=-300.0)
    parser.add_argument("--boundary-window", type=int, default=10)
    args = parser.parse_args()
    if args.num_tasks <= 0 or args.episodes_per_task <= 0:
        raise ValueError("Task and episode counts must be positive.")
    if args.boundary_window < 0:
        raise ValueError("--boundary-window must be non-negative.")

    frame, metadata = load_evaluation_frame(
        args.comparison,
        split=args.split,
        num_tasks=args.num_tasks,
        episodes_per_task=args.episodes_per_task,
        failure_reward=args.failure_reward,
        boundary_window=args.boundary_window,
    )
    slices, tasks, episodes, phases, candidates = summarize_upstream_quality(
        frame,
        num_tasks=args.num_tasks,
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "aligned_frames": output_dir / "critic_upstream_aligned_frames.parquet",
        "slices": output_dir / "critic_upstream_by_slice.csv",
        "tasks": output_dir / "critic_upstream_by_task.csv",
        "episodes": output_dir / "critic_upstream_by_episode.csv",
        "phases": output_dir / "critic_upstream_by_phase.csv",
        "candidates": output_dir / "critic_upstream_paper_candidates.csv",
        "summary": output_dir / "critic_upstream_summary.json",
        "text": output_dir / "critic_upstream_summary.txt",
    }
    frame.to_parquet(paths["aligned_frames"], index=False)
    slices.to_csv(paths["slices"], index=False)
    tasks.to_csv(paths["tasks"], index=False)
    episodes.to_csv(paths["episodes"], index=False)
    phases.to_csv(paths["phases"], index=False)
    candidates.to_csv(paths["candidates"], index=False)
    report = {
        **metadata,
        "episode_win_rate": float(episodes["fused_wins"].mean()),
        "task_win_rate": float((tasks["fused_mae"] < tasks["raw_mae"]).mean()),
        "candidate_metrics": candidates.to_dict("records"),
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    paths["summary"].write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    text = "\n".join(
        [
            "HELD-OUT UPSTREAM CRITIC QUALITY",
            json.dumps(metadata, indent=2, ensure_ascii=False),
            "",
            "PAPER CANDIDATES",
            candidates.to_string(
                index=False,
                float_format=lambda value: f"{value:.6f}",
            ),
            "",
            "SLICE METRICS",
            slices.to_string(
                index=False,
                float_format=lambda value: f"{value:.6f}",
            ),
            "",
        ]
    )
    paths["text"].write_text(text, encoding="utf-8")
    print(text)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
