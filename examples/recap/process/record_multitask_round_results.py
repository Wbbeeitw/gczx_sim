"""Aggregate per-task policy and critic metrics for one multitask round."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from examples.recap.process.record_round_results import (
        _error_metrics,
        _load_comparison_frame,
        _load_fusion_metrics,
        _load_zp_metrics,
        _policy_summary,
        _read_json,
    )
else:
    from record_round_results import (
        _error_metrics,
        _load_comparison_frame,
        _load_fusion_metrics,
        _load_zp_metrics,
        _policy_summary,
        _read_json,
    )


def _parse_path_mapping(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected TASK=PATH, got {value!r}")
    task, raw_path = value.split("=", 1)
    return task, Path(raw_path)


def _parse_range(value: str) -> tuple[str, tuple[int, int]]:
    if "=" not in value or ":" not in value:
        raise ValueError(f"Expected TASK=START:END, got {value!r}")
    task, raw_range = value.split("=", 1)
    raw_start, raw_end = raw_range.split(":", 1)
    start, end = int(raw_start), int(raw_end)
    if start < 0 or end <= start:
        raise ValueError(f"Invalid task range {value!r}")
    return task, (start, end)


def _pooled_success_length_metrics(
    task_metrics: dict[str, dict[str, Any]],
) -> tuple[float | None, float | None]:
    stats = []
    for metrics in task_metrics.values():
        count = metrics.get("successes")
        mean = metrics.get("success_only_act")
        std = metrics.get("success_only_act_std")
        if count and mean is not None and std is not None:
            stats.append((int(count), float(mean), float(std)))
    total = sum(count for count, _, _ in stats)
    if total == 0:
        return None, None
    mean = sum(count * task_mean for count, task_mean, _ in stats) / total
    second_moment = sum(
        count * (task_std**2 + task_mean**2)
        for count, task_mean, task_std in stats
    ) / total
    return mean, math.sqrt(max(0.0, second_moment - mean**2))


def record_multitask_results(
    *,
    round_index: int,
    policy_label: str,
    critic_label: str,
    checkpoint_path: str,
    task_ranges: dict[str, tuple[int, int]],
    eval_summaries: dict[str, Path],
    zp_metrics: dict[str, Path],
    fusion_metrics: dict[str, Path],
    comparison_path: Path,
    policy_data_report_path: Path | None,
    output_dir: Path,
    output_name: str,
) -> dict[str, Any]:
    """Write detailed JSON and flat per-task CSV results."""
    tasks = list(task_ranges)
    if set(tasks) != set(eval_summaries):
        raise ValueError("task ranges and evaluation summaries must match")

    comparison = _read_json(comparison_path)
    comparison_frame = _load_comparison_frame(comparison)
    policy_data_report = (
        _read_json(policy_data_report_path)
        if policy_data_report_path and policy_data_report_path.exists()
        else None
    )

    task_results: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for task in tasks:
        start, end = task_ranges[task]
        policy_metrics = _policy_summary(eval_summaries[task])
        task_frame = comparison_frame[
            (comparison_frame["episode_index"] >= start)
            & (comparison_frame["episode_index"] < end)
        ]
        critic_metrics = _error_metrics(task_frame)
        zp = _load_zp_metrics(zp_metrics.get(task))
        fusion = _load_fusion_metrics(fusion_metrics.get(task))
        policy_data = (
            policy_data_report.get("tasks", {}).get(task)
            if policy_data_report
            else None
        )
        task_result = {
            "episode_range": [start, end],
            "policy": policy_metrics,
            "critic": critic_metrics,
            "zp": zp,
            "fusion": fusion,
            "policy_data": policy_data,
        }
        task_results[task] = task_result
        csv_rows.append(
            {
                "task": task,
                "round": round_index,
                "policy": policy_label,
                "critic": critic_label,
                "num_trajectories": policy_metrics.get("episodes"),
                "successes": policy_metrics.get("successes"),
                "success_rate": policy_metrics.get("success_rate"),
                "success_episode_act_mean": policy_metrics.get(
                    "success_only_act"
                ),
                "success_episode_act_std": policy_metrics.get(
                    "success_only_act_std"
                ),
                "raw_frame_mae": critic_metrics.get("base_mae"),
                "fused_frame_mae": critic_metrics.get("fused_mae"),
                "positive_ratio": (
                    policy_data.get("positive_ratio") if policy_data else None
                ),
            }
        )

    success_rates = [
        result["policy"]["success_rate"]
        for result in task_results.values()
        if result["policy"].get("success_rate") is not None
    ]
    total_trajectories = sum(
        int(result["policy"].get("episodes") or 0)
        for result in task_results.values()
    )
    total_successes = sum(
        int(result["policy"].get("successes") or 0)
        for result in task_results.values()
    )
    success_act_mean, success_act_std = _pooled_success_length_metrics(
        {task: result["policy"] for task, result in task_results.items()}
    )
    fused_mae = [
        result["critic"].get("fused_mae") for result in task_results.values()
    ]
    fused_mae = [value for value in fused_mae if value is not None]
    aggregate = {
        "tasks": len(task_results),
        "num_trajectories": total_trajectories,
        "successes": total_successes,
        "macro_success_rate": (
            float(np.mean(success_rates)) if success_rates else None
        ),
        "micro_success_rate": (
            total_successes / total_trajectories if total_trajectories else None
        ),
        "success_episode_act_mean": success_act_mean,
        "success_episode_act_std": success_act_std,
        "macro_fused_frame_mae": (
            float(np.mean(fused_mae)) if fused_mae else None
        ),
    }
    result = {
        "schema_version": "1.0",
        "round": round_index,
        "policy_label": policy_label,
        "critic_label": critic_label,
        "policy_checkpoint": checkpoint_path,
        "tasks": task_results,
        "aggregate": aggregate,
        "primary_metrics": {
            "macro_success_rate": aggregate["macro_success_rate"],
            "micro_success_rate": aggregate["micro_success_rate"],
            "num_trajectories": aggregate["num_trajectories"],
            "success_episode_act_mean": aggregate[
                "success_episode_act_mean"
            ],
            "success_episode_act_std": aggregate["success_episode_act_std"],
            "macro_fused_frame_mae": aggregate["macro_fused_frame_mae"],
        },
        "source_paths": {
            "comparison": str(comparison_path),
            "policy_data_report": (
                str(policy_data_report_path) if policy_data_report_path else None
            ),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{output_name}.json"
    csv_path = output_dir / f"{output_name}.csv"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--policy-label", required=True)
    parser.add_argument("--critic-label", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--task-range", action="append", required=True)
    parser.add_argument("--eval-summary", action="append", required=True)
    parser.add_argument("--zp-metrics", action="append", default=[])
    parser.add_argument("--fusion-metrics", action="append", default=[])
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--policy-data-report", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()

    result = record_multitask_results(
        round_index=args.round,
        policy_label=args.policy_label,
        critic_label=args.critic_label,
        checkpoint_path=args.checkpoint_path,
        task_ranges=dict(_parse_range(value) for value in args.task_range),
        eval_summaries=dict(
            _parse_path_mapping(value) for value in args.eval_summary
        ),
        zp_metrics=dict(_parse_path_mapping(value) for value in args.zp_metrics),
        fusion_metrics=dict(
            _parse_path_mapping(value) for value in args.fusion_metrics
        ),
        comparison_path=args.comparison,
        policy_data_report_path=args.policy_data_report,
        output_dir=args.output_dir,
        output_name=args.output_name,
    )
    print(json.dumps(result["primary_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
