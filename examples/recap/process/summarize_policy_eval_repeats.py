#!/usr/bin/env python
"""Summarize repeated multitask policy evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _metric_value(metrics: dict[str, Any], name: str) -> Any:
    if name in metrics:
        return metrics[name]
    matches = [
        value for key, value in metrics.items() if key.endswith(f"/{name}")
    ]
    return matches[0] if len(matches) == 1 else None


def _read_task_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation summary not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics") or payload
    success_rate = _metric_value(metrics, "success_rate")
    if success_rate is None:
        raise ValueError(f"Success rate not found in evaluation summary: {path}")
    episodes = _metric_value(metrics, "num_trajectories")
    successes = _metric_value(metrics, "success_count")
    return {
        "success_rate": float(success_rate),
        "episodes": int(float(episodes)) if episodes is not None else None,
        "successes": int(float(successes)) if successes is not None else None,
    }


def _parse_repeat(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Repeat must use NAME=PATH syntax: {raw}")
    name, path_text = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Repeat name cannot be empty: {raw}")
    return name, Path(path_text).expanduser().resolve()


def summarize_repeats(
    repeats: list[tuple[str, Path]],
    num_tasks: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return repeat-task rows, task aggregates, and overall aggregates."""
    rows = []
    for repeat, root in repeats:
        for task_index in range(num_tasks):
            task = f"task{task_index}"
            summary_path = root / task / "eval_policy_summary.json"
            rows.append(
                {
                    "repeat": repeat,
                    "task": task,
                    "task_index": task_index,
                    "summary_path": str(summary_path),
                    **_read_task_summary(summary_path),
                }
            )
    repeat_task = pd.DataFrame(rows)

    repeat_order = [name for name, _ in repeats]
    task_rows = []
    for task_index in range(num_tasks):
        task = f"task{task_index}"
        part = repeat_task[repeat_task["task"] == task]
        row: dict[str, Any] = {"task": task, "task_index": task_index}
        for repeat in repeat_order:
            value = part.loc[part["repeat"] == repeat, "success_rate"]
            row[f"{repeat}_success_rate"] = float(value.iloc[0])
        row["mean_success_rate"] = float(part["success_rate"].mean())
        row["std_success_rate"] = float(part["success_rate"].std(ddof=1))
        task_rows.append(row)
    by_task = pd.DataFrame(task_rows)

    repeat_macro = {
        repeat: float(
            repeat_task.loc[
                repeat_task["repeat"] == repeat, "success_rate"
            ].mean()
        )
        for repeat in repeat_order
    }
    aggregate = {
        "repeats": repeat_order,
        "tasks": num_tasks,
        "repeat_macro_success_rate": repeat_macro,
        "three_repeat_task_macro_mean": float(
            repeat_task["success_rate"].mean()
        ),
        "repeat_macro_mean": float(pd.Series(repeat_macro).mean()),
        "repeat_macro_std": float(pd.Series(repeat_macro).std(ddof=1)),
        "std_definition": "sample standard deviation across evaluation repeats",
    }
    return repeat_task, by_task, aggregate


def main() -> None:
    """Write detailed success-rate summaries for repeated policy evaluations."""
    parser = argparse.ArgumentParser(
        description="Summarize task success rates across policy evaluation repeats."
    )
    parser.add_argument(
        "--repeat",
        action="append",
        required=True,
        help="Repeat evaluation root in NAME=PATH syntax; pass once per repeat.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-tasks", type=int, default=10)
    args = parser.parse_args()
    if args.num_tasks <= 0:
        raise ValueError("--num-tasks must be positive.")

    repeats = [_parse_repeat(raw) for raw in args.repeat]
    names = [name for name, _ in repeats]
    if len(set(names)) != len(names):
        raise ValueError(f"Repeat names must be unique: {names}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    repeat_task, by_task, aggregate = summarize_repeats(repeats, args.num_tasks)

    repeat_task_path = output_dir / "policy_success_by_repeat_task.csv"
    by_task_path = output_dir / "policy_success_by_task.csv"
    summary_path = output_dir / "policy_success_summary.json"
    text_path = output_dir / "policy_success_summary.txt"
    repeat_task.to_csv(repeat_task_path, index=False)
    by_task.to_csv(by_task_path, index=False)
    summary_path.write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    text = "\n".join(
        [
            "Success rates are fractions in [0, 1].",
            "",
            by_task.to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            ),
            "",
            "AGGREGATE",
            json.dumps(aggregate, indent=2, ensure_ascii=False),
            "",
        ]
    )
    text_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"Repeat/task CSV: {repeat_task_path}")
    print(f"Per-task CSV: {by_task_path}")
    print(f"JSON summary: {summary_path}")
    print(f"Text summary: {text_path}")


if __name__ == "__main__":
    main()
