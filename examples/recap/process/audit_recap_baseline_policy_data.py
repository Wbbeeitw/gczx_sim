#!/usr/bin/env python
"""Audit raw-Value policy labels for the multitask ReCap baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_mapping(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected TASK=PATH, got {value!r}")
    task, raw_path = value.split("=", 1)
    if not task or not raw_path:
        raise ValueError(f"Expected TASK=PATH, got {value!r}")
    return task, Path(raw_path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def audit_recap_baseline_policy_data(
    *,
    summary_path: Path,
    raw_reports: dict[str, Path],
    source_advantages_path: Path,
    source_feature_manifest: Path,
    value_checkpoint: Path,
    advantage_tag: str,
    expected_tasks: int = 10,
    expected_episodes_per_task: int = 30,
    expected_rollout_episodes: int = 20,
    expected_expert_episodes: int = 10,
    positive_quantile: float = 0.3,
    positive_ratio_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Validate coverage, provenance, and ungated raw top-quantile labels."""

    summary = _read_json(summary_path)
    feature_manifest = _read_json(source_feature_manifest)
    recorded_checkpoint = Path(str(feature_manifest["value_checkpoint"])).resolve()
    expected_checkpoint = value_checkpoint.resolve()
    if recorded_checkpoint != expected_checkpoint:
        raise ValueError(
            f"raw features use Value checkpoint {recorded_checkpoint}, "
            f"expected {expected_checkpoint}"
        )
    task_summaries = summary.get("tasks") or {}
    if len(task_summaries) != expected_tasks:
        raise ValueError(
            f"expected {expected_tasks} tasks, got {len(task_summaries)}"
        )
    if set(task_summaries) != set(raw_reports):
        raise ValueError("summary tasks and raw export reports do not match")
    if summary.get("advantage_tag") != advantage_tag:
        raise ValueError(
            f"expected advantage tag {advantage_tag!r}, "
            f"got {summary.get('advantage_tag')!r}"
        )

    expected_total_episodes = expected_tasks * expected_episodes_per_task
    aggregate = summary.get("aggregate") or {}
    if int(aggregate.get("episodes", -1)) != expected_total_episodes:
        raise ValueError(
            f"expected {expected_total_episodes} aggregate episodes, "
            f"got {aggregate.get('episodes')}"
        )

    resolved_source = source_advantages_path.resolve()
    audited_tasks: dict[str, Any] = {}
    for task in sorted(task_summaries):
        task_summary = task_summaries[task]
        episodes = int(task_summary.get("episodes", -1))
        if episodes != expected_episodes_per_task:
            raise ValueError(
                f"{task}: expected {expected_episodes_per_task} episodes, "
                f"got {episodes}"
            )

        sources = task_summary.get("sources") or {}
        rollout_episodes = int((sources.get("rollout") or {}).get("episodes", -1))
        expert_episodes = int((sources.get("expert") or {}).get("episodes", -1))
        if rollout_episodes != expected_rollout_episodes:
            raise ValueError(
                f"{task}: expected {expected_rollout_episodes} rollout episodes, "
                f"got {rollout_episodes}"
            )
        if expert_episodes != expected_expert_episodes:
            raise ValueError(
                f"{task}: expected {expected_expert_episodes} expert episodes, "
                f"got {expert_episodes}"
            )

        positive_ratio = float(task_summary.get("positive_ratio", -1.0))
        if abs(positive_ratio - positive_quantile) > positive_ratio_tolerance:
            raise ValueError(
                f"{task}: raw positive ratio {positive_ratio:.6f} is outside "
                f"{positive_quantile:.6f} +/- {positive_ratio_tolerance:.6f}"
            )

        export_report = _read_json(raw_reports[task])
        if export_report.get("mode") != "raw":
            raise ValueError(f"{task}: expected raw export mode")
        if export_report.get("output_tag") != advantage_tag:
            raise ValueError(f"{task}: raw export tag does not match {advantage_tag}")
        report_source = Path(str(export_report["source_advantages_path"])).resolve()
        if report_source != resolved_source:
            raise ValueError(
                f"{task}: source advantages {report_source} do not match "
                f"{resolved_source}"
            )
        if bool(export_report.get("success_gate", False)):
            raise ValueError(f"{task}: ReCap baseline must not enable success_gate")
        if bool(export_report.get("demo_backstop", False)):
            raise ValueError(f"{task}: ReCap baseline must not enable demo_backstop")
        if export_report.get("predictions_path") is not None:
            raise ValueError(f"{task}: raw baseline unexpectedly uses predictions")
        if int(export_report.get("episodes_exported", -1)) != episodes:
            raise ValueError(f"{task}: export episode coverage mismatch")
        if int(export_report.get("rows_exported", -1)) != int(
            task_summary.get("frames", -2)
        ):
            raise ValueError(f"{task}: export frame coverage mismatch")
        report_quantile = float(export_report.get("positive_quantile", -1.0))
        if abs(report_quantile - positive_quantile) > 1e-12:
            raise ValueError(f"{task}: export positive quantile mismatch")

        audited_tasks[task] = {
            "episodes": episodes,
            "rollout_episodes": rollout_episodes,
            "expert_episodes": expert_episodes,
            "frames": int(task_summary["frames"]),
            "positive_ratio": positive_ratio,
            "raw_export_report": str(raw_reports[task]),
        }

    return {
        "passed": True,
        "advantage_tag": advantage_tag,
        "source_advantages_path": str(resolved_source),
        "source_feature_manifest": str(source_feature_manifest.resolve()),
        "value_checkpoint": str(expected_checkpoint),
        "positive_quantile": positive_quantile,
        "positive_ratio_tolerance": positive_ratio_tolerance,
        "tasks": audited_tasks,
        "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-path", required=True, type=Path)
    parser.add_argument("--raw-report", action="append", required=True)
    parser.add_argument("--source-advantages-path", required=True, type=Path)
    parser.add_argument("--source-feature-manifest", required=True, type=Path)
    parser.add_argument("--value-checkpoint", required=True, type=Path)
    parser.add_argument("--advantage-tag", required=True)
    parser.add_argument("--expected-tasks", type=int, default=10)
    parser.add_argument("--expected-episodes-per-task", type=int, default=30)
    parser.add_argument("--expected-rollout-episodes", type=int, default=20)
    parser.add_argument("--expected-expert-episodes", type=int, default=10)
    parser.add_argument("--positive-quantile", type=float, default=0.3)
    parser.add_argument("--positive-ratio-tolerance", type=float, default=0.02)
    parser.add_argument("--output-path", required=True, type=Path)
    args = parser.parse_args()

    raw_reports = dict(_parse_mapping(value) for value in args.raw_report)
    audit = audit_recap_baseline_policy_data(
        summary_path=args.summary_path,
        raw_reports=raw_reports,
        source_advantages_path=args.source_advantages_path,
        source_feature_manifest=args.source_feature_manifest,
        value_checkpoint=args.value_checkpoint,
        advantage_tag=args.advantage_tag,
        expected_tasks=args.expected_tasks,
        expected_episodes_per_task=args.expected_episodes_per_task,
        expected_rollout_episodes=args.expected_rollout_episodes,
        expected_expert_episodes=args.expected_expert_episodes,
        positive_quantile=args.positive_quantile,
        positive_ratio_tolerance=args.positive_ratio_tolerance,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
