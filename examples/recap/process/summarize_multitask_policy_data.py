"""Validate and summarize per-task policy advantage datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _parse_mapping(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected TASK=PATH, got {value!r}")
    task, raw_path = value.split("=", 1)
    if not task or not raw_path:
        raise ValueError(f"Expected TASK=PATH, got {value!r}")
    return task, Path(raw_path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def summarize_policy_data(
    task_datasets: dict[str, Path],
    manifests: dict[str, Path],
    advantage_tag: str,
    report_path: Path,
    combined_manifest_path: Path,
) -> dict[str, Any]:
    """Validate advantage coverage and combine per-task manifests."""
    if set(task_datasets) != set(manifests):
        raise ValueError("task datasets and manifests must contain identical tasks")

    combined_manifest: dict[str, Any] = {}
    task_reports: dict[str, Any] = {}
    total_frames = 0
    total_positive = 0
    total_episodes = 0

    for task, dataset_path in task_datasets.items():
        info = _read_json(dataset_path / "meta" / "info.json")
        advantage_path = (
            dataset_path / "meta" / f"advantages_{advantage_tag}.parquet"
        )
        advantages = pd.read_parquet(advantage_path)
        required = {"episode_index", "frame_index", "advantage"}
        missing = required.difference(advantages.columns)
        if missing:
            raise ValueError(
                f"{task}: advantage table missing columns {sorted(missing)}"
            )

        expected_frames = int(info["total_frames"])
        expected_episodes = int(info["total_episodes"])
        exported_episodes = int(advantages["episode_index"].nunique())
        if len(advantages) != expected_frames:
            raise ValueError(
                f"{task}: expected {expected_frames} advantage rows, "
                f"got {len(advantages)}"
            )
        if exported_episodes != expected_episodes:
            raise ValueError(
                f"{task}: expected {expected_episodes} advantage episodes, "
                f"got {exported_episodes}"
            )

        manifest = _read_json(manifests[task])
        for key, value in manifest.items():
            if key in combined_manifest:
                raise ValueError(f"Duplicate manifest dataset key: {key}")
            combined_manifest[key] = value

        provenance_rows = _read_jsonl(
            dataset_path / "meta" / "episode_provenance.jsonl"
        )
        source_by_episode = {
            int(row["episode_index"]): str(row.get("source_type", "unknown"))
            for row in provenance_rows
        }
        default_source = "unknown" if provenance_rows else "rollout"
        frame_sources = advantages["episode_index"].astype(int).map(
            lambda episode: source_by_episode.get(episode, default_source)
        )
        positive = advantages["advantage"].astype(bool)
        source_reports: dict[str, Any] = {}
        for source_type in sorted(set(frame_sources)):
            source_mask = frame_sources == source_type
            source_positive = positive[source_mask]
            source_reports[source_type] = {
                "frames": int(source_mask.sum()),
                "positive_frames": int(source_positive.sum()),
                "positive_ratio": (
                    float(source_positive.mean()) if len(source_positive) else 0.0
                ),
                "episodes": int(
                    advantages.loc[source_mask, "episode_index"].nunique()
                ),
            }

        positive_count = int(positive.sum())
        task_reports[task] = {
            "dataset_path": str(dataset_path),
            "advantage_path": str(advantage_path),
            "episodes": expected_episodes,
            "frames": expected_frames,
            "positive_frames": positive_count,
            "positive_ratio": float(positive.mean()),
            "sources": source_reports,
        }
        total_frames += expected_frames
        total_positive += positive_count
        total_episodes += expected_episodes

    combined_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    combined_manifest_path.write_text(
        json.dumps(combined_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = {
        "advantage_tag": advantage_tag,
        "tasks": task_reports,
        "aggregate": {
            "tasks": len(task_reports),
            "episodes": total_episodes,
            "frames": total_frames,
            "positive_frames": total_positive,
            "positive_ratio": (
                total_positive / total_frames if total_frames else 0.0
            ),
        },
        "combined_manifest_path": str(combined_manifest_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dataset", action="append", required=True)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--advantage-tag", required=True)
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--combined-manifest-path", required=True)
    args = parser.parse_args()

    task_datasets = dict(_parse_mapping(value) for value in args.task_dataset)
    manifests = dict(_parse_mapping(value) for value in args.manifest)
    report = summarize_policy_data(
        task_datasets,
        manifests,
        args.advantage_tag,
        Path(args.report_path),
        Path(args.combined_manifest_path),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
