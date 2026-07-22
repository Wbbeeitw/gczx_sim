#!/usr/bin/env python
"""Rebuild phase sidecars from complete raw semantic traces.

The repair is conservative: missing raw-trace episodes are never invented.
Existing sidecars and collection summaries receive ``.bak`` backups before
they are replaced.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from rlinf.revalue.semantic_trace import (
    build_task1_phase_labels,
    build_task2_phase_labels,
    build_task3_phase_labels,
    build_task4_phase_labels,
    build_task5_phase_labels,
    build_task6_phase_labels,
    build_task7_phase_labels,
    build_task9_phase_labels,
    write_task0_semantic_artifacts,
    write_task1_semantic_artifacts,
    write_task2_semantic_artifacts,
    write_task3_semantic_artifacts,
    write_task4_semantic_artifacts,
    write_task5_semantic_artifacts,
    write_task6_semantic_artifacts,
    write_task7_semantic_artifacts,
    write_task8_semantic_artifacts,
    write_task9_semantic_artifacts,
)

Builder = Callable[..., tuple[pd.DataFrame, pd.DataFrame]]
Writer = Callable[..., dict[str, str]]

BUILDERS: dict[str, Builder] = {
    "task0": build_task1_phase_labels,
    "task1": build_task1_phase_labels,
    "task2": build_task2_phase_labels,
    "task3": build_task3_phase_labels,
    "task4": build_task4_phase_labels,
    "task5": build_task5_phase_labels,
    "task6": build_task6_phase_labels,
    "task7": build_task7_phase_labels,
    "task8": build_task7_phase_labels,
    "task9": build_task9_phase_labels,
}
WRITERS: dict[str, Writer] = {
    "task0": write_task0_semantic_artifacts,
    "task1": write_task1_semantic_artifacts,
    "task2": write_task2_semantic_artifacts,
    "task3": write_task3_semantic_artifacts,
    "task4": write_task4_semantic_artifacts,
    "task5": write_task5_semantic_artifacts,
    "task6": write_task6_semantic_artifacts,
    "task7": write_task7_semantic_artifacts,
    "task8": write_task8_semantic_artifacts,
    "task9": write_task9_semantic_artifacts,
}
DEFAULT_STABLE_FRAMES = {"task0": 3, "task1": 5}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _backup(path: Path) -> None:
    backup = path.with_name(path.name + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)


def _parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected TASK=DATASET, got {value!r}")
    task, raw_path = value.split("=", 1)
    if task not in BUILDERS or not raw_path:
        raise ValueError(f"Invalid task dataset mapping: {value!r}")
    return task, Path(raw_path)


def repair_dataset(
    task: str,
    dataset: Path,
    *,
    stable_frames: int | None,
    fix_summary: bool,
    min_trainable: int,
    dry_run: bool,
) -> dict[str, Any]:
    meta = dataset / "meta"
    episodes = _read_jsonl(meta / "episodes.jsonl")
    raw_path = meta / f"semantic_trace_{task}.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"{task}: missing raw trace {raw_path}")
    raw = pd.read_parquet(raw_path)
    expected_ids = {int(row["episode_index"]) for row in episodes}
    actual_ids = {int(value) for value in raw["episode_index"].unique()}
    if actual_ids != expected_ids:
        raise ValueError(
            f"{task}: raw trace covers {len(actual_ids)} episodes, but dataset "
            f"contains {len(expected_ids)}; missing raw trace cannot be repaired"
        )
    stable = int(
        stable_frames
        if stable_frames is not None
        else DEFAULT_STABLE_FRAMES.get(task, 3)
    )
    _, audit = BUILDERS[task](raw, stable_frames=stable)
    trainable = int(audit["trainable"].astype(bool).sum())
    successes = int(audit["is_success"].astype(bool).sum())
    if trainable < min_trainable:
        raise ValueError(
            f"{task}: rebuilt trainable={trainable}, below minimum {min_trainable}"
        )
    result = {
        "task": task,
        "dataset": str(dataset),
        "episodes": len(expected_ids),
        "successes": successes,
        "trainable": trainable,
        "stable_frames": stable,
        "dry_run": dry_run,
    }
    if dry_run:
        return result

    metadata_path = meta / f"semantic_trace_{task}_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    targets = [
        raw_path,
        meta / f"phase_progress_semantic_trace_{task}.parquet",
        meta / f"semantic_trace_{task}_audit.csv",
        metadata_path,
    ]
    for path in targets:
        if path.exists():
            _backup(path)
    WRITERS[task](
        dataset,
        raw.to_dict(orient="records"),
        metadata,
        output_name=f"semantic_trace_{task}",
        stable_frames=stable,
    )
    if fix_summary:
        summary_path = dataset / "collection_summary.json"
        if summary_path.exists():
            _backup(summary_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            info = json.loads((meta / "info.json").read_text(encoding="utf-8"))
            summary["num_episodes"] = int(info["total_episodes"])
            summary["total_frames"] = int(info["total_frames"])
            summary["successes"] = successes
            summary["success_rate"] = successes / len(expected_ids) if expected_ids else 0.0
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--stable-frames", type=int)
    parser.add_argument("--fix-summary", action="store_true")
    parser.add_argument("--min-trainable", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    results = [
        repair_dataset(
            task,
            dataset,
            stable_frames=args.stable_frames,
            fix_summary=args.fix_summary,
            min_trainable=args.min_trainable,
            dry_run=args.dry_run,
        )
        for task, dataset in (_parse_dataset(value) for value in args.dataset)
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
