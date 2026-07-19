#!/usr/bin/env python
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Repair episode_index numbering inside the task58 replayed merged dataset.

The replay pipeline merged task5 (41 episodes) and task8 (29 episodes) into
one dataset but left its meta files with three different numbering schemes:

- ``meta/episodes.jsonl``: correct global numbering (0..69).
- ``meta/episodes_stats.jsonl``: positionally aligned with episodes.jsonl but
  each record's own ``episode_index`` field keeps stale per-segment numbering
  (0..40 for task5, 0..28 for task8).
- ``phase_progress_*`` / ``semantic_trace_*`` files: rows are keyed by the
  *source* demo ids (e.g. 3..68 for task5, 41..110 for task8), not the local
  episode indices.

This script rewrites every meta file so all of them use the episodes.jsonl
numbering (task5 -> 0..40, task8 -> 41..69). Originals are backed up as
``<name>.bak`` on first run. The script is idempotent.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

TASK_SEGMENTS = {
    "task5": (0, 41),  # episodes.jsonl local indices 0..40
    "task8": (41, 70),  # episodes.jsonl local indices 41..69
}


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _backup(path: Path, dry_run: bool) -> None:
    backup = path.with_name(path.name + ".bak")
    if backup.exists():
        return
    if dry_run:
        print(f"  would back up {path.name} -> {backup.name}")
        return
    shutil.copy2(path, backup)


def _build_id_mapping(
    phase_path: Path, segment: tuple[int, int], task: str
) -> dict[int, int]:
    """Map sorted source demo ids to consecutive local episode indices."""
    start, end = segment
    expected = end - start
    frame = pd.read_parquet(phase_path, columns=["episode_index"])
    source_ids = sorted(int(value) for value in frame["episode_index"].unique())
    if len(source_ids) != expected:
        raise ValueError(
            f"{phase_path.name}: {len(source_ids)} unique episode ids, "
            f"expected {expected} for {task}; cannot build an ordered mapping"
        )
    return {source_id: start + offset for offset, source_id in enumerate(source_ids)}


def _remap_frame(path: Path, mapping: dict[int, int], dry_run: bool) -> str:
    """Remap the episode_index column of a parquet/csv file in place."""
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    if "episode_index" not in frame.columns:
        return f"  {path.name}: no episode_index column, skipped"
    ids = {int(value) for value in frame["episode_index"].unique()}
    unknown = sorted(ids.difference(mapping))
    if unknown:
        raise ValueError(
            f"{path.name}: episode ids {unknown} not covered by the mapping"
        )
    if ids and all(mapping[identifier] == identifier for identifier in ids):
        return f"  {path.name}: already aligned, skipped"
    frame["episode_index"] = frame["episode_index"].map(mapping).astype("int64")
    if dry_run:
        return f"  would remap {path.name}: {sorted(ids)} rows={len(frame)}"
    _backup(path, dry_run)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    return f"  remapped {path.name}: rows={len(frame)}"


def repair(dataset: Path, dry_run: bool) -> None:
    episodes_path = dataset / "meta" / "episodes.jsonl"
    episodes = _read_jsonl(episodes_path)
    if len(episodes) != 70:
        raise ValueError(
            f"{episodes_path}: expected 70 episodes (41 task5 + 29 task8), "
            f"got {len(episodes)}"
        )

    # 1. episodes_stats.jsonl: rewrite the stale field with the positional index.
    stats_path = dataset / "meta" / "episodes_stats.jsonl"
    stats = _read_jsonl(stats_path)
    if len(stats) != len(episodes):
        raise ValueError(
            f"{stats_path.name}: {len(stats)} records for {len(episodes)} episodes"
        )
    if all(int(record["episode_index"]) == i for i, record in enumerate(stats)):
        print("  episodes_stats.jsonl: already aligned, skipped")
    elif dry_run:
        print("  would rewrite episodes_stats.jsonl episode_index -> 0..69")
    else:
        _backup(stats_path, dry_run)
        for position, record in enumerate(stats):
            record["episode_index"] = position
        _write_jsonl(stats_path, stats)
        print("  rewrote episodes_stats.jsonl episode_index -> 0..69")

    # 2. Build per-task source-id -> local-index mappings from the phase files.
    mappings: dict[str, dict[int, int]] = {}
    for task, segment in TASK_SEGMENTS.items():
        phase_path = dataset / "meta" / f"phase_progress_semantic_trace_{task}.parquet"
        mappings[task] = _build_id_mapping(phase_path, segment, task)
        print(
            f"  {task}: {len(mappings[task])} source ids -> "
            f"local {segment[0]}..{segment[1] - 1}"
        )

    # 3. Remap every file keyed by source demo ids.
    targets = []
    for task in TASK_SEGMENTS:
        targets += [
            (dataset / "meta" / f"phase_progress_semantic_trace_{task}.parquet", task),
            (dataset / "meta" / f"semantic_trace_{task}.parquet", task),
            (dataset / "meta" / f"semantic_trace_{task}_audit.csv", task),
        ]
    for path, task in targets:
        if path.exists():
            print(_remap_frame(path, mappings[task], dry_run))

    # 4. Rebuild phase_progress_multitask.parquet from the repaired task files.
    multitask_path = dataset / "meta" / "phase_progress_multitask.parquet"
    combined = pd.concat(
        [
            pd.read_parquet(dataset / "meta" / f"phase_progress_semantic_trace_{task}.parquet")
            for task in TASK_SEGMENTS
        ],
        ignore_index=True,
    )
    sort_columns = [
        column for column in ("episode_index", "frame_index") if column in combined.columns
    ]
    if sort_columns:
        combined = combined.sort_values(sort_columns)
    if dry_run:
        print(f"  would rebuild {multitask_path.name}: rows={len(combined)}")
    else:
        _backup(multitask_path, dry_run)
        combined.to_parquet(multitask_path, index=False)
        print(f"  rebuilt {multitask_path.name}: rows={len(combined)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        default="/data/libero_long/libero_task58_replayed_merged",
        help="Path to the task58 replayed merged dataset",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repair(Path(args.dataset), args.dry_run)
    print("done.")


if __name__ == "__main__":
    main()
