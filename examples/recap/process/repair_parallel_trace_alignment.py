#!/usr/bin/env python
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Repair phase-label episode alignment for parallel-collected datasets.

With multi-env collection, episodes complete out of order: the dataset
numbers episodes in *completion* order, but trace/phase artifacts were keyed
by *attempt* index, so phase rows get attached to the wrong episodes
(permuted). Downstream stages then fail with "Missing phase label" or,
worse, silently train on mismatched labels.

Since attempt indices carry no episode content, the only reliable
correspondence is per-episode length: each written episode's length
(episodes.jsonl) equals the row count of its trace group. This script
rebuilds the mapping by matching written episodes to trace groups by
length, preferring the smallest unused attempt id on ties (harmless for
equal-length episodes, whose labels are interchangeable at task level).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _backup(path: Path, dry_run: bool) -> None:
    backup = path.with_name(path.name + ".bak")
    if backup.exists() or dry_run:
        return
    shutil.copy2(path, backup)


def _build_mapping(
    episodes: list[dict], trace_counts: dict[int, int], name: str
) -> dict[int, int]:
    """Map each trace id to its written episode index by length matching."""
    remaining = dict(trace_counts)  # trace_id -> row_count
    mapping: dict[int, int] = {}
    for record in episodes:
        written_index = int(record["episode_index"])
        length = int(record["length"])
        candidates = [
            trace_id
            for trace_id, count in remaining.items()
            if count == length
        ]
        if not candidates:
            closest = min(
                remaining,
                key=lambda trace_id: abs(remaining[trace_id] - length),
                default=None,
            )
            if closest is None:
                raise ValueError(
                    f"{name}: no trace group left for episode {written_index} "
                    f"(length {length})"
                )
            print(
                f"  WARN {name}: episode {written_index} length {length} has no "
                f"exact trace group; using closest id {closest} "
                f"(rows {remaining[closest]})"
            )
            candidates = [closest]
        trace_id = min(candidates)
        mapping[trace_id] = written_index
        del remaining[trace_id]
    if remaining:
        raise ValueError(
            f"{name}: {len(remaining)} trace groups left unassigned: "
            f"{sorted(remaining)[:10]}"
        )
    return mapping


def _remap_frame(path: Path, mapping: dict[int, int], dry_run: bool) -> str:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    if "episode_index" not in frame.columns:
        return f"  {path.name}: no episode_index column, skipped"
    ids = {int(value) for value in frame["episode_index"].unique()}
    unknown = sorted(ids.difference(mapping))
    if unknown:
        raise ValueError(f"{path.name}: episode ids {unknown} not in mapping")
    if ids and all(mapping[identifier] == identifier for identifier in ids):
        return f"  {path.name}: already aligned, skipped"
    frame["episode_index"] = frame["episode_index"].map(mapping).astype("int64")
    if dry_run:
        return f"  would remap {path.name}: rows={len(frame)}"
    _backup(path, dry_run)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    return f"  remapped {path.name}: rows={len(frame)}"


def repair(dataset: Path, dry_run: bool) -> None:
    meta = dataset / "meta"
    episodes = _read_jsonl(meta / "episodes.jsonl")
    phase_files = sorted(meta.glob("phase_progress_semantic_trace_*.parquet"))
    if not phase_files:
        phase_files = sorted(meta.glob("phase_progress_multitask.parquet"))
    if len(phase_files) != 1:
        raise ValueError(
            f"{dataset}: expected exactly one phase file, found "
            f"{[path.name for path in phase_files]}"
        )
    phase_path = phase_files[0]
    task = phase_path.name.removeprefix("phase_progress_semantic_trace_").removesuffix(
        ".parquet"
    )
    phase = pd.read_parquet(phase_path, columns=["episode_index"])
    trace_counts = {
        int(trace_id): int(count)
        for trace_id, count in phase["episode_index"].value_counts().items()
    }
    mapping = _build_mapping(episodes, trace_counts, dataset.name)
    n_moved = sum(1 for key, value in mapping.items() if key != value)
    print(
        f"{dataset.name}: {len(episodes)} episodes, "
        f"{n_moved}/{len(mapping)} trace ids renumbered ({task})"
    )
    targets = [phase_path]
    raw_trace = meta / f"semantic_trace_{task}.parquet"
    audit = meta / f"semantic_trace_{task}_audit.csv"
    for path in (raw_trace, audit):
        if path.exists():
            targets.append(path)
    for path in targets:
        print(_remap_frame(path, mapping, dry_run))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", help="dataset dirs to repair")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for raw in args.datasets:
        repair(Path(raw), args.dry_run)
    print("done.")


if __name__ == "__main__":
    main()
