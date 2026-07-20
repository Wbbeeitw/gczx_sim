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

"""Repair episode_index alignment in success-only rollout datasets.

Success-only collection writes only successful episodes, so the dataset
numbers them sequentially (0..N-1). Trace artifacts captured during
collection, however, were keyed by the *attempt* index (failures discarded
in between), so `phase_progress_*.parquet`, `semantic_trace_*.parquet` and
`*_audit.csv` no longer line up with `meta/episodes.jsonl` and downstream
pool building fails with "phase labels missing episodes".

Since successful episodes were written in attempt order, the sorted unique
attempt indices map 1:1 onto 0..N-1. This script rewrites the trace files
accordingly and backs up originals as ``<name>.bak``. Idempotent.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def _backup(path: Path, dry_run: bool) -> None:
    backup = path.with_name(path.name + ".bak")
    if backup.exists() or dry_run:
        return
    shutil.copy2(path, backup)


def _episode_count(dataset: Path) -> int:
    episodes_path = dataset / "meta" / "episodes.jsonl"
    with episodes_path.open(encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def _remap_file(path: Path, mapping: dict[int, int], dry_run: bool) -> str:
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
    phase_files = sorted(meta.glob("phase_progress_semantic_trace_*.parquet"))
    if len(phase_files) != 1:
        raise ValueError(
            f"{dataset}: expected exactly one phase_progress file, found "
            f"{[path.name for path in phase_files]}"
        )
    phase_path = phase_files[0]
    task = phase_path.name.removeprefix("phase_progress_semantic_trace_").removesuffix(
        ".parquet"
    )
    episode_count = _episode_count(dataset)
    phase = pd.read_parquet(phase_path, columns=["episode_index"])
    source_ids = sorted(int(value) for value in phase["episode_index"].unique())
    if len(source_ids) != episode_count:
        raise ValueError(
            f"{dataset.name}: {len(source_ids)} unique trace ids for "
            f"{episode_count} episodes; cannot build an ordered mapping"
        )
    mapping = {
        source_id: index for index, source_id in enumerate(source_ids)
    }
    print(
        f"{dataset.name}: {episode_count} episodes, "
        f"remapping {len(mapping)} trace ids ({task})"
    )

    targets = [
        phase_path,
        meta / f"semantic_trace_{task}.parquet",
        meta / f"semantic_trace_{task}_audit.csv",
    ]
    for path in targets:
        if path.exists():
            print(_remap_file(path, mapping, dry_run))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", help="success-only dataset dirs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for raw in args.datasets:
        repair(Path(raw), args.dry_run)
    print("done.")


if __name__ == "__main__":
    main()
