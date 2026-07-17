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

"""Download demos of selected LIBERO tasks and prune to a subset dataset.

Pulls only the episodes whose task description matches the given keywords
out of ``physical-intelligence/libero`` (or another LeRobot dataset repo),
then rewrites the metadata so the result is a self-contained small dataset
usable for targeted BC fine-tuning of weak tasks.

Example:
    python examples/recap/process/download_libero_task_demos.py \
        --out /data/gczx_sim/datasets/libero_task58_demos \
        --keywords "back compartment of the caddy" "both moka pots"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="physical-intelligence/libero")
    parser.add_argument("--out", required=True)
    parser.add_argument("--keywords", nargs="+", required=True)
    parser.add_argument("--limit-per-task", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    endpoint = os.environ.get("HF_ENDPOINT") or None
    api = HfApi(endpoint=endpoint)
    out = Path(args.out)
    meta_dir = out / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        allow_patterns=["meta/*"],
        local_dir=out,
        endpoint=endpoint,
    )

    tasks = _read_jsonl(meta_dir / "tasks.jsonl")
    episodes = _read_jsonl(meta_dir / "episodes.jsonl")
    keywords = [keyword.lower() for keyword in args.keywords]
    selected: list[dict] = []
    per_task_count: dict[str, int] = {}
    for episode in episodes:
        task_text = " ".join(episode.get("tasks", [])).lower()
        if not any(keyword in task_text for keyword in keywords):
            continue
        task_key = task_text
        if args.limit_per_task and per_task_count.get(task_key, 0) >= args.limit_per_task:
            continue
        per_task_count[task_key] = per_task_count.get(task_key, 0) + 1
        selected.append(episode)
    if not selected:
        raise SystemExit(f"no episodes matched keywords: {args.keywords}")

    selected_indices = sorted(int(ep["episode_index"]) for ep in selected)
    print(f"matched {len(selected)} episodes for {len(per_task_count)} task(s):")
    for task_text, count in sorted(per_task_count.items()):
        print(f"  {count:4d}  {task_text}")

    repo_files = api.list_repo_files(args.repo, repo_type="dataset")
    wanted = []
    for relpath in repo_files:
        if not relpath.startswith(("data/", "videos/")):
            continue
        stem = Path(relpath).stem
        if not stem.startswith("episode_"):
            continue
        try:
            index = int(stem.removeprefix("episode_"))
        except ValueError:
            continue
        if index in selected_indices:
            wanted.append(relpath)
    print(f"downloading {len(wanted)} episode files ...")
    for relpath in wanted:
        target = out / relpath
        if target.exists():
            continue
        if args.dry_run:
            print("  [dry-run]", relpath)
            continue
        hf_hub_download(
            repo_id=args.repo,
            repo_type="dataset",
            filename=relpath,
            local_dir=out,
            endpoint=endpoint,
        )

    if args.dry_run:
        print("dry-run finished; metadata left untouched")
        return

    index_set = set(selected_indices)
    pruned_episodes = [
        episode
        for episode in episodes
        if int(episode["episode_index"]) in index_set
    ]
    _write_jsonl(meta_dir / "episodes.jsonl", pruned_episodes)

    stats_path = meta_dir / "episodes_stats.jsonl"
    if stats_path.exists():
        pruned_stats = [
            record
            for record in _read_jsonl(stats_path)
            if int(record["episode_index"]) in index_set
        ]
        _write_jsonl(stats_path, pruned_stats)

    info_path = meta_dir / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    total_frames = sum(int(ep["length"]) for ep in pruned_episodes)
    info["total_episodes"] = len(pruned_episodes)
    info["total_frames"] = total_frames
    info["total_tasks"] = len(tasks)
    info["total_videos"] = sum(1 for path in wanted if path.startswith("videos/"))
    info["splits"] = {"train": f"0:{len(pruned_episodes)}"}
    info_path.write_text(
        json.dumps(info, ensure_ascii=False, indent=4) + "\n", encoding="utf-8"
    )

    summary = {
        "repo": args.repo,
        "keywords": args.keywords,
        "episodes": len(pruned_episodes),
        "frames": total_frames,
        "episode_indices": selected_indices,
        "tasks": sorted(per_task_count),
    }
    (out / "download_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
