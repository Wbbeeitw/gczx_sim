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
import time


def _download_with_retry(
    repo: str,
    relpath: str,
    out: Path,
    endpoint: str | None,
    position: int,
    total: int,
    attempts: int = 8,
) -> None:
    """Download one repo file, backing off politely on HTTP 429."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

    delay = 5.0
    for attempt in range(1, attempts + 1):
        try:
            hf_hub_download(
                repo_id=repo,
                repo_type="dataset",
                filename=relpath,
                local_dir=out,
                endpoint=endpoint,
            )
            print(f"  [{position}/{total}] {relpath}")
            return
        except (HfHubHTTPError, LocalEntryNotFoundError) as exc:
            is_rate_limit = "429" in str(exc)
            if not is_rate_limit or attempt == attempts:
                raise
            print(
                f"  [{position}/{total}] rate limited, "
                f"retry {attempt}/{attempts} in {delay:.0f}s"
            )
            time.sleep(delay)
            delay = min(delay * 2.0, 120.0)


def _reindex_episodes(
    out: Path,
    selected_indices: list[int],
    pruned_episodes: list[dict],
    pruned_stats: list[dict],
) -> None:
    """Renumber downloaded episodes to a contiguous 0..N-1 range.

    Some LeRobot loaders expect contiguous indices; two-phase renames avoid
    collisions between source and target filenames.
    """
    mapping = {old: new for new, old in enumerate(selected_indices)}
    for directory, pattern in (
        (out / "data", "**/episode_0*.parquet"),
        (out / "videos", "**/episode_0*.mp4"),
    ):
        if not directory.exists():
            continue
        for stale in directory.rglob("episode_tmp_*"):
            stale.unlink()
    for directory, pattern in (
        (out / "data", "**/episode_0*.parquet"),
        (out / "videos", "**/episode_0*.mp4"),
    ):
        if not directory.exists():
            continue
        # Materialize before renaming: rglob is lazy and would otherwise
        # re-match the episode_tmp_* files produced by this loop.
        for file in list(directory.rglob(pattern)):
            old_index = int(file.stem.removeprefix("episode_"))
            if old_index not in mapping:
                continue
            target = file.with_name(f"episode_tmp_{mapping[old_index]:06d}{file.suffix}")
            if target.exists():
                target.unlink()
            file.rename(target)
        for file in list(directory.rglob("episode_tmp_*")):
            new_index = int(file.stem.removeprefix("episode_tmp_"))
            target = file.with_name(f"episode_{new_index:06d}{file.suffix}")
            if target.exists():
                target.unlink()
            file.rename(target)
    for episode in pruned_episodes:
        episode["episode_index"] = mapping[int(episode["episode_index"])]
    for record in pruned_stats:
        record["episode_index"] = mapping[int(record["episode_index"])]


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
    parser.add_argument(
        "--keep-indices",
        action="store_true",
        help="keep original episode indices instead of renumbering from 0",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=2.0,
        help="seconds to wait between episode downloads (rate-limit politeness)",
    )
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
    for position, relpath in enumerate(wanted, 1):
        target = out / relpath
        if target.exists():
            continue
        if args.dry_run:
            print("  [dry-run]", relpath)
            continue
        _download_with_retry(args.repo, relpath, out, endpoint, position, len(wanted))
        time.sleep(args.pause)

    if args.dry_run:
        print("dry-run finished; metadata left untouched")
        return

    index_set = set(selected_indices)
    pruned_episodes = [
        episode
        for episode in episodes
        if int(episode["episode_index"]) in index_set
    ]

    stats_path = meta_dir / "episodes_stats.jsonl"
    pruned_stats = []
    if stats_path.exists():
        pruned_stats = [
            record
            for record in _read_jsonl(stats_path)
            if int(record["episode_index"]) in index_set
        ]

    if not args.keep_indices:
        _reindex_episodes(out, selected_indices, pruned_episodes, pruned_stats)
    _write_jsonl(meta_dir / "episodes.jsonl", pruned_episodes)
    if stats_path.exists():
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
