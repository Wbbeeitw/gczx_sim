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

"""Split a merged feature cache into per-task subsets by episode ranges."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _parse_range(spec: str) -> tuple[str, int, int]:
    name, rng = spec.split("=")
    start, end = rng.split("-")
    return name, int(start), int(end)


def _subset(cache: dict, mask: torch.Tensor) -> dict:
    out = {}
    rows = int(mask.shape[0])
    for key, value in cache.items():
        if torch.is_tensor(value) and value.shape[0] == rows:
            out[key] = value[mask]
        else:
            out[key] = value
    return out


def _load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - older caches may contain metadata
        return torch.load(path, map_location="cpu", weights_only=False)


def _concat_caches(caches: list[dict]) -> dict:
    if not caches:
        raise ValueError("At least one feature cache is required")
    row_counts = [int(cache["episode_index"].shape[0]) for cache in caches]
    keys = set(caches[0])
    if any(set(cache) != keys for cache in caches[1:]):
        raise ValueError("Feature caches must contain identical keys")
    combined = {}
    for key in keys:
        values = [cache[key] for cache in caches]
        if all(
            torch.is_tensor(value)
            and value.ndim > 0
            and value.shape[0] == row_count
            for value, row_count in zip(values, row_counts, strict=True)
        ):
            combined[key] = torch.cat(values, dim=0)
        else:
            first = values[0]
            if torch.is_tensor(first):
                if any(
                    not torch.is_tensor(value) or not torch.equal(first, value)
                    for value in values[1:]
                ):
                    raise ValueError(f"Feature cache metadata differs for key={key!r}")
            elif any(value != first for value in values[1:]):
                raise ValueError(f"Feature cache metadata differs for key={key!r}")
            combined[key] = first
    return combined


def _task_seed(seed: int, name: str) -> int:
    return seed + sum((index + 1) * ord(char) for index, char in enumerate(name))


def split_feature_caches_by_task(
    features_dir: Path,
    ranges: list[str],
    out_dir: Path,
    *,
    val_episode_ratio: float = 0.2,
    seed: int = 42,
    allow_single_episode_overlap: bool = False,
) -> dict[str, dict[str, int | bool]]:
    """Combine global caches and create an independent split for each task."""
    if not 0.0 <= val_episode_ratio < 1.0:
        raise ValueError("val_episode_ratio must be in [0, 1)")
    source_paths = [
        features_dir / f"{split}.pt"
        for split in ("train", "val")
        if (features_dir / f"{split}.pt").exists()
    ]
    if not source_paths:
        raise FileNotFoundError(f"No train.pt or val.pt found in {features_dir}")
    combined = _concat_caches([_load(path) for path in source_paths])
    episode_index = combined["episode_index"].long()
    reports: dict[str, dict[str, int | bool]] = {}

    for spec in ranges:
        name, start, end = _parse_range(spec)
        task_mask = (episode_index >= start) & (episode_index < end)
        task_episodes = torch.unique(episode_index[task_mask], sorted=True)
        if len(task_episodes) == 0:
            raise ValueError(f"{name}: no feature rows in episode range {start}-{end}")

        overlap = False
        if len(task_episodes) == 1:
            if not allow_single_episode_overlap:
                raise ValueError(
                    f"{name}: one episode cannot form disjoint train/val splits; "
                    "collect at least two episodes or pass "
                    "--allow_single_episode_overlap for a linkage-only smoke test"
                )
            train_episodes = task_episodes
            val_episodes = task_episodes
            overlap = True
        else:
            generator = torch.Generator().manual_seed(_task_seed(seed, name))
            shuffled = task_episodes[
                torch.randperm(len(task_episodes), generator=generator)
            ]
            num_val = max(1, int(round(len(task_episodes) * val_episode_ratio)))
            num_val = min(num_val, len(task_episodes) - 1)
            val_episodes = shuffled[:num_val]
            train_episodes = shuffled[num_val:]

        target_dir = out_dir / name
        target_dir.mkdir(parents=True, exist_ok=True)
        split_rows: dict[str, int] = {}
        for split, selected_episodes in (
            ("train", train_episodes),
            ("val", val_episodes),
        ):
            mask = task_mask & torch.isin(episode_index, selected_episodes)
            torch.save(_subset(combined, mask), target_dir / f"{split}.pt")
            split_rows[split] = int(mask.sum())
            print(
                f"{name}/{split}.pt: rows={split_rows[split]} "
                f"episodes={len(selected_episodes)}"
            )
        reports[name] = {
            "episodes": int(len(task_episodes)),
            "train_rows": split_rows["train"],
            "val_rows": split_rows["val"],
            "single_episode_overlap": overlap,
        }
    return reports


def main() -> None:
    """Split train/val feature caches by explicit episode ranges."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features_dir", required=True)
    parser.add_argument(
        "--ranges",
        nargs="+",
        required=True,
        help="task_name=start-end entries; end is exclusive",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--val_episode_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_single_episode_overlap", action="store_true")
    args = parser.parse_args()

    split_feature_caches_by_task(
        Path(args.features_dir),
        args.ranges,
        Path(args.out_dir),
        val_episode_ratio=args.val_episode_ratio,
        seed=args.seed,
        allow_single_episode_overlap=args.allow_single_episode_overlap,
    )


if __name__ == "__main__":
    main()
