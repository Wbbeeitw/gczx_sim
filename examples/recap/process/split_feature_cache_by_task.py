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
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    out_dir = Path(args.out_dir)
    for spec in args.ranges:
        name, start, end = _parse_range(spec)
        for split in ("train", "val"):
            source = features_dir / f"{split}.pt"
            if not source.exists():
                continue
            cache = _load(source)
            episode_index = cache["episode_index"].long()
            mask = (episode_index >= start) & (episode_index < end)
            target_dir = out_dir / name
            target_dir.mkdir(parents=True, exist_ok=True)
            torch.save(_subset(cache, mask), target_dir / f"{split}.pt")
            print(f"{name}/{split}.pt: {int(mask.sum())} rows")


if __name__ == "__main__":
    main()
