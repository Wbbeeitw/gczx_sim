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

"""Resplit existing Revalue feature caches by episode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from rlinf.revalue.data.episode_manifest import split_episode_ids
from rlinf.revalue.io import save_json


@dataclass(frozen=True)
class FeatureResplitConfig:
    """Options for resplitting cached features without re-extraction."""

    source_dir: str
    output_dir: str
    val_episode_ratio: float = 0.2
    test_episode_ratio: float = 0.0
    seed: int = 42
    manifest_path: str | None = None
    overwrite: bool = False


def _load_existing_caches(source_dir: Path) -> list[dict[str, Any]]:
    caches = []
    for name in ("train.pt", "val.pt", "test.pt"):
        path = source_dir / name
        if path.exists():
            caches.append(torch.load(path, map_location="cpu", weights_only=False))
    if not caches:
        raise FileNotFoundError(f"No train.pt/val.pt/test.pt found in {source_dir}")
    return caches


def _concat_caches(caches: list[dict[str, Any]]) -> dict[str, Any]:
    keys = set(caches[0])
    for cache in caches[1:]:
        keys &= set(cache)

    out: dict[str, Any] = {}
    row_counts = [int(cache["episode_index"].shape[0]) for cache in caches]
    for key in sorted(keys):
        values = [cache[key] for cache in caches]
        if not all(torch.is_tensor(value) for value in values):
            if all(value == values[0] for value in values):
                out[key] = values[0]
            continue
        if values[0].ndim == 0:
            if all(torch.equal(value, values[0]) for value in values):
                out[key] = values[0]
            continue
        if all(int(value.shape[0]) == row_count for value, row_count in zip(values, row_counts)):
            out[key] = torch.cat(values, dim=0)
        elif all(torch.equal(value, values[0]) for value in values):
            out[key] = values[0]

    if "episode_index" not in out:
        raise ValueError("Feature caches must contain episode_index")

    order = torch.argsort(out["episode_index"].long() * 1_000_000 + out["frame_index"].long())
    for key, value in list(out.items()):
        if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == len(order):
            out[key] = value.index_select(0, order)
    return out


def _subset_cache(cache: dict[str, Any], episodes: list[int]) -> dict[str, Any]:
    episode_tensor = cache["episode_index"].long()
    keep = torch.zeros_like(episode_tensor, dtype=torch.bool)
    for episode in episodes:
        keep |= episode_tensor == int(episode)
    indices = keep.nonzero(as_tuple=False).squeeze(-1)

    out: dict[str, Any] = {}
    total_rows = int(episode_tensor.shape[0])
    for key, value in cache.items():
        if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == total_rows:
            out[key] = value.index_select(0, indices)
        else:
            out[key] = value
    return out


def resplit_feature_cache(cfg: FeatureResplitConfig) -> dict[str, Any]:
    """Resplit existing feature cache files by episode and save new pt files."""
    source_dir = Path(cfg.source_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_outputs = [output_dir / "train.pt", output_dir / "val.pt"]
    if cfg.test_episode_ratio > 0.0:
        required_outputs.append(output_dir / "test.pt")
    if not cfg.overwrite and all(path.exists() for path in required_outputs):
        return {
            "source_dir": str(source_dir),
            "output_dir": str(output_dir),
            "skipped_existing": True,
        }

    merged = _concat_caches(_load_existing_caches(source_dir))
    episodes = sorted(set(int(ep) for ep in merged["episode_index"].tolist()))
    train_eps, val_eps, test_eps = split_episode_ids(
        episodes,
        val_episode_ratio=cfg.val_episode_ratio,
        test_episode_ratio=cfg.test_episode_ratio,
        seed=cfg.seed,
    )

    split_to_eps = {
        "train": train_eps,
        "val": val_eps,
        "test": test_eps,
    }
    split_rows = {}
    for split, eps in split_to_eps.items():
        if split == "test" and not eps:
            continue
        split_cache = _subset_cache(merged, eps)
        torch.save(split_cache, output_dir / f"{split}.pt")
        split_rows[split] = int(split_cache["episode_index"].shape[0])

    manifest = {
        "selected_episodes": episodes,
        "train_episodes": train_eps,
        "val_episodes": val_eps,
        "test_episodes": test_eps,
        "num_frames": int(merged["episode_index"].shape[0]),
        "seed": int(cfg.seed),
        "val_episode_ratio": float(cfg.val_episode_ratio),
        "test_episode_ratio": float(cfg.test_episode_ratio),
        "source_features_dir": str(source_dir),
        "output_features_dir": str(output_dir),
        "split_rows": split_rows,
    }
    manifest_path = Path(cfg.manifest_path) if cfg.manifest_path else output_dir / "feature_split_manifest.json"
    save_json(manifest, manifest_path)
    return manifest
