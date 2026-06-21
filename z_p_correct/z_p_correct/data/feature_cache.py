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

"""Feature cache utilities for pre-extracted frozen VLM features."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class FeatureCache(Dataset):
    """In-memory dataset backed by a ``.pt`` cache of features and labels.

    Supports optional temporal windowing for heads that consume feature windows.
    """

    def __init__(self, cache_path: str, window_size: int = 1):
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        self.features = data["features"].float()
        self.episode_index = data["episode_index"].long()
        self.frame_index = data["frame_index"].long()
        self.phase = data["phase"].long()
        self.phase_progress = data["phase_progress"].float()
        self.global_progress = data["global_progress"].float()

        # Optional: raw logits/values precomputed by the value critic.
        self.raw_logits = data.get("raw_logits")
        self.atoms = data.get("atoms")

        self.window_size = window_size
        self.half_window = window_size // 2
        self._build_index()

    def _build_index(self) -> None:
        """Build per-episode index for temporal windowing."""
        self._episodes: dict[int, list[int]] = {}
        for idx, ep in enumerate(self.episode_index.tolist()):
            self._episodes.setdefault(ep, []).append(idx)
        # Sort each episode's indices by frame_index.
        for ep in self._episodes:
            self._episodes[ep] = sorted(
                self._episodes[ep],
                key=lambda i: int(self.frame_index[i].item()),
            )
        self._index_to_ep_pos = []
        for ep in sorted(self._episodes.keys()):
            for pos, idx in enumerate(self._episodes[ep]):
                self._index_to_ep_pos.append((ep, pos, idx))

    def __len__(self) -> int:
        return len(self.features)

    def _get_window(self, global_idx: int) -> torch.Tensor:
        """Build a feature window centered at global_idx with edge replication."""
        ep, pos, _center_idx = self._index_to_ep_pos[global_idx]
        episode_indices = self._episodes[ep]
        length = len(episode_indices)
        window_indices = []
        for offset in range(-self.half_window, self.half_window + 1):
            neighbor_pos = max(0, min(length - 1, pos + offset))
            window_indices.append(episode_indices[neighbor_pos])
        return self.features[window_indices]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item: dict[str, Any] = {
            "episode_index": self.episode_index[idx],
            "frame_index": self.frame_index[idx],
            "phase": self.phase[idx],
            "phase_progress": self.phase_progress[idx],
            "global_progress": self.global_progress[idx],
        }
        if self.window_size > 1:
            item["feature_window"] = self._get_window(idx)
        else:
            item["features"] = self.features[idx]
        if self.raw_logits is not None:
            item["raw_logits"] = self.raw_logits[idx]
        return item

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[-1])

    @property
    def num_samples(self) -> int:
        return len(self)


def _collate_features(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate a list of cached feature samples."""
    out: dict[str, Any] = {}
    keys = batch[0].keys()
    for key in keys:
        values = [b[key] for b in batch]
        if isinstance(values[0], torch.Tensor):
            out[key] = torch.stack(values, dim=0)
        else:
            out[key] = values
    return out


def build_feature_loaders(
    features_dir: str,
    window_size: int = 1,
    batch_size: int = 256,
    num_workers: int = 0,
    train_shuffle: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders from a feature cache directory.

    The directory must contain ``train.pt`` and ``val.pt`` produced by the feature
    extraction script.
    """
    features_dir = Path(features_dir)
    train_cache = FeatureCache(str(features_dir / "train.pt"), window_size=window_size)
    val_cache = FeatureCache(str(features_dir / "val.pt"), window_size=window_size)

    train_loader = DataLoader(
        train_cache,
        batch_size=batch_size,
        shuffle=train_shuffle,
        num_workers=num_workers,
        collate_fn=_collate_features,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_cache,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_features,
        pin_memory=True,
    )

    logger.info(
        "Feature loaders: train=%d val=%d feature_dim=%d window_size=%d",
        len(train_cache),
        len(val_cache),
        train_cache.feature_dim,
        window_size,
    )
    return train_loader, val_loader
