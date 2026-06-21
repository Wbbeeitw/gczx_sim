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

"""Feature cache dataset used by Revalue two-stage training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

_REQUIRED_KEYS = {
    "features",
    "episode_index",
    "frame_index",
    "phase",
    "phase_progress",
    "global_progress",
}


class FeatureCache(Dataset):
    """In-memory dataset backed by one Revalue ``.pt`` cache file."""

    def __init__(self, cache_path: str | Path) -> None:
        self.cache_path = Path(cache_path)
        data = torch.load(self.cache_path, map_location="cpu", weights_only=False)
        missing = _REQUIRED_KEYS - set(data)
        if missing:
            raise ValueError(
                f"Feature cache {self.cache_path} missing keys: {sorted(missing)}"
            )

        self.features = data["features"].float()
        self.episode_index = data["episode_index"].long()
        self.frame_index = data["frame_index"].long()
        self.phase = data["phase"].long()
        self.phase_progress = data["phase_progress"].float()
        self.global_progress = data["global_progress"].float()

        self.raw_logits = (
            data["raw_logits"].float() if data.get("raw_logits") is not None else None
        )
        self.raw_value = (
            data["raw_value"].float() if data.get("raw_value") is not None else None
        )
        self.atoms = data.get("atoms")
        if self.atoms is not None:
            self.atoms = self.atoms.float()

        row_count = len(self.features)
        for key in _REQUIRED_KEYS - {"features"}:
            value = getattr(self, key)
            if len(value) != row_count:
                raise ValueError(
                    f"Feature cache {self.cache_path} key {key!r} has "
                    f"{len(value)} rows, expected {row_count}"
                )
        if self.raw_logits is not None and len(self.raw_logits) != row_count:
            raise ValueError(
                f"Feature cache {self.cache_path} raw_logits has "
                f"{len(self.raw_logits)} rows, expected {row_count}"
            )

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "features": self.features[index],
            "episode_index": self.episode_index[index],
            "frame_index": self.frame_index[index],
            "phase": self.phase[index],
            "phase_progress": self.phase_progress[index],
            "global_progress": self.global_progress[index],
        }
        if self.raw_logits is not None:
            item["raw_logits"] = self.raw_logits[index]
        if self.raw_value is not None:
            item["raw_value"] = self.raw_value[index]
        return item

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[-1])

    @property
    def has_raw_logits(self) -> bool:
        return self.raw_logits is not None


def collate_feature_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate feature-cache rows into tensors."""
    output: dict[str, Any] = {}
    for key in batch[0]:
        values = [row[key] for row in batch]
        if isinstance(values[0], torch.Tensor):
            output[key] = torch.stack(values, dim=0)
        else:
            output[key] = values
    return output


def build_feature_loaders(
    features_dir: str | Path,
    *,
    batch_size: int = 256,
    num_workers: int = 0,
    train_shuffle: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val loaders from ``train.pt`` and ``val.pt``."""
    features_dir = Path(features_dir)
    train_cache = FeatureCache(features_dir / "train.pt")
    val_cache = FeatureCache(features_dir / "val.pt")

    train_loader = DataLoader(
        train_cache,
        batch_size=batch_size,
        shuffle=train_shuffle,
        num_workers=num_workers,
        collate_fn=collate_feature_batch,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_cache,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_feature_batch,
        pin_memory=True,
    )
    return train_loader, val_loader
