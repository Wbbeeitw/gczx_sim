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
HEAD_TYPE_SHARED_MLP = "shared_mlp"
HEAD_TYPE_TEMPORAL_Z_MLP_P = "temporal_z_mlp_p"


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


class TemporalWindowDataset(Dataset):
    """Episode-aware temporal window wrapper over frame-level cache rows."""

    def __init__(self, base_dataset: Dataset, window_size: int) -> None:
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError(
                f"window_size must be a positive odd integer, got {window_size}"
            )

        self.base_dataset = base_dataset
        self.window_size = int(window_size)
        self.radius = self.window_size // 2
        self.cache = (
            base_dataset.feature_cache
            if hasattr(base_dataset, "feature_cache")
            else base_dataset
        )
        required = (
            "features",
            "episode_index",
            "frame_index",
            "phase",
            "phase_progress",
            "global_progress",
        )
        missing = [name for name in required if not hasattr(self.cache, name)]
        if missing:
            raise ValueError(
                "TemporalWindowDataset requires cache-style tensors for "
                f"{sorted(missing)}"
            )

        self.features = self.cache.features.float()
        self.episode_index = self.cache.episode_index.long()
        self.frame_index = self.cache.frame_index.long()
        self.phase = self.cache.phase.long()
        self.phase_progress = self.cache.phase_progress.float()
        self.global_progress = self.cache.global_progress.float()
        self._episode_to_rows: dict[int, torch.Tensor] = {}
        for episode in torch.unique(self.episode_index).tolist():
            row_idx = torch.nonzero(
                self.episode_index == int(episode),
                as_tuple=False,
            ).squeeze(1)
            order = torch.argsort(self.frame_index.index_select(0, row_idx))
            self._episode_to_rows[int(episode)] = row_idx.index_select(0, order)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        center_item = self.base_dataset[index]
        episode = int(self.episode_index[index].item())
        rows = self._episode_to_rows[episode]
        pos = int(torch.nonzero(rows == index, as_tuple=False).item())

        window_tokens: list[torch.Tensor] = []
        phase_tokens: list[torch.Tensor] = []
        phase_progress_tokens: list[torch.Tensor] = []
        global_progress_tokens: list[torch.Tensor] = []
        valid_mask: list[torch.Tensor] = []
        center_row = rows[pos]

        for offset in range(-self.radius, self.radius + 1):
            tok_pos = pos + offset
            if tok_pos < 0:
                row = rows[0]
                valid = 0.0
            elif tok_pos >= len(rows):
                row = rows[-1]
                valid = 0.0
            else:
                row = rows[tok_pos]
                valid = 1.0

            window_tokens.append(self.features[row].unsqueeze(0))
            phase_tokens.append(self.phase[row].unsqueeze(0))
            phase_progress_tokens.append(self.phase_progress[row].unsqueeze(0))
            global_progress_tokens.append(self.global_progress[row].unsqueeze(0))
            valid_mask.append(torch.tensor([valid], dtype=torch.float32))

        return {
            **center_item,
            "feature_window": torch.cat(window_tokens, dim=0),
            "phase_window": torch.cat(phase_tokens, dim=0),
            "phase_progress_window": torch.cat(phase_progress_tokens, dim=0),
            "global_progress_window": torch.cat(global_progress_tokens, dim=0),
            "valid_mask": torch.cat(valid_mask, dim=0),
            "phase_center": self.phase[center_row],
            "phase_progress_center": self.phase_progress[center_row],
            "global_progress_center": self.global_progress[center_row],
        }

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[-1])

    @property
    def atoms(self) -> torch.Tensor | None:
        return getattr(self.cache, "atoms", None)


def _maybe_wrap_temporal(
    dataset: Dataset,
    *,
    head_type: str,
    window_size: int,
) -> Dataset:
    if head_type == HEAD_TYPE_TEMPORAL_Z_MLP_P:
        return TemporalWindowDataset(dataset, window_size=window_size)
    return dataset


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
    head_type: str = HEAD_TYPE_SHARED_MLP,
    window_size: int = 5,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val loaders from ``train.pt`` and ``val.pt``."""
    features_dir = Path(features_dir)
    train_cache = _maybe_wrap_temporal(
        FeatureCache(features_dir / "train.pt"),
        head_type=head_type,
        window_size=window_size,
    )
    val_cache = _maybe_wrap_temporal(
        FeatureCache(features_dir / "val.pt"),
        head_type=head_type,
        window_size=window_size,
    )

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
