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

"""Helpers for reading and writing ReCap advantage tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from rlinf.revalue.data.feature_cache import FeatureCache, collate_feature_batch

_ADVANTAGE_REQUIRED_FOR_FUSION = {
    "episode_index",
    "frame_index",
    "return",
    "value_current",
    "value_logits_current",
}


def resolve_advantage_path(dataset_path: str | Path, tag: str | None) -> Path:
    """Return the standard ReCap advantage parquet path for a dataset/tag."""
    dataset_path = Path(dataset_path)
    filename = f"advantages_{tag}.parquet" if tag else "advantages.parquet"
    return dataset_path / "meta" / filename


def read_advantages(path: str | Path) -> pd.DataFrame:
    """Read an advantage parquet and validate basic key columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Advantage parquet not found: {path}")
    df = pd.read_parquet(path)
    required = {"episode_index", "frame_index"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Advantage parquet {path} missing keys: {sorted(missing)}")
    return df


def validate_fusion_advantages(df: pd.DataFrame, *, source: str | Path) -> None:
    """Validate that an advantage table can train logit fusion."""
    missing = _ADVANTAGE_REQUIRED_FOR_FUSION - set(df.columns)
    if missing:
        raise ValueError(
            f"Fusion requires source advantages with columns {sorted(missing)}. "
            f"Regenerate {source} with advantage.save_value_distribution=true."
        )


class FeatureAdvantageDataset(Dataset):
    """Feature cache rows joined with source advantage rows by episode/frame."""

    def __init__(
        self,
        feature_cache: FeatureCache | str | Path,
        advantages: pd.DataFrame | str | Path,
    ) -> None:
        self.feature_cache = (
            feature_cache
            if isinstance(feature_cache, FeatureCache)
            else FeatureCache(feature_cache)
        )
        self.advantages_df = (
            advantages.copy() if isinstance(advantages, pd.DataFrame) else read_advantages(advantages)
        )
        validate_fusion_advantages(self.advantages_df, source="advantages")
        self._rows = self._align_rows(self.advantages_df)

    def _align_rows(self, advantages_df: pd.DataFrame) -> list[dict[str, Any]]:
        key_to_row: dict[tuple[int, int], dict[str, Any]] = {}
        for row in advantages_df.to_dict("records"):
            key_to_row[(int(row["episode_index"]), int(row["frame_index"]))] = row

        rows: list[dict[str, Any]] = []
        missing: list[tuple[int, int]] = []
        for index in range(len(self.feature_cache)):
            ep = int(self.feature_cache.episode_index[index].item())
            frame = int(self.feature_cache.frame_index[index].item())
            key = (ep, frame)
            row = key_to_row.get(key)
            if row is None:
                missing.append(key)
                continue
            rows.append(row)

        if missing:
            preview = ", ".join(str(key) for key in missing[:5])
            raise ValueError(
                f"{len(missing)} feature rows have no source advantage row. "
                f"First missing keys: {preview}"
            )
        return rows

    def __len__(self) -> int:
        return len(self.feature_cache)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.feature_cache[index]
        row = self._rows[index]
        logits = np.asarray(row["value_logits_current"], dtype=np.float32)
        item["raw_logits"] = torch.tensor(logits, dtype=torch.float32)
        item["raw_value"] = torch.tensor(float(row["value_current"]), dtype=torch.float32)
        item["return"] = torch.tensor(float(row["return"]), dtype=torch.float32)
        return item

    @property
    def feature_dim(self) -> int:
        return self.feature_cache.feature_dim

    @property
    def atoms(self) -> torch.Tensor | None:
        return self.feature_cache.atoms


def build_fusion_loaders(
    features_dir: str | Path,
    advantages_path: str | Path,
    *,
    batch_size: int = 256,
    num_workers: int = 0,
    train_shuffle: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val fusion loaders from feature caches and source advantages."""
    advantages_df = read_advantages(advantages_path)
    validate_fusion_advantages(advantages_df, source=advantages_path)

    train_dataset = FeatureAdvantageDataset(
        Path(features_dir) / "train.pt",
        advantages_df,
    )
    val_dataset = FeatureAdvantageDataset(
        Path(features_dir) / "val.pt",
        advantages_df,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        num_workers=num_workers,
        collate_fn=collate_feature_batch,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_feature_batch,
        pin_memory=True,
    )
    return train_loader, val_loader
