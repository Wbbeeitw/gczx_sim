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

"""Build fusion MLP datasets from predictions + advantages DataFrames."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def _normalize_return(
    values: np.ndarray,
    return_min: float,
    return_max: float,
) -> np.ndarray:
    """Normalize returns to [-1, 1]."""
    rng = return_max - return_min
    if rng <= 0:
        return np.full(len(values), -0.5, dtype=np.float32)
    return (values - return_min) / rng - 1.0


def build_fusion_dataset(
    predictions_df: pd.DataFrame,
    advantages_df: pd.DataFrame,
    return_min: float = -1.0,
    return_max: float = 0.0,
    split: Optional[str] = None,
) -> TensorDataset:
    """Merge predictions with advantages and build a TensorDataset.

    Args:
        predictions_df: DataFrame with ``episode_index``, ``frame_index``,
            ``phase_probs_pred``, ``phase_progress_pred``, ``global_progress_pred``.
        advantages_df: DataFrame with ``episode_index``, ``frame_index``,
            ``return``, ``value_current``, ``value_logits_current``.
        return_min: Minimum return for normalization.
        return_max: Maximum return for normalization.
        split: If provided, filter predictions_df by ``split`` column.

    Returns:
        TensorDataset with (raw_logits, phase_probs, phase_progress, global_progress,
        raw_value, return_norm).
    """
    pred_df = predictions_df.copy()
    if split is not None and "split" in pred_df.columns:
        pred_df = pred_df[pred_df["split"] == split].reset_index(drop=True)

    required_pred = {
        "episode_index",
        "frame_index",
        "phase_probs_pred",
        "phase_progress_pred",
        "global_progress_pred",
    }
    missing = required_pred - set(pred_df.columns)
    if missing:
        raise ValueError(f"Predictions missing columns: {sorted(missing)}")

    required_adv = {"episode_index", "frame_index", "return", "value_current", "value_logits_current"}
    missing_adv = required_adv - set(advantages_df.columns)
    if missing_adv:
        raise ValueError(f"Advantages missing columns: {sorted(missing_adv)}")

    merged = advantages_df.merge(
        pred_df,
        on=["episode_index", "frame_index"],
        how="inner",
    )
    if merged.empty:
        raise ValueError(f"No overlapping rows for split={split}")

    merged = merged.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    merged["return_norm"] = _normalize_return(
        merged["return"].to_numpy(dtype=np.float64),
        return_min=return_min,
        return_max=return_max,
    )

    logits = np.stack(merged["value_logits_current"].to_numpy()).astype(np.float32)
    phase_probs = np.stack(merged["phase_probs_pred"].to_numpy()).astype(np.float32)
    phase_progress = merged["phase_progress_pred"].to_numpy(dtype=np.float32)
    global_progress = merged["global_progress_pred"].to_numpy(dtype=np.float32)
    raw_value = merged["value_current"].to_numpy(dtype=np.float32)
    return_norm = merged["return_norm"].astype(np.float32)

    return TensorDataset(
        torch.tensor(logits, dtype=torch.float32),
        torch.tensor(phase_probs, dtype=torch.float32),
        torch.tensor(phase_progress, dtype=torch.float32),
        torch.tensor(global_progress, dtype=torch.float32),
        torch.tensor(raw_value, dtype=torch.float32),
        torch.tensor(return_norm, dtype=torch.float32),
    )


def build_fusion_loaders(
    predictions_df: pd.DataFrame,
    advantages_df: pd.DataFrame,
    return_min: float = -1.0,
    return_max: float = 0.0,
    batch_size: int = 256,
    train_shuffle: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val fusion DataLoaders from merged DataFrames."""
    train_ds = build_fusion_dataset(
        predictions_df, advantages_df, return_min, return_max, split="train"
    )
    val_ds = build_fusion_dataset(
        predictions_df, advantages_df, return_min, return_max, split="val"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_shuffle,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )
    return train_loader, val_loader
