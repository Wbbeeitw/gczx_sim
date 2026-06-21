"""Unit tests for integration helpers."""

import numpy as np
import pandas as pd
import pytest
import torch

from z_p_correct.data.fusion_dataset import build_fusion_dataset


def _make_logits(num_bins: int, batch_size: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((batch_size, num_bins)).astype(np.float32)


def test_build_fusion_dataset():
    num_bins = 201
    batch_size = 20
    predictions_df = pd.DataFrame({
        "episode_index": np.zeros(batch_size, dtype=np.int64),
        "frame_index": np.arange(batch_size, dtype=np.int64),
        "split": ["train"] * (batch_size // 2) + ["val"] * (batch_size // 2),
        "phase_probs_pred": list(np.eye(5, dtype=np.float32)[np.arange(batch_size) % 5]),
        "phase_progress_pred": np.random.rand(batch_size).astype(np.float32),
        "global_progress_pred": np.random.rand(batch_size).astype(np.float32),
    })
    advantages_df = pd.DataFrame({
        "episode_index": np.zeros(batch_size, dtype=np.int64),
        "frame_index": np.arange(batch_size, dtype=np.int64),
        "return": np.random.uniform(-1.0, 0.0, size=batch_size).astype(np.float64),
        "value_current": np.random.uniform(-1.0, 0.0, size=batch_size).astype(np.float64),
        "value_logits_current": list(_make_logits(num_bins, batch_size)),
    })

    train_ds = build_fusion_dataset(
        predictions_df, advantages_df, return_min=-1.0, return_max=0.0, split="train"
    )
    assert len(train_ds) == batch_size // 2
    sample = train_ds[0]
    assert len(sample) == 6
    assert sample[0].shape == (num_bins,)
    assert sample[1].shape == (5,)
