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

"""Prediction stage for trained Revalue fusion models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rlinf.revalue.data import collate_feature_batch, read_advantages
from rlinf.revalue.models.fusion import fuse_logits, value_from_logits
from rlinf.revalue.pipeline.train import load_fusion, load_zp_head


@dataclass
class PredictionConfig:
    """Options for generating frame-level fused value predictions."""

    features_dir: str
    advantages_path: str
    zp_head_path: str
    fusion_path: str
    output_path: str
    batch_size: int = 512
    device: str = "cuda"


def _predict_split(
    *,
    cache_path: str | Path,
    advantages_df: pd.DataFrame,
    zp_head,
    fusion,
    atoms: torch.Tensor,
    alpha: float,
    split: str,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    from rlinf.revalue.data.advantage_table import FeatureAdvantageDataset

    dataset = FeatureAdvantageDataset(cache_path, advantages_df)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_feature_batch,
    )
    device_obj = torch.device(device)
    zp_head.to(device_obj).eval()
    fusion.to(device_obj).eval()
    atoms = atoms.to(device_obj)

    records = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device_obj)
            raw_logits = batch["raw_logits"].to(device_obj)
            head_out = zp_head(features)
            delta_logits = fusion(
                raw_logits,
                head_out["phase_probs"],
                head_out["phase_progress"],
                head_out["global_progress"],
            )
            fused_logits = fuse_logits(raw_logits, delta_logits, alpha)
            fused_values = value_from_logits(fused_logits, atoms)
            raw_values = value_from_logits(raw_logits, atoms)

            phase_probs = head_out["phase_probs"].detach().cpu().numpy()
            for row_index in range(features.shape[0]):
                records.append(
                    {
                        "split": split,
                        "episode_index": int(batch["episode_index"][row_index].item()),
                        "frame_index": int(batch["frame_index"][row_index].item()),
                        "phase_true": int(batch["phase"][row_index].item()),
                        "phase_pred": int(head_out["phase_pred"][row_index].item()),
                        "phase_probs_pred": phase_probs[row_index].astype(np.float32),
                        "phase_progress_true": float(
                            batch["phase_progress"][row_index].item()
                        ),
                        "phase_progress_pred": float(
                            head_out["phase_progress"][row_index].item()
                        ),
                        "global_progress_true": float(
                            batch["global_progress"][row_index].item()
                        ),
                        "global_progress_pred": float(
                            head_out["global_progress"][row_index].item()
                        ),
                        "raw_value": float(raw_values[row_index].item()),
                        "value_fused": float(fused_values[row_index].item()),
                    }
                )
    return pd.DataFrame(records)


def predict_fused_values(cfg: PredictionConfig) -> Path:
    """Generate ``predictions.parquet`` with ``value_fused``."""
    advantages_df = read_advantages(cfg.advantages_path)
    zp_head = load_zp_head(cfg.zp_head_path, device=cfg.device)
    fusion, atoms, alpha = load_fusion(cfg.fusion_path, device=cfg.device)

    train_df = _predict_split(
        cache_path=Path(cfg.features_dir) / "train.pt",
        advantages_df=advantages_df,
        zp_head=zp_head,
        fusion=fusion,
        atoms=atoms,
        alpha=alpha,
        split="train",
        batch_size=cfg.batch_size,
        device=cfg.device,
    )
    val_df = _predict_split(
        cache_path=Path(cfg.features_dir) / "val.pt",
        advantages_df=advantages_df,
        zp_head=zp_head,
        fusion=fusion,
        atoms=atoms,
        alpha=alpha,
        split="val",
        batch_size=cfg.batch_size,
        device=cfg.device,
    )
    frames = [train_df, val_df]
    test_cache = Path(cfg.features_dir) / "test.pt"
    if test_cache.exists():
        test_df = _predict_split(
            cache_path=test_cache,
            advantages_df=advantages_df,
            zp_head=zp_head,
            fusion=fusion,
            atoms=atoms,
            alpha=alpha,
            split="test",
            batch_size=cfg.batch_size,
            device=cfg.device,
        )
        frames.append(test_df)
    out_path = Path(cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(out_path, index=False)
    return out_path
