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

"""Predict phase/progress from a frozen head and feature cache."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.feature_cache import FeatureCache, _collate_features

logger = logging.getLogger(__name__)


class PhaseProgressPredictor:
    """Run inference with a frozen phase/progress head on cached features."""

    def __init__(
        self,
        head: nn.Module,
        device: str = "cuda",
        batch_size: int = 512,
    ):
        self.head = head
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.head.to(self.device).eval()

    @torch.no_grad()
    def predict(
        self,
        feature_cache: FeatureCache,
        split: str = "val",
    ) -> pd.DataFrame:
        """Predict z/p for all frames in a feature cache and return a DataFrame."""
        loader = DataLoader(
            feature_cache,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=_collate_features,
        )

        records: list[dict[str, Any]] = []
        for batch in loader:
            if "feature_window" in batch:
                head_input = batch["feature_window"].to(self.device)
            else:
                head_input = batch["features"].to(self.device)

            out = self.head(head_input)

            batch_size = head_input.shape[0]
            for i in range(batch_size):
                record: dict[str, Any] = {
                    "split": split,
                    "episode_index": int(batch["episode_index"][i].item()),
                    "frame_index": int(batch["frame_index"][i].item()),
                    "phase_true": int(batch["phase"][i].item()),
                    "phase_progress_true": float(batch["phase_progress"][i].item()),
                    "global_progress_true": float(batch["global_progress"][i].item()),
                    "phase_pred": int(out["phase_pred"][i].item()),
                    "phase_progress_pred": float(out["phase_progress"][i].item()),
                    "global_progress_pred": float(out["global_progress"][i].item()),
                }
                phase_probs = out["phase_probs"][i].cpu().numpy()
                record["phase_probs_pred"] = phase_probs.astype(np.float32)
                records.append(record)

        df = pd.DataFrame(records)
        logger.info("Predicted %d frames for split=%s", len(df), split)
        return df

    def save_predictions(
        self,
        feature_cache: FeatureCache,
        output_path: str,
        split: str = "val",
    ) -> None:
        """Predict and save to parquet."""
        df = self.predict(feature_cache, split=split)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info("Saved predictions to %s", output_path)
