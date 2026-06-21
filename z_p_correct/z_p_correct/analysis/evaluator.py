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

"""Strict evaluator: raw critic vs fused critic on a held-out validation set."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.fusion.logit_fusion import fuse_logits, LogitFusionMLP
from ..utils.metrics import compute_fusion_metrics, value_from_logits

logger = logging.getLogger(__name__)


class FusionEvaluator:
    """Evaluate raw vs fused value predictions on cached features."""

    def __init__(
        self,
        head: nn.Module,
        fusion: LogitFusionMLP,
        atoms: torch.Tensor,
        alpha: float = 1.0,
        device: str = "cuda",
        batch_size: int = 512,
    ):
        self.head = head.to(device).eval()
        self.fusion = fusion.to(device).eval()
        self.atoms = atoms.to(device)
        self.alpha = alpha
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def evaluate_loader(
        self,
        data_loader: DataLoader,
    ) -> dict[str, float]:
        """Evaluate on a DataLoader with raw_logits and return_norm."""
        all_pred: list[torch.Tensor] = []
        all_true: list[torch.Tensor] = []
        all_raw: list[torch.Tensor] = []

        for batch in data_loader:
            raw_logits = batch["raw_logits"].to(self.device)
            return_norm = batch["return_norm"].to(self.device)
            raw_value = batch.get("raw_value")

            if "feature_window" in batch:
                head_input = batch["feature_window"].to(self.device)
            else:
                head_input = batch["features"].to(self.device)

            head_out = self.head(head_input)
            delta = self.fusion(
                raw_logits,
                head_out["phase_probs"],
                head_out["phase_progress"],
                head_out["global_progress"],
            )
            fused_logits = fuse_logits(raw_logits, delta, self.alpha)
            fused_value = value_from_logits(fused_logits, self.atoms)

            all_pred.append(fused_value.cpu())
            all_true.append(return_norm.cpu())
            if raw_value is not None:
                all_raw.append(raw_value.cpu())

        pred_all = torch.cat(all_pred)
        true_all = torch.cat(all_true)
        raw_all = torch.cat(all_raw) if all_raw else pred_all.clone()

        return compute_fusion_metrics(pred_all, true_all, raw_all)

    def evaluate_predictions(
        self,
        predictions_df: pd.DataFrame,
        advantages_df: pd.DataFrame,
        return_min: float = -1.0,
        return_max: float = 0.0,
    ) -> dict[str, float]:
        """Evaluate from prediction + advantage DataFrames.

        This mirrors the original strict evaluation pipeline for portability.
        """
        merged = advantages_df.merge(
            predictions_df,
            on=["episode_index", "frame_index"],
            how="inner",
        )

        def _norm(v):
            rng = return_max - return_min
            if rng <= 0:
                return np.full(len(v), -0.5, dtype=np.float32)
            return (v - return_min) / rng - 1.0

        merged["return_norm"] = _norm(merged["return"].to_numpy())

        logits = np.stack(merged["value_logits_current"].to_numpy()).astype(np.float32)
        phase_probs = np.stack(merged["phase_probs_pred"].to_numpy()).astype(np.float32)
        phase_progress = merged["phase_progress_pred"].to_numpy(dtype=np.float32)
        global_progress = merged["global_progress_pred"].to_numpy(dtype=np.float32)
        return_norm = merged["return_norm"].to_numpy(dtype=np.float32)
        raw_value = merged["value_current"].to_numpy(dtype=np.float32)

        logits_t = torch.tensor(logits, device=self.device)
        phase_probs_t = torch.tensor(phase_probs, device=self.device)
        phase_progress_t = torch.tensor(phase_progress, device=self.device)
        global_progress_t = torch.tensor(global_progress, device=self.device)

        with torch.no_grad():
            delta = self.fusion(logits_t, phase_probs_t, phase_progress_t, global_progress_t)
            fused_logits = fuse_logits(logits_t, delta, self.alpha)
            fused_value = value_from_logits(fused_logits, self.atoms).cpu().numpy()

        pred_t = torch.tensor(fused_value, dtype=torch.float32)
        true_t = torch.tensor(return_norm, dtype=torch.float32)
        raw_t = torch.tensor(raw_value, dtype=torch.float32)
        return compute_fusion_metrics(pred_t, true_t, raw_t)

    def save_report(
        self,
        metrics: dict[str, float],
        output_path: str,
    ) -> None:
        """Save evaluation metrics to JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        import json

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved evaluation report to %s", output_path)
