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

"""Metric helpers for Revalue."""

from __future__ import annotations

import torch


def phase_progress_metrics(
    *,
    phase_logits: torch.Tensor,
    phase_progress_pred: torch.Tensor,
    global_progress_pred: torch.Tensor,
    phase_true: torch.Tensor,
    phase_progress_true: torch.Tensor,
    global_progress_true: torch.Tensor,
) -> dict[str, float]:
    """Compute basic z/p prediction metrics."""
    phase_pred = torch.argmax(phase_logits, dim=-1)
    phase_acc = (phase_pred == phase_true).float().mean()
    progress_mae = torch.mean(torch.abs(phase_progress_pred - phase_progress_true))
    progress_mse = torch.mean((phase_progress_pred - phase_progress_true) ** 2)
    global_mae = torch.mean(torch.abs(global_progress_pred - global_progress_true))
    global_mse = torch.mean((global_progress_pred - global_progress_true) ** 2)
    return {
        "phase_acc": float(phase_acc.item()),
        "progress_mae": float(progress_mae.item()),
        "progress_mse": float(progress_mse.item()),
        "global_progress_mae": float(global_mae.item()),
        "global_progress_mse": float(global_mse.item()),
    }


def value_regression_metrics(
    pred_value: torch.Tensor,
    target_value: torch.Tensor,
    raw_value: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute value MSE/MAE and optional improvement over raw critic values."""
    mse = torch.mean((pred_value - target_value) ** 2)
    mae = torch.mean(torch.abs(pred_value - target_value))
    metrics = {
        "value_mse": float(mse.item()),
        "value_mae": float(mae.item()),
    }
    if raw_value is not None:
        raw_mse = torch.mean((raw_value - target_value) ** 2)
        metrics["raw_value_mse"] = float(raw_mse.item())
        if float(raw_mse.item()) > 0.0:
            metrics["improvement_pct"] = float(
                (1.0 - mse.item() / raw_mse.item()) * 100.0
            )
        else:
            metrics["improvement_pct"] = 0.0
    return metrics
