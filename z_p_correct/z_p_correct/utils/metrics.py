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

"""Metrics for head and fusion training/evaluation."""

from __future__ import annotations

from typing import Any

import torch


def compute_head_metrics(
    phase_logits: torch.Tensor,
    phase_progress_pred: torch.Tensor,
    global_progress_pred: torch.Tensor,
    phase_true: torch.Tensor,
    phase_progress_true: torch.Tensor,
    global_progress_true: torch.Tensor,
    num_phases: int,
) -> dict[str, float]:
    """Compute metrics for a phase/progress head."""
    phase_pred = phase_logits.argmax(dim=-1)
    acc = (phase_pred == phase_true).float().mean().item()

    per_phase_acc: dict[str, float] = {}
    for ph in range(num_phases):
        mask = phase_true == ph
        if mask.any():
            per_phase_acc[f"phase_{ph}_acc"] = (
                (phase_pred[mask] == phase_true[mask]).float().mean().item()
            )
        else:
            per_phase_acc[f"phase_{ph}_acc"] = float("nan")

    progress_mae = (phase_progress_pred - phase_progress_true).abs().mean().item()
    progress_mse = ((phase_progress_pred - phase_progress_true) ** 2).mean().item()
    global_mae = (global_progress_pred - global_progress_true).abs().mean().item()
    global_mse = ((global_progress_pred - global_progress_true) ** 2).mean().item()

    return {
        "phase_acc": acc,
        "progress_mae": progress_mae,
        "progress_mse": progress_mse,
        "global_progress_mae": global_mae,
        "global_progress_mse": global_mse,
        **per_phase_acc,
    }


def compute_fusion_metrics(
    pred_values: torch.Tensor,
    true_values: torch.Tensor,
    raw_values: torch.Tensor,
) -> dict[str, float]:
    """Compute metrics for fused value predictions.

    Args:
        pred_values: Fused value predictions.
        true_values: Ground-truth normalized returns.
        raw_values: Raw critic value predictions.

    Returns:
        Dict with MSE metrics and improvement percentage over raw critic.
    """
    mse = ((pred_values - true_values) ** 2).mean().item()
    raw_mse = ((raw_values - true_values) ** 2).mean().item()
    improvement_pct = (1 - mse / raw_mse) * 100 if raw_mse > 0 else 0.0

    return {
        "value_mse": mse,
        "raw_value_mse": raw_mse,
        "improvement_pct": improvement_pct,
    }


def value_from_logits(logits: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
    """Convert categorical value logits to scalar value expectation."""
    probs = torch.softmax(logits, dim=-1)
    return (probs * atoms.unsqueeze(0)).sum(dim=-1)
