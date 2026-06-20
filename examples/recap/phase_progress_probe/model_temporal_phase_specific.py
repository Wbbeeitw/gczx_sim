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

"""Temporal phase-progress probe heads over frozen feature windows."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalPhaseSpecificProgressHead(nn.Module):
    """Small temporal Conv1d head over local frozen-feature windows.

    The backbone remains fully frozen. This module only consumes pre-extracted
    feature windows of shape ``[B, T, D]`` and predicts the center frame's
    semantic phase plus phase-specific progress.
    """

    def __init__(
        self,
        feature_dim: int,
        num_phases: int = 5,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        window_size: int = 5,
        temporal_kernel_size: int = 3,
        temporal_layers: int = 2,
    ):
        super().__init__()
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError(
                f"window_size must be a positive odd integer, got {window_size}"
            )
        if temporal_layers < 1:
            raise ValueError(
                f"temporal_layers must be >= 1, got {temporal_layers}"
            )
        if temporal_kernel_size < 1 or temporal_kernel_size % 2 == 0:
            raise ValueError(
                "temporal_kernel_size must be a positive odd integer, "
                f"got {temporal_kernel_size}"
            )

        self.feature_dim = feature_dim
        self.num_phases = num_phases
        self.window_size = window_size
        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_layers = temporal_layers
        self.center_index = window_size // 2

        proj_layers: list[nn.Module] = [
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        ]
        self.input_proj = nn.Sequential(*proj_layers)

        temporal_blocks: list[nn.Module] = []
        for _ in range(temporal_layers):
            temporal_blocks.extend(
                [
                    nn.Conv1d(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        kernel_size=temporal_kernel_size,
                        padding=temporal_kernel_size // 2,
                    ),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                ]
            )
        self.temporal_trunk = nn.Sequential(*temporal_blocks)

        self.phase_head = nn.Linear(hidden_dim, num_phases)
        self.progress_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid(),
                )
                for _ in range(num_phases)
            ]
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feature_window: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict the center frame's phase and progress from a local window.

        Args:
            feature_window: Float tensor of shape ``[B, T, D]``.
        """
        if feature_window.ndim != 3:
            raise ValueError(
                f"feature_window must have shape [B, T, D], got {feature_window.shape}"
            )

        x = self.input_proj(feature_window)
        x = x.transpose(1, 2)
        x = self.temporal_trunk(x)
        x = x.transpose(1, 2)
        center_feat = x[:, self.center_index, :]

        phase_logits = self.phase_head(center_feat)
        phase_probs = F.softmax(phase_logits, dim=-1)
        phase_pred = phase_logits.argmax(dim=-1)

        progress_all = torch.cat(
            [head(center_feat) for head in self.progress_heads], dim=-1
        )
        progress_all = progress_all.clamp(0.0, 1.0)

        phase_indices = torch.arange(
            self.num_phases, dtype=progress_all.dtype, device=progress_all.device
        )
        expected_phase = (phase_probs * phase_indices).sum(dim=-1)
        phase_progress_soft = (phase_probs * progress_all).sum(dim=-1)
        phase_progress_hard = progress_all.gather(1, phase_pred.unsqueeze(1)).squeeze(1)
        global_progress_soft = (expected_phase + phase_progress_soft) / self.num_phases
        global_progress_hard = (
            phase_pred.to(progress_all.dtype) + phase_progress_hard
        ) / self.num_phases

        return {
            "phase_logits": phase_logits,
            "phase_probs": phase_probs,
            "phase_pred": phase_pred,
            "phase_progress_all": progress_all,
            "phase_progress": phase_progress_soft,
            "phase_progress_soft": phase_progress_soft,
            "phase_progress_hard": phase_progress_hard,
            "global_progress": global_progress_soft,
            "global_progress_soft": global_progress_soft,
            "global_progress_hard": global_progress_hard,
        }
