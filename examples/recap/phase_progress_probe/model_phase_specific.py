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

"""Alternative z/p head variants for phase-progress probing."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseSpecificProgressHead(nn.Module):
    """Shared trunk + phase classifier + per-phase progress experts.

    Compared with the baseline ``PhaseProgressHead``, this variant keeps a
    single phase classifier but replaces the single shared progress regressor
    with one progress expert per phase. This is the minimal structural change
    that preserves the original frozen-feature setting while allowing each phase
    to have its own phase-progress geometry.
    """

    def __init__(
        self,
        feature_dim: int,
        num_phases: int = 5,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        trunk_depth: int = 1,
    ):
        super().__init__()
        if trunk_depth < 1:
            raise ValueError(f"trunk_depth must be >= 1, got {trunk_depth}")

        self.feature_dim = feature_dim
        self.num_phases = num_phases
        self.trunk_depth = trunk_depth

        trunk_layers: list[nn.Module] = []
        in_dim = feature_dim
        for _ in range(trunk_depth):
            trunk_layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                ]
            )
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*trunk_layers)

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

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict phase logits plus per-phase progress estimates."""
        x = self.trunk(features)
        phase_logits = self.phase_head(x)
        phase_probs = F.softmax(phase_logits, dim=-1)
        phase_pred = phase_logits.argmax(dim=-1)

        progress_all = torch.cat([head(x) for head in self.progress_heads], dim=-1)
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
