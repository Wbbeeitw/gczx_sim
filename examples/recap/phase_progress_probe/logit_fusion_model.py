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

"""Logit-space fusion MLP for 201-bin value distribution correction."""

from __future__ import annotations

import torch
import torch.nn as nn


class LogitFusionMLP(nn.Module):
    """Predict 201-bin delta logits from raw logits and z/p signals."""

    def __init__(
        self,
        num_bins: int = 201,
        num_phases: int = 5,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        depth: int = 2,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        self.num_bins = num_bins
        self.num_phases = num_phases
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.depth = depth
        self.input_dim = num_bins + num_phases + 2

        layers: list[nn.Module] = []
        in_dim = self.input_dim
        for _ in range(depth):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                ]
            )
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.delta_head = nn.Linear(hidden_dim, num_bins)

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

    def forward(
        self,
        raw_logits: torch.Tensor,
        phase_repr: torch.Tensor,
        phase_progress: torch.Tensor,
        global_progress: torch.Tensor,
    ) -> torch.Tensor:
        """Predict additive delta logits.

        Args:
            raw_logits: Raw 201-bin logits, shape ``[B, num_bins]``.
            phase_repr: Phase one-hot or phase probability vector, shape ``[B, num_phases]``.
            phase_progress: Progress within phase, shape ``[B]``.
            global_progress: Global normalized progress, shape ``[B]``.

        Returns:
            Delta logits of shape ``[B, num_bins]``.
        """
        fused = torch.cat(
            [
                raw_logits,
                phase_repr,
                phase_progress.unsqueeze(-1),
                global_progress.unsqueeze(-1),
            ],
            dim=-1,
        )
        hidden = self.trunk(fused)
        return self.delta_head(hidden)
