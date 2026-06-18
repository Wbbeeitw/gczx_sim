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

"""Small fusion MLP for value-space z/p correction."""

from __future__ import annotations

import torch
import torch.nn as nn


class FusionMLP(nn.Module):
    """Predict a scalar value bias from frozen VLM features and z/p signals."""

    def __init__(
        self,
        feature_dim: int,
        num_phases: int = 5,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        depth: int = 2,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        self.feature_dim = feature_dim
        self.num_phases = num_phases
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.depth = depth
        self.input_dim = feature_dim + num_phases + 3

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
        self.out_proj = nn.Linear(hidden_dim, 1)

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
        features: torch.Tensor,
        phase_repr: torch.Tensor,
        phase_progress: torch.Tensor,
        global_progress: torch.Tensor,
        raw_value: torch.Tensor,
    ) -> torch.Tensor:
        """Predict scalar bias for the current frame.

        Args:
            features: Frozen VLM features, shape ``[B, feature_dim]``.
            phase_repr: Phase distribution / one-hot, shape ``[B, num_phases]``.
            phase_progress: Progress within phase, shape ``[B]``.
            global_progress: Global normalized progress, shape ``[B]``.
            raw_value: Raw critic value, shape ``[B]``.

        Returns:
            Predicted scalar bias, shape ``[B]``.
        """
        fused = torch.cat(
            [
                features,
                phase_repr,
                phase_progress.unsqueeze(-1),
                global_progress.unsqueeze(-1),
                raw_value.unsqueeze(-1),
            ],
            dim=-1,
        )
        hidden = self.trunk(fused)
        return self.out_proj(hidden).squeeze(-1)
