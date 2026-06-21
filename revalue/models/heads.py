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

"""Phase/progress heads used by Revalue."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SharedMLPPhaseProgressHead(nn.Module):
    """Shared MLP trunk with a phase classifier and progress regressor.

    This is the only z/p head supported by the maintained Revalue fusion path.
    It consumes one frozen VLM feature vector per frame.
    """

    def __init__(
        self,
        feature_dim: int,
        num_phases: int = 5,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        trunk_depth: int = 1,
    ) -> None:
        super().__init__()
        if trunk_depth < 1:
            raise ValueError(f"trunk_depth must be >= 1, got {trunk_depth}")

        self.feature_dim = int(feature_dim)
        self.num_phases = int(num_phases)
        self.hidden_dim = int(hidden_dim)
        self.trunk_depth = int(trunk_depth)

        layers: list[nn.Module] = []
        in_dim = self.feature_dim
        for _ in range(self.trunk_depth):
            layers.extend(
                [
                    nn.Linear(in_dim, self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                ]
            )
            in_dim = self.hidden_dim

        self.trunk = nn.Sequential(*layers)
        self.phase_head = nn.Linear(self.hidden_dim, self.num_phases)
        self.progress_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict phase and phase-local progress.

        Args:
            features: Tensor of shape ``[batch, feature_dim]``.

        Returns:
            Dictionary with phase logits/probabilities, argmax phase, phase-local
            progress, and expected global progress.
        """
        hidden = self.trunk(features)
        phase_logits = self.phase_head(hidden)
        phase_probs = F.softmax(phase_logits, dim=-1)
        phase_pred = torch.argmax(phase_logits, dim=-1)
        phase_progress = self.progress_head(hidden).squeeze(-1)

        phase_indices = torch.arange(
            self.num_phases,
            dtype=phase_progress.dtype,
            device=phase_progress.device,
        )
        expected_phase = torch.sum(phase_probs * phase_indices, dim=-1)
        global_progress = (expected_phase + phase_progress) / self.num_phases

        return {
            "phase_logits": phase_logits,
            "phase_probs": phase_probs,
            "phase_pred": phase_pred,
            "phase_progress": phase_progress,
            "global_progress": global_progress,
        }
