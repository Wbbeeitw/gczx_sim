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

"""Logit-space fusion modules for value distribution correction."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class LogitFusionMLP(nn.Module):
    """Predict additive delta logits from raw critic logits and z/p signals.

    The maintained Revalue fusion rule is:

        fused_logits = raw_logits + alpha * delta_logits

    where ``delta_logits`` is produced from ``[raw_logits, phase_probs,
    phase_progress, global_progress]``.
    """

    def __init__(
        self,
        num_bins: int = 201,
        num_phases: int = 5,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        depth: int = 2,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        self.num_bins = int(num_bins)
        self.num_phases = int(num_phases)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.input_dim = self.num_bins + self.num_phases + 2

        layers: list[nn.Module] = []
        in_dim = self.input_dim
        for _ in range(self.depth):
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
        self.delta_head = nn.Linear(self.hidden_dim, self.num_bins)
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

    def forward(
        self,
        raw_logits: torch.Tensor,
        phase_repr: torch.Tensor,
        phase_progress: torch.Tensor,
        global_progress: torch.Tensor,
    ) -> torch.Tensor:
        """Return additive delta logits.

        Args:
            raw_logits: Raw critic logits, shape ``[batch, num_bins]``.
            phase_repr: Phase probabilities, shape ``[batch, num_phases]``.
            phase_progress: Phase-local progress, shape ``[batch]``.
            global_progress: Expected global progress, shape ``[batch]``.

        Returns:
            Delta logits of shape ``[batch, num_bins]``.
        """
        fused_input = torch.cat(
            [
                raw_logits,
                phase_repr,
                phase_progress.unsqueeze(-1),
                global_progress.unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.delta_head(self.trunk(fused_input))


def fuse_logits(
    raw_logits: torch.Tensor,
    delta_logits: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Apply additive logit-space fusion."""
    return raw_logits + float(alpha) * delta_logits


def value_from_logits(logits: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
    """Compute scalar values from categorical value logits and support atoms."""
    probs = F.softmax(logits, dim=-1)
    return torch.sum(probs * atoms.to(device=logits.device, dtype=probs.dtype), dim=-1)
