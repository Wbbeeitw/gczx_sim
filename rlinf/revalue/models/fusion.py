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
        phase_progress, global_progress]`` or, for richer temporal heads,
        ``[raw_logits, phase_probs, phase_progress_all, global_progress]``.
    """

    def __init__(
        self,
        num_bins: int = 201,
        num_phases: int = 5,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        depth: int = 2,
        use_phase_progress_all: bool = False,
        shared_hidden_dim: int | None = None,
        shared_hidden_proj_dim: int = 0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        self.num_bins = int(num_bins)
        self.num_phases = int(num_phases)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.use_phase_progress_all = bool(use_phase_progress_all)
        self.shared_hidden_dim = (
            int(shared_hidden_dim) if shared_hidden_dim is not None else None
        )
        self.shared_hidden_proj_dim = int(shared_hidden_proj_dim)
        progress_dim = self.num_phases if self.use_phase_progress_all else 1
        self.input_dim = self.num_bins + self.num_phases + progress_dim + 1
        if self.shared_hidden_proj_dim > 0:
            if self.shared_hidden_dim is None or self.shared_hidden_dim < 1:
                raise ValueError(
                    "shared_hidden_dim must be provided when shared_hidden_proj_dim > 0"
                )
            self.shared_hidden_proj = nn.Sequential(
                nn.Linear(self.shared_hidden_dim, self.shared_hidden_proj_dim),
                nn.LayerNorm(self.shared_hidden_proj_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            )
            self.input_dim += self.shared_hidden_proj_dim
        else:
            self.shared_hidden_proj = None

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
        phase_progress_all: torch.Tensor | None = None,
        shared_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return additive delta logits.

        Args:
            raw_logits: Raw critic logits, shape ``[batch, num_bins]``.
            phase_repr: Phase probabilities, shape ``[batch, num_phases]``.
            phase_progress: Phase-local progress, shape ``[batch]``.
            global_progress: Expected global progress, shape ``[batch]``.
            phase_progress_all: Per-phase local progress hypotheses, shape
                ``[batch, num_phases]`` when enabled.
            shared_hidden: Refined shared temporal feature for the center frame.

        Returns:
            Delta logits of shape ``[batch, num_bins]``.
        """
        if self.use_phase_progress_all:
            if phase_progress_all is None:
                raise ValueError(
                    "Fusion head was configured with use_phase_progress_all=True "
                    "but no phase_progress_all tensor was provided."
                )
            progress_input = phase_progress_all
        else:
            progress_input = phase_progress.unsqueeze(-1)
        inputs = [
            raw_logits,
            phase_repr,
            progress_input,
            global_progress.unsqueeze(-1),
        ]
        if self.shared_hidden_proj is not None:
            if shared_hidden is None:
                raise ValueError(
                    "Fusion head was configured with shared hidden context but "
                    "no shared_hidden tensor was provided."
                )
            inputs.append(self.shared_hidden_proj(shared_hidden))
        fused_input = torch.cat(inputs, dim=-1)
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
