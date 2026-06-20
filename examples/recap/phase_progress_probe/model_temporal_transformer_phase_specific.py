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

"""Temporal transformer phase-progress probe heads over frozen feature windows."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalTransformerPhaseSpecificProgressHead(nn.Module):
    """Small transformer head over local frozen-feature windows.

    The VLM backbone remains frozen. This module only consumes cached feature
    windows of shape ``[B, T, D]`` and predicts the center frame's phase plus
    phase-specific progress.
    """

    def __init__(
        self,
        feature_dim: int,
        num_phases: int = 5,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        window_size: int = 5,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = 512,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError(
                f"window_size must be a positive odd integer, got {window_size}"
            )
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        if ffn_dim < hidden_dim:
            raise ValueError(
                f"ffn_dim ({ffn_dim}) must be >= hidden_dim ({hidden_dim})"
            )

        self.feature_dim = feature_dim
        self.num_phases = num_phases
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.center_index = window_size // 2

        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, window_size, hidden_dim)
        )
        self.input_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

        self.phase_head = nn.Linear(hidden_dim, num_phases)
        self.progress_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(attention_dropout)
                    if attention_dropout > 0.0
                    else nn.Identity(),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Sigmoid(),
                )
                for _ in range(num_phases)
            ]
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
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
        if feature_window.shape[1] != self.window_size:
            raise ValueError(
                "feature_window length does not match configured window_size: "
                f"{feature_window.shape[1]} vs {self.window_size}"
            )

        x = self.input_proj(feature_window)
        x = x + self.position_embedding
        x = self.input_dropout(x)
        x = self.temporal_encoder(x)
        center_feat = self.output_norm(x[:, self.center_index, :])

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
