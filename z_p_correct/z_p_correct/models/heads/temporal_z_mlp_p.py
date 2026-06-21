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

"""Temporal z->MLP->p phase/progress head."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...registry.head_registry import register_head


@register_head("temporal_z_mlp_p")
class TemporalZMLPProgressHead(nn.Module):
    """Temporal phase head with center-token z-conditioned MLP progress."""

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
        stage_embedding_dim: int = 32,
        progress_hidden_dim: int = 256,
        progress_depth: int = 2,
        phase_span_priors: Optional[List[float]] = None,
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
        if stage_embedding_dim < 1:
            raise ValueError(
                f"stage_embedding_dim must be >= 1, got {stage_embedding_dim}"
            )
        if progress_hidden_dim < 1:
            raise ValueError(
                f"progress_hidden_dim must be >= 1, got {progress_hidden_dim}"
            )
        if progress_depth < 1:
            raise ValueError(f"progress_depth must be >= 1, got {progress_depth}")

        self.feature_dim = feature_dim
        self.num_phases = num_phases
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.stage_embedding_dim = stage_embedding_dim
        self.progress_hidden_dim = progress_hidden_dim
        self.progress_depth = progress_depth
        self.center_index = window_size // 2

        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, window_size, hidden_dim))
        self.input_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.phase_head = nn.Linear(hidden_dim, num_phases)

        self.stage_embedding = nn.Linear(num_phases, stage_embedding_dim, bias=False)

        progress_layers: list[nn.Module] = []
        progress_in_dim = hidden_dim + stage_embedding_dim
        for _ in range(progress_depth):
            progress_layers.extend(
                [
                    nn.Linear(progress_in_dim, progress_hidden_dim),
                    nn.LayerNorm(progress_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                ]
            )
            progress_in_dim = progress_hidden_dim
        progress_layers.extend([nn.Linear(progress_hidden_dim, 1), nn.Sigmoid()])
        self.progress_head = nn.Sequential(*progress_layers)

        default_spans = [1.0 / num_phases for _ in range(num_phases)]
        phase_spans = phase_span_priors or default_spans
        if len(phase_spans) != num_phases:
            raise ValueError(
                "phase_span_priors length must match num_phases, got "
                f"{len(phase_spans)} vs {num_phases}"
            )
        phase_span_tensor = torch.tensor(phase_spans, dtype=torch.float32)
        phase_span_tensor = phase_span_tensor / phase_span_tensor.sum().clamp(min=1e-6)
        phase_prefix_tensor = torch.cat(
            [torch.zeros(1, dtype=torch.float32), phase_span_tensor.cumsum(dim=0)[:-1]]
        )
        self.register_buffer("phase_span_priors", phase_span_tensor, persistent=True)
        self.register_buffer("phase_prefix_priors", phase_prefix_tensor, persistent=True)

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

    def _compute_global_progress(
        self,
        phase_probs: torch.Tensor,
        phase_pred: torch.Tensor,
        phase_progress: torch.Tensor,
    ) -> torch.Tensor:
        """Map stage-local progress to global progress using phase priors."""
        span = self.phase_span_priors.to(phase_probs.dtype)
        prefix = self.phase_prefix_priors.to(phase_probs.dtype)

        prefix_soft = phase_probs @ prefix
        span_soft = phase_probs @ span
        global_progress = prefix_soft + span_soft * phase_progress

        return global_progress.clamp(0.0, 1.0)

    def forward(
        self,
        feature_window: torch.Tensor,
        stage_prior: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Predict phase/progress from a feature window.

        Args:
            feature_window: Float tensor of shape ``[B, T, D]``.
            stage_prior: Optional phase prior of shape ``[B, K]``.

        Returns:
            Dict with ``phase_logits`` [B, K], ``phase_probs`` [B, K],
            ``phase_pred`` [B], ``phase_progress`` [B], ``global_progress`` [B].
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

        hidden = self.input_proj(feature_window)
        hidden = hidden + self.position_embedding
        hidden = self.input_dropout(hidden)
        hidden = self.temporal_encoder(hidden)
        hidden = self.temporal_norm(hidden)

        center_hidden = hidden[:, self.center_index, :]
        phase_logits = self.phase_head(center_hidden)
        phase_probs = F.softmax(phase_logits, dim=-1)
        phase_pred = phase_logits.argmax(dim=-1)

        if stage_prior is None:
            prior = phase_probs
        else:
            prior = stage_prior.to(phase_probs.dtype)
        stage_emb = self.stage_embedding(prior)
        progress_input = torch.cat([center_hidden, stage_emb], dim=-1)
        phase_progress = self.progress_head(progress_input).squeeze(-1)

        global_progress = self._compute_global_progress(
            phase_probs,
            phase_pred,
            phase_progress,
        )

        return {
            "phase_logits": phase_logits,
            "phase_probs": phase_probs,
            "phase_pred": phase_pred,
            "phase_progress": phase_progress,
            "global_progress": global_progress,
        }
