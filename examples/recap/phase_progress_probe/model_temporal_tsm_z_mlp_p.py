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

"""Temporal TSM + MLP z->p head over frozen feature windows."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalShift(nn.Module):
    """Temporal Shift Module over `[B, T, C]` features."""

    def __init__(self, shift_div: int = 8):
        super().__init__()
        if shift_div < 2:
            raise ValueError(f"shift_div must be >= 2, got {shift_div}")
        self.shift_div = shift_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Shift a small channel subset left/right along time."""
        if x.ndim != 3:
            raise ValueError(f"expected [B, T, C], got {x.shape}")

        batch, time, channels = x.shape
        fold = channels // self.shift_div
        if fold == 0 or time == 1:
            return x

        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, 1:, fold : 2 * fold] = x[:, :-1, fold : 2 * fold]
        out[:, :, 2 * fold :] = x[:, :, 2 * fold :]
        return out


class TemporalShiftMLPBlock(nn.Module):
    """Residual TSM block with token-wise MLP mixing."""

    def __init__(
        self,
        hidden_dim: int,
        mlp_hidden_dim: int,
        dropout: float = 0.1,
        shift_div: int = 8,
    ):
        super().__init__()
        self.shift = TemporalShift(shift_div=shift_div)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.temporal_mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shifted = self.shift(x)
        x = x + self.temporal_mlp(self.norm1(shifted))
        x = x + self.channel_mlp(self.norm2(x))
        return x


class TemporalTSMZMLPProgressHead(nn.Module):
    """Window -> TSM+MLP temporal encoder -> center h_t -> z -> p."""

    def __init__(
        self,
        feature_dim: int,
        num_phases: int = 5,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        window_size: int = 5,
        num_blocks: int = 4,
        mlp_hidden_dim: int = 512,
        shift_div: int = 8,
        stage_embedding_dim: int = 32,
        progress_hidden_dim: int = 256,
        progress_depth: int = 2,
        phase_span_priors: list[float] | None = None,
    ):
        super().__init__()
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError(
                f"window_size must be a positive odd integer, got {window_size}"
            )
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
        if mlp_hidden_dim < hidden_dim:
            raise ValueError(
                f"mlp_hidden_dim ({mlp_hidden_dim}) must be >= hidden_dim ({hidden_dim})"
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
        self.num_blocks = num_blocks
        self.mlp_hidden_dim = mlp_hidden_dim
        self.shift_div = shift_div
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

        self.temporal_blocks = nn.ModuleList(
            [
                TemporalShiftMLPBlock(
                    hidden_dim=hidden_dim,
                    mlp_hidden_dim=mlp_hidden_dim,
                    dropout=dropout,
                    shift_div=shift_div,
                )
                for _ in range(num_blocks)
            ]
        )
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
        phase_progress_soft: torch.Tensor,
        phase_progress_hard: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map stage-local progress to global progress using phase priors."""
        span = self.phase_span_priors.to(phase_probs.dtype)
        prefix = self.phase_prefix_priors.to(phase_probs.dtype)

        prefix_soft = phase_probs @ prefix
        span_soft = phase_probs @ span
        global_progress_soft = prefix_soft + span_soft * phase_progress_soft

        prefix_hard = prefix.index_select(0, phase_pred)
        span_hard = span.index_select(0, phase_pred)
        global_progress_hard = prefix_hard + span_hard * phase_progress_hard

        return global_progress_soft.clamp(0.0, 1.0), global_progress_hard.clamp(0.0, 1.0)

    def forward(
        self,
        feature_window: torch.Tensor,
        stage_prior: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict phase/progress from a feature window."""
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
        for block in self.temporal_blocks:
            hidden = block(hidden)
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
        phase_progress_soft = self.progress_head(progress_input).squeeze(-1)
        phase_progress_hard = phase_progress_soft

        global_progress_soft, global_progress_hard = self._compute_global_progress(
            phase_probs,
            phase_pred,
            phase_progress_soft,
            phase_progress_hard,
        )

        return {
            "phase_logits": phase_logits,
            "phase_probs": phase_probs,
            "phase_pred": phase_pred,
            "phase_progress": phase_progress_soft,
            "phase_progress_soft": phase_progress_soft,
            "phase_progress_hard": phase_progress_hard,
            "global_progress": global_progress_soft,
            "global_progress_soft": global_progress_soft,
            "global_progress_hard": global_progress_hard,
            "center_hidden": center_hidden,
            "stage_prior_used": prior,
        }
