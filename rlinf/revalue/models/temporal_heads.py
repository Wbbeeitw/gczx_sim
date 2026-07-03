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

"""Temporal phase/progress heads used by Revalue."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _make_mlp_block(
    input_dim: int,
    output_dim: int,
    *,
    dropout: float,
) -> list[nn.Module]:
    return [
        nn.Linear(input_dim, output_dim),
        nn.LayerNorm(output_dim),
        nn.GELU(),
        nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
    ]


def _make_progress_head(
    *,
    input_dim: int,
    hidden_dim: int,
    depth: int,
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(depth):
        layers.extend(_make_mlp_block(current_dim, hidden_dim, dropout=dropout))
        current_dim = hidden_dim
    layers.extend([nn.Linear(current_dim, 1), nn.Sigmoid()])
    return nn.Sequential(*layers)


class TemporalZMLPProgressHead(nn.Module):
    """Temporal phase head with center-token z-conditioned MLP progress.

    This head consumes a local frozen-feature window ``[B, T, D]``. A small
    temporal encoder predicts the center frame phase ``z_t`` and then regresses
    phase-local progress ``p_t`` conditioned on a stage prior.
    """

    HEAD_TYPE = "temporal_z_mlp_p"

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
        phase_span_priors: list[float] | None = None,
    ) -> None:
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

        self.feature_dim = int(feature_dim)
        self.num_phases = int(num_phases)
        self.hidden_dim = int(hidden_dim)
        self.window_size = int(window_size)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.ffn_dim = int(ffn_dim)
        self.stage_embedding_dim = int(stage_embedding_dim)
        self.progress_hidden_dim = int(progress_hidden_dim)
        self.progress_depth = int(progress_depth)
        self.center_index = self.window_size // 2

        self.input_proj = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.window_size, self.hidden_dim)
        )
        self.input_dropout = (
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
        )
        self.temporal_norm = nn.LayerNorm(self.hidden_dim)
        self.phase_head = nn.Linear(self.hidden_dim, self.num_phases)
        self.stage_embedding = nn.Linear(
            self.num_phases,
            self.stage_embedding_dim,
            bias=False,
        )

        progress_layers: list[nn.Module] = []
        progress_in_dim = self.hidden_dim + self.stage_embedding_dim
        for _ in range(self.progress_depth):
            progress_layers.extend(
                _make_mlp_block(
                    progress_in_dim,
                    self.progress_hidden_dim,
                    dropout=dropout,
                )
            )
            progress_in_dim = self.progress_hidden_dim
        progress_layers.extend([nn.Linear(self.progress_hidden_dim, 1), nn.Sigmoid()])
        self.progress_head = nn.Sequential(*progress_layers)

        default_spans = [1.0 / self.num_phases for _ in range(self.num_phases)]
        phase_spans = phase_span_priors or default_spans
        if len(phase_spans) != self.num_phases:
            raise ValueError(
                "phase_span_priors length must match num_phases, got "
                f"{len(phase_spans)} vs {self.num_phases}"
            )
        phase_span_tensor = torch.tensor(phase_spans, dtype=torch.float32)
        phase_span_tensor = phase_span_tensor / phase_span_tensor.sum().clamp(min=1e-6)
        phase_prefix_tensor = torch.cat(
            [torch.zeros(1, dtype=torch.float32), phase_span_tensor.cumsum(dim=0)[:-1]]
        )
        self.register_buffer("phase_span_priors", phase_span_tensor, persistent=True)
        self.register_buffer(
            "phase_prefix_priors",
            phase_prefix_tensor,
            persistent=True,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _compute_global_progress(
        self,
        phase_probs: torch.Tensor,
        phase_pred: torch.Tensor,
        phase_progress_soft: torch.Tensor,
        phase_progress_hard: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        span = self.phase_span_priors.to(phase_probs.dtype)
        prefix = self.phase_prefix_priors.to(phase_probs.dtype)

        prefix_soft = phase_probs @ prefix
        span_soft = phase_probs @ span
        global_progress_soft = prefix_soft + span_soft * phase_progress_soft

        prefix_hard = prefix.index_select(0, phase_pred)
        span_hard = span.index_select(0, phase_pred)
        global_progress_hard = prefix_hard + span_hard * phase_progress_hard

        return global_progress_soft.clamp(0.0, 1.0), global_progress_hard.clamp(
            0.0,
            1.0,
        )

    def forward(
        self,
        feature_window: torch.Tensor,
        stage_prior: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
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

        prior = phase_probs if stage_prior is None else stage_prior.to(phase_probs.dtype)
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


class TemporalStageExpertsProgressHead(nn.Module):
    """Temporal stage-then-experts head over local frozen feature windows.

    The head first predicts the center-frame stage from a shared temporal
    representation and then regresses per-stage local progress with dedicated
    expert MLPs conditioned on a stage prior. A shared post-transformer MLP
    refines the center token before branching.
    """

    HEAD_TYPE = "temporal_stage_experts"

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
        trunk_depth: int = 1,
        phase_span_priors: list[float] | None = None,
    ) -> None:
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
        if trunk_depth < 0:
            raise ValueError(f"trunk_depth must be >= 0, got {trunk_depth}")

        self.feature_dim = int(feature_dim)
        self.num_phases = int(num_phases)
        self.hidden_dim = int(hidden_dim)
        self.window_size = int(window_size)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.ffn_dim = int(ffn_dim)
        self.stage_embedding_dim = int(stage_embedding_dim)
        self.progress_hidden_dim = int(progress_hidden_dim)
        self.progress_depth = int(progress_depth)
        self.trunk_depth = int(trunk_depth)
        self.center_index = self.window_size // 2

        self.input_proj = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.window_size, self.hidden_dim)
        )
        self.input_dropout = (
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
        )
        self.temporal_norm = nn.LayerNorm(self.hidden_dim)

        refine_layers: list[nn.Module] = []
        current_dim = self.hidden_dim
        for _ in range(self.trunk_depth):
            refine_layers.extend(
                _make_mlp_block(
                    current_dim,
                    self.hidden_dim,
                    dropout=dropout,
                )
            )
            current_dim = self.hidden_dim
        self.shared_fusion = (
            nn.Sequential(*refine_layers)
            if refine_layers
            else nn.Identity()
        )
        self.has_shared_fusion = bool(refine_layers)
        self.shared_norm = nn.LayerNorm(self.hidden_dim)

        self.phase_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(self.hidden_dim, self.num_phases),
        )
        self.stage_embedding = nn.Linear(
            self.num_phases,
            self.stage_embedding_dim,
            bias=False,
        )
        expert_in_dim = self.hidden_dim + self.stage_embedding_dim
        self.progress_experts = nn.ModuleList(
            [
                _make_progress_head(
                    input_dim=expert_in_dim,
                    hidden_dim=self.progress_hidden_dim,
                    depth=self.progress_depth,
                    dropout=dropout,
                )
                for _ in range(self.num_phases)
            ]
        )

        default_spans = [1.0 / self.num_phases for _ in range(self.num_phases)]
        phase_spans = phase_span_priors or default_spans
        if len(phase_spans) != self.num_phases:
            raise ValueError(
                "phase_span_priors length must match num_phases, got "
                f"{len(phase_spans)} vs {self.num_phases}"
            )
        phase_span_tensor = torch.tensor(phase_spans, dtype=torch.float32)
        phase_span_tensor = phase_span_tensor / phase_span_tensor.sum().clamp(min=1e-6)
        phase_prefix_tensor = torch.cat(
            [torch.zeros(1, dtype=torch.float32), phase_span_tensor.cumsum(dim=0)[:-1]]
        )
        self.register_buffer("phase_span_priors", phase_span_tensor, persistent=True)
        self.register_buffer(
            "phase_prefix_priors",
            phase_prefix_tensor,
            persistent=True,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _compute_global_progress(
        self,
        phase_weights: torch.Tensor,
        phase_pred: torch.Tensor,
        phase_progress_soft: torch.Tensor,
        phase_progress_hard: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        span = self.phase_span_priors.to(phase_weights.dtype)
        prefix = self.phase_prefix_priors.to(phase_weights.dtype)

        prefix_soft = phase_weights @ prefix
        span_soft = phase_weights @ span
        global_progress_soft = prefix_soft + span_soft * phase_progress_soft

        prefix_hard = prefix.index_select(0, phase_pred)
        span_hard = span.index_select(0, phase_pred)
        global_progress_hard = prefix_hard + span_hard * phase_progress_hard

        return global_progress_soft.clamp(0.0, 1.0), global_progress_hard.clamp(
            0.0,
            1.0,
        )

    def forward(
        self,
        feature_window: torch.Tensor,
        stage_prior: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
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
        refined_hidden = self.shared_fusion(center_hidden)
        if self.has_shared_fusion and refined_hidden.shape == center_hidden.shape:
            refined_hidden = refined_hidden + center_hidden
        shared_hidden = self.shared_norm(refined_hidden)

        phase_logits = self.phase_head(shared_hidden)
        phase_probs = F.softmax(phase_logits, dim=-1)
        phase_pred = phase_logits.argmax(dim=-1)

        prior = phase_probs if stage_prior is None else stage_prior.to(phase_probs.dtype)
        stage_emb = self.stage_embedding(prior)
        expert_input = torch.cat([shared_hidden, stage_emb], dim=-1)
        phase_progress_all = torch.cat(
            [expert(expert_input) for expert in self.progress_experts],
            dim=-1,
        ).clamp(0.0, 1.0)

        phase_progress_soft = (prior * phase_progress_all).sum(dim=-1)
        phase_progress_hard = phase_progress_all.gather(
            1,
            phase_pred.unsqueeze(1),
        ).squeeze(1)
        global_progress_soft, global_progress_hard = self._compute_global_progress(
            prior,
            phase_pred,
            phase_progress_soft,
            phase_progress_hard,
        )

        return {
            "phase_logits": phase_logits,
            "phase_probs": phase_probs,
            "phase_pred": phase_pred,
            "phase_progress_all": phase_progress_all,
            "phase_progress": phase_progress_soft,
            "phase_progress_soft": phase_progress_soft,
            "phase_progress_hard": phase_progress_hard,
            "global_progress": global_progress_soft,
            "global_progress_soft": global_progress_soft,
            "global_progress_hard": global_progress_hard,
            "center_hidden": center_hidden,
            "shared_hidden": shared_hidden,
            "stage_prior_used": prior,
        }
