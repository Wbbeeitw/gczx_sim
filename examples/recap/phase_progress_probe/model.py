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

"""Feature extractor and lightweight head for z/p probing."""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from rlinf.models.embodiment.value_model import ValueCriticModel

logger = logging.getLogger(__name__)


def _make_4d_attention_mask(att_2d_masks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert a 2D bool mask to a 4D additive mask."""
    att_2d_masks_4d = att_2d_masks[:, None, :, :]
    return torch.where(
        att_2d_masks_4d,
        torch.tensor(0.0, dtype=dtype, device=att_2d_masks.device),
        torch.tensor(-2.3819763e38, dtype=dtype, device=att_2d_masks.device),
    )


class VLMBackboneFeatureExtractor(nn.Module):
    """Frozen ValueCriticModel VLM backbone that emits mean-pooled prefix features.

    Extracts the Gemma3 prefix output (image patches + language tokens) and
    returns a fixed-dimensional feature vector for each sample.
    """

    def __init__(self, value_model: ValueCriticModel):
        super().__init__()
        self.value_model = value_model

        # Freeze everything and set to eval mode.
        self.value_model.eval()
        for param in self.value_model.parameters():
            param.requires_grad = False

    @property
    def feature_dim(self) -> int:
        """Feature dimension after mean-pooling."""
        return self.value_model.value_expert.gemma3.config.hidden_size

    @property
    def device(self) -> torch.device:
        return next(self.value_model.parameters()).device

    @torch.no_grad()
    def extract_prefix_features(self, observation: dict[str, Any]) -> torch.Tensor:
        """Extract mean-pooled prefix features from the VLM backbone.

        Args:
            observation: Dict as returned by ``ValueCriticModel._prepare_observation``,
                containing ``images``, ``image_masks``, ``tokenized_prompt``,
                ``tokenized_prompt_mask``.

        Returns:
            Float32 tensor of shape ``[B, feature_dim]``.
        """
        value_model = self.value_model

        (
            images,
            image_masks,
            lang_tokens,
            lang_masks,
            _,
            _,
        ) = value_model._preprocess_observation(observation)

        # Build prefix embeddings and padding mask.
        prefix_embs, prefix_pad_masks = value_model.embed_prefix(
            images, image_masks, lang_tokens, lang_masks
        )

        # 4D additive attention mask for bidirectional prefix.
        prefix_attn = prefix_pad_masks[:, None, :] * prefix_pad_masks[:, :, None]
        dtype = value_model._get_model_dtype()
        prefix_attn_4d = _make_4d_attention_mask(prefix_attn, dtype)
        prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Run Gemma3 on the prefix only (Mode A). Pre-create DynamicCache so this
        # works regardless of whether the model is in train or eval mode.
        (prefix_out, _), _ = value_model.value_expert.forward(
            attention_mask=prefix_attn_4d,
            position_ids=prefix_pos,
            past_key_values=DynamicCache(),
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # Mean pool over valid prefix tokens.
        mask = prefix_pad_masks.unsqueeze(-1).to(prefix_out.dtype)
        masked = prefix_out * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        features = summed / counts

        return features.to(torch.float32)

    def forward(self, observation: dict[str, Any]) -> torch.Tensor:
        """Alias for ``extract_prefix_features``."""
        return self.extract_prefix_features(observation)


class PhaseProgressHead(nn.Module):
    """Lightweight MLP head predicting semantic phase and phase progress."""

    def __init__(
        self,
        feature_dim: int,
        num_phases: int = 5,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_phases = num_phases

        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )
        self.phase_head = nn.Linear(hidden_dim, num_phases)
        self.progress_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
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
        """Forward pass.

        Args:
            features: Float tensor of shape ``[B, feature_dim]``.

        Returns:
            Dict with ``phase_logits`` [B, num_phases], ``phase_progress`` [B],
            and ``global_progress`` [B].
        """
        x = self.trunk(features)
        phase_logits = self.phase_head(x)
        phase_progress = self.progress_head(x).squeeze(-1)

        # global_progress = (predicted_phase_soft * 1 + phase_progress) / num_phases
        phase_probs = F.softmax(phase_logits, dim=-1)
        phase_indices = torch.arange(
            self.num_phases, dtype=phase_progress.dtype, device=phase_progress.device
        )
        expected_phase = (phase_probs * phase_indices).sum(dim=-1)
        global_progress = (expected_phase + phase_progress) / self.num_phases

        return {
            "phase_logits": phase_logits,
            "phase_progress": phase_progress,
            "global_progress": global_progress,
        }


class PhaseProgressProbe(nn.Module):
    """End-to-end wrapper: frozen VLM feature extractor + trainable z/p head."""

    def __init__(
        self,
        feature_extractor: VLMBackboneFeatureExtractor,
        head: PhaseProgressHead,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.head = head

    @classmethod
    def from_value_checkpoint(
        cls,
        checkpoint_dir: str,
        num_phases: int = 5,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.1,
        device: str = "cuda",
        **checkpoint_kwargs: Any,
    ) -> "PhaseProgressProbe":
        """Build a probe from a ValueCriticModel checkpoint.

        Args:
            checkpoint_dir: Path to the value model checkpoint.
            num_phases: Number of semantic phases.
            head_hidden_dim: Hidden dimension of the head.
            head_dropout: Dropout rate in the head.
            device: Device to load the model on.
            **checkpoint_kwargs: Passed to ``ValueCriticModel.from_checkpoint``.

        Returns:
            A ``PhaseProgressProbe`` instance on the requested device.
        """
        value_model = ValueCriticModel.from_checkpoint(
            checkpoint_dir,
            device=device,
            **checkpoint_kwargs,
        )
        feature_extractor = VLMBackboneFeatureExtractor(value_model)
        head = PhaseProgressHead(
            feature_dim=feature_extractor.feature_dim,
            num_phases=num_phases,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
        )
        return cls(feature_extractor, head).to(device)

    def forward(self, observation: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Extract features and predict z/p in one call."""
        features = self.feature_extractor.extract_prefix_features(observation)
        return self.head(features)

    @property
    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return only the head parameters (VLM is frozen)."""
        return list(self.head.parameters())
