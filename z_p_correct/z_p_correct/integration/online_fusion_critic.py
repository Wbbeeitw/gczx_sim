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

"""Online fusion critic that corrects raw value predictions with z/p signals."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from ..models.fusion.logit_fusion import LogitFusionMLP, fuse_logits
from ..utils.metrics import value_from_logits


def _make_4d_attention_mask(att_2d_masks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert a 2D bool mask to a 4D additive mask."""
    att_2d_masks_4d = att_2d_masks[:, None, :, :]
    return torch.where(
        att_2d_masks_4d,
        torch.tensor(0.0, dtype=dtype, device=att_2d_masks.device),
        torch.tensor(-2.3819763e38, dtype=dtype, device=att_2d_masks.device),
    )


class VLMFeatureExtractor(nn.Module):
    """Frozen ValueCriticModel prefix feature extractor.

    Extracts the Gemma3 prefix output (image patches + language tokens) and
    returns a fixed-dimensional feature vector for each sample.
    """

    def __init__(self, value_model: nn.Module):
        super().__init__()
        self.value_model = value_model
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

        prefix_embs, prefix_pad_masks = value_model.embed_prefix(
            images, image_masks, lang_tokens, lang_masks
        )

        prefix_attn = prefix_pad_masks[:, None, :] * prefix_pad_masks[:, :, None]
        dtype = value_model._get_model_dtype()
        prefix_attn_4d = _make_4d_attention_mask(prefix_attn, dtype)
        prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1

        (prefix_out, _), _ = value_model.value_expert.forward(
            attention_mask=prefix_attn_4d,
            position_ids=prefix_pos,
            past_key_values=DynamicCache(),
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        mask = prefix_pad_masks.unsqueeze(-1).to(prefix_out.dtype)
        masked = prefix_out * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        features = summed / counts

        return features.to(torch.float32)

    def forward(self, observation: dict[str, Any]) -> torch.Tensor:
        """Alias for ``extract_prefix_features``."""
        return self.extract_prefix_features(observation)


class OnlineFusionCritic(nn.Module):
    """Value critic wrapper that fuses raw value logits with z/p corrections.

    This module is intended for online inference and end-to-end training. It
    keeps the base ValueCriticModel frozen, uses a frozen phase/progress head to
    extract z/p signals, and applies a learned fusion MLP to correct the raw
    value distribution.
    """

    def __init__(
        self,
        value_model: nn.Module,
        head: nn.Module,
        fusion: LogitFusionMLP,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.value_model = value_model
        self.feature_extractor = VLMFeatureExtractor(value_model)
        self.head = head
        self.fusion = fusion
        self.alpha = alpha

        self.head.eval()
        for param in self.head.parameters():
            param.requires_grad = False

    @property
    def atoms(self) -> torch.Tensor:
        """Return value distribution atoms."""
        return self.value_model.value_head.atoms

    @torch.no_grad()
    def forward(self, observation: dict[str, Any]) -> torch.Tensor:
        """Predict fused scalar value for an observation.

        Args:
            observation: Prepared observation dict.

        Returns:
            Fused scalar values of shape ``[B]``.
        """
        # Raw critic distribution.
        critic_out = self.value_model.predict(observation)
        raw_logits = critic_out.logits

        # Frozen VLM features -> frozen z/p head.
        features = self.feature_extractor.extract_prefix_features(observation)
        head_out = self.head(features)

        # Fusion correction.
        delta = self.fusion(
            raw_logits,
            head_out["phase_probs"],
            head_out["phase_progress"],
            head_out["global_progress"],
        )
        fused_logits = fuse_logits(raw_logits, delta, self.alpha)

        return value_from_logits(fused_logits, self.atoms)

    def predict_distribution(
        self, observation: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return fused value, raw logits, fused logits, and atoms."""
        critic_out = self.value_model.predict(observation)
        raw_logits = critic_out.logits
        features = self.feature_extractor.extract_prefix_features(observation)
        head_out = self.head(features)
        delta = self.fusion(
            raw_logits,
            head_out["phase_probs"],
            head_out["phase_progress"],
            head_out["global_progress"],
        )
        fused_logits = fuse_logits(raw_logits, delta, self.alpha)
        fused_value = value_from_logits(fused_logits, self.atoms)
        return fused_value, raw_logits, fused_logits, self.atoms
