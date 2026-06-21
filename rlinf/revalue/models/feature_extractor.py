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

"""Frozen VLM prefix feature extractor for ValueCriticModel."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _make_4d_attention_mask(
    att_2d_masks: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert a 2D bool attention mask to a 4D additive mask."""
    att_2d_masks_4d = att_2d_masks[:, None, :, :]
    return torch.where(
        att_2d_masks_4d,
        torch.tensor(0.0, dtype=dtype, device=att_2d_masks.device),
        torch.tensor(-2.3819763e38, dtype=dtype, device=att_2d_masks.device),
    )


class VLMFeatureExtractor(nn.Module):
    """Frozen ValueCriticModel VLM prefix extractor.

    It emits a mean-pooled hidden vector over valid prefix tokens. The wrapped
    value model is set to eval mode and all parameters are frozen.
    """

    def __init__(self, value_model: nn.Module) -> None:
        super().__init__()
        self.value_model = value_model
        self.value_model.eval()
        for param in self.value_model.parameters():
            param.requires_grad = False

    @property
    def feature_dim(self) -> int:
        return int(self.value_model.value_expert.gemma3.config.hidden_size)

    @property
    def device(self) -> torch.device:
        return next(self.value_model.parameters()).device

    @torch.no_grad()
    def extract_prefix_features(self, observation: dict[str, Any]) -> torch.Tensor:
        """Extract one feature vector per observation row."""
        from transformers.cache_utils import DynamicCache

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
            images,
            image_masks,
            lang_tokens,
            lang_masks,
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
        features = (prefix_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return features.to(torch.float32)

    def forward(self, observation: dict[str, Any]) -> torch.Tensor:
        """Alias for :meth:`extract_prefix_features`."""
        return self.extract_prefix_features(observation)
