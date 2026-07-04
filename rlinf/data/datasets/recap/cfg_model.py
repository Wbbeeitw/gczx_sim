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

"""CFG-model dataset and dataloader helpers for ReCap training."""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Iterator

import torch

from rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model import (
    Observation as CFGObservation,
)

from .common import BaseDataLoaderImpl, ReCapMixtureDataset

logger = logging.getLogger(__name__)


class AdvantagePreservingDataset:
    """Wrapper to preserve advantage through the OpenPI transform pipeline."""

    def __init__(
        self,
        base_dataset: Any,
        transformed_dataset: Any,
        advantages_lookup: dict[tuple[int, int], Any] | None = None,
    ):
        self._transformed_dataset = transformed_dataset
        self._metadata_by_index = self._build_metadata_index(
            base_dataset, advantages_lookup
        )
        self._base_dataset = base_dataset if self._metadata_by_index is None else None

    @staticmethod
    def _get_hf_dataset(dataset: Any) -> Any:
        current = dataset
        while current is not None:
            if hasattr(current, "hf_dataset"):
                return current.hf_dataset
            if hasattr(current, "_dataset"):
                current = current._dataset
            else:
                return None
        return None

    def _build_metadata_index(
        self,
        base_dataset: Any,
        advantages_lookup: dict[tuple[int, int], Any] | None,
    ) -> dict[int, dict[str, Any]] | None:
        hf_dataset = self._get_hf_dataset(base_dataset)
        if hf_dataset is None:
            logger.warning(
                "Cannot access underlying HF dataset, "
                "falling back to per-sample advantage loading (slower)."
            )
            return None

        if advantages_lookup is not None:
            ep_indices = hf_dataset["episode_index"]
            frame_indices = hf_dataset["frame_index"]
            metadata_by_index: dict[int, dict[str, Any]] = {}
            missing_keys = []
            for i in range(len(hf_dataset)):
                key = (int(ep_indices[i]), int(frame_indices[i]))
                if key in advantages_lookup:
                    record = advantages_lookup[key]
                    if isinstance(record, dict):
                        metadata_by_index[i] = dict(record)
                    else:
                        metadata_by_index[i] = {"advantage": bool(record)}
                else:
                    missing_keys.append(key)
            if missing_keys:
                raise ValueError(
                    f"[AdvantagePreservingDataset] {len(missing_keys)} samples not found "
                    f"in advantages lookup (first 5: {missing_keys[:5]}). "
                    f"The advantages parquet does not match this dataset. "
                    f"Re-run compute_advantages.py."
                )
            return metadata_by_index

        if "advantage" in hf_dataset.column_names:
            advantages = hf_dataset["advantage"]
            return {i: {"advantage": bool(v)} for i, v in enumerate(advantages)}

        raise ValueError(
            "[AdvantagePreservingDataset] No advantage data found: "
            "advantages_lookup is None, and 'advantage' column not in dataset. "
            "Run compute_advantages.py first."
        )

    def __len__(self) -> int:
        return len(self._transformed_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self._transformed_dataset[idx]

        if self._metadata_by_index is not None:
            if idx not in self._metadata_by_index:
                raise KeyError(
                    f"[AdvantagePreservingDataset] Index {idx} not found in advantage index. "
                    f"Dataset size: {len(self._transformed_dataset)}, "
                    f"advantage index size: {len(self._metadata_by_index)}."
                )
            sample.update(self._metadata_by_index[idx])
        else:
            base_sample = self._base_dataset[idx]
            if "advantage" not in base_sample:
                raise KeyError(
                    f"[AdvantagePreservingDataset] 'advantage' key not found in base_sample "
                    f"at index {idx}. Run compute_advantages.py first."
                )
            advantage = base_sample["advantage"]
            if isinstance(advantage, torch.Tensor):
                advantage = bool(advantage.item())
            sample["advantage"] = advantage

        return sample


class CFGDataLoaderImpl(BaseDataLoaderImpl):
    """DataLoader wrapper that yields CFG training tuples."""

    def __iter__(self) -> Iterator[tuple[Any, Any, dict[str, torch.Tensor]]]:
        for batch in self._data_loader:
            observation = CFGObservation.from_dict(batch)
            actions = batch["actions"]

            advantage = batch["advantage"]
            if not isinstance(advantage, torch.Tensor):
                advantage = torch.tensor(advantage, dtype=torch.bool)
            metadata: dict[str, torch.Tensor] = {
                "advantage": advantage.to(dtype=torch.bool),
            }
            for key, dtype in (
                ("cfg_quality_label", torch.long),
                ("cfg_percentile_rank", torch.float32),
                ("cfg_loss_weight", torch.float32),
            ):
                if key in batch:
                    value = batch[key]
                    if not isinstance(value, torch.Tensor):
                        value = torch.tensor(value, dtype=dtype)
                    metadata[key] = value.to(dtype=dtype)

            yield observation, actions, metadata


@dataclasses.dataclass(frozen=True)
class TokenizePromptWithGuidance:
    """Tokenize both original prompt and guidance prompts for CFG models."""

    tokenizer: Any
    discrete_state_input: bool = False

    def __call__(self, data: dict) -> dict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)

        positive_prompt = f"{prompt}\nAdvantage: positive"
        neutral_prompt = f"{prompt}\nAdvantage: neutral"
        negative_prompt = f"{prompt}\nAdvantage: negative"

        positive_tokens, positive_masks = self.tokenizer.tokenize(
            positive_prompt, state
        )
        neutral_tokens, neutral_masks = self.tokenizer.tokenize(neutral_prompt, state)
        negative_tokens, negative_masks = self.tokenizer.tokenize(
            negative_prompt, state
        )

        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_masks,
            "tokenized_positive_guidance_prompt": positive_tokens,
            "tokenized_positive_guidance_prompt_mask": positive_masks,
            "tokenized_neutral_guidance_prompt": neutral_tokens,
            "tokenized_neutral_guidance_prompt_mask": neutral_masks,
            "tokenized_negative_guidance_prompt": negative_tokens,
            "tokenized_negative_guidance_prompt_mask": negative_masks,
        }


class CfgMixtureDataset(ReCapMixtureDataset):
    """Mixture of multiple datasets with weighted sampling for CFG-RL training."""

    mixture_name = "CfgMixtureDataset"
