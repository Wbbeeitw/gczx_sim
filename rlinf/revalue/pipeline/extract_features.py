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

"""Feature extraction stage for Revalue."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from rlinf.revalue.data.phase_dataset import build_phase_datasets
from rlinf.revalue.models.feature_extractor import VLMFeatureExtractor

logger = logging.getLogger(__name__)


@dataclass
class FeatureExtractionConfig:
    """Arguments for extracting frozen VLM features."""

    dataset_path: str
    value_checkpoint: str
    output_dir: str
    siglip_path: str
    gemma3_path: str
    tokenizer_path: str | None = None
    robot_type: str = "libero"
    env_type: str = "libero"
    model_type: str = "pi05"
    critic_expert_variant: str = "gemma_1m"
    num_return_bins: int = 201
    value_min: float = -1.0
    value_max: float = 0.0
    action_dim: int = 32
    default_prompt: str | None = None
    val_episode_ratio: float = 0.2
    label_name: str = "phase_progress_semantic"
    seed: int = 42
    max_episodes: int | None = None
    episode_subset_path: str | None = None
    episode_split_path: str | None = None
    batch_size: int = 16
    num_workers: int = 0
    device: str = "cuda"


def _collate_phase_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    labels = {
        "episode_index": torch.tensor(
            [row["episode_index"] for row in batch], dtype=torch.long
        ),
        "frame_index": torch.tensor(
            [row["frame_index"] for row in batch], dtype=torch.long
        ),
        "phase": torch.tensor([row["phase"] for row in batch], dtype=torch.long),
        "phase_progress": torch.tensor(
            [row["phase_progress"] for row in batch], dtype=torch.float32
        ),
        "global_progress": torch.tensor(
            [row["global_progress"] for row in batch], dtype=torch.float32
        ),
    }
    return {
        "labels": labels,
        "transformed": [row["transformed"] for row in batch],
    }


@torch.no_grad()
def _extract_split(
    *,
    value_model,
    feature_extractor: VLMFeatureExtractor,
    data_loader: DataLoader,
    output_path: Path,
) -> None:
    value_model.eval()
    all_features: list[torch.Tensor] = []
    all_logits: list[torch.Tensor] = []
    all_values: list[torch.Tensor] = []
    all_labels: list[dict[str, torch.Tensor]] = []
    atoms = None

    for batch in tqdm(data_loader, desc=f"extract_{output_path.stem}"):
        observation = value_model._prepare_observation_batch(batch["transformed"])
        features = feature_extractor.extract_prefix_features(observation)
        critic_out = value_model.predict(observation)
        all_features.append(features.cpu())
        all_logits.append(critic_out.logits.cpu())
        all_values.append(critic_out.predicted_values.cpu())
        all_labels.append({key: value.cpu() for key, value in batch["labels"].items()})
        atoms = critic_out.atoms.detach().cpu()

    if not all_features:
        raise ValueError(f"No samples extracted for {output_path}")

    labels = {
        key: torch.cat([label[key] for label in all_labels], dim=0)
        for key in all_labels[0]
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": torch.cat(all_features, dim=0),
            "raw_logits": torch.cat(all_logits, dim=0),
            "raw_value": torch.cat(all_values, dim=0),
            "atoms": atoms,
            **labels,
        },
        output_path,
    )
    logger.info("saved feature cache %s", output_path)


def extract_features(cfg: FeatureExtractionConfig) -> None:
    """Extract train/val feature caches."""
    from rlinf.models.embodiment.value_model import ValueCriticModel

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as file:
        json.dump(asdict(cfg), file, indent=2)

    value_model = ValueCriticModel.from_checkpoint(
        cfg.value_checkpoint,
        device=cfg.device,
        env_type=cfg.env_type,
        model_type=cfg.model_type,
        num_return_bins=cfg.num_return_bins,
        return_min=cfg.value_min,
        return_max=cfg.value_max,
        critic_expert_variant=cfg.critic_expert_variant,
        tokenizer_path=cfg.tokenizer_path,
        siglip_path=cfg.siglip_path,
        gemma3_path=cfg.gemma3_path,
    )
    feature_extractor = VLMFeatureExtractor(value_model)

    common_dataset_kwargs = {
        "robot_type": cfg.robot_type,
        "model_type": cfg.model_type,
        "action_dim": cfg.action_dim,
        "default_prompt": cfg.default_prompt,
        "val_episode_ratio": cfg.val_episode_ratio,
        "label_name": cfg.label_name,
        "seed": cfg.seed,
        "max_episodes": cfg.max_episodes,
        "episode_subset_path": cfg.episode_subset_path,
        "episode_split_path": cfg.episode_split_path,
    }
    train_dataset, val_dataset = build_phase_datasets(
        cfg.dataset_path,
        **common_dataset_kwargs,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=_collate_phase_batch,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=_collate_phase_batch,
        pin_memory=True,
    )

    _extract_split(
        value_model=value_model,
        feature_extractor=feature_extractor,
        data_loader=train_loader,
        output_path=output_dir / "train.pt",
    )
    _extract_split(
        value_model=value_model,
        feature_extractor=feature_extractor,
        data_loader=val_loader,
        output_path=output_dir / "val.pt",
    )
