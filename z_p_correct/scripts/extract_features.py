#!/usr/bin/env python
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

"""Pre-extract frozen VLM prefix features + raw critic logits for z_p_correct."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from z_p_correct.data.dataset import build_datasets
from z_p_correct.integration.online_fusion_critic import VLMFeatureExtractor
from rlinf.models.embodiment.value_model import ValueCriticModel

logger = logging.getLogger(__name__)


def _collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate dataset samples; keep transformed as a list for model prep."""
    labels = {
        "episode_index": torch.tensor([b["episode_index"] for b in batch], dtype=torch.long),
        "frame_index": torch.tensor([b["frame_index"] for b in batch], dtype=torch.long),
        "phase": torch.tensor([b["phase"] for b in batch], dtype=torch.long),
        "phase_progress": torch.tensor(
            [b["phase_progress"] for b in batch], dtype=torch.float32
        ),
        "global_progress": torch.tensor(
            [b["global_progress"] for b in batch], dtype=torch.float32
        ),
    }
    transformed = [b["transformed"] for b in batch]
    return {"labels": labels, "transformed": transformed}


@torch.no_grad()
def extract_split(
    value_model: ValueCriticModel,
    feature_extractor: VLMFeatureExtractor,
    data_loader: DataLoader,
    output_path: Path,
) -> None:
    """Extract features and raw logits for one split and cache to disk."""
    value_model.eval()

    all_features: list[torch.Tensor] = []
    all_logits: list[torch.Tensor] = []
    all_labels: list[dict[str, torch.Tensor]] = []

    for batch in tqdm(data_loader, desc=f"Extracting {output_path.stem}"):
        transformed = batch["transformed"]
        observation = value_model._prepare_observation_batch(transformed)

        features = feature_extractor.extract_prefix_features(observation)
        critic_out = value_model(observation)

        all_features.append(features.cpu())
        all_logits.append(critic_out.logits.cpu())
        all_labels.append({k: v.cpu() for k, v in batch["labels"].items()})

    features = torch.cat(all_features, dim=0)
    logits = torch.cat(all_logits, dim=0)
    labels = {k: torch.cat([lb[k] for lb in all_labels], dim=0) for k in all_labels[0]}

    torch.save(
        {
            "features": features,
            "raw_logits": logits,
            "atoms": critic_out.atoms.cpu(),
            **labels,
        },
        output_path,
    )
    logger.info(
        "Saved %s features to %s (shape=%s)",
        output_path.stem,
        output_path,
        tuple(features.shape),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--value_checkpoint", required=True)
    parser.add_argument("--siglip_path", required=True)
    parser.add_argument("--gemma3_path", required=True)
    parser.add_argument(
        "--output_dir",
        default="/workspace/results/z_p_correct/features",
    )
    parser.add_argument("--robot_type", default="libero")
    parser.add_argument("--env_type", default="libero")
    parser.add_argument("--model_type", default="pi05")
    parser.add_argument("--critic_expert_variant", default="gemma_100m")
    parser.add_argument("--num_return_bins", type=int, default=201)
    parser.add_argument("--return_min", type=float, default=-1.0)
    parser.add_argument("--return_max", type=float, default=0.0)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--action_dim", type=int, default=32)
    parser.add_argument("--default_prompt", default=None)
    parser.add_argument("--val_episode_ratio", type=float, default=0.2)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--episode_subset_path", default=None)
    parser.add_argument("--episode_split_path", default=None)
    parser.add_argument("--label_name", default="phase_progress_semantic")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "extract_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    logger.info("Loading ValueCriticModel from %s", args.value_checkpoint)
    value_model = ValueCriticModel.from_checkpoint(
        args.value_checkpoint,
        env_type=args.env_type,
        model_type=args.model_type,
        num_return_bins=args.num_return_bins,
        return_min=args.return_min,
        return_max=args.return_max,
        critic_expert_variant=args.critic_expert_variant,
        tokenizer_path=args.tokenizer_path,
        siglip_path=args.siglip_path,
        gemma3_path=args.gemma3_path,
    )
    feature_extractor = VLMFeatureExtractor(value_model)
    logger.info(
        "Feature extractor ready. feature_dim=%d device=%s",
        feature_extractor.feature_dim,
        feature_extractor.device,
    )

    logger.info("Building datasets from %s", args.dataset_path)
    train_ds, val_ds = build_datasets(
        dataset_path=args.dataset_path,
        robot_type=args.robot_type,
        model_type=args.model_type,
        action_dim=args.action_dim,
        default_prompt=args.default_prompt,
        val_episode_ratio=args.val_episode_ratio,
        max_episodes=args.max_episodes,
        episode_subset_path=args.episode_subset_path,
        episode_split_path=args.episode_split_path,
        label_name=args.label_name,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate_fn,
        pin_memory=True,
    )

    extract_split(value_model, feature_extractor, train_loader, output_dir / "train.pt")
    extract_split(value_model, feature_extractor, val_loader, output_dir / "val.pt")

    logger.info("Feature extraction complete. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
