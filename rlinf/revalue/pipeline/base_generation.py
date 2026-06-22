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

"""Base return and raw critic advantage generation for Revalue."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlinf.revalue.data.advantage_table import resolve_advantage_path


@dataclass(frozen=True)
class ReturnGenerationConfig:
    """Options for computing Revalue return sidecars."""

    dataset_path: str
    tag: str
    dataset_type: str = "rollout"
    gamma: float = 1.0
    failure_reward: float = -300.0
    num_workers: int = 64


@dataclass(frozen=True)
class BaseAdvantageGenerationConfig:
    """Options for computing raw critic advantages with saved logits."""

    dataset_path: str
    tag: str
    value_checkpoint: str
    returns_tag: str | None = None
    episode_subset_path: str | None = None
    dataset_type: str = "rollout"
    robot_type: str = "libero"
    model_type: str = "pi05"
    critic_expert_variant: str = "gemma_1m"
    tokenizer_path: str | None = None
    siglip_path: str | None = None
    gemma3_path: str | None = None
    num_bins: int = 201
    value_min: float = -1.0
    value_max: float = 0.0
    return_min: float = -700.0
    return_max: float = 0.0
    lookahead_step: int = 10
    gamma: float = 1.0
    positive_quantile: float = 0.3
    discount_next_value: bool = True
    batch_size: int = 256
    num_workers_per_gpu: int = 12
    prefetch_factor: int = 2
    flush_interval: int = 256
    max_samples: int | None = None


def compute_revalue_returns(cfg: ReturnGenerationConfig) -> Path:
    """Compute `meta/returns_<tag>.parquet` through the Revalue API."""
    from examples.recap.process.compute_returns import process_dataset

    process_dataset(
        dataset_path=Path(cfg.dataset_path),
        output_path=None,
        dataset_type=cfg.dataset_type,
        gamma=cfg.gamma,
        failure_reward=cfg.failure_reward,
        num_workers=cfg.num_workers,
        tag=cfg.tag,
    )
    return Path(cfg.dataset_path) / "meta" / f"returns_{cfg.tag}.parquet"


def _base_advantage_cfg(cfg: BaseAdvantageGenerationConfig) -> Any:
    """Build the config consumed by the existing value inference engine."""
    from omegaconf import OmegaConf

    returns_tag = cfg.returns_tag if cfg.returns_tag is not None else cfg.tag
    return OmegaConf.create(
        {
            "advantage": {
                "value_checkpoint": cfg.value_checkpoint,
                "batch_size": cfg.batch_size,
                "flush_interval": cfg.flush_interval,
                "num_dataloader_workers_per_gpu": cfg.num_workers_per_gpu,
                "prefetch_factor": cfg.prefetch_factor,
                "discount_next_value": cfg.discount_next_value,
                "positive_quantile": cfg.positive_quantile,
                "tag": cfg.tag,
                "returns_tag": returns_tag,
                "episode_subset_path": cfg.episode_subset_path,
                "max_samples": cfg.max_samples,
                "save_value_distribution": True,
                "model": {
                    "critic_expert_variant": cfg.critic_expert_variant,
                    "tokenizer_path": cfg.tokenizer_path,
                    "siglip_path": cfg.siglip_path,
                    "gemma3_path": cfg.gemma3_path,
                    "num_bins": cfg.num_bins,
                    "v_min": cfg.value_min,
                    "v_max": cfg.value_max,
                },
            },
            "data": {
                "model_type": cfg.model_type,
                "robot_type": cfg.robot_type,
                "train_data_paths": [
                    {
                        "dataset_path": cfg.dataset_path,
                        "robot_type": cfg.robot_type,
                        "type": cfg.dataset_type,
                        "weight": 1.0,
                    }
                ],
                "advantage_lookahead_step": cfg.lookahead_step,
                "gamma": cfg.gamma,
                "return_min": cfg.return_min,
                "return_max": cfg.return_max,
            },
            "distributed": {
                "enabled": True,
                "backend": "nccl",
                "timeout": 3600,
            },
        }
    )


def compute_revalue_base_advantages(cfg: BaseAdvantageGenerationConfig) -> Path:
    """Compute `meta/advantages_<tag>.parquet` with raw value logits saved."""
    from examples.recap.process.compute_advantages import main as compute_main

    target = getattr(compute_main, "__wrapped__", compute_main)
    target(_base_advantage_cfg(cfg))
    return resolve_advantage_path(cfg.dataset_path, cfg.tag)
