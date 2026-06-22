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

"""Configuration dataclasses for the Revalue CLI."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RevalueDataConfig:
    """Dataset and split settings."""

    dataset_path: str = "/home/enine/rlinf_workspace/datasets/example"
    robot_type: str = "libero"
    env_type: str = "libero"
    model_type: str = "pi05"
    action_dim: int = 32
    default_prompt: str | None = None
    label_name: str = "phase_progress_semantic"
    val_episode_ratio: float = 0.2
    seed: int = 42
    max_episodes: int | None = None
    episode_subset_path: str | None = None
    episode_split_path: str | None = None


@dataclass
class RevalueManifestConfig:
    """Episode sampling and split manifest settings."""

    output_path: str | None = None
    num_episodes: int | None = None
    success_ratio: float = 0.5
    val_episode_ratio: float = 0.2
    test_episode_ratio: float = 0.0
    success_phase: int = 4
    overwrite: bool = False


@dataclass
class RevalueValueConfig:
    """Value critic checkpoint settings."""

    checkpoint: str = "/home/enine/rlinf_workspace/results/value/checkpoint"
    siglip_path: str = "/home/enine/rlinf_workspace/models/siglip2-so400m-patch14-224"
    gemma3_path: str = "/home/enine/rlinf_workspace/models/gemma-3-270m"
    tokenizer_path: str | None = None
    critic_expert_variant: str = "gemma_1m"
    num_bins: int = 201
    v_min: float = -1.0
    v_max: float = 0.0


@dataclass
class RevalueReturnsConfig:
    """Raw return range used to normalize targets into value space."""

    global_min: float = -700.0
    global_max: float = 0.0
    tag: str | None = None
    dataset_type: str = "rollout"
    failure_reward: float = -300.0
    num_workers: int = 64
    compute: bool = True


@dataclass
class RevalueBaseConfig:
    """Raw critic base advantage generation settings."""

    tag: str = "base"
    returns_tag: str | None = None
    compute_returns: bool = True
    compute_advantages: bool = True
    batch_size: int = 256
    num_workers_per_gpu: int = 12
    prefetch_factor: int = 2
    flush_interval: int = 256
    max_samples: int | None = None


@dataclass
class RevalueOutputConfig:
    """Output locations outside the source tree."""

    root: str = "/home/enine/rlinf_workspace/results/revalue/run"
    features_dir: str | None = None
    zp_dir: str | None = None
    fusion_dir: str | None = None
    predictions_path: str | None = None
    comparison_path: str | None = None


@dataclass
class RevalueZPConfig:
    """Stage-1 z/p head settings."""

    hidden_dim: int = 256
    dropout: float = 0.1
    trunk_depth: int = 1
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    max_epochs: int = 100
    early_stop_patience: int = 10


@dataclass
class RevalueFusionConfig:
    """Stage-2 fusion MLP settings."""

    hidden_dim: int = 256
    depth: int = 2
    dropout: float = 0.1
    alpha: float = 1.0
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    max_epochs: int = 100
    early_stop_patience: int = 10


@dataclass
class RevalueTrainConfig:
    """Common training options."""

    batch_size: int = 256
    extract_batch_size: int = 16
    num_workers: int = 0
    device: str = "cuda"
    seed: int = 42


@dataclass
class RevalueRecapConfig:
    """ReCap source/output tag settings."""

    source_tag: str | None = None
    source_advantages_path: str | None = None
    output_tag: str = "zp_fused"
    lookahead_step: int = 10
    gamma: float = 1.0
    positive_quantile: float = 0.3
    discount_next_value: bool = True
    export_split: str | None = "train"


@dataclass
class RevalueConfig:
    """Top-level Revalue CLI config."""

    method: str = "shared_mlp_fusion"
    stage: str = "all"
    data: RevalueDataConfig = field(default_factory=RevalueDataConfig)
    manifest: RevalueManifestConfig = field(default_factory=RevalueManifestConfig)
    value: RevalueValueConfig = field(default_factory=RevalueValueConfig)
    returns: RevalueReturnsConfig = field(default_factory=RevalueReturnsConfig)
    base: RevalueBaseConfig = field(default_factory=RevalueBaseConfig)
    output: RevalueOutputConfig = field(default_factory=RevalueOutputConfig)
    zp: RevalueZPConfig = field(default_factory=RevalueZPConfig)
    fusion: RevalueFusionConfig = field(default_factory=RevalueFusionConfig)
    train: RevalueTrainConfig = field(default_factory=RevalueTrainConfig)
    recap: RevalueRecapConfig = field(default_factory=RevalueRecapConfig)
