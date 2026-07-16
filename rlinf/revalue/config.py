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
    num_phases: int = 5
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
    source_features_dir: str | None = None
    zp_dir: str | None = None
    fusion_dir: str | None = None
    predictions_path: str | None = None
    comparison_path: str | None = None


@dataclass
class RevalueZPConfig:
    """Stage-1 z/p head settings."""

    head_type: str = "temporal_z_mlp_p"
    hidden_dim: int = 256
    dropout: float = 0.1
    trunk_depth: int = 1
    window_size: int = 5
    num_layers: int = 2
    num_heads: int = 4
    ffn_dim: int = 512
    stage_embedding_dim: int = 32
    progress_hidden_dim: int = 256
    progress_depth: int = 2
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    max_epochs: int = 100
    early_stop_patience: int = 10
    phase_loss_weight: float = 1.0
    progress_loss_weight: float = 1.0
    global_progress_loss_weight: float = 0.25
    label_smoothing: float = 0.05
    use_class_weights: bool = True
    progress_beta: float = 0.05
    max_grad_norm: float | None = None
    stage_only_epochs: int = 4
    gt_stage_prior_epochs: int = 6
    stage_prior_ramp_epochs: int = 10
    max_pred_stage_prior_weight: float = 0.7


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
class RevalueCFGTrainConfig:
    """Downstream CFG training settings driven by exported advantages."""

    enabled: bool = False
    config_name: str = "libero_cfg_openpi"
    advantage_tag: str | None = None
    dataset_path: str | None = None
    base_model_path: str | None = None
    init_checkpoint_path: str | None = None
    episode_split_path: str | None = None
    episode_split_name: str = "train"
    experiment_name: str = "revalue_cfg_train"
    log_dir: str | None = None
    model_type: str = "cfg_model"
    openpi_config_name: str = "pi05_libero"
    strategy: str = "binary"
    guidance_type: str = "positive"
    positive_only_conditional: bool = True
    unconditional_prob: float = 0.1
    negative_guidance_scale: float = 0.0
    csa_positive_quantile: float = 0.30
    csa_bottom_quantile: float = 0.15
    csa_positive_prompt_prob: float = 0.85
    csa_bottom_negative_prob: float = 0.50
    csa_weight_lambda: float = 0.20
    positive_residual_alpha: float = 0.50
    max_epochs: int = -1
    max_steps: int = 5000
    save_interval: int = 5000
    val_check_interval: int = -1
    total_training_steps: int | None = None
    lr_warmup_steps: int | None = None
    global_batch_size: int = 256
    micro_batch_size: int = 16
    data_type: str = "rollout"
    dataset_weight: float = 1.0
    python_bin: str | None = None
    extra_overrides: list[str] = field(default_factory=list)


@dataclass
class RevaluePolicyEvalConfig:
    """Embodied evaluation settings for a policy checkpoint."""

    enabled: bool = False
    config_name: str = "libero_10_pi05_sft_eval"
    model_path: str | None = None
    model_type: str = "openpi"
    checkpoint_path: str | None = None
    experiment_name: str = "revalue_policy_eval"
    log_dir: str | None = None
    eval_rollout_epoch: int = 10
    total_num_envs: int = 5
    save_video: bool = False
    task_suite_name: str | None = None
    task_id_filter: list[int] = field(default_factory=list)
    openpi_config_name: str = "pi05_libero"
    guidance_type: str = "positive"
    positive_only_conditional: bool = True
    negative_guidance_scale: float = 0.0
    python_bin: str | None = None
    extra_overrides: list[str] = field(default_factory=list)


@dataclass
class RevalueRolloutCollectConfig:
    """LIBERO rollout collection settings for a specified policy."""

    enabled: bool = False
    output_dir: str | None = None
    model_path: str | None = None
    checkpoint_path: str | None = None
    model_type: str = "openpi"
    openpi_config_name: str = "pi05_libero"
    task_suite_name: str = "libero_10"
    task_id: int = 0
    num_episodes: int = 64
    noise_scale: float = 0.0
    noise_clip: float = 0.3
    action_chunk: int = 5
    num_steps: int = 5
    num_steps_wait: int = 10
    seed: int = 42
    gpu_id: int = 0
    fps: int = 10
    overwrite: bool = False
    failure_reward: float | None = None
    semantic_trace: bool = False
    semantic_trace_task: str = "task1"
    semantic_trace_output_name: str = "semantic_trace_task1"
    python_bin: str | None = None


@dataclass
class RevalueExportViewConfig:
    """Child-dataset view export settings (re-indexed fused advantages)."""

    source_advantages_path: str | None = None
    predictions_path: str | None = None
    child_dataset_path: str | None = None
    output_tag: str = "fused_child"
    source_episode_start: int = 0
    source_episode_end: int | None = None
    child_episode_offset: int = 0
    lookahead_step: int = 10
    gamma: float = 1.0
    positive_quantile: float = 0.3
    discount_next_value: bool = True
    expected_episodes: int | None = None
    report_path: str | None = None


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
    cfg_train: RevalueCFGTrainConfig = field(default_factory=RevalueCFGTrainConfig)
    policy_eval: RevaluePolicyEvalConfig = field(default_factory=RevaluePolicyEvalConfig)
    rollout_collect: RevalueRolloutCollectConfig = field(
        default_factory=RevalueRolloutCollectConfig
    )
    export_view: RevalueExportViewConfig = field(
        default_factory=RevalueExportViewConfig
    )
