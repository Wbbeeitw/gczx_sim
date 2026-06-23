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

"""Top-level orchestration for Revalue."""

from __future__ import annotations

import logging
from pathlib import Path

from rlinf.revalue.config import RevalueConfig
from rlinf.revalue.constants import (
    METHOD_BASE,
    METHOD_SHARED_MLP_FUSION,
    STAGE_ALL,
    STAGE_BUILD_BASE,
    STAGE_COLLECT_ROLLOUTS,
    STAGE_COMPARE_RETURNS,
    STAGE_EVAL_POLICY,
    STAGE_EXPORT,
    STAGE_EXTRACT_FEATURES,
    STAGE_PREPARE_DATA,
    STAGE_PREDICT,
    STAGE_RESPLIT_FEATURES,
    STAGE_TRAIN_CFG,
    STAGE_TRAIN_FUSION,
    STAGE_TRAIN_ZP,
    SUPPORTED_METHODS,
    SUPPORTED_STAGES,
)
from rlinf.revalue.data import resolve_advantage_path, validate_fusion_advantages
from rlinf.revalue.data.advantage_table import read_advantages
from rlinf.revalue.data.episode_manifest import (
    EpisodeManifestConfig,
    build_episode_manifest,
)
from rlinf.revalue.data.feature_resplit import (
    FeatureResplitConfig,
    resplit_feature_cache,
)
from rlinf.revalue.evaluation import (
    ReturnComparisonConfig,
    compare_return_predictions,
)
from rlinf.revalue.pipeline.base_generation import (
    BaseAdvantageGenerationConfig,
    ReturnGenerationConfig,
    compute_revalue_base_advantages,
    compute_revalue_returns,
)
from rlinf.revalue.pipeline.embodied import (
    DownstreamCFGTrainingConfig,
    LiberoRolloutCollectionConfig,
    PolicyEvaluationConfig,
    collect_libero_rollouts,
    evaluate_policy_checkpoint,
    resolve_latest_trained_checkpoint,
    train_cfg_from_advantages,
)
from rlinf.revalue.pipeline.extract_features import (
    FeatureExtractionConfig,
    extract_features,
)
from rlinf.revalue.pipeline.predict import PredictionConfig, predict_fused_values
from rlinf.revalue.pipeline.train import (
    FusionTrainingConfig,
    ZPTrainingConfig,
    train_fusion,
    train_zp_head,
)
from rlinf.revalue.recap.export import ExportConfig, export_fused_advantages

logger = logging.getLogger(__name__)


def _output_paths(cfg: RevalueConfig) -> dict[str, Path]:
    root = Path(cfg.output.root)
    return {
        "root": root,
        "features": (
            Path(cfg.output.features_dir)
            if cfg.output.features_dir
            else root / "features"
        ),
        "zp": Path(cfg.output.zp_dir) if cfg.output.zp_dir else root / "zp_head",
        "fusion": (
            Path(cfg.output.fusion_dir)
            if cfg.output.fusion_dir
            else root / "fusion"
        ),
        "predictions": (
            Path(cfg.output.predictions_path)
            if cfg.output.predictions_path
            else root / "predictions.parquet"
        ),
        "comparison": (
            Path(cfg.output.comparison_path)
            if cfg.output.comparison_path
            else root / "return_compare.json"
        ),
        "manifest": (
            Path(cfg.manifest.output_path)
            if cfg.manifest.output_path
            else root / "episode_manifest.json"
        ),
        "train_cfg": root / "downstream_train",
        "policy_eval": root / "policy_eval",
        "rollout_collect": root / "collected_rollouts",
    }


def _source_advantage_path(cfg: RevalueConfig) -> Path:
    if cfg.recap.source_advantages_path:
        return Path(cfg.recap.source_advantages_path)
    source_tag = cfg.recap.source_tag or cfg.base.tag
    return resolve_advantage_path(cfg.data.dataset_path, source_tag)


def _manifest_path(cfg: RevalueConfig, paths: dict[str, Path]) -> Path | None:
    if cfg.data.episode_split_path:
        return Path(cfg.data.episode_split_path)
    if cfg.data.episode_subset_path:
        return Path(cfg.data.episode_subset_path)
    return paths["manifest"]


def _feature_subset_path(cfg: RevalueConfig) -> Path | None:
    if cfg.data.episode_split_path:
        return None
    if cfg.data.episode_subset_path:
        return Path(cfg.data.episode_subset_path)
    return None


def _feature_split_path(cfg: RevalueConfig, paths: dict[str, Path]) -> Path | None:
    if cfg.data.episode_split_path:
        return Path(cfg.data.episode_split_path)
    if cfg.data.episode_subset_path:
        return None
    return paths["manifest"]


def _validate_base(cfg: RevalueConfig) -> Path:
    path = _source_advantage_path(cfg)
    df = read_advantages(path)
    logger.info("base advantages available: %s rows=%d", path, len(df))
    return path


def _stages_for_request(stage: str) -> list[str]:
    if stage == STAGE_ALL:
        if_requested = [
            STAGE_PREPARE_DATA,
            STAGE_BUILD_BASE,
        ]
        return if_requested + [
            STAGE_EXTRACT_FEATURES,
            STAGE_TRAIN_ZP,
            STAGE_TRAIN_FUSION,
            STAGE_PREDICT,
            STAGE_EXPORT,
            STAGE_COMPARE_RETURNS,
        ]
    return [stage]


def _stages_for_method(cfg: RevalueConfig) -> list[str]:
    if cfg.stage != STAGE_ALL:
        return _stages_for_request(cfg.stage)
    if cfg.method == METHOD_BASE:
        stages = [
            STAGE_PREPARE_DATA,
            STAGE_BUILD_BASE,
        ]
    else:
        stages = _stages_for_request(cfg.stage)
    if cfg.cfg_train.enabled:
        stages.append(STAGE_TRAIN_CFG)
    if cfg.policy_eval.enabled:
        stages.append(STAGE_EVAL_POLICY)
    if cfg.rollout_collect.enabled:
        stages.append(STAGE_COLLECT_ROLLOUTS)
    return stages


def _run_prepare_data(cfg: RevalueConfig, paths: dict[str, Path]) -> Path:
    existing_manifest = _manifest_path(cfg, paths)
    if cfg.data.episode_split_path or cfg.data.episode_subset_path:
        if existing_manifest is None:
            raise ValueError("Expected an existing episode manifest path.")
        logger.info("using existing episode manifest: %s", existing_manifest)
        return existing_manifest

    manifest_path = build_episode_manifest(
        EpisodeManifestConfig(
            dataset_path=cfg.data.dataset_path,
            output_path=str(paths["manifest"]),
            label_name=cfg.data.label_name,
            num_episodes=cfg.manifest.num_episodes or cfg.data.max_episodes,
            success_ratio=cfg.manifest.success_ratio,
            val_episode_ratio=cfg.manifest.val_episode_ratio,
            test_episode_ratio=cfg.manifest.test_episode_ratio,
            seed=cfg.data.seed,
            success_phase=cfg.manifest.success_phase,
            overwrite=cfg.manifest.overwrite,
        )
    )
    logger.info("prepared Revalue episode manifest: %s", manifest_path)
    return manifest_path


def _run_build_base(cfg: RevalueConfig, paths: dict[str, Path]) -> Path:
    tag = cfg.base.tag
    returns_tag = cfg.base.returns_tag or cfg.returns.tag or tag
    manifest_path = _manifest_path(cfg, paths)
    if cfg.base.compute_returns and cfg.returns.compute:
        returns_path = compute_revalue_returns(
            ReturnGenerationConfig(
                dataset_path=cfg.data.dataset_path,
                tag=returns_tag,
                dataset_type=cfg.returns.dataset_type,
                gamma=cfg.recap.gamma,
                failure_reward=cfg.returns.failure_reward,
                num_workers=cfg.returns.num_workers,
            )
        )
        logger.info("computed Revalue returns: %s", returns_path)

    if cfg.base.compute_advantages:
        advantages_path = compute_revalue_base_advantages(
            BaseAdvantageGenerationConfig(
                dataset_path=cfg.data.dataset_path,
                tag=tag,
                value_checkpoint=cfg.value.checkpoint,
                returns_tag=returns_tag,
                episode_subset_path=str(manifest_path) if manifest_path else None,
                dataset_type=cfg.returns.dataset_type,
                robot_type=cfg.data.robot_type,
                model_type=cfg.data.model_type,
                critic_expert_variant=cfg.value.critic_expert_variant,
                tokenizer_path=cfg.value.tokenizer_path,
                siglip_path=cfg.value.siglip_path,
                gemma3_path=cfg.value.gemma3_path,
                num_bins=cfg.value.num_bins,
                value_min=cfg.value.v_min,
                value_max=cfg.value.v_max,
                return_min=cfg.returns.global_min,
                return_max=cfg.returns.global_max,
                lookahead_step=cfg.recap.lookahead_step,
                gamma=cfg.recap.gamma,
                positive_quantile=cfg.recap.positive_quantile,
                discount_next_value=cfg.recap.discount_next_value,
                batch_size=cfg.base.batch_size,
                num_workers_per_gpu=cfg.base.num_workers_per_gpu,
                prefetch_factor=cfg.base.prefetch_factor,
                flush_interval=cfg.base.flush_interval,
                max_samples=cfg.base.max_samples,
            )
        )
        logger.info("computed Revalue base advantages: %s", advantages_path)
        return advantages_path

    return resolve_advantage_path(cfg.data.dataset_path, tag)


def _run_resplit_features(cfg: RevalueConfig, paths: dict[str, Path]) -> None:
    if not cfg.output.source_features_dir:
        raise ValueError("output.source_features_dir is required for resplit_features")
    manifest = resplit_feature_cache(
        FeatureResplitConfig(
            source_dir=cfg.output.source_features_dir,
            output_dir=str(paths["features"]),
            val_episode_ratio=cfg.manifest.val_episode_ratio,
            test_episode_ratio=cfg.manifest.test_episode_ratio,
            seed=cfg.data.seed,
            manifest_path=str(paths["manifest"]),
            overwrite=cfg.manifest.overwrite,
        )
    )
    logger.info(
        "resplit Revalue features: source=%s output=%s rows=%s",
        cfg.output.source_features_dir,
        paths["features"],
        manifest.get("split_rows"),
    )


def _advantage_tag_for_cfg_train(cfg: RevalueConfig) -> str:
    if cfg.cfg_train.advantage_tag:
        return cfg.cfg_train.advantage_tag
    if cfg.method == METHOD_BASE:
        return cfg.base.tag
    return cfg.recap.output_tag


def _cfg_train_dataset_path(cfg: RevalueConfig) -> str:
    return cfg.cfg_train.dataset_path or cfg.data.dataset_path


def _cfg_train_base_model_path(cfg: RevalueConfig) -> str:
    if cfg.cfg_train.base_model_path:
        return cfg.cfg_train.base_model_path
    raise ValueError("cfg_train.base_model_path is required for train_cfg stage")


def _cfg_train_episode_split_path(cfg: RevalueConfig, paths: dict[str, Path]) -> str | None:
    if cfg.cfg_train.episode_split_path:
        return cfg.cfg_train.episode_split_path
    manifest_path = _manifest_path(cfg, paths)
    return str(manifest_path) if manifest_path else None


def _policy_eval_model_path(cfg: RevalueConfig) -> str:
    if cfg.policy_eval.model_path:
        return cfg.policy_eval.model_path
    base_model = cfg.cfg_train.base_model_path
    if base_model:
        return base_model
    raise ValueError("policy_eval.model_path is required for eval_policy stage")


def _resolve_policy_eval_checkpoint(cfg: RevalueConfig, paths: dict[str, Path]) -> str | None:
    if cfg.policy_eval.checkpoint_path:
        return cfg.policy_eval.checkpoint_path
    summary_path = paths["train_cfg"] / "train_cfg_summary.json"
    if summary_path.exists():
        return resolve_latest_trained_checkpoint(summary_path)
    return None


def _resolve_rollout_collect_model_path(cfg: RevalueConfig) -> str:
    if cfg.rollout_collect.model_path:
        return cfg.rollout_collect.model_path
    if cfg.policy_eval.model_path:
        return cfg.policy_eval.model_path
    if cfg.cfg_train.base_model_path:
        return cfg.cfg_train.base_model_path
    raise ValueError(
        "rollout_collect.model_path is required for collect_rollouts stage"
    )


def _resolve_rollout_collect_checkpoint(cfg: RevalueConfig, paths: dict[str, Path]) -> str | None:
    if cfg.rollout_collect.checkpoint_path:
        return cfg.rollout_collect.checkpoint_path
    summary_path = paths["train_cfg"] / "train_cfg_summary.json"
    if summary_path.exists():
        return resolve_latest_trained_checkpoint(summary_path)
    return None


def run_revalue(cfg: RevalueConfig) -> None:
    """Run the requested Revalue method/stage."""
    if cfg.method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported method={cfg.method!r}; expected {SUPPORTED_METHODS}"
        )
    if cfg.stage not in SUPPORTED_STAGES:
        raise ValueError(
            f"Unsupported stage={cfg.stage!r}; expected {SUPPORTED_STAGES}"
        )

    paths = _output_paths(cfg)
    paths["root"].mkdir(parents=True, exist_ok=True)

    requested_stages = _stages_for_method(cfg)

    if cfg.method not in {METHOD_BASE, METHOD_SHARED_MLP_FUSION}:
        raise ValueError(f"Unsupported method={cfg.method!r}")

    source_advantages = _source_advantage_path(cfg)

    for stage in requested_stages:
        if stage == STAGE_PREPARE_DATA:
            _run_prepare_data(cfg, paths)
        elif stage == STAGE_BUILD_BASE:
            source_advantages = _run_build_base(cfg, paths)
        elif stage == STAGE_RESPLIT_FEATURES:
            _run_resplit_features(cfg, paths)
        elif stage == STAGE_EXTRACT_FEATURES:
            feature_subset_path = _feature_subset_path(cfg)
            feature_split_path = _feature_split_path(cfg, paths)
            extract_features(
                FeatureExtractionConfig(
                    dataset_path=cfg.data.dataset_path,
                    value_checkpoint=cfg.value.checkpoint,
                    output_dir=str(paths["features"]),
                    siglip_path=cfg.value.siglip_path,
                    gemma3_path=cfg.value.gemma3_path,
                    tokenizer_path=cfg.value.tokenizer_path,
                    robot_type=cfg.data.robot_type,
                    env_type=cfg.data.env_type,
                    model_type=cfg.data.model_type,
                    critic_expert_variant=cfg.value.critic_expert_variant,
                    num_return_bins=cfg.value.num_bins,
                    value_min=cfg.value.v_min,
                    value_max=cfg.value.v_max,
                    action_dim=cfg.data.action_dim,
                    default_prompt=cfg.data.default_prompt,
                    val_episode_ratio=cfg.data.val_episode_ratio,
                    label_name=cfg.data.label_name,
                    seed=cfg.data.seed,
                    max_episodes=cfg.data.max_episodes,
                    episode_subset_path=(
                        str(feature_subset_path) if feature_subset_path else None
                    ),
                    episode_split_path=(
                        str(feature_split_path) if feature_split_path else None
                    ),
                    batch_size=cfg.train.extract_batch_size,
                    num_workers=cfg.train.num_workers,
                    device=cfg.train.device,
                )
            )
        elif stage == STAGE_TRAIN_ZP:
            train_zp_head(
                ZPTrainingConfig(
                    features_dir=str(paths["features"]),
                    output_dir=str(paths["zp"]),
                    num_phases=5,
                    hidden_dim=cfg.zp.hidden_dim,
                    dropout=cfg.zp.dropout,
                    trunk_depth=cfg.zp.trunk_depth,
                    batch_size=cfg.train.batch_size,
                    num_workers=cfg.train.num_workers,
                    lr=cfg.zp.lr,
                    weight_decay=cfg.zp.weight_decay,
                    max_epochs=cfg.zp.max_epochs,
                    early_stop_patience=cfg.zp.early_stop_patience,
                    device=cfg.train.device,
                    seed=cfg.train.seed,
                )
            )
        elif stage == STAGE_TRAIN_FUSION:
            source_df = read_advantages(source_advantages)
            validate_fusion_advantages(source_df, source=source_advantages)
            train_fusion(
                FusionTrainingConfig(
                    features_dir=str(paths["features"]),
                    advantages_path=str(source_advantages),
                    zp_head_path=str(paths["zp"] / "zp_head.pt"),
                    output_dir=str(paths["fusion"]),
                    return_min=cfg.returns.global_min,
                    return_max=cfg.returns.global_max,
                    value_min=cfg.value.v_min,
                    value_max=cfg.value.v_max,
                    num_bins=cfg.value.num_bins,
                    num_phases=5,
                    fusion_hidden_dim=cfg.fusion.hidden_dim,
                    fusion_depth=cfg.fusion.depth,
                    fusion_dropout=cfg.fusion.dropout,
                    alpha=cfg.fusion.alpha,
                    batch_size=cfg.train.batch_size,
                    num_workers=cfg.train.num_workers,
                    lr=cfg.fusion.lr,
                    weight_decay=cfg.fusion.weight_decay,
                    max_epochs=cfg.fusion.max_epochs,
                    early_stop_patience=cfg.fusion.early_stop_patience,
                    device=cfg.train.device,
                    seed=cfg.train.seed,
                )
            )
        elif stage == STAGE_PREDICT:
            source_df = read_advantages(source_advantages)
            validate_fusion_advantages(source_df, source=source_advantages)
            predict_fused_values(
                PredictionConfig(
                    features_dir=str(paths["features"]),
                    advantages_path=str(source_advantages),
                    zp_head_path=str(paths["zp"] / "zp_head.pt"),
                    fusion_path=str(paths["fusion"] / "fusion.pt"),
                    output_path=str(paths["predictions"]),
                    batch_size=cfg.train.batch_size,
                    device=cfg.train.device,
                )
            )
        elif stage == STAGE_EXPORT:
            read_advantages(source_advantages)
            export_fused_advantages(
                ExportConfig(
                    dataset_path=cfg.data.dataset_path,
                    source_advantages_path=str(source_advantages),
                    predictions_path=str(paths["predictions"]),
                    output_tag=cfg.recap.output_tag,
                    lookahead_step=cfg.recap.lookahead_step,
                    gamma=cfg.recap.gamma,
                    positive_quantile=cfg.recap.positive_quantile,
                    discount_next_value=cfg.recap.discount_next_value,
                    split=cfg.recap.export_split,
                )
            )
        elif stage == STAGE_COMPARE_RETURNS:
            read_advantages(source_advantages)
            report = compare_return_predictions(
                ReturnComparisonConfig(
                    advantages_path=str(source_advantages),
                    predictions_path=str(paths["predictions"]),
                    output_path=str(paths["comparison"]),
                    return_min=cfg.returns.global_min,
                    return_max=cfg.returns.global_max,
                )
            )
            logger.info(
                "saved Revalue return comparison to %s splits=%s",
                paths["comparison"],
                sorted(report.get("frame_level", {})),
            )
        elif stage == STAGE_TRAIN_CFG:
            summary = train_cfg_from_advantages(
                DownstreamCFGTrainingConfig(
                    repo_root=str(Path(__file__).resolve().parents[3]),
                    dataset_path=_cfg_train_dataset_path(cfg),
                    base_model_path=_cfg_train_base_model_path(cfg),
                    advantage_tag=_advantage_tag_for_cfg_train(cfg),
                    experiment_name=cfg.cfg_train.experiment_name,
                    log_dir=cfg.cfg_train.log_dir or str(paths["train_cfg"]),
                    config_name=cfg.cfg_train.config_name,
                    episode_split_path=_cfg_train_episode_split_path(cfg, paths),
                    episode_split_name=cfg.cfg_train.episode_split_name,
                    model_type=cfg.cfg_train.model_type,
                    openpi_config_name=cfg.cfg_train.openpi_config_name,
                    guidance_type=cfg.cfg_train.guidance_type,
                    positive_only_conditional=cfg.cfg_train.positive_only_conditional,
                    max_epochs=cfg.cfg_train.max_epochs,
                    max_steps=cfg.cfg_train.max_steps,
                    save_interval=cfg.cfg_train.save_interval,
                    val_check_interval=cfg.cfg_train.val_check_interval,
                    total_training_steps=cfg.cfg_train.total_training_steps,
                    lr_warmup_steps=cfg.cfg_train.lr_warmup_steps,
                    global_batch_size=cfg.cfg_train.global_batch_size,
                    micro_batch_size=cfg.cfg_train.micro_batch_size,
                    data_type=cfg.cfg_train.data_type,
                    dataset_weight=cfg.cfg_train.dataset_weight,
                    python_bin=cfg.cfg_train.python_bin,
                    extra_overrides=cfg.cfg_train.extra_overrides,
                )
            )
            logger.info(
                "saved Revalue downstream CFG training summary to %s checkpoint=%s",
                Path(cfg.cfg_train.log_dir or str(paths["train_cfg"])) / "train_cfg_summary.json",
                summary.get("checkpoint_path"),
            )
        elif stage == STAGE_EVAL_POLICY:
            summary = evaluate_policy_checkpoint(
                PolicyEvaluationConfig(
                    repo_root=str(Path(__file__).resolve().parents[3]),
                    model_path=_policy_eval_model_path(cfg),
                    checkpoint_path=_resolve_policy_eval_checkpoint(cfg, paths),
                    experiment_name=cfg.policy_eval.experiment_name,
                    log_dir=cfg.policy_eval.log_dir or str(paths["policy_eval"]),
                    config_name=cfg.policy_eval.config_name,
                    model_type=cfg.policy_eval.model_type,
                    openpi_config_name=cfg.policy_eval.openpi_config_name,
                    guidance_type=cfg.policy_eval.guidance_type,
                    positive_only_conditional=cfg.policy_eval.positive_only_conditional,
                    eval_rollout_epoch=cfg.policy_eval.eval_rollout_epoch,
                    total_num_envs=cfg.policy_eval.total_num_envs,
                    save_video=cfg.policy_eval.save_video,
                    task_suite_name=cfg.policy_eval.task_suite_name,
                    task_id_filter=cfg.policy_eval.task_id_filter,
                    python_bin=cfg.policy_eval.python_bin,
                    extra_overrides=cfg.policy_eval.extra_overrides,
                )
            )
            logger.info(
                "saved Revalue policy eval summary to %s metrics=%s",
                Path(cfg.policy_eval.log_dir or str(paths["policy_eval"])) / "eval_policy_summary.json",
                summary.get("metrics"),
            )
        elif stage == STAGE_COLLECT_ROLLOUTS:
            output_dir = (
                cfg.rollout_collect.output_dir or str(paths["rollout_collect"])
            )
            summary = collect_libero_rollouts(
                LiberoRolloutCollectionConfig(
                    output_dir=output_dir,
                    model_path=_resolve_rollout_collect_model_path(cfg),
                    checkpoint_path=_resolve_rollout_collect_checkpoint(cfg, paths),
                    model_type=cfg.rollout_collect.model_type,
                    openpi_config_name=cfg.rollout_collect.openpi_config_name,
                    task_suite_name=cfg.rollout_collect.task_suite_name,
                    task_id=cfg.rollout_collect.task_id,
                    num_episodes=cfg.rollout_collect.num_episodes,
                    noise_scale=cfg.rollout_collect.noise_scale,
                    noise_clip=cfg.rollout_collect.noise_clip,
                    action_chunk=cfg.rollout_collect.action_chunk,
                    num_steps=cfg.rollout_collect.num_steps,
                    num_steps_wait=cfg.rollout_collect.num_steps_wait,
                    seed=cfg.rollout_collect.seed,
                    gpu_id=cfg.rollout_collect.gpu_id,
                    fps=cfg.rollout_collect.fps,
                    overwrite=cfg.rollout_collect.overwrite,
                    failure_reward=cfg.rollout_collect.failure_reward,
                )
            )
            logger.info(
                "saved Revalue rollout collection summary to %s success_rate=%.4f",
                Path(output_dir) / "collection_summary.json",
                float(summary.get("success_rate", 0.0)),
            )
        else:
            raise ValueError(f"Unhandled stage={stage!r}")
