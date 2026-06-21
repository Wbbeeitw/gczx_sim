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
    STAGE_EXPORT,
    STAGE_EXTRACT_FEATURES,
    STAGE_PREDICT,
    STAGE_TRAIN_FUSION,
    STAGE_TRAIN_ZP,
    SUPPORTED_METHODS,
    SUPPORTED_STAGES,
)
from rlinf.revalue.data import resolve_advantage_path, validate_fusion_advantages
from rlinf.revalue.data.advantage_table import read_advantages
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
    }


def _source_advantage_path(cfg: RevalueConfig) -> Path:
    if cfg.recap.source_advantages_path:
        return Path(cfg.recap.source_advantages_path)
    return resolve_advantage_path(cfg.data.dataset_path, cfg.recap.source_tag)


def _validate_base(cfg: RevalueConfig) -> Path:
    path = _source_advantage_path(cfg)
    df = read_advantages(path)
    logger.info("base advantages available: %s rows=%d", path, len(df))
    return path


def _stages_for_request(stage: str) -> list[str]:
    if stage == STAGE_ALL:
        return [
            STAGE_EXTRACT_FEATURES,
            STAGE_TRAIN_ZP,
            STAGE_TRAIN_FUSION,
            STAGE_PREDICT,
            STAGE_EXPORT,
        ]
    return [stage]


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

    if cfg.method == METHOD_BASE:
        _validate_base(cfg)
        logger.info(
            "method=base does not train z/p or fusion. Use data.advantage_tag=%s "
            "in CFG/ReCap training.",
            cfg.recap.source_tag,
        )
        return

    if cfg.method != METHOD_SHARED_MLP_FUSION:
        raise ValueError(f"Unsupported method={cfg.method!r}")

    requested_stages = _stages_for_request(cfg.stage)
    source_advantages = _source_advantage_path(cfg)
    if any(
        stage in {STAGE_TRAIN_FUSION, STAGE_PREDICT}
        for stage in requested_stages
    ):
        source_df = read_advantages(source_advantages)
        validate_fusion_advantages(source_df, source=source_advantages)
    elif STAGE_EXPORT in requested_stages:
        read_advantages(source_advantages)

    for stage in requested_stages:
        if stage == STAGE_EXTRACT_FEATURES:
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
                    episode_subset_path=cfg.data.episode_subset_path,
                    episode_split_path=cfg.data.episode_split_path,
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
        else:
            raise ValueError(f"Unhandled stage={stage!r}")
