import importlib.util
from pathlib import Path
import sys

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


cfg_strategy = _load_module(
    "cfg_strategy_module",
    "rlinf/data/datasets/recap/cfg_strategy.py",
)
cfg_routing = _load_module(
    "cfg_routing_module",
    "rlinf/models/embodiment/openpi_cfg/cfg_routing.py",
)

CFG_STRATEGY_BINARY = cfg_strategy.CFG_STRATEGY_BINARY
CFG_STRATEGY_BINARY_WEIGHTED = cfg_strategy.CFG_STRATEGY_BINARY_WEIGHTED
CFG_STRATEGY_ACP_CFG = cfg_strategy.CFG_STRATEGY_ACP_CFG
CFG_STRATEGY_CSA_RESIDUAL = cfg_strategy.CFG_STRATEGY_CSA_RESIDUAL
CFG_STRATEGY_CSA_SOFT = cfg_strategy.CFG_STRATEGY_CSA_SOFT
CFGStrategyConfig = cfg_strategy.CFGStrategyConfig
QUALITY_LABEL_NEGATIVE = cfg_strategy.QUALITY_LABEL_NEGATIVE
QUALITY_LABEL_NEUTRAL = cfg_strategy.QUALITY_LABEL_NEUTRAL
QUALITY_LABEL_POSITIVE = cfg_strategy.QUALITY_LABEL_POSITIVE
build_cfg_sample_metadata = cfg_strategy.build_cfg_sample_metadata
build_weighted_loss = cfg_routing.build_weighted_loss
compute_binary_cfg_routing_masks = cfg_routing.compute_binary_cfg_routing_masks
compute_csa_cfg_routing_masks = cfg_routing.compute_csa_cfg_routing_masks


def test_build_cfg_sample_metadata_binary_preserves_advantage() -> None:
    df = pd.DataFrame(
        {
            "episode_index": [0, 0, 1],
            "frame_index": [0, 1, 0],
            "advantage": [True, False, True],
        }
    )

    metadata = build_cfg_sample_metadata(
        df,
        strategy_cfg=CFGStrategyConfig(strategy=CFG_STRATEGY_BINARY),
        dataset_id="dataset_a",
    )

    assert metadata[(0, 0)] == {"advantage": True}
    assert metadata[(0, 1)] == {"advantage": False}
    assert metadata[(1, 0)] == {"advantage": True}


def test_build_cfg_sample_metadata_csa_soft_assigns_labels_and_weights() -> None:
    df = pd.DataFrame(
        {
            "episode_index": [0] * 10,
            "frame_index": list(range(10)),
            "advantage": [False] * 10,
            "advantage_continuous": list(range(10)),
        }
    )

    metadata = build_cfg_sample_metadata(
        df,
        strategy_cfg=CFGStrategyConfig(
            strategy=CFG_STRATEGY_CSA_SOFT,
            positive_quantile=0.30,
            bottom_quantile=0.20,
            bottom_negative_prob=1.0,
            weight_lambda=0.20,
            seed=7,
        ),
        dataset_id="dataset_b",
    )

    assert metadata[(0, 9)]["cfg_quality_label"] == QUALITY_LABEL_POSITIVE
    assert metadata[(0, 8)]["cfg_quality_label"] == QUALITY_LABEL_POSITIVE
    assert metadata[(0, 7)]["cfg_quality_label"] == QUALITY_LABEL_POSITIVE
    assert metadata[(0, 0)]["cfg_quality_label"] == QUALITY_LABEL_NEGATIVE
    assert metadata[(0, 1)]["cfg_quality_label"] == QUALITY_LABEL_NEGATIVE
    assert metadata[(0, 4)]["cfg_quality_label"] == QUALITY_LABEL_NEUTRAL
    assert metadata[(0, 9)]["advantage"] is True
    assert metadata[(0, 4)]["advantage"] is False
    assert abs(metadata[(0, 0)]["cfg_loss_weight"] - 0.8) < 1.0e-6
    assert abs(metadata[(0, 9)]["cfg_loss_weight"] - 1.2) < 1.0e-6


def test_build_cfg_sample_metadata_binary_weighted_preserves_binary_routing() -> None:
    df = pd.DataFrame(
        {
            "episode_index": [0] * 10,
            "frame_index": list(range(10)),
            "advantage": [False] * 10,
            "advantage_continuous": list(range(10)),
        }
    )

    metadata = build_cfg_sample_metadata(
        df,
        strategy_cfg=CFGStrategyConfig(
            strategy=CFG_STRATEGY_BINARY_WEIGHTED,
            positive_quantile=0.30,
            weight_lambda=0.20,
        ),
        dataset_id="dataset_weighted_binary",
    )

    assert metadata[(0, 9)]["advantage"] is True
    assert metadata[(0, 7)]["advantage"] is True
    assert metadata[(0, 6)]["advantage"] is False
    assert abs(metadata[(0, 0)]["cfg_loss_weight"] - 0.8) < 1.0e-6
    assert abs(metadata[(0, 9)]["cfg_loss_weight"] - 1.2) < 1.0e-6
    assert "cfg_quality_label" not in metadata[(0, 9)]
    assert "cfg_residual_positive_mask" not in metadata[(0, 9)]


def test_build_cfg_sample_metadata_acp_cfg_matches_weighted_binary_behavior() -> None:
    df = pd.DataFrame(
        {
            "episode_index": [0] * 10,
            "frame_index": list(range(10)),
            "advantage": [False] * 10,
            "advantage_continuous": list(range(10)),
        }
    )

    metadata = build_cfg_sample_metadata(
        df,
        strategy_cfg=CFGStrategyConfig(
            strategy=CFG_STRATEGY_ACP_CFG,
            positive_quantile=0.30,
            weight_lambda=0.10,
        ),
        dataset_id="dataset_acp_cfg",
    )

    assert metadata[(0, 9)]["advantage"] is True
    assert metadata[(0, 6)]["advantage"] is False
    assert abs(metadata[(0, 0)]["cfg_loss_weight"] - 0.9) < 1.0e-6
    assert abs(metadata[(0, 9)]["cfg_loss_weight"] - 1.1) < 1.0e-6
    assert "cfg_quality_label" not in metadata[(0, 9)]
    assert "cfg_residual_positive_mask" not in metadata[(0, 9)]


def test_build_cfg_sample_metadata_csa_soft_bottom_split_is_deterministic() -> None:
    df = pd.DataFrame(
        {
            "episode_index": [0] * 6,
            "frame_index": list(range(6)),
            "advantage": [False] * 6,
            "advantage_continuous": [0.0, 0.1, 0.2, 0.8, 0.9, 1.0],
        }
    )

    cfg = CFGStrategyConfig(
        strategy=CFG_STRATEGY_CSA_SOFT,
        positive_quantile=0.30,
        bottom_quantile=0.34,
        bottom_negative_prob=0.50,
        seed=11,
    )
    first = build_cfg_sample_metadata(df, strategy_cfg=cfg, dataset_id="dataset_c")
    second = build_cfg_sample_metadata(df, strategy_cfg=cfg, dataset_id="dataset_c")

    assert first == second


def test_build_cfg_sample_metadata_csa_residual_marks_only_top_bucket() -> None:
    df = pd.DataFrame(
        {
            "episode_index": [0] * 10,
            "frame_index": list(range(10)),
            "advantage": [False] * 10,
            "advantage_continuous": list(range(10)),
        }
    )

    metadata = build_cfg_sample_metadata(
        df,
        strategy_cfg=CFGStrategyConfig(
            strategy=CFG_STRATEGY_CSA_RESIDUAL,
            positive_quantile=0.30,
        ),
        dataset_id="dataset_residual",
    )

    assert metadata[(0, 9)]["advantage"] is True
    assert metadata[(0, 8)]["cfg_residual_positive_mask"] is True
    assert metadata[(0, 7)]["cfg_residual_positive_mask"] is True
    assert metadata[(0, 6)]["cfg_residual_positive_mask"] is False
    assert metadata[(0, 0)]["advantage"] is False
    assert "cfg_quality_label" not in metadata[(0, 9)]
    assert "cfg_loss_weight" not in metadata[(0, 9)]


def test_compute_binary_cfg_routing_masks_matches_legacy_semantics() -> None:
    masks = compute_binary_cfg_routing_masks(
        torch.tensor([True, False, True]),
        positive_only_conditional=False,
        unconditional_prob=0.1,
        random_values=torch.tensor([0.2, 0.2, 0.05]),
    )

    assert masks["positive_conditional_mask"].tolist() == [True, False, False]
    assert masks["negative_conditional_mask"].tolist() == [False, True, False]
    assert masks["positive_unconditional_mask"].tolist() == [False, False, True]


def test_compute_csa_cfg_routing_masks_routes_positive_neutral_negative() -> None:
    masks = compute_csa_cfg_routing_masks(
        torch.tensor(
            [
                QUALITY_LABEL_POSITIVE,
                QUALITY_LABEL_POSITIVE,
                QUALITY_LABEL_NEUTRAL,
                QUALITY_LABEL_NEGATIVE,
            ]
        ),
        unconditional_prob=0.1,
        positive_prompt_prob=0.85,
        unconditional_random=torch.tensor([0.05, 0.2, 0.2, 0.2]),
        positive_prompt_random=torch.tensor([0.1, 0.9, 0.0, 0.0]),
    )

    assert masks["positive_unconditional_mask"].tolist() == [True, False, False, False]
    assert masks["positive_conditional_mask"].tolist() == [False, False, False, False]
    assert masks["positive_to_neutral_mask"].tolist() == [False, True, False, False]
    assert masks["neutral_conditional_mask"].tolist() == [False, True, True, False]
    assert masks["negative_conditional_mask"].tolist() == [False, False, False, True]


def test_build_weighted_loss_uses_normalized_weighted_mean() -> None:
    per_sample_loss = torch.tensor([1.0, 3.0], dtype=torch.float32)
    sample_weight = torch.tensor([1.0, 3.0], dtype=torch.float32)

    loss = build_weighted_loss(per_sample_loss, sample_weight)

    assert abs(float(loss.item()) - 2.5) < 1.0e-6
