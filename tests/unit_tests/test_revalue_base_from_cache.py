from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
import torch

from revalue_test_utils import install_omegaconf_stub

install_omegaconf_stub()

from rlinf.revalue.pipeline.base_from_cache import (  # noqa: E402
    BaseFromCacheConfig,
    build_revalue_base_from_cache,
)


def _write_cache(
    path: Path,
    episode_index,
    frame_index,
    raw_value,
    raw_logits,
    *,
    atoms=None,
) -> None:
    row_count = len(episode_index)
    if atoms is None:
        atoms = [-1.0, -0.5, 0.0]
    torch.save(
        {
            "features": torch.zeros(row_count, 4, dtype=torch.float32),
            "episode_index": torch.tensor(episode_index, dtype=torch.long),
            "frame_index": torch.tensor(frame_index, dtype=torch.long),
            "phase": torch.zeros(row_count, dtype=torch.long),
            "phase_progress": torch.zeros(row_count, dtype=torch.float32),
            "global_progress": torch.zeros(row_count, dtype=torch.float32),
            "raw_value": torch.tensor(raw_value, dtype=torch.float32),
            "raw_logits": torch.tensor(raw_logits, dtype=torch.float32),
            "atoms": torch.tensor(atoms, dtype=torch.float32),
        },
        path,
    )


def test_build_revalue_base_from_cache_writes_standard_advantages(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    meta_dir = dataset_path / "meta"
    features_dir = tmp_path / "features"
    meta_dir.mkdir(parents=True)
    features_dir.mkdir(parents=True)

    returns_df = pd.DataFrame(
        {
            "episode_index": [0, 0, 0, 1, 1],
            "frame_index": [0, 1, 2, 0, 1],
            "return": [-2.0, -1.0, 0.0, -6.0, -5.0],
            "reward": [-1.0, -1.0, 0.0, -1.0, -5.0],
        }
    )
    returns_df.to_parquet(meta_dir / "returns_demo.parquet", index=False)

    _write_cache(
        features_dir / "train.pt",
        episode_index=[0, 0, 0],
        frame_index=[0, 1, 2],
        raw_value=[-0.8, -0.5, -0.2],
        raw_logits=[
            [3.0, 1.0, -1.0],
            [2.5, 0.5, -1.0],
            [1.0, 1.0, 1.0],
        ],
    )
    _write_cache(
        features_dir / "val.pt",
        episode_index=[1, 1],
        frame_index=[0, 1],
        raw_value=[-0.6, -0.1],
        raw_logits=[
            [0.0, 2.0, 1.0],
            [0.5, 1.5, 0.0],
        ],
    )

    out_path = build_revalue_base_from_cache(
        BaseFromCacheConfig(
            dataset_path=str(dataset_path),
            features_dir=str(features_dir),
            tag="cache_base",
            returns_tag="demo",
            lookahead_step=2,
            gamma=1.0,
            positive_quantile=0.4,
            discount_next_value=True,
            return_min=-6.0,
            return_max=0.0,
            value_min=-1.0,
            value_max=0.0,
            report_path=str(tmp_path / "report.json"),
        )
    )

    saved = pd.read_parquet(out_path).sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    assert saved["episode_index"].tolist() == [0, 0, 0, 1, 1]
    assert saved["frame_index"].tolist() == [0, 1, 2, 0, 1]
    assert saved["num_valid_rewards"].tolist() == [2, 2, 1, 2, 1]
    assert saved["reward_sum_raw"].tolist() == [-2.0, -1.0, 0.0, -6.0, -5.0]

    expected_reward_sum = np.array(
        [
            (-2.0 + 6.0) / 6.0 - 1.0,
            (-1.0 + 6.0) / 6.0 - 1.0,
            (0.0 + 6.0) / 6.0 - 1.0,
            (-6.0 + 6.0) / 6.0 - 1.0,
            (-5.0 + 6.0) / 6.0 - 1.0,
        ]
    )
    expected_value_next = np.array([-0.2, 0.0, 0.0, 0.0, 0.0])
    expected_adv = expected_reward_sum + expected_value_next - np.array(
        [-0.8, -0.5, -0.2, -0.6, -0.1]
    )

    np.testing.assert_allclose(saved["reward_sum"].to_numpy(dtype=np.float64), expected_reward_sum)
    np.testing.assert_allclose(saved["value_next"].to_numpy(dtype=np.float64), expected_value_next)
    np.testing.assert_allclose(
        saved["advantage_continuous"].to_numpy(dtype=np.float64),
        expected_adv,
    )

    current_logits = saved.loc[0, "value_logits_current"]
    next_logits = saved.loc[0, "value_logits_next"]
    tail_next_logits = saved.loc[1, "value_logits_next"]
    tail_next_probs = saved.loc[1, "value_probs_next"]
    np.testing.assert_allclose(current_logits, [3.0, 1.0, -1.0])
    np.testing.assert_allclose(next_logits, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(tail_next_logits, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(tail_next_probs, [0.0, 0.0, 0.0])

    threshold = float(np.percentile(expected_adv, 60.0))
    expected_labels = [value >= threshold for value in expected_adv]
    assert saved["advantage"].tolist() == expected_labels

    with open(tmp_path / "report.json", "r", encoding="utf-8") as file:
        report = json.load(file)
    assert report["num_rows"] == 5
    assert report["feature_splits"]["train"]["rows"] == 3
    assert report["feature_splits"]["val"]["rows"] == 2


def test_build_revalue_base_from_cache_supports_custom_value_scale(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset"
    meta_dir = dataset_path / "meta"
    features_dir = tmp_path / "features"
    meta_dir.mkdir(parents=True)
    features_dir.mkdir(parents=True)

    returns_df = pd.DataFrame(
        {
            "episode_index": [0, 0, 0, 1, 1],
            "frame_index": [0, 1, 2, 0, 1],
            "return": [-2.0, -1.0, 0.0, -6.0, -5.0],
            "reward": [-1.0, -1.0, 0.0, -1.0, -5.0],
        }
    )
    returns_df.to_parquet(meta_dir / "returns_demo.parquet", index=False)

    _write_cache(
        features_dir / "train.pt",
        episode_index=[0, 0, 0],
        frame_index=[0, 1, 2],
        raw_value=[-4.8, -3.0, -1.2],
        raw_logits=[
            [3.0, 1.0, -1.0],
            [2.5, 0.5, -1.0],
            [1.0, 1.0, 1.0],
        ],
        atoms=[-6.0, -3.0, 0.0],
    )
    _write_cache(
        features_dir / "val.pt",
        episode_index=[1, 1],
        frame_index=[0, 1],
        raw_value=[-3.6, -0.6],
        raw_logits=[
            [0.0, 2.0, 1.0],
            [0.5, 1.5, 0.0],
        ],
        atoms=[-6.0, -3.0, 0.0],
    )

    out_path = build_revalue_base_from_cache(
        BaseFromCacheConfig(
            dataset_path=str(dataset_path),
            features_dir=str(features_dir),
            tag="cache_base_raw",
            returns_tag="demo",
            lookahead_step=2,
            gamma=1.0,
            positive_quantile=0.4,
            discount_next_value=True,
            return_min=-6.0,
            return_max=0.0,
            value_min=-6.0,
            value_max=0.0,
        )
    )

    saved = (
        pd.read_parquet(out_path)
        .sort_values(["episode_index", "frame_index"])
        .reset_index(drop=True)
    )

    expected_reward_sum = np.array([-2.0, -1.0, 0.0, -6.0, -5.0])
    expected_value_next = np.array([-1.2, 0.0, 0.0, 0.0, 0.0])
    expected_adv = expected_reward_sum + expected_value_next - np.array(
        [-4.8, -3.0, -1.2, -3.6, -0.6]
    )

    np.testing.assert_allclose(
        saved["reward_sum"].to_numpy(dtype=np.float64),
        expected_reward_sum,
    )
    np.testing.assert_allclose(
        saved["value_next"].to_numpy(dtype=np.float64),
        expected_value_next,
    )
    np.testing.assert_allclose(
        saved["advantage_continuous"].to_numpy(dtype=np.float64),
        expected_adv,
    )


def test_build_revalue_base_from_cache_rejects_atom_scale_mismatch(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset"
    meta_dir = dataset_path / "meta"
    features_dir = tmp_path / "features"
    meta_dir.mkdir(parents=True)
    features_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "episode_index": [0],
            "frame_index": [0],
            "return": [-1.0],
            "reward": [-1.0],
        }
    ).to_parquet(meta_dir / "returns_demo.parquet", index=False)

    _write_cache(
        features_dir / "train.pt",
        episode_index=[0],
        frame_index=[0],
        raw_value=[-0.8],
        raw_logits=[[1.0, 0.0, -1.0]],
        atoms=[-6.0, -3.0, 0.0],
    )
    _write_cache(
        features_dir / "val.pt",
        episode_index=[0],
        frame_index=[0],
        raw_value=[-0.8],
        raw_logits=[[1.0, 0.0, -1.0]],
        atoms=[-6.0, -3.0, 0.0],
    )

    with pytest.raises(ValueError, match="atoms range"):
        build_revalue_base_from_cache(
            BaseFromCacheConfig(
                dataset_path=str(dataset_path),
                features_dir=str(features_dir),
                tag="cache_base_bad_scale",
                returns_tag="demo",
                return_min=-6.0,
                return_max=0.0,
                value_min=-1.0,
                value_max=0.0,
            )
        )
