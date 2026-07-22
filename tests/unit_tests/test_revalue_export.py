import json

import pandas as pd
import pytest

from revalue_test_utils import install_omegaconf_stub

install_omegaconf_stub()

from rlinf.revalue.recap.export import (
    build_save_advantages_df,
    compute_fused_advantages,
)
from rlinf.revalue.recap.export_view import (
    ExportDatasetViewConfig,
    export_dataset_view,
)


def test_compute_fused_advantages_uses_frame_lookahead() -> None:
    source = pd.DataFrame(
        {
            "episode_index": [0, 0, 0],
            "frame_index": [0, 1, 2],
            "return": [-3.0, -2.0, -1.0],
            "reward_sum": [-0.1, -0.1, -0.1],
            "reward_sum_raw": [-1.0, -1.0, -1.0],
            "num_valid_rewards": [2, 1, 1],
            "dataset_name": ["d", "d", "d"],
        }
    )
    predictions = pd.DataFrame(
        {
            "episode_index": [0, 0, 0],
            "frame_index": [0, 1, 2],
            "value_fused": [-0.8, -0.4, 0.0],
        }
    )

    out = compute_fused_advantages(
        source_advantages=source,
        predictions=predictions,
        lookahead_step=1,
        gamma=1.0,
        discount_next_value=True,
    )

    assert out.loc[0, "value_current"] == -0.8
    assert out.loc[0, "value_next"] == 0.0
    assert out.loc[0, "advantage_continuous"] == pytest.approx(0.7)
    assert out.loc[1, "value_next"] == 0.0
    assert out.loc[2, "value_next"] == 0.0


def test_build_save_advantages_df_adds_boolean_label() -> None:
    fused = pd.DataFrame(
        {
            "episode_index": [0, 0],
            "frame_index": [0, 1],
            "advantage_continuous": [-0.2, 0.3],
            "return": [-2.0, -1.0],
            "value_current": [-0.5, -0.1],
            "value_next": [-0.1, 0.0],
            "reward_sum": [-0.1, -0.1],
        }
    )

    save_df = build_save_advantages_df(fused, threshold=0.0)

    assert save_df["advantage"].tolist() == [False, True]


def test_raw_dataset_view_reindexes_without_fused_predictions(
    tmp_path,
) -> None:
    child = tmp_path / "child"
    (child / "meta").mkdir(parents=True)
    (child / "meta" / "info.json").write_text(
        json.dumps({"total_frames": 2, "total_episodes": 1}),
        encoding="utf-8",
    )
    source = tmp_path / "raw.parquet"
    pd.DataFrame(
        {
            "episode_index": [1, 1],
            "frame_index": [0, 1],
            "advantage_continuous": [0.1, 0.9],
            "return": [-10.0, -5.0],
            "value_current": [-0.5, -0.4],
            "value_next": [-0.4, 0.0],
            "reward_sum": [0.0, 0.0],
        }
    ).to_parquet(source)

    output = export_dataset_view(
        ExportDatasetViewConfig(
            mode="raw",
            source_advantages_path=str(source),
            predictions_path=None,
            child_dataset_path=str(child),
            output_tag="raw_top50",
            source_episode_start=1,
            source_episode_end=2,
            child_episode_offset=-1,
            positive_quantile=0.5,
            expected_episodes=1,
        )
    )

    exported = pd.read_parquet(output)
    assert exported["episode_index"].tolist() == [0, 0]
    assert exported["advantage"].tolist() == [False, True]
    report = json.loads(
        (child / "meta" / "raw_top50_revalue_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["mode"] == "raw"
    assert report["predictions_path"] is None


def test_raw_dataset_view_applies_expert_backstop_and_rollout_failure_cap(
    tmp_path,
) -> None:
    child = tmp_path / "child"
    (child / "meta").mkdir(parents=True)
    (child / "meta" / "info.json").write_text(
        json.dumps({"total_frames": 30, "total_episodes": 3}),
        encoding="utf-8",
    )
    (child / "meta" / "full_positive_episodes.json").write_text(
        json.dumps([2]),
        encoding="utf-8",
    )
    source = tmp_path / "raw.parquet"
    rows = []
    for episode, first_return, advantage_offset in (
        (0, -10.0, 0.0),
        (1, -310.0, 10.0),
        (2, -10.0, -10.0),
    ):
        for frame in range(10):
            rows.append(
                {
                    "episode_index": episode,
                    "frame_index": frame,
                    "advantage_continuous": advantage_offset + frame / 100.0,
                    "return": first_return + frame,
                    "value_current": 0.0,
                    "value_next": 0.0,
                    "reward_sum": 0.0,
                }
            )
    pd.DataFrame(rows).to_parquet(source)

    output = export_dataset_view(
        ExportDatasetViewConfig(
            mode="raw",
            source_advantages_path=str(source),
            predictions_path=None,
            child_dataset_path=str(child),
            output_tag="facd",
            source_episode_start=0,
            source_episode_end=3,
            positive_quantile=0.5,
            failure_positive_cap=0.2,
            failure_reward=-300.0,
            success_gate=True,
            demo_backstop=True,
            expected_episodes=3,
        )
    )

    exported = pd.read_parquet(output)
    positives = exported.groupby("episode_index")["advantage"].sum().to_dict()
    assert positives == {0: 8, 1: 2, 2: 10}
    report = json.loads(
        (child / "meta" / "facd_revalue_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["num_full_positive_frames"] == 10
    assert report["rollout_positive_budget"] == 10
    assert report["num_rollout_positive_frames"] == 10
    assert report["num_rollout_failure_positive_frames"] == 2
    assert report["rollout_failure_positive_ratio"] == pytest.approx(0.2)
    assert report["rollout_budget_feasible"] is True
    assert report["rollout_max_feasible_filled"] is True
    assert report["rollout_positive_shortfall"] == 0
    assert report["rollout_positive_shortfall_reason"] is None
    assert report["failure_cap_satisfied"] is True
