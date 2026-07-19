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
