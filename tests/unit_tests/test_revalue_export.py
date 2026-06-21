import pandas as pd
import pytest

from revalue_test_utils import install_omegaconf_stub

install_omegaconf_stub()

from rlinf.revalue.recap.export import (
    build_save_advantages_df,
    compute_fused_advantages,
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
