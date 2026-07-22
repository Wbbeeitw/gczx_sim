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

"""Unit tests for success-gated positive selection in advantage export."""

import pandas as pd

from revalue_test_utils import install_omegaconf_stub

install_omegaconf_stub()

from rlinf.revalue.recap.export import (
    compute_gated_positive_mask,
    infer_episode_success,
)


def _make_episode(episode_index: int, num_frames: int, first_return: float, adv_offset: float) -> pd.DataFrame:
    """Build one synthetic episode's advantage rows."""
    return pd.DataFrame(
        {
            "episode_index": [episode_index] * num_frames,
            "frame_index": list(range(num_frames)),
            # Return-to-go increases by 1 per frame from the first-frame value.
            "return": [first_return + t for t in range(num_frames)],
            "advantage_continuous": [adv_offset + t * 0.001 for t in range(num_frames)],
        }
    )


def test_infer_episode_success_separates_by_failure_reward() -> None:
    success = _make_episode(0, num_frames=100, first_return=-100.0, adv_offset=0.0)
    failure = _make_episode(1, num_frames=100, first_return=-400.0, adv_offset=0.0)
    df = pd.concat([success, failure], ignore_index=True)

    inferred = infer_episode_success(df, failure_reward=-300.0)

    assert inferred[0] is True or inferred[0] == True  # noqa: E712
    assert inferred[1] is False or inferred[1] == False  # noqa: E712


def test_gate_caps_failure_positives_and_prefers_top_advantage() -> None:
    # 3 success episodes (300 frames) + 2 failure episodes (200 frames).
    dfs = [
        _make_episode(ep, num_frames=100, first_return=-100.0, adv_offset=0.0)
        for ep in range(3)
    ]
    dfs += [
        _make_episode(ep, num_frames=100, first_return=-400.0, adv_offset=10.0 * ep)
        for ep in range(3, 5)
    ]
    df = pd.concat(dfs, ignore_index=True)

    mask = compute_gated_positive_mask(
        df,
        positive_quantile=0.3,
        failure_positive_cap=0.2,
        failure_reward=-300.0,
    )

    # budget = round(0.3 * 500) = 150; failure cap = 30, success gets 120.
    assert mask.sum() == 150
    failure_positive = mask[df["episode_index"] >= 3].sum()
    assert failure_positive == 30
    # Failure positives must be the highest-advantage failure frames: episode 4
    # has the larger adv_offset, so all 30 failure positives come from it.
    assert mask[df["episode_index"] == 4].sum() == 30
    assert mask[df["episode_index"] == 3].sum() == 0


def test_gate_does_not_reserve_failure_quota() -> None:
    success = _make_episode(
        0,
        num_frames=100,
        first_return=-100.0,
        adv_offset=10.0,
    )
    failure = _make_episode(
        1,
        num_frames=100,
        first_return=-400.0,
        adv_offset=0.0,
    )
    df = pd.concat([success, failure], ignore_index=True)

    mask = compute_gated_positive_mask(
        df,
        positive_quantile=0.3,
        failure_positive_cap=0.2,
        failure_reward=-300.0,
    )

    assert mask.sum() == 60
    assert mask[df["episode_index"] == 0].sum() == 60
    assert mask[df["episode_index"] == 1].sum() == 0


def test_demo_backstop_forces_all_frames_and_stays_outside_budget() -> None:
    dfs = [
        _make_episode(ep, num_frames=100, first_return=-100.0, adv_offset=0.0)
        for ep in range(3)
    ]
    df = pd.concat(dfs, ignore_index=True)

    mask = compute_gated_positive_mask(
        df,
        positive_quantile=0.3,
        failure_positive_cap=0.2,
        failure_reward=-300.0,
        full_positive_episodes={2},
    )

    # Episode 2 is fully positive; the remaining 200 frames still get their
    # own budget of round(0.3 * 200) = 60.
    assert mask[df["episode_index"] == 2].sum() == 100
    assert mask[df["episode_index"] < 2].sum() == 60
    assert mask.sum() == 160


def test_all_failure_pool_only_uses_capped_budget() -> None:
    dfs = [
        _make_episode(ep, num_frames=100, first_return=-400.0, adv_offset=0.0)
        for ep in range(3)
    ]
    df = pd.concat(dfs, ignore_index=True)

    mask = compute_gated_positive_mask(
        df,
        positive_quantile=0.3,
        failure_positive_cap=0.2,
        failure_reward=-300.0,
    )

    # budget = 90, but with no success frames only the failure cap (18) is used.
    assert mask.sum() == 18
