"""Unit tests for simulator-state semantic phase labels."""

from __future__ import annotations

import pandas as pd

from rlinf.revalue.semantic_trace import build_task1_phase_labels


def _trace_row(
    frame_index: int,
    *,
    success: bool,
    object_a_controlled: bool = False,
    object_b_controlled: bool = False,
    object_a_in_basket: bool = False,
    object_b_in_basket: bool = False,
    object_a_near: bool = False,
    object_b_near: bool = False,
) -> dict[str, object]:
    return {
        "episode_index": 0,
        "frame_index": frame_index,
        "is_success": success,
        "object_a_controlled": object_a_controlled,
        "object_b_controlled": object_b_controlled,
        "object_a_basket_contact": object_a_in_basket,
        "object_b_basket_contact": object_b_in_basket,
        "object_a_near_basket": object_a_near,
        "object_b_near_basket": object_b_near,
    }


def test_task1_joint_transfer_creates_b2_before_completion() -> None:
    rows = []
    for frame_index in range(20):
        rows.append(
            _trace_row(
                frame_index,
                success=True,
                object_a_controlled=2 <= frame_index < 12,
                object_b_controlled=2 <= frame_index < 12,
                object_a_near=7 <= frame_index < 12,
                object_b_near=7 <= frame_index < 12,
                object_a_in_basket=12 <= frame_index,
                object_b_in_basket=12 <= frame_index,
            )
        )

    labels, audit = build_task1_phase_labels(pd.DataFrame(rows), stable_frames=3)

    assert audit.loc[0, "b1_frame"] == 2
    assert audit.loc[0, "b2_frame"] == 7
    assert audit.loc[0, "b3_frame"] == 12
    assert bool(audit.loc[0, "b2_joint_transfer_detected"])
    assert labels.loc[labels["frame_index"] == 6, "phase"].item() == 1
    assert labels.loc[labels["frame_index"] == 11, "phase"].item() == 2
    assert labels.loc[labels["frame_index"] == 14, "phase"].item() == 3
    assert set(labels["semantic_confidence"]) == {"state_verified"}


def test_task1_drop_does_not_reset_phase_one() -> None:
    rows = []
    for frame_index in range(20):
        rows.append(
            _trace_row(
                frame_index,
                success=False,
                object_a_controlled=2 <= frame_index < 6,
                object_b_controlled=9 <= frame_index < 13,
                object_a_in_basket=14 <= frame_index,
            )
        )

    labels, audit = build_task1_phase_labels(pd.DataFrame(rows), stable_frames=3)

    assert audit.loc[0, "b1_frame"] == 2
    assert audit.loc[0, "b2_frame"] == 14
    assert pd.isna(audit.loc[0, "b3_frame"])
    assert bool(audit.loc[0, "trainable"])
    assert labels.loc[labels["frame_index"] == 8, "phase"].item() == 1
    assert labels.loc[labels["frame_index"] == 17, "phase"].item() == 2
    assert labels["phase"].max() == 2
