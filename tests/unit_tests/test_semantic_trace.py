"""Unit tests for simulator-state semantic phase labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rlinf.revalue.semantic_trace import (
    _phase_progress,
    build_task1_phase_labels,
    build_task2_phase_labels,
    build_task3_phase_labels,
    build_task4_phase_labels,
    build_task5_phase_labels,
    build_task6_phase_labels,
)


def test_success_phase_progress_completes_each_entered_phase() -> None:
    phase = np.array([0, 0, 1, 1, 1, 2, 2, 3])

    phase_progress, global_progress = _phase_progress(phase, is_success=True)

    np.testing.assert_allclose(
        phase_progress,
        [0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 1.0, 1.0],
    )
    assert global_progress[-1] == 1.0


def test_failed_terminal_phase_reaches_half_then_plateaus() -> None:
    phase = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2])

    phase_progress, global_progress = _phase_progress(phase, is_success=False)

    np.testing.assert_allclose(
        phase_progress,
        [0.0, 0.0, 0.5, 1.0, 0.0, 0.25, 0.5, 0.5, 0.5],
    )
    assert global_progress[-1] == 0.625


def test_failed_episode_without_b1_keeps_phase_zero_at_zero() -> None:
    phase = np.zeros(4, dtype=np.int64)

    phase_progress, global_progress = _phase_progress(phase, is_success=False)

    np.testing.assert_allclose(phase_progress, 0.0)
    np.testing.assert_allclose(global_progress, 0.0)


def _trace_row(
    frame_index: int,
    *,
    success: bool,
    env_success: bool = False,
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
        "env_success": env_success,
        "object_a_controlled": object_a_controlled,
        "object_b_controlled": object_b_controlled,
        "object_a_basket_contact": object_a_in_basket,
        "object_b_basket_contact": object_b_in_basket,
        "object_a_near_basket": object_a_near,
        "object_b_near_basket": object_b_near,
    }


def _task2_trace_row(
    frame_index: int,
    *,
    success: bool,
    env_success: bool = False,
    moka_pot_controlled: bool = False,
    stove_button_interacted: bool = False,
    stove_turn_on: bool = False,
    moka_pot_on_cook_region: bool = False,
    frypan_gripper_contact: bool = False,
) -> dict[str, object]:
    return {
        "episode_index": 0,
        "frame_index": frame_index,
        "is_success": success,
        "env_success": env_success,
        "moka_pot_controlled": moka_pot_controlled,
        "stove_button_interacted": stove_button_interacted,
        "stove_turn_on": stove_turn_on,
        "moka_pot_on_cook_region": moka_pot_on_cook_region,
        "frypan_gripper_contact": frypan_gripper_contact,
    }


def _task3_trace_row(
    frame_index: int,
    *,
    success: bool,
    env_success: bool = False,
    black_bowl_controlled: bool = False,
    bottom_drawer_interacted: bool = False,
    bottom_drawer_is_close: bool = False,
    black_bowl_in_bottom_drawer: bool = False,
    wine_bottle_gripper_contact: bool = False,
    wine_rack_gripper_contact: bool = False,
) -> dict[str, object]:
    return {
        "episode_index": 0,
        "frame_index": frame_index,
        "is_success": success,
        "env_success": env_success,
        "black_bowl_controlled": black_bowl_controlled,
        "bottom_drawer_interacted": bottom_drawer_interacted,
        "bottom_drawer_is_close": bottom_drawer_is_close,
        "black_bowl_in_bottom_drawer": black_bowl_in_bottom_drawer,
        "wine_bottle_gripper_contact": wine_bottle_gripper_contact,
        "wine_rack_gripper_contact": wine_rack_gripper_contact,
    }


def _task4_trace_row(
    frame_index: int,
    *,
    success: bool,
    env_success: bool = False,
    porcelain_mug_controlled: bool = False,
    white_yellow_mug_controlled: bool = False,
    porcelain_mug_on_left_plate: bool = False,
    white_yellow_mug_on_right_plate: bool = False,
    porcelain_mug_on_right_plate: bool = False,
    white_yellow_mug_on_left_plate: bool = False,
    red_coffee_mug_gripper_contact: bool = False,
) -> dict[str, object]:
    return {
        "episode_index": 0,
        "frame_index": frame_index,
        "is_success": success,
        "env_success": env_success,
        "porcelain_mug_controlled": porcelain_mug_controlled,
        "white_yellow_mug_controlled": white_yellow_mug_controlled,
        "porcelain_mug_on_left_plate": porcelain_mug_on_left_plate,
        "white_yellow_mug_on_right_plate": white_yellow_mug_on_right_plate,
        "porcelain_mug_on_right_plate": porcelain_mug_on_right_plate,
        "white_yellow_mug_on_left_plate": white_yellow_mug_on_left_plate,
        "red_coffee_mug_gripper_contact": red_coffee_mug_gripper_contact,
    }


def _task5_trace_row(
    frame_index: int,
    *,
    success: bool,
    env_success: bool = False,
    black_book_controlled: bool = False,
    black_book_final_insertion_zone: bool = False,
    black_book_in_back_compartment: bool = False,
    black_book_back_distance: float = 0.30,
    black_book_caddy_contact: bool = False,
    white_yellow_mug_gripper_contact: bool = False,
) -> dict[str, object]:
    return {
        "episode_index": 0,
        "frame_index": frame_index,
        "is_success": success,
        "env_success": env_success,
        "black_book_controlled": black_book_controlled,
        "black_book_final_insertion_zone": black_book_final_insertion_zone,
        "black_book_in_back_compartment": black_book_in_back_compartment,
        "black_book_back_distance": black_book_back_distance,
        "black_book_caddy_contact": black_book_caddy_contact,
        "white_yellow_mug_gripper_contact": white_yellow_mug_gripper_contact,
    }


def _task6_trace_row(
    frame_index: int,
    *,
    success: bool,
    env_success: bool = False,
    porcelain_mug_controlled: bool = False,
    chocolate_pudding_controlled: bool = False,
    porcelain_mug_on_plate: bool = False,
    chocolate_pudding_on_plate: bool = False,
    chocolate_pudding_on_plate_left_region: bool = False,
    chocolate_pudding_on_plate_right_region: bool = False,
    red_coffee_mug_gripper_contact: bool = False,
) -> dict[str, object]:
    return {
        "episode_index": 0,
        "frame_index": frame_index,
        "is_success": success,
        "env_success": env_success,
        "porcelain_mug_controlled": porcelain_mug_controlled,
        "chocolate_pudding_controlled": chocolate_pudding_controlled,
        "porcelain_mug_on_plate": porcelain_mug_on_plate,
        "chocolate_pudding_on_plate": chocolate_pudding_on_plate,
        "chocolate_pudding_on_plate_left_region": chocolate_pudding_on_plate_left_region,
        "chocolate_pudding_on_plate_right_region": chocolate_pudding_on_plate_right_region,
        "red_coffee_mug_gripper_contact": red_coffee_mug_gripper_contact,
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


def test_task1_terminal_success_accepts_short_verified_completion() -> None:
    rows = []
    for frame_index in range(20):
        rows.append(
            _trace_row(
                frame_index,
                success=True,
                object_a_controlled=2 <= frame_index < 8,
                object_a_in_basket=8 <= frame_index,
                object_b_controlled=12 <= frame_index < 17,
                object_b_in_basket=17 <= frame_index,
            )
        )

    labels, audit = build_task1_phase_labels(pd.DataFrame(rows), stable_frames=5)

    assert audit.loc[0, "b1_frame"] == 2
    assert audit.loc[0, "b2_frame"] == 8
    assert audit.loc[0, "b3_frame"] == 17
    assert audit.loc[0, "b3_source"] == "state_terminal_success"
    assert bool(audit.loc[0, "b3_consistent_with_success"])
    assert labels.loc[labels["frame_index"] == 17, "phase"].item() == 3


def test_task1_env_success_completes_without_basket_contact() -> None:
    rows = []
    for frame_index in range(20):
        rows.append(
            _trace_row(
                frame_index,
                success=True,
                env_success=frame_index == 19,
                object_a_controlled=2 <= frame_index < 8,
                object_a_in_basket=8 <= frame_index,
                object_b_controlled=12 <= frame_index < 19,
                object_b_near=17 <= frame_index,
            )
        )

    labels, audit = build_task1_phase_labels(pd.DataFrame(rows), stable_frames=5)

    assert audit.loc[0, "b3_frame"] == 19
    assert audit.loc[0, "b3_source"] == "env_success_terminal"
    assert bool(audit.loc[0, "b3_consistent_with_success"])
    assert labels.loc[labels["frame_index"] == 19, "phase"].item() == 3


def test_task1_env_success_resolves_same_frame_b2_and_b3() -> None:
    rows = []
    for frame_index in range(20):
        rows.append(
            _trace_row(
                frame_index,
                success=True,
                env_success=frame_index == 19,
                object_a_controlled=2 <= frame_index < 10,
                object_b_controlled=2 <= frame_index < 10,
                object_a_in_basket=10 <= frame_index,
                object_b_in_basket=10 <= frame_index,
            )
        )

    labels, audit = build_task1_phase_labels(pd.DataFrame(rows), stable_frames=5)

    assert audit.loc[0, "b2_frame"] == 10
    assert audit.loc[0, "b3_frame"] == 19
    assert audit.loc[0, "b3_source"] == "env_success_terminal"
    assert labels.loc[labels["frame_index"] == 19, "phase"].item() == 3


def test_task2_stove_first_creates_b2_before_moka_completion() -> None:
    rows = []
    for frame_index in range(20):
        rows.append(
            _task2_trace_row(
                frame_index,
                success=True,
                stove_button_interacted=frame_index == 2,
                stove_turn_on=4 <= frame_index,
                moka_pot_controlled=6 <= frame_index < 11,
                moka_pot_on_cook_region=12 <= frame_index,
            )
        )

    labels, audit = build_task2_phase_labels(pd.DataFrame(rows), stable_frames=3)

    assert audit.loc[0, "b1_frame"] == 2
    assert audit.loc[0, "b1_source"] == "stove_button_interaction"
    assert audit.loc[0, "b2_frame"] == 4
    assert audit.loc[0, "b2_source"] == "stove_turn_on"
    assert audit.loc[0, "b3_frame"] == 12
    assert audit.loc[0, "b3_source"] == "state_stable"
    assert audit.loc[0, "subgoal_order"] == "stove_then_moka_pot"
    assert labels.loc[labels["frame_index"] == 11, "phase"].item() == 2
    assert labels.loc[labels["frame_index"] == 14, "phase"].item() == 3


def test_task2_moka_first_is_a_supported_subgoal_order() -> None:
    rows = []
    for frame_index in range(20):
        rows.append(
            _task2_trace_row(
                frame_index,
                success=True,
                moka_pot_controlled=2 <= frame_index < 8,
                moka_pot_on_cook_region=8 <= frame_index,
                stove_button_interacted=13 == frame_index,
                stove_turn_on=15 <= frame_index,
            )
        )

    labels, audit = build_task2_phase_labels(pd.DataFrame(rows), stable_frames=3)

    assert audit.loc[0, "b1_frame"] == 2
    assert audit.loc[0, "b2_frame"] == 8
    assert audit.loc[0, "b2_source"] == "moka_pot_on_cook_region"
    assert audit.loc[0, "b3_frame"] == 15
    assert audit.loc[0, "subgoal_order"] == "moka_pot_then_stove"
    assert labels.loc[labels["frame_index"] == 14, "phase"].item() == 2
    assert labels.loc[labels["frame_index"] == 17, "phase"].item() == 3


def test_task2_frypan_contact_does_not_start_a_phase() -> None:
    rows = [
        _task2_trace_row(
            frame_index,
            success=False,
            frypan_gripper_contact=2 <= frame_index < 12,
        )
        for frame_index in range(20)
    ]

    labels, audit = build_task2_phase_labels(pd.DataFrame(rows), stable_frames=3)

    assert pd.isna(audit.loc[0, "b1_frame"])
    assert bool(audit.loc[0, "frypan_gripper_contact_observed"])
    assert not bool(audit.loc[0, "trainable"])
    assert labels["phase"].eq(0).all()


def test_task2_drop_does_not_reset_phase_one() -> None:
    rows = []
    for frame_index in range(20):
        rows.append(
            _task2_trace_row(
                frame_index,
                success=False,
                moka_pot_controlled=2 <= frame_index < 6,
            )
        )

    labels, audit = build_task2_phase_labels(pd.DataFrame(rows), stable_frames=3)

    assert audit.loc[0, "b1_frame"] == 2
    assert pd.isna(audit.loc[0, "b2_frame"])
    assert pd.isna(audit.loc[0, "b3_frame"])
    assert bool(audit.loc[0, "trainable"])
    assert labels.loc[labels["frame_index"] == 10, "phase"].item() == 1
    assert labels["phase"].max() == 1


def test_task2_env_success_terminal_completes_short_final_state() -> None:
    rows = []
    for frame_index in range(20):
        rows.append(
            _task2_trace_row(
                frame_index,
                success=True,
                env_success=frame_index == 19,
                stove_button_interacted=frame_index == 2,
                stove_turn_on=4 <= frame_index,
                moka_pot_controlled=12 <= frame_index < 19,
            )
        )

    labels, audit = build_task2_phase_labels(pd.DataFrame(rows), stable_frames=3)

    assert audit.loc[0, "b1_frame"] == 2
    assert audit.loc[0, "b2_frame"] == 4
    assert audit.loc[0, "b3_frame"] == 19
    assert audit.loc[0, "b3_source"] == "env_success_terminal"
    assert bool(audit.loc[0, "b3_consistent_with_success"])
    assert labels.loc[labels["frame_index"] == 19, "phase"].item() == 3


def test_task3_bowl_then_close_creates_all_boundaries() -> None:
    rows = []
    for frame_index in range(20):
        rows.append(
            _task3_trace_row(
                frame_index,
                success=True,
                black_bowl_controlled=2 <= frame_index < 9,
                black_bowl_in_bottom_drawer=9 <= frame_index,
                bottom_drawer_interacted=13 == frame_index,
                bottom_drawer_is_close=15 <= frame_index,
            )
        )

    labels, audit = build_task3_phase_labels(pd.DataFrame(rows), stable_frames=3)

    assert audit.loc[0, "b1_frame"] == 2
    assert audit.loc[0, "b1_source"] == "black_bowl_controlled"
    assert audit.loc[0, "b2_frame"] == 9
    assert audit.loc[0, "b2_source"] == "black_bowl_in_bottom_drawer"
    assert audit.loc[0, "b3_frame"] == 15
    assert audit.loc[0, "b3_source"] == "state_stable"
    assert not bool(audit.loc[0, "drawer_closed_before_bowl_observed"])
    assert labels.loc[labels["frame_index"] == 14, "phase"].item() == 2
    assert labels.loc[labels["frame_index"] == 17, "phase"].item() == 3


def test_task3_early_drawer_close_does_not_create_b2() -> None:
    rows = []
    for frame_index in range(22):
        rows.append(
            _task3_trace_row(
                frame_index,
                success=True,
                bottom_drawer_interacted=frame_index == 2,
                bottom_drawer_is_close=4 <= frame_index < 7 or 16 <= frame_index,
                black_bowl_controlled=8 <= frame_index < 11,
                black_bowl_in_bottom_drawer=11 <= frame_index,
            )
        )

    labels, audit = build_task3_phase_labels(pd.DataFrame(rows), stable_frames=3)

    assert audit.loc[0, "b1_frame"] == 2
    assert audit.loc[0, "b2_frame"] == 11
    assert audit.loc[0, "b3_frame"] == 16
    assert bool(audit.loc[0, "drawer_closed_before_bowl_observed"])
    assert labels.loc[labels["frame_index"] == 6, "phase"].item() == 1
    assert labels.loc[labels["frame_index"] == 13, "phase"].item() == 2


def test_task3_distractor_contact_does_not_start_a_phase() -> None:
    rows = [
        _task3_trace_row(
            frame_index,
            success=False,
            wine_bottle_gripper_contact=2 <= frame_index < 8,
            wine_rack_gripper_contact=10 <= frame_index < 16,
        )
        for frame_index in range(20)
    ]

    labels, audit = build_task3_phase_labels(pd.DataFrame(rows), stable_frames=3)

    assert pd.isna(audit.loc[0, "b1_frame"])
    assert bool(audit.loc[0, "wine_bottle_gripper_contact_observed"])
    assert bool(audit.loc[0, "wine_rack_gripper_contact_observed"])
    assert not bool(audit.loc[0, "trainable"])
    assert labels["phase"].eq(0).all()


def test_task3_bowl_drop_does_not_reset_phase_one() -> None:
    rows = []
    for frame_index in range(20):
        rows.append(
            _task3_trace_row(
                frame_index,
                success=False,
                black_bowl_controlled=2 <= frame_index < 6,
            )
        )

    labels, audit = build_task3_phase_labels(pd.DataFrame(rows), stable_frames=3)

    assert audit.loc[0, "b1_frame"] == 2
    assert pd.isna(audit.loc[0, "b2_frame"])
    assert pd.isna(audit.loc[0, "b3_frame"])
    assert bool(audit.loc[0, "trainable"])
    assert labels.loc[labels["frame_index"] == 10, "phase"].item() == 1
    assert labels["phase"].max() == 1


def test_task3_env_success_terminal_completes_short_final_state() -> None:
    rows = []
    for frame_index in range(20):
        rows.append(
            _task3_trace_row(
                frame_index,
                success=True,
                env_success=frame_index == 19,
                black_bowl_controlled=2 <= frame_index < 8,
                black_bowl_in_bottom_drawer=8 <= frame_index,
            )
        )

    labels, audit = build_task3_phase_labels(pd.DataFrame(rows), stable_frames=3)

    assert audit.loc[0, "b1_frame"] == 2
    assert audit.loc[0, "b2_frame"] == 8
    assert audit.loc[0, "b3_frame"] == 19
    assert audit.loc[0, "b3_source"] == "env_success_terminal"
    assert bool(audit.loc[0, "b3_consistent_with_success"])
    assert labels.loc[labels["frame_index"] == 19, "phase"].item() == 3


def test_task4_porcelain_first_creates_b2_before_completion() -> None:
    rows = []
    for frame_index in range(22):
        rows.append(
            _task4_trace_row(
                frame_index,
                success=True,
                porcelain_mug_controlled=2 <= frame_index < 9,
                porcelain_mug_on_left_plate=9 <= frame_index,
                white_yellow_mug_controlled=12 <= frame_index < 17,
                white_yellow_mug_on_right_plate=17 <= frame_index,
            )
        )

    labels, audit = build_task4_phase_labels(pd.DataFrame(rows), stable_frames=3)

    assert audit.loc[0, "b1_frame"] == 2
    assert audit.loc[0, "b2_frame"] == 9
    assert audit.loc[0, "b2_source"] == "porcelain_mug_on_left_plate"
    assert audit.loc[0, "b3_frame"] == 17
    assert labels.loc[labels["frame_index"] == 15, "phase"].item() == 2
    assert labels.loc[labels["frame_index"] == 19, "phase"].item() == 3


def test_task4_white_yellow_first_is_a_supported_order() -> None:
    rows = []
    for frame_index in range(22):
        rows.append(
            _task4_trace_row(
                frame_index,
                success=True,
                white_yellow_mug_controlled=2 <= frame_index < 8,
                white_yellow_mug_on_right_plate=8 <= frame_index,
                porcelain_mug_controlled=11 <= frame_index < 16,
                porcelain_mug_on_left_plate=16 <= frame_index,
            )
        )

    labels, audit = build_task4_phase_labels(pd.DataFrame(rows), stable_frames=3)

    assert audit.loc[0, "b1_frame"] == 2
    assert audit.loc[0, "b2_frame"] == 8
    assert audit.loc[0, "b2_source"] == "white_yellow_mug_on_right_plate"
    assert audit.loc[0, "b3_frame"] == 16
    assert labels.loc[labels["frame_index"] == 14, "phase"].item() == 2


def test_task4_wrong_plate_does_not_create_b2() -> None:
    rows = []
    for frame_index in range(20):
        rows.append(
            _task4_trace_row(
                frame_index,
                success=False,
                porcelain_mug_controlled=2 <= frame_index < 7,
                porcelain_mug_on_right_plate=7 <= frame_index,
            )
        )

    labels, audit = build_task4_phase_labels(pd.DataFrame(rows), stable_frames=3)

    assert audit.loc[0, "b1_frame"] == 2
    assert pd.isna(audit.loc[0, "b2_frame"])
    assert bool(audit.loc[0, "incorrect_plate_placement_observed"])
    assert labels["phase"].max() == 1


def test_task4_red_coffee_contact_does_not_start_a_phase() -> None:
    rows = [
        _task4_trace_row(
            frame_index,
            success=False,
            red_coffee_mug_gripper_contact=2 <= frame_index < 12,
        )
        for frame_index in range(20)
    ]

    labels, audit = build_task4_phase_labels(pd.DataFrame(rows), stable_frames=3)

    assert pd.isna(audit.loc[0, "b1_frame"])
    assert bool(audit.loc[0, "red_coffee_mug_gripper_contact_observed"])
    assert not bool(audit.loc[0, "trainable"])
    assert labels["phase"].eq(0).all()


def test_task4_env_success_terminal_completes_short_final_state() -> None:
    rows = []
    for frame_index in range(20):
        rows.append(
            _task4_trace_row(
                frame_index,
                success=True,
                env_success=frame_index == 19,
                porcelain_mug_controlled=2 <= frame_index < 8,
                porcelain_mug_on_left_plate=8 <= frame_index,
                white_yellow_mug_controlled=12 <= frame_index < 19,
            )
        )

    labels, audit = build_task4_phase_labels(pd.DataFrame(rows), stable_frames=3)

    assert audit.loc[0, "b1_frame"] == 2
    assert audit.loc[0, "b2_frame"] == 8
    assert audit.loc[0, "b3_frame"] == 19
    assert audit.loc[0, "b3_source"] == "env_success_terminal"
    assert bool(audit.loc[0, "b3_consistent_with_success"])
    assert labels.loc[labels["frame_index"] == 19, "phase"].item() == 3


def test_task5_final_insertion_zone_creates_b2_before_completion() -> None:
    rows = []
    for frame_index in range(22):
        rows.append(
            _task5_trace_row(
                frame_index,
                success=True,
                black_book_controlled=2 <= frame_index < 10,
                black_book_final_insertion_zone=8 <= frame_index < 14,
                black_book_in_back_compartment=14 <= frame_index,
                black_book_back_distance=0.08 if frame_index >= 8 else 0.30,
                black_book_caddy_contact=8 <= frame_index,
            )
        )

    labels, audit = build_task5_phase_labels(pd.DataFrame(rows), stable_frames=3)

    assert audit.loc[0, "b1_frame"] == 2
    assert audit.loc[0, "b2_frame"] == 8
    assert audit.loc[0, "b2_source"] == "black_book_final_insertion_zone"
    assert audit.loc[0, "b3_frame"] == 14
    assert audit.loc[0, "b3_source"] == "state_stable"
    assert labels.loc[labels["frame_index"] == 12, "phase"].item() == 2
    assert labels.loc[labels["frame_index"] == 16, "phase"].item() == 3


def test_task5_book_drop_does_not_reset_phase_one() -> None:
    rows = []
    for frame_index in range(20):
        rows.append(
            _task5_trace_row(
                frame_index,
                success=False,
                black_book_controlled=2 <= frame_index < 6,
            )
        )

    labels, audit = build_task5_phase_labels(pd.DataFrame(rows), stable_frames=3)

    assert audit.loc[0, "b1_frame"] == 2
    assert pd.isna(audit.loc[0, "b2_frame"])
    assert pd.isna(audit.loc[0, "b3_frame"])
    assert bool(audit.loc[0, "trainable"])
    assert labels.loc[labels["frame_index"] == 10, "phase"].item() == 1


def test_task5_env_success_terminal_completes_short_final_state() -> None:
    rows = []
    for frame_index in range(20):
        rows.append(
            _task5_trace_row(
                frame_index,
                success=True,
                env_success=frame_index == 19,
                black_book_controlled=2 <= frame_index < 8,
                black_book_final_insertion_zone=9 <= frame_index < 19,
                black_book_back_distance=0.08 if frame_index >= 9 else 0.30,
                black_book_caddy_contact=9 <= frame_index,
            )
        )

    labels, audit = build_task5_phase_labels(pd.DataFrame(rows), stable_frames=3)

    assert audit.loc[0, "b1_frame"] == 2
    assert audit.loc[0, "b2_frame"] == 9
    assert audit.loc[0, "b3_frame"] == 19
    assert audit.loc[0, "b3_source"] == "env_success_terminal"
    assert bool(audit.loc[0, "b3_consistent_with_success"])
    assert labels.loc[labels["frame_index"] == 19, "phase"].item() == 3


def test_task6_porcelain_mug_first_creates_b2_before_completion() -> None:
    rows = [
        _task6_trace_row(
            frame,
            success=True,
            porcelain_mug_controlled=2 <= frame < 8,
            porcelain_mug_on_plate=8 <= frame,
            chocolate_pudding_controlled=11 <= frame < 16,
            chocolate_pudding_on_plate_right_region=16 <= frame,
        )
        for frame in range(22)
    ]
    labels, audit = build_task6_phase_labels(pd.DataFrame(rows), stable_frames=3)
    assert (
        audit.loc[0, "b1_frame"],
        audit.loc[0, "b2_frame"],
        audit.loc[0, "b3_frame"],
    ) == (2, 8, 16)
    assert audit.loc[0, "b2_source"] == "porcelain_mug_on_plate"
    assert labels.loc[labels["frame_index"] == 14, "phase"].item() == 2


def test_task6_chocolate_pudding_first_creates_b2_before_completion() -> None:
    rows = [
        _task6_trace_row(
            frame,
            success=True,
            chocolate_pudding_controlled=2 <= frame < 8,
            chocolate_pudding_on_plate_right_region=8 <= frame,
            porcelain_mug_controlled=11 <= frame < 16,
            porcelain_mug_on_plate=16 <= frame,
        )
        for frame in range(22)
    ]
    labels, audit = build_task6_phase_labels(pd.DataFrame(rows), stable_frames=3)
    assert audit.loc[0, "b1_frame"] == 2
    assert audit.loc[0, "b2_frame"] == 8
    assert audit.loc[0, "b2_source"] == "chocolate_pudding_on_plate_right_region"
    assert audit.loc[0, "b3_frame"] == 16
    assert labels.loc[labels["frame_index"] == 14, "phase"].item() == 2


def test_task6_incorrect_pudding_placement_does_not_create_b2() -> None:
    rows = [
        _task6_trace_row(
            frame,
            success=False,
            chocolate_pudding_controlled=2 <= frame < 8,
            chocolate_pudding_on_plate=8 <= frame,
            chocolate_pudding_on_plate_left_region=8 <= frame,
        )
        for frame in range(20)
    ]
    labels, audit = build_task6_phase_labels(pd.DataFrame(rows), stable_frames=3)
    assert audit.loc[0, "b1_frame"] == 2
    assert pd.isna(audit.loc[0, "b2_frame"])
    assert bool(audit.loc[0, "chocolate_pudding_incorrect_placement_observed"])
    assert labels["phase"].max() == 1


def test_task6_distractor_contact_does_not_start_phase() -> None:
    rows = [
        _task6_trace_row(
            frame,
            success=False,
            red_coffee_mug_gripper_contact=True,
        )
        for frame in range(20)
    ]
    labels, audit = build_task6_phase_labels(pd.DataFrame(rows), stable_frames=3)
    assert pd.isna(audit.loc[0, "b1_frame"])
    assert bool(audit.loc[0, "red_coffee_mug_gripper_contact_observed"])
    assert labels["phase"].eq(0).all()
