"""Phase/progress annotation for LIBERO-10 Task 1.

Task instruction: "Put both the cream cheese box and the butter in the basket"
Semantic structure (6 phases):
    phase 0: approaching and grasping the first object
    phase 1: transporting the first object to the basket
    phase 2: placing the first object into the basket
    phase 3: approaching and grasping the second object
    phase 4: transporting the second object to the basket
    phase 5: placing the second object into the basket / task completion

The annotation uses the actual gripper width from the state vector rather than
the commanded gripper action. Spatial zones (object A, basket, object B) are
estimated from the end-effector trajectory during stable gripper events, so no
external task knowledge is required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common_annotator import (
    annotate_dataset,
    detect_segments,
    smooth_1d,
)

# --------------------------------------------------------------------------- #
# State layout
# --------------------------------------------------------------------------- #
# State vector: [eef_pos(3), axis_angle(3), gripper_qpos(2)]
_EEF_POS_DIM = 3
_GRIPPER_QPOS_START = 6

# --------------------------------------------------------------------------- #
# Hyperparameters for gripper-event detection
# --------------------------------------------------------------------------- #
_SMOOTH_WINDOW = 11
_MIN_GRASP_FRAMES = 10
_MIN_OPEN_FRAMES = 5

# Percentiles for adaptive open/closed thresholds. These are intentionally
# conservative: a frame must be in the lower/upper quartile to count as
# closed/open, and the thresholds are clamped to avoid crossing.
_CLOSE_PERCENTILE = 25
_OPEN_PERCENTILE = 75
_CLOSE_CLAMP_PCT = 40
_OPEN_CLAMP_PCT = 60

# --------------------------------------------------------------------------- #
# Hyperparameters for spatial-zone estimation
# --------------------------------------------------------------------------- #
_ZONE_STD_SCALE = 3.0
# Basket is a target region: be slightly more generous than object zones.
_BASKET_RADIUS_SCALE = 1.5
_MIN_ZONE_RADIUS_RATIO = 0.05

# Index convention for estimated zones.
_OBJ_A = 0
_BASKET = 1
_OBJ_B = 2


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _gripper_width(state: np.ndarray) -> np.ndarray:
    """Extract actual gripper width from the state vector.

    LIBERO/robosuite returns two mirrored qpos values (one per finger). The
    physical width is the sum of their absolute values.
    """
    if state.shape[1] > _GRIPPER_QPOS_START:
        qpos = state[:, _GRIPPER_QPOS_START:]
        return np.abs(qpos).sum(axis=-1)
    return np.zeros(len(state), dtype=np.float32)


def _adaptive_gripper_thresholds(width_smooth: np.ndarray) -> tuple[float, float]:
    """Compute closed/open width thresholds from the episode statistics."""
    close_thr = float(np.percentile(width_smooth, _CLOSE_PERCENTILE))
    open_thr = float(np.percentile(width_smooth, _OPEN_PERCENTILE))

    # Safety clamps so that close_thr and open_thr do not cross.
    close_thr = min(close_thr, float(np.percentile(width_smooth, _CLOSE_CLAMP_PCT)))
    open_thr = max(open_thr, float(np.percentile(width_smooth, _OPEN_CLAMP_PCT)))

    return close_thr, open_thr


def _zone_radius(positions: np.ndarray, global_range: float) -> float:
    """Estimate a spatial radius from a cluster of 3-D positions."""
    std = float(positions.std(axis=0).mean()) if len(positions) > 1 else 0.0
    return max(_ZONE_STD_SCALE * std, _MIN_ZONE_RADIUS_RATIO * global_range)


def _estimate_zones(
    eef_pos: np.ndarray,
    closed_segments: list[tuple[int, int]],
    open_segments: list[tuple[int, int]],
) -> tuple[list[np.ndarray | None], list[float | None]]:
    """Estimate object-A, basket, and object-B zones from the trajectory.

    Zones are inferred purely from gripper events in this episode:
        * first stable closed segment  -> object A location
        * first stable open segment    -> basket location
        * second stable closed segment -> object B location

    Returns:
        (centers, radii) where each is a 3-element list indexed by _OBJ_A,
        _BASKET, _OBJ_B. Missing zones are None.
    """
    global_range = float(np.ptp(eef_pos, axis=0).max())
    centers: list[np.ndarray | None] = [None, None, None]
    radii: list[float | None] = [None, None, None]

    # Object A: mean eef pose during the first stable grasp.
    if closed_segments:
        s, e = closed_segments[_OBJ_A]
        pos = eef_pos[s:e]
        centers[_OBJ_A] = pos.mean(axis=0)
        radii[_OBJ_A] = _zone_radius(pos, global_range)

    # Basket: mean eef pose during the first stable release after grasping A.
    if closed_segments and open_segments:
        grasp_a_end = closed_segments[_OBJ_A][1]
        first_release = next(
            (seg for seg in open_segments if seg[0] >= grasp_a_end), None
        )
        if first_release is not None:
            s, e = first_release
            pos = eef_pos[s:e]
            centers[_BASKET] = pos.mean(axis=0)
            radii[_BASKET] = _zone_radius(pos, global_range) * _BASKET_RADIUS_SCALE

    # Object B: mean eef pose during the second stable grasp.
    if len(closed_segments) >= 2:
        s, e = closed_segments[_OBJ_B]
        pos = eef_pos[s:e]
        centers[_OBJ_B] = pos.mean(axis=0)
        radii[_OBJ_B] = _zone_radius(pos, global_range)

    return centers, radii


def _first_entry_frame(
    eef_pos: np.ndarray,
    center: np.ndarray | None,
    radius: float | None,
    start: int,
) -> int | None:
    """Return the first frame >= start whose eef position lies inside a zone."""
    if center is None or radius is None:
        return None
    for t in range(start, len(eef_pos)):
        if float(np.linalg.norm(eef_pos[t] - center)) < radius:
            return t
    return None


# --------------------------------------------------------------------------- #
# Main annotator
# --------------------------------------------------------------------------- #
def annotate_task1(
    actions: np.ndarray,
    state: np.ndarray,
    episode_length: int,
) -> tuple[np.ndarray, int]:
    """Annotate one episode for task1.

    Args:
        actions: episode actions, shape (T, action_dim). Last dim is gripper.
        state: episode state, shape (T, state_dim). Contains eef_pos and
            actual gripper qpos.
        episode_length: T (redundant, kept for interface consistency).

    Returns:
        phase array of shape (T,) and total number of phases (6).
    """
    _ = episode_length
    num_phases = 6
    phase = np.zeros(len(actions), dtype=int)

    eef_pos = state[:, :_EEF_POS_DIM]
    width = _gripper_width(state)
    width_smooth = smooth_1d(width.astype(np.float32), window=_SMOOTH_WINDOW)
    close_thr, open_thr = _adaptive_gripper_thresholds(width_smooth)

    closed = width_smooth < close_thr
    opened = width_smooth > open_thr

    closed_segments = detect_segments(closed, min_len=_MIN_GRASP_FRAMES)
    open_segments = detect_segments(opened, min_len=_MIN_OPEN_FRAMES)

    centers, radii = _estimate_zones(eef_pos, closed_segments, open_segments)

    # ------------------------------------------------------------------ #
    # Degenerate case: no stable grasp at all.
    # ------------------------------------------------------------------ #
    if not closed_segments:
        return phase, num_phases

    # ------------------------------------------------------------------ #
    # First object
    # ------------------------------------------------------------------ #
    grasp_a_end = closed_segments[_OBJ_A][1]
    enter_basket_a = _first_entry_frame(
        eef_pos, centers[_BASKET], radii[_BASKET], start=grasp_a_end
    )
    release_a = next(
        (seg for seg in open_segments if seg[0] >= grasp_a_end), None
    )

    # Phase 0: approach and grasp the first object.
    phase[:grasp_a_end] = 0

    if enter_basket_a is not None and release_a is not None:
        release_a_end = release_a[1]
        # Phase 1: transport A to basket.
        phase[grasp_a_end:enter_basket_a] = 1
        # Phase 2: place A into basket.
        phase[enter_basket_a:release_a_end] = 2
        cursor = release_a_end
    elif release_a is not None:
        # Basket zone not found; treat everything until release as placing.
        release_a_end = release_a[1]
        phase[grasp_a_end:release_a_end] = 2
        cursor = release_a_end
    elif enter_basket_a is not None:
        # Reached basket but never released A.
        phase[grasp_a_end:enter_basket_a] = 1
        phase[enter_basket_a:] = 2
        return phase, num_phases
    else:
        # Grasped A but nothing else happened.
        phase[grasp_a_end:] = 1
        return phase, num_phases

    # ------------------------------------------------------------------ #
    # Second object
    # ------------------------------------------------------------------ #
    if len(closed_segments) < 2:
        # Placed A but never grasped B.
        phase[cursor:] = 3
        return phase, num_phases

    grasp_b_end = closed_segments[_OBJ_B][1]
    enter_basket_b = _first_entry_frame(
        eef_pos, centers[_BASKET], radii[_BASKET], start=grasp_b_end
    )
    release_b = next(
        (seg for seg in open_segments if seg[0] >= grasp_b_end), None
    )

    # Phase 3: approach and grasp the second object.
    phase[cursor:grasp_b_end] = 3

    if enter_basket_b is not None and release_b is not None:
        release_b_end = release_b[1]
        # Phase 4: transport B to basket.
        phase[grasp_b_end:enter_basket_b] = 4
        # Phase 5: place B into basket / completion.
        phase[enter_basket_b:release_b_end] = 5
        phase[release_b_end:] = 5
    elif release_b is not None:
        release_b_end = release_b[1]
        phase[grasp_b_end:release_b_end] = 5
        phase[release_b_end:] = 5
    elif enter_basket_b is not None:
        phase[grasp_b_end:enter_basket_b] = 4
        phase[enter_basket_b:] = 5
    else:
        # Grasped B but neither basket entry nor release detected.
        phase[grasp_b_end:] = 4

    return phase, num_phases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate phases for LIBERO-10 Task 1 rollouts."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to LeRobot dataset root (contains data/ and meta/).",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="phase_progress_semantic",
        help="Output parquet name (saved under dataset/meta/).",
    )
    args = parser.parse_args()

    out_path = annotate_dataset(
        dataset_path=args.dataset_path,
        annotator=annotate_task1,
        output_name=args.output_name,
    )
    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
