"""Phase/progress annotation for LIBERO-10 Task 1.

Task instruction: "Put both the cream cheese box and the butter in the basket"
Semantic structure (7 phases):
    phase 0: approaching the first object
    phase 1: stably grasping the first object
    phase 2: transporting the first object to the basket
    phase 3: placing/releasing the first object into the basket
    phase 4: approaching the second object
    phase 5: stably grasping the second object
    phase 6: transporting and placing the second object / task completion

The annotation uses the actual gripper width from the state vector rather than
 the commanded gripper action, so that unstable grasps and slips are not
 incorrectly treated as successful phase transitions.
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


# Number of dimensions before gripper qpos in the state vector.
# State layout: [eef_pos(3), axis_angle(3), gripper_qpos(...)]
_GRIPPER_QPOS_START = 6

# Minimum duration (frames) for a stable grasp or a release.
_MIN_GRASP_FRAMES = 10
_MIN_OPEN_FRAMES = 5


def _gripper_width(state: np.ndarray) -> np.ndarray:
    """Extract actual gripper width from the state vector."""
    if state.shape[1] > _GRIPPER_QPOS_START:
        return state[:, _GRIPPER_QPOS_START:].sum(axis=-1)
    # Fallback if state is not available.
    return np.zeros(len(state), dtype=np.float32)


def _adaptive_gripper_thresholds(
    width_smooth: np.ndarray,
    gripper_action: np.ndarray,
) -> tuple[float, float]:
    """Compute closed/open width thresholds from data statistics.

    Uses the commanded gripper action to find frames where the policy intended
    to close or open the gripper, then takes percentiles of the actual width
    during those intervals.
    """
    # In LIBERO/OpenPI action space, positive gripper action means close.
    close_intent = gripper_action > 0.0
    open_intent = gripper_action <= 0.0

    if close_intent.any():
        close_thr = float(np.percentile(width_smooth[close_intent], 25))
    else:
        close_thr = float(np.percentile(width_smooth, 25))

    if open_intent.any():
        open_thr = float(np.percentile(width_smooth[open_intent], 75))
    else:
        open_thr = float(np.percentile(width_smooth, 75))

    # Safety clamps so that close_thr and open_thr do not cross.
    close_thr = min(close_thr, float(np.percentile(width_smooth, 40)))
    open_thr = max(open_thr, float(np.percentile(width_smooth, 60)))

    return close_thr, open_thr


def annotate_task1(
    actions: np.ndarray,
    state: np.ndarray,
    episode_length: int,
) -> tuple[np.ndarray, int]:
    """Annotate one episode for task1.

    Args:
        actions: episode actions, shape (T, action_dim). Last dim is gripper.
        state: episode state, shape (T, state_dim). Contains actual gripper qpos.
        episode_length: T (redundant, kept for interface consistency).

    Returns:
        phase array of shape (T,) and total number of phases (7).
    """
    _ = episode_length
    gripper_action = actions[:, -1]
    width = _gripper_width(state)

    # Smooth the actual gripper width to reduce single-frame noise.
    width_smooth = smooth_1d(width.astype(np.float32), window=11)

    close_thr, open_thr = _adaptive_gripper_thresholds(width_smooth, gripper_action)

    closed = width_smooth < close_thr
    opened = width_smooth > open_thr

    closed_segments = detect_segments(closed, min_len=_MIN_GRASP_FRAMES)
    open_segments = detect_segments(opened, min_len=_MIN_OPEN_FRAMES)

    num_phases = 7
    phase = np.zeros(len(actions), dtype=int)

    if not closed_segments:
        # No stable grasp detected; entire episode stays in phase 0.
        return phase, num_phases

    # First object: approach -> grasp -> transport -> place
    s0, e0 = closed_segments[0]
    phase[:s0] = 0          # approach A
    phase[s0:e0] = 1        # grasp A

    first_open = next((seg for seg in open_segments if seg[0] >= e0), None)
    if first_open is None:
        # Grasped A but never released it (episode will be a failure).
        phase[e0:] = 2      # transport A (incomplete)
        return phase, num_phases

    os0, oe0 = first_open
    phase[e0:os0] = 2       # transport A
    phase[os0:oe0] = 3      # place A

    # Second object: approach -> grasp -> transport/place
    if len(closed_segments) < 2:
        # Placed A but never stably grasped B.
        phase[oe0:] = 4     # approach B (incomplete)
        return phase, num_phases

    s1, e1 = closed_segments[1]
    phase[oe0:s1] = 4       # approach B
    phase[s1:e1] = 5        # grasp B
    phase[e1:] = 6          # transport & place B / completion

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
