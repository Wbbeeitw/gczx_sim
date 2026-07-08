"""Phase/progress annotation for LIBERO-10 Task 1.

Task instruction: "Put both the cream cheese box and the butter in the basket"
Semantic structure (5 phases):
    phase 0: approaching / pre-first-grasp
    phase 1: grasping/holding the first object
    phase 2: between first release and second grasp
    phase 3: grasping/holding the second object
    phase 4: after second release / task completion

The annotation is based on detecting two sustained gripper-close segments from
 the action vector's last dimension (gripper).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common_annotator import annotate_dataset, detect_grasp_segments


def annotate_task1(actions: np.ndarray, episode_length: int) -> tuple[np.ndarray, int]:
    """Annotate one episode for task1.

    Args:
        actions: episode actions, shape (T, action_dim). Last dim is gripper.
        episode_length: T (redundant, kept for interface consistency).

    Returns:
        phase array of shape (T,) and total number of phases (5).
    """
    _ = episode_length
    gripper = actions[:, -1]

    # Detect all sustained gripper-close segments.
    segments = detect_grasp_segments(
        gripper,
        threshold=0.0,
        smooth_window=11,
        min_len=15,
    )

    # Keep the two longest segments (the two grasps), then sort by time.
    segments = sorted(segments, key=lambda s: s[1] - s[0], reverse=True)[:2]
    segments = sorted(segments, key=lambda s: s[0])

    num_phases = 5
    phase = np.zeros(len(actions), dtype=int)
    if not segments:
        return phase, num_phases

    # Build boundaries: start, seg1_start, seg1_end, seg2_start, seg2_end, end.
    boundaries = [0]
    labels = [0]
    for seg in segments:
        boundaries.append(seg[0])
        labels.append(labels[-1] + 1)
        boundaries.append(seg[1])
        labels.append(labels[-1] + 1)
    boundaries.append(len(actions))
    labels = labels[: len(boundaries) - 1]

    for li in range(len(boundaries) - 1):
        phase[boundaries[li] : boundaries[li + 1]] = labels[li]

    # Clip phase labels to valid range in case of unexpected segments.
    phase = np.clip(phase, 0, num_phases - 1)
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
