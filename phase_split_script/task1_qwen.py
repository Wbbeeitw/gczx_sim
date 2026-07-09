"""Qwen-VL based phase annotation for LIBERO-10 Task 1.

Task instruction: "Put both the cream cheese box and the butter in the basket"
Semantic structure (7 phases):
    phase 0: approaching the first object
    phase 1: grasping the first object
    phase 2: transporting the first object to the basket
    phase 3: placing/releasing the first object into the basket
    phase 4: approaching the second object
    phase 5: grasping the second object
    phase 6: transporting and placing the second object / task completion
"""

from __future__ import annotations

import argparse

from qwen_annotator import annotate_dataset_with_qwen


TASK_DESCRIPTION = "Put both the cream cheese box and the butter in the basket"

PHASE_DEFINITIONS = {
    0: "approaching the first object (gripper open, moving toward it)",
    1: "grasping the first object (gripper closing or closed around it)",
    2: "transporting the first object (gripper closed, moving toward the basket)",
    3: "placing the first object (gripper opening above the basket to release it)",
    4: "approaching the second object",
    5: "grasping the second object",
    6: "transporting and placing the second object / task completion",
}

NUM_PHASES = 7


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate phases for LIBERO-10 Task 1 rollouts using Qwen-VL."
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
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=10,
        help="Call Qwen-VL every N frames.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen-vl-plus",
        help="DashScope model name (e.g. qwen-vl-plus, qwen-vl-max).",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=2,
        help="Number of episodes to process in parallel.",
    )
    parser.add_argument(
        "--no_wrist",
        action="store_true",
        help="Do not pass the wrist camera image to Qwen-VL.",
    )
    args = parser.parse_args()

    out_path = annotate_dataset_with_qwen(
        dataset_path=args.dataset_path,
        task_description=TASK_DESCRIPTION,
        phase_definitions=PHASE_DEFINITIONS,
        num_phases=NUM_PHASES,
        output_name=args.output_name,
        sample_interval=args.sample_interval,
        model=args.model,
        wrist_video_key=None if args.no_wrist else "wrist_image",
        max_workers=args.max_workers,
    )
    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
