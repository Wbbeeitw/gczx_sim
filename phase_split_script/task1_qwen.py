"""Qwen-VL based phase annotation for LIBERO-10 Task 1.

Task instruction: "Put both the cream cheese box and the butter in the basket"
Semantic structure (6 phases):
    phase 0: approaching and grasping the first object
    phase 1: transporting the first object to the basket
    phase 2: placing the first object into the basket
    phase 3: approaching and grasping the second object
    phase 4: transporting the second object to the basket
    phase 5: placing the second object into the basket / task completion

Usage:
    # Local vLLM (default when QWEN_API_BASE is set)
    export QWEN_API_BASE=http://localhost:8001/v1
    export QWEN_MODEL=qwen-local
    python task1_qwen.py --dataset_path /data/libero_long/task1

    # DashScope (cloud)
    export DASHSCOPE_API_KEY=your_key
    python task1_qwen.py --dataset_path /data/libero_long/task1 \
        --model qwen3-vl-flash
"""

from __future__ import annotations

import argparse
import os

from qwen_annotator import annotate_dataset_with_qwen


TASK_DESCRIPTION = "Put both the cream cheese box and the butter in the basket"

PHASE_DEFINITIONS = {
    0: "approaching and grasping the first object (gripper open, moving toward it)",
    1: "transporting the first object to the basket (gripper closed, moving toward basket)",
    2: "placing the first object into the basket (gripper opening above basket)",
    3: "approaching and grasping the second object",
    4: "transporting the second object to the basket (gripper closed)",
    5: "placing the second object into the basket / task completion (gripper opening)",
}

NUM_PHASES = 6


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
        default="phase_progress_semantic_qwen",
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
        default=os.environ.get("QWEN_MODEL", "qwen-local"),
        help="Model name (local vLLM served name or DashScope model name).",
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
    parser.add_argument(
        "--enable_reasoning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable model reasoning/thinking. Default: enabled.",
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
        enable_reasoning=args.enable_reasoning,
    )
    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
