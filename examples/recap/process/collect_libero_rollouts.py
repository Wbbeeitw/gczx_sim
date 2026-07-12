#!/usr/bin/env python
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

"""Collect LIBERO rollouts and save them as a LeRobot dataset."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlinf.revalue.pipeline.embodied import (  # noqa: E402
    LiberoRolloutCollectionConfig,
    collect_libero_rollouts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task_suite_name", default="libero_10")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--num_episodes", type=int, default=64)
    parser.add_argument("--noise_scale", type=float, default=0.0)
    parser.add_argument("--noise_clip", type=float, default=0.3)
    parser.add_argument("--action_chunk", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--config_name", default="pi05_libero")
    parser.add_argument("--model_type", default="openpi")
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--failure_reward", type=float, default=None)
    parser.add_argument("--semantic_trace", action="store_true")
    parser.add_argument(
        "--semantic_trace_task",
        choices=("task1", "task2", "task3", "task4", "task5", "task6", "task7"),
        default="task1",
        help="Task-specific privileged semantic trace to record.",
    )
    parser.add_argument(
        "--semantic_trace_output_name", default="semantic_trace_task1"
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    summary = collect_libero_rollouts(
        LiberoRolloutCollectionConfig(
            output_dir=args.output_dir,
            model_path=args.pretrained_path,
            checkpoint_path=args.checkpoint_path,
            model_type=args.model_type,
            openpi_config_name=args.config_name,
            task_suite_name=args.task_suite_name,
            task_id=args.task_id,
            num_episodes=args.num_episodes,
            noise_scale=args.noise_scale,
            noise_clip=args.noise_clip,
            action_chunk=args.action_chunk,
            num_steps=args.num_steps,
            num_steps_wait=args.num_steps_wait,
            seed=args.seed,
            gpu_id=args.gpu_id,
            fps=args.fps,
            overwrite=args.overwrite,
            failure_reward=args.failure_reward,
            semantic_trace=args.semantic_trace,
            semantic_trace_task=args.semantic_trace_task,
            semantic_trace_output_name=args.semantic_trace_output_name,
        )
    )
    logging.getLogger(__name__).info("rollout collection summary: %s", summary)


if __name__ == "__main__":
    main()
