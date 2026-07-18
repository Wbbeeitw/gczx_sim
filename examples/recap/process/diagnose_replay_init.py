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

"""Diagnose demo-to-init-state mapping by comparing first frames.

Open-loop replay starts each demo from task init_states[ordinal]. If the
dataset's demo order does not match the init_states order, replays start
from the wrong object poses and are doomed. This renders frame zero for
every candidate init state and ranks them by pixel difference against the
demo's stored first frame, revealing the true mapping.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlinf.revalue.pipeline.embodied import (  # noqa: E402
    LIBERO_ENV_RESOLUTION,
    _get_libero_env,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--task_suite_name", default="libero_10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--top", type=int, default=5)
    return parser.parse_args()


def _demo_first_frame(dataset: Path, episode_index: int) -> np.ndarray:
    matches = sorted(dataset.glob(f"data/**/episode_{episode_index:06d}.parquet"))
    if len(matches) != 1:
        raise ValueError(f"expected one parquet for episode {episode_index}")
    frame = pd.read_parquet(matches[0])
    image_columns = [
        column
        for column in frame.columns
        if "image" in column and "wrist" not in column and "hand" not in column
    ]
    if not image_columns:
        raise ValueError(f"no image column in {matches[0]}: {list(frame.columns)}")
    image = np.asarray(frame[image_columns[0]].iloc[0])
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return image


def _render_first_frame(env, init_state) -> np.ndarray:
    obs = env.set_init_state(init_state)
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    from libero.libero import benchmark

    dataset = Path(args.dataset)
    reference = _demo_first_frame(dataset, args.episode).astype(np.float32)

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    init_states = task_suite.get_task_init_states(args.task_id)
    env, task_description = _get_libero_env(
        task, LIBERO_ENV_RESOLUTION, args.seed, args.gpu_id
    )
    logger.info("task %d: %s", args.task_id, task_description)

    scores = []
    for index in range(len(init_states)):
        candidate = _render_first_frame(env, init_states[index]).astype(np.float32)
        if candidate.shape != reference.shape:
            raise ValueError(
                f"shape mismatch: demo {reference.shape} vs env {candidate.shape}"
            )
        score = float(np.abs(candidate - reference).mean())
        scores.append((score, index))
    scores.sort()
    print(f"demo episode {args.episode} first-frame matches (score=mean|diff|):")
    for score, index in scores[: args.top]:
        print(f"  init_states[{index:2d}]  diff={score:8.2f}")
    best_score, best_index = scores[0]
    median = float(np.median([score for score, _ in scores]))
    print(
        f"best=init_states[{best_index}] diff={best_score:.2f} "
        f"median={median:.2f} ratio={best_score / max(median, 1e-6):.3f}"
    )
    print(
        "hint: ratio << 1 means a clear true match exists; "
        "best != your ordinal means the mapping is wrong."
    )


if __name__ == "__main__":
    main()
