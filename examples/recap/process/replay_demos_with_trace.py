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

"""Replay LeRobot demo episodes in the simulator and record semantic traces.

Official LIBERO demo datasets contain observations and actions but no
privileged simulator state (object poses, contacts). To feed demos into the
critic pipeline, this script replays each demo's action sequence in the
matching LIBERO environment while the task's semantic trace recorder
captures the privileged state, then writes a rollout-format dataset with
phase labels — indistinguishable from a freshly collected rollout dataset.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import logging
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlinf.data.lerobot_writer import LeRobotDatasetWriter  # noqa: E402
from rlinf.revalue.pipeline.embodied import (  # noqa: E402
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    _build_features,
    _compute_returns,
    _get_libero_env,
    _quat2axisangle,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="source demo dataset")
    parser.add_argument("--out", required=True, help="output replayed dataset")
    parser.add_argument("--task_suite_name", default="libero_10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--fps", type=int, default=0, help="0 = keep source fps")
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--max_episodes", "--max-episodes", dest="max_episodes", type=int, default=0)
    parser.add_argument("--task_ids", type=int, nargs="*", default=None)
    parser.add_argument("--no_match_init", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _decode_image(value) -> np.ndarray:
    """Decode a LeRobot image cell (ndarray or {bytes, path} dict) to uint8."""
    if isinstance(value, dict):
        payload = value.get("bytes")
        if payload is not None:
            from io import BytesIO

            from PIL import Image

            return np.array(Image.open(BytesIO(payload)).convert("RGB"))
        raise ValueError(f"image dict has no bytes: {sorted(value.keys())}")
    return np.asarray(value)


def _episode_first_frame(dataset: Path, episode_index: int) -> np.ndarray:
    matches = sorted(dataset.glob(f"data/**/episode_{episode_index:06d}.parquet"))
    frame = pd.read_parquet(matches[0])
    column = next(
        column
        for column in frame.columns
        if "image" in column and "wrist" not in column and "hand" not in column
    )
    image = _decode_image(frame[column].iloc[0])
    return image.astype(np.float32)


def _render_init_candidates(env, init_states, num_steps_wait) -> list[np.ndarray]:
    """Render the settled first frame of every candidate init state once."""
    candidates = []
    for init_state in init_states:
        env.reset()
        obs = env.set_init_state(init_state)
        for _ in range(num_steps_wait):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        candidates.append(
            np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]).astype(
                np.float32
            )
        )
    return candidates


def _match_init_index(reference: np.ndarray, candidates: list[np.ndarray]) -> tuple[int, float, float]:
    """Pick the init candidate closest to the demo first frame.

    The 90th-percentile pixel difference emphasizes object placement over
    background similarity. Returns (index, score, score/median).
    """
    if reference.shape != candidates[0].shape:
        raise ValueError(
            f"shape mismatch: demo {reference.shape} vs env {candidates[0].shape}"
        )
    scores = []
    for candidate in candidates:
        diff = np.abs(candidate - reference)
        scores.append(float(np.percentile(diff, 90)))
    best = int(np.argmin(scores))
    median = float(np.median(scores))
    return best, scores[best], scores[best] / max(median, 1e-6)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _episode_actions(dataset: Path, episode_index: int) -> np.ndarray:
    matches = sorted(dataset.glob(f"data/**/episode_{episode_index:06d}.parquet"))
    if len(matches) != 1:
        raise ValueError(f"expected one parquet for episode {episode_index}")
    frame = pd.read_parquet(matches[0])
    for column in ("action", "actions"):
        if column in frame.columns:
            values = frame[column].to_numpy()
            return np.stack([np.asarray(value, dtype=np.float32) for value in values])
    raise ValueError(f"no action column found in {matches[0]}")


def _task_groups(episodes: list[dict[str, Any]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for episode in episodes:
        task_text = " ".join(episode.get("tasks", []))
        groups[task_text].append(int(episode["episode_index"]))
    return groups


def _trace_specs():
    from rlinf.revalue import semantic_trace as st

    return [
        ("alphabet soup and the tomato sauce", 0, st.Task0SemanticTraceRecorder, st.write_task0_semantic_artifacts, "semantic_trace_task0"),
        ("cream cheese box and the butter", 1, st.Task1SemanticTraceRecorder, st.write_task1_semantic_artifacts, "semantic_trace_task1"),
        ("turn on the stove and put the moka pot", 2, st.Task2SemanticTraceRecorder, st.write_task2_semantic_artifacts, "semantic_trace_task2"),
        ("black bowl in the bottom drawer", 3, st.Task3SemanticTraceRecorder, st.write_task3_semantic_artifacts, "semantic_trace_task3"),
        ("white mug on the left plate", 4, st.Task4SemanticTraceRecorder, st.write_task4_semantic_artifacts, "semantic_trace_task4"),
        ("back compartment of the caddy", 5, st.Task5SemanticTraceRecorder, st.write_task5_semantic_artifacts, "semantic_trace_task5"),
        ("chocolate pudding to the right of the plate", 6, st.Task6SemanticTraceRecorder, st.write_task6_semantic_artifacts, "semantic_trace_task6"),
        ("alphabet soup and the cream cheese", 7, st.Task7SemanticTraceRecorder, st.write_task7_semantic_artifacts, "semantic_trace_task7"),
        ("both moka pots", 8, st.Task8SemanticTraceRecorder, st.write_task8_semantic_artifacts, "semantic_trace_task8"),
        ("microwave and close it", 9, st.Task9SemanticTraceRecorder, st.write_task9_semantic_artifacts, "semantic_trace_task9"),
    ]


def _match_spec(task_text: str):
    lowered = task_text.lower()
    for needle, task_id, recorder, writer, name in _trace_specs():
        if needle in lowered:
            return task_id, recorder, writer, name
    raise ValueError(f"no semantic trace spec matches task: {task_text!r}")


def _replay_episode(env, recorder, actions, episode_index, num_steps_wait):
    frames: list[dict[str, Any]] = []
    trace_records: list[dict[str, Any]] = []
    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    done = False
    for frame_index, action in enumerate(actions):
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        state = np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ).astype(np.float32)
        obs, reward, done, _ = env.step(np.asarray(action, dtype=np.float32).tolist())
        trace_records.append(
            recorder.capture(env, episode_index, frame_index, observation=obs)
        )
        frames.append(
            {
                "image": img,
                "wrist_image": wrist_img,
                "state": state,
                "actions": np.asarray(action, dtype=np.float32),
                "reward": np.array([float(reward)], dtype=np.float32),
                "task": "",
                "done": np.array([False], dtype=bool),
                "is_success": np.array([False], dtype=bool),
            }
        )
        if done:
            break
    return frames, trace_records, bool(done)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    from libero.libero import benchmark

    dataset = Path(args.dataset)
    out = Path(args.out)
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {out}; use --overwrite")
        shutil.rmtree(out)

    episodes = _read_jsonl(dataset / "meta" / "episodes.jsonl")
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
    info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = args.fps or int(info.get("fps", 20))

    groups = _task_groups(episodes)
    if args.task_ids is not None:
        wanted = set(args.task_ids)
        groups = {
            task_text: indices
            for task_text, indices in groups.items()
            if _match_spec(task_text)[0] in wanted
        }
        if not groups:
            raise SystemExit(f"no episodes matched task_ids={args.task_ids}")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()

    writer = LeRobotDatasetWriter()
    summary: dict[str, Any] = {"dataset": str(dataset), "out": str(out), "tasks": {}}
    pending_artifacts: list[tuple] = []
    mapping_report: list[dict[str, Any]] = []
    for task_text, episode_indices in sorted(groups.items()):
        task_id, recorder_cls, writer_fn, trace_name = _match_spec(task_text)
        task = task_suite.get_task(task_id)
        init_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(
            task, LIBERO_ENV_RESOLUTION, args.seed, args.gpu_id
        )
        recorder = recorder_cls(env)
        init_candidates = None
        if not args.no_match_init:
            init_candidates = _render_init_candidates(
                env, init_states, args.num_steps_wait
            )
            logger.info(
                "task%d: rendered %d init candidates for first-frame matching",
                task_id,
                len(init_candidates),
            )
        task_trace_records: list[dict[str, Any]] = []
        successes = 0
        for ordinal, episode_index in enumerate(sorted(episode_indices)):
            actions = _episode_actions(dataset, episode_index)
            if init_candidates is not None:
                reference = _episode_first_frame(dataset, episode_index)
                chosen_index, score, ratio = _match_init_index(
                    reference, init_candidates
                )
                mapping_report.append(
                    {
                        "task_id": task_id,
                        "episode_index": episode_index,
                        "init_index": chosen_index,
                        "score": score,
                        "ratio": ratio,
                    }
                )
            else:
                chosen_index = ordinal % len(init_states)
            env.reset()
            obs = env.set_init_state(init_states[chosen_index])
            frames, trace_records, is_success = _replay_episode(
                env, recorder, actions, episode_index, args.num_steps_wait
            )
            if not frames:
                logger.warning("episode %d replayed empty, skipping", episode_index)
                continue
            successes += int(is_success)
            returns = _compute_returns(
                [float(frame["reward"][0]) for frame in frames], gamma=1.0
            )
            for frame_index, frame in enumerate(frames):
                frame["is_success"] = np.array([is_success], dtype=bool)
                frame["return"] = np.array([returns[frame_index]], dtype=np.float32)
                frame["task"] = str(task_description)
            for record in trace_records:
                record["is_success"] = is_success
            frames[-1]["done"] = np.array([True], dtype=bool)
            task_trace_records.extend(trace_records)

            if writer.dataset is None:
                first = frames[0]
                writer.create(
                    repo_id=str(out),
                    robot_type="franka_panda",
                    fps=fps,
                    features=_build_features(
                        first["image"].shape,
                        int(first["state"].shape[-1]),
                        int(first["actions"].shape[-1]),
                    ),
                    image_shape=first["image"].shape,
                    state_dim=int(first["state"].shape[-1]),
                    action_dim=int(first["actions"].shape[-1]),
                    has_image=True,
                    wrist_image_keys={"wrist_image": first["wrist_image"].shape},
                    has_intervene_flag=False,
                )
            writer.add_episode(frames)
            logger.info(
                "task%d episode %d: len=%d success=%s total=%d/%d",
                task_id,
                episode_index,
                len(frames),
                is_success,
                successes,
                ordinal + 1,
            )
        # NB: no env.close() here; EGL teardown between task groups has been
        # a crash point, contexts are released at process exit instead.
        pending_artifacts.append(
            (
                writer_fn,
                trace_name,
                task_trace_records,
                recorder.metadata,
                recorder.config.stable_frames,
                task_id,
                task_description,
                len(episode_indices),
                successes,
            )
        )
        # Crash-safe staging: persist this task's trace artifacts outside the
        # dataset root immediately (lerobot's root-wide parquet glob forbids
        # writing them into meta/ before finalize).
        staging_dir = out.parent / f"{out.name}_trace_staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        staged = writer_fn(
            staging_dir,
            task_trace_records,
            recorder.metadata,
            output_name=trace_name,
            stable_frames=recorder.config.stable_frames,
        )
        logger.info("staged trace artifacts for task%d: %s", task_id, staged)

    writer.finalize()
    staging_dir = out.parent / f"{out.name}_trace_staging"
    staging_meta = staging_dir / "meta"
    for staged_file in sorted(staging_meta.rglob("*")):
        if staged_file.is_file():
            target = out / "meta" / staged_file.relative_to(staging_meta)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staged_file, target)
    for (
        writer_fn,
        trace_name,
        task_trace_records,
        recorder_metadata,
        stable_frames,
        task_id,
        task_description,
        num_episodes,
        successes,
    ) in pending_artifacts:
        artifacts = {
            key: str(out / "meta" / Path(value).relative_to(staging_dir / "meta"))
            for key, value in writer_fn(
                staging_dir,
                task_trace_records,
                recorder_metadata,
                output_name=trace_name,
                stable_frames=stable_frames,
            ).items()
        }
        summary["tasks"][str(task_id)] = {
            "task": task_description,
            "episodes": num_episodes,
            "replayed_success": successes,
            "trace_artifacts": artifacts,
        }
    out.mkdir(parents=True, exist_ok=True)
    if mapping_report:
        with (out / "init_mapping_report.json").open("w", encoding="utf-8") as file:
            json.dump(mapping_report, file, indent=2)
    with (out / "replay_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    logger.info("replay summary: %s", json.dumps(summary["tasks"], ensure_ascii=False))


if __name__ == "__main__":
    main()
