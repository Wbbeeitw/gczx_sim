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

"""
Collect LIBERO rollouts using a trained pi0/pi0.5 policy and save as a LeRobot dataset.

Usage:
    cd examples/recap/process
    python collect_libero_rollouts.py \
        --pretrained_path /workspace/models/RLinf-Pi05-LIBERO-SFT \
        --output_dir /workspace/datasets/libero10_task0_rollouts \
        --num_episodes 64 \
        --noise_scale 0.1 \
        --task_suite_name libero_10 \
        --task_id 0
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from rlinf.data.lerobot_writer import LeRobotDatasetWriter
from toolkits.eval_scripts_openpi import setup_policy


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def _quat2axisangle(quat):
    """Convert quaternion to axis-angle (robosuite convention)."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = math.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_libero_env(task, resolution, seed, gpu_id=0):
    task_description = task.language
    task_bddl_file = (
        Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "render_gpu_device_id": gpu_id,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _max_steps_for_suite(suite_name):
    return {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }.get(suite_name, 520)


def _build_features(image_shape, state_dim, action_dim):
    return {
        "image": {
            "dtype": "image",
            "shape": list(image_shape),
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "image",
            "shape": list(image_shape),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["actions"],
        },
        "done": {
            "dtype": "bool",
            "shape": (1,),
            "names": ["done"],
        },
        "is_success": {
            "dtype": "bool",
            "shape": (1,),
            "names": ["is_success"],
        },
        "reward": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["reward"],
        },
        "return": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["return"],
        },
        "task": {
            "dtype": "string",
            "shape": (1,),
            "names": ["task"],
        },
    }


def _compute_returns(rewards, gamma=1.0):
    returns = np.zeros(len(rewards), dtype=np.float32)
    g = 0.0
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g
        returns[t] = g
    return returns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task_suite_name", default="libero_10")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--num_episodes", type=int, default=64)
    parser.add_argument("--noise_scale", type=float, default=0.0, help="Gaussian action noise std")
    parser.add_argument("--noise_clip", type=float, default=0.3, help="Clip noise per dimension")
    parser.add_argument("--action_chunk", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=5, help="Policy diffusion sampling steps")
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--config_name",
        default="pi05_libero",
        help="Model config name in openpi dataconfig",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output_dir if it already exists",
    )
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"Loading policy from {args.pretrained_path}")
    policy_args = argparse.Namespace(
        config_name=args.config_name,
        pretrained_path=args.pretrained_path,
        num_steps=args.num_steps,
        action_chunk=args.action_chunk,
        num_steps_wait=args.num_steps_wait,
    )
    policy = setup_policy(policy_args)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    max_steps = _max_steps_for_suite(args.task_suite_name)

    env, task_description = _get_libero_env(
        task, LIBERO_ENV_RESOLUTION, args.seed, args.gpu_id
    )

    writer = LeRobotDatasetWriter()
    output_path = Path(args.output_dir)
    if output_path.exists():
        if args.overwrite:
            import shutil
            shutil.rmtree(output_path)
            print(f"Removed existing output dir: {output_path}")
        else:
            raise FileExistsError(
                f"Output dir already exists: {output_path}. Use --overwrite to replace."
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    successes = 0
    all_returns = []
    episode_lengths = []

    for ep_idx in range(args.num_episodes):
        policy.reset()
        env.reset()
        obs = env.set_init_state(initial_states[ep_idx % len(initial_states)])

        # Stabilization steps
        for _ in range(args.num_steps_wait):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

        frames = []
        rewards = []
        done = False
        action_plan = []

        for t in range(max_steps):
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(
                obs["robot0_eye_in_hand_image"][::-1, ::-1]
            )
            state = np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            )

            if not action_plan:
                observation = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": state,
                    "task": str(task_description),
                }
                action_chunk = policy.infer(observation)["actions"]
                action_plan = list(action_chunk[: args.action_chunk])

            action = np.asarray(action_plan.pop(0), dtype=np.float32)
            if args.noise_scale > 0:
                noise = np.random.randn(*action.shape) * args.noise_scale
                noise = np.clip(noise, -args.noise_clip, args.noise_clip)
                action = action + noise

            obs, reward, done, info = env.step(action.tolist())
            rewards.append(float(reward))

            frames.append(
                {
                    "image": img,
                    "wrist_image": wrist_img,
                    "state": state.astype(np.float32),
                    "actions": action.astype(np.float32),
                    "reward": np.array([float(reward)], dtype=np.float32),
                    "task": str(task_description),
                    "done": np.array([False], dtype=bool),
                    "is_success": np.array([False], dtype=bool),
                }
            )

            if done:
                successes += 1
                break

        # Finalize episode
        ep_len = len(frames)
        if ep_len == 0:
            print(f"Episode {ep_idx}: empty, skipping")
            continue
        is_success = bool(done)
        returns = _compute_returns(rewards, gamma=1.0)
        for i, frame in enumerate(frames):
            frame["is_success"] = np.array([is_success], dtype=bool)
            frame["return"] = np.array([returns[i]], dtype=np.float32)
        frames[-1]["done"] = np.array([True], dtype=bool)

        # Initialize writer on first episode
        if writer.dataset is None:
            first = frames[0]
            writer.create(
                repo_id=str(output_path),
                robot_type="franka_panda",
                fps=args.fps,
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
        all_returns.append(returns[0] if len(returns) > 0 else 0.0)
        episode_lengths.append(ep_len)
        print(
            f"Episode {ep_idx}: len={ep_len}, success={is_success}, "
            f"return={returns[0]:.2f}, total_success={successes}/{ep_idx+1}"
        )

    writer.finalize()
    env.close()

    print("\n=== Rollout summary ===")
    print(f"Total episodes: {args.num_episodes}")
    print(f"Successes: {successes} ({successes/args.num_episodes*100:.1f}%)")
    print(f"Mean episode length: {np.mean(episode_lengths):.1f}")
    print(f"Mean initial return: {np.mean(all_returns):.2f}")
    print(f"Dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
