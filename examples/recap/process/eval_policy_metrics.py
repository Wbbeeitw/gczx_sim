#!/usr/bin/env python
"""Evaluate a LIBERO policy and save success/ACT metrics only."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlinf.revalue.pipeline.embodied import (  # noqa: E402
    LIBERO_DUMMY_ACTION,
    LiberoRolloutCollectionConfig,
    _get_libero_env,
    _load_rollout_policy,
    _max_steps_for_suite,
    _quat2axisangle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--noise-scale", type=float, default=0.0)
    parser.add_argument("--noise-clip", type=float, default=0.3)
    parser.add_argument("--action-chunk", type=int, default=5)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--model-type", default="cfg_model")
    parser.add_argument("--config-name", default="pi05_libero")
    return parser.parse_args()


def _predict_action_chunk(policy, observation, task_description):
    import torch

    action_chunk_out, _ = policy.predict_action_batch(
        {
            "main_images": torch.from_numpy(
                np.stack([observation["image"]])
            ),
            "wrist_images": torch.from_numpy(
                np.stack([observation["wrist_image"]])
            ),
            "states": torch.from_numpy(np.stack([observation["state"]])),
            "task_descriptions": [str(task_description)],
        },
        mode="eval",
    )
    if hasattr(action_chunk_out, "detach"):
        action_chunk_out = action_chunk_out.detach().cpu().numpy()
    action_chunk = np.asarray(action_chunk_out, dtype=np.float32)
    if action_chunk.ndim == 3:
        action_chunk = action_chunk[0]
    return action_chunk


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    from libero.libero import benchmark

    config = LiberoRolloutCollectionConfig(
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
    )
    policy = _load_rollout_policy(config)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    max_steps = _max_steps_for_suite(args.task_suite_name)
    env, task_description = _get_libero_env(
        task, (256, 256), args.seed, args.gpu_id
    )

    successes = 0
    episode_lengths: list[int] = []
    success_episode_lengths: list[int] = []
    try:
        for episode_index in range(args.num_episodes):
            if hasattr(policy, "reset"):
                policy.reset()
            env.reset()
            obs = env.set_init_state(
                initial_states[episode_index % len(initial_states)]
            )
            for _ in range(args.num_steps_wait):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

            action_plan = []
            done = False
            episode_length = 0
            for _ in range(max_steps):
                image = np.ascontiguousarray(
                    obs["agentview_image"][::-1, ::-1]
                )
                wrist_image = np.ascontiguousarray(
                    obs["robot0_eye_in_hand_image"][::-1, ::-1]
                )
                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                ).astype(np.float32)
                if not action_plan:
                    action_plan = list(
                        _predict_action_chunk(
                            policy,
                            {
                                "image": image,
                                "wrist_image": wrist_image,
                                "state": state,
                            },
                            task_description,
                        )[: args.action_chunk]
                    )

                action = np.asarray(action_plan.pop(0), dtype=np.float32)
                if args.noise_scale > 0:
                    noise = np.random.randn(*action.shape) * args.noise_scale
                    action += np.clip(noise, -args.noise_clip, args.noise_clip)
                obs, _, done, _ = env.step(action.tolist())
                episode_length += 1
                if done:
                    break

            is_success = bool(done)
            episode_lengths.append(episode_length)
            if is_success:
                successes += 1
                success_episode_lengths.append(episode_length)
            logging.info(
                "episode=%d/%d success=%s length=%d",
                episode_index + 1,
                args.num_episodes,
                is_success,
                episode_length,
            )
    finally:
        env.close()

    success_rate = successes / args.num_episodes if args.num_episodes else 0.0
    result = {
        "task_suite_name": args.task_suite_name,
        "task_id": args.task_id,
        "model_type": args.model_type,
        "pretrained_path": args.pretrained_path,
        "checkpoint_path": args.checkpoint_path,
        "num_episodes": args.num_episodes,
        "successes": successes,
        "failures": args.num_episodes - successes,
        "success_rate": success_rate,
        "all_episode_act_mean": (
            float(np.mean(episode_lengths)) if episode_lengths else None
        ),
        "success_episode_act_mean": (
            float(np.mean(success_episode_lengths))
            if success_episode_lengths
            else None
        ),
        "success_episode_act_std": (
            float(np.std(success_episode_lengths))
            if success_episode_lengths
            else None
        ),
        "success_episode_lengths": success_episode_lengths,
        "episode_lengths": episode_lengths,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    logging.info(
        "evaluation summary: success=%d/%d (%.4f), success_act=%.2f +/- %.2f",
        successes,
        args.num_episodes,
        success_rate,
        result["success_episode_act_mean"]
        if result["success_episode_act_mean"] is not None
        else float("nan"),
        result["success_episode_act_std"]
        if result["success_episode_act_std"] is not None
        else float("nan"),
    )
    logging.info("saved metrics summary: %s", output_dir / "metrics_summary.json")
    return result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    evaluate(parse_args())


if __name__ == "__main__":
    main()
