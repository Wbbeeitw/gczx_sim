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

"""Run an isolated fixed-Value, binary-CFG LIBERO baseline.

The baseline deliberately avoids the ReCap z/p and fusion stages:

1. Collect fresh rollouts for every requested LIBERO task.
2. Merge the rollouts and compute returns.
3. Score them with an existing Value checkpoint.
4. Export raw Value advantages back to each task dataset.
5. Train one binary (pure) CFG policy and evaluate it per task.

All generated data lives under ``--run-root``. The root must not exist, which
keeps this comparison experiment from replacing a ReCap round or its datasets.
Use ``--dry-run`` to inspect every generated command without creating files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.recap.rounds.run_round import (
    Step,
    _build_ctx,
    _build_env,
    _run_step,
    _step_merge_multitask,
    _steps_collect_multitask,
    _steps_eval_policy,
    _steps_export_multitask_raw,
    _steps_score_critic_multitask,
    _steps_train_policy,
)


DEFAULT_BASE_MODEL = "/workspace/models/RLinf-Pi05-LIBERO-SFT"
DEFAULT_SIGLIP = "/workspace/models/siglip2-so400m-patch14-224"
DEFAULT_GEMMA = "/workspace/models/gemma-3-270m"


def _parse_tasks(raw_tasks: str) -> list[str]:
    """Parse and validate a comma-separated list of LIBERO ``task<N>`` names."""
    tasks = [task.strip() for task in raw_tasks.split(",") if task.strip()]
    if not tasks:
        raise ValueError("--tasks must contain at least one task")
    if len(tasks) != len(set(tasks)):
        raise ValueError("--tasks must not contain duplicates")
    for task in tasks:
        if not task.startswith("task") or not task[4:].isdigit():
            raise ValueError(
                f"invalid task {task!r}; expected a LIBERO name such as 'task0'"
            )
    return tasks


def _path_under_run_root(run_root: Path, *parts: str) -> str:
    return str(run_root.joinpath(*parts))


def _build_round_config(args: argparse.Namespace) -> dict:
    """Translate the baseline CLI into the existing round step-builder config."""
    run_root = Path(args.run_root).absolute()
    tasks = _parse_tasks(args.tasks)
    return {
        "round_id": 1,
        "task": "fixed_value_purecfg",
        "task_id": 0,
        "task_suite_name": args.task_suite_name,
        "method": "fixed_value_purecfg",
        "repo_root": args.repo_root,
        "bootstrap": False,
        "pipeline": [
            "collect_multitask",
            "score_critic_multitask",
            "export_multitask_raw",
            "train_policy",
            "eval_policy",
        ],
        "tasks": tasks,
        "gpu": {
            "cuda_visible_devices": args.cuda_visible_devices,
            "gpu_id": args.gpu_id,
        },
        "paths": {
            "base_model": args.base_model,
            "siglip": args.siglip_path,
            "gemma3": args.gemma3_path,
            "tokenizer": args.tokenizer_path,
            # Unused by this baseline, but retained for the round context schema.
            "parent_dataset": _path_under_run_root(run_root, "unused_parent"),
            "child_pattern": _path_under_run_root(run_root, "datasets", "{task}"),
            "merged_dataset": _path_under_run_root(run_root, "merged"),
            "exp_root": _path_under_run_root(run_root, "artifacts"),
            "results_root": _path_under_run_root(run_root, "checkpoints"),
            "value_checkpoint": args.value_checkpoint,
        },
        "datasets": {"parent_episodes": 0},
        "parent_policy": {"checkpoint": "", "label": "SFT"},
        "tags": {
            "returns": args.returns_tag,
            "merged_base": args.base_advantage_tag,
            "child_fused": args.raw_advantage_tag,
            "policy_advantage": args.raw_advantage_tag,
        },
        "collect": {
            "num_episodes": args.episodes_per_task,
            "parallel": 1,
            "seed": args.seed,
            "fps": args.fps,
            "action_chunk": args.action_chunk,
            "num_steps": args.num_steps,
            "num_steps_wait": args.num_steps_wait,
            "model_type": "openpi",
            "openpi_config_name": args.openpi_config_name,
            "warmup_before_env": args.warmup_before_env,
            "guidance_type": args.collection_guidance_type,
            "positive_only_conditional": args.collection_positive_only_conditional,
            "guidance_scale": args.collection_guidance_scale,
            "negative_guidance_scale": args.collection_negative_guidance_scale,
            "visualize": False,
        },
        "returns": {
            "failure_reward": args.failure_reward,
            "gamma": args.gamma,
            "global_min": args.return_global_min,
            "global_max": args.return_global_max,
            "num_workers": args.return_workers,
        },
        "value": {
            # The external checkpoint is never trained here. The round context
            # still needs these fields because score_critic_multitask builds and
            # then removes the Value-SFT step before returning its plan.
            "steps": args.fixed_value_steps,
            "save_interval": 1,
            "micro_batch_size": 1,
            "global_batch_size": 1,
            "lr": 0.0,
            "value_lr": 0.0,
            "lr_warmup_steps": 0,
            "action_dim": args.action_dim,
            "action_horizon": args.action_horizon,
            "critic_expert_variant": args.critic_expert_variant,
        },
        "revalue": {
            "label_name": args.merged_label_name,
            "num_phases": args.num_phases,
            "success_phase": args.success_phase,
            "seed": args.seed,
            "lookahead_step": args.lookahead_step,
            "positive_quantile": args.positive_quantile,
            "extract_batch_size": args.extract_batch_size,
        },
        "policy": {
            # "binary" is the project strategy for unweighted, pure CFG.
            "strategy": "binary",
            "advantage_source": "raw",
            "guidance_type": args.policy_guidance_type,
            "positive_only_conditional": args.policy_positive_only_conditional,
            "negative_guidance_scale": args.policy_negative_guidance_scale,
            "unconditional_prob": args.policy_unconditional_prob,
            "positive_quantile": args.positive_quantile,
            # The shared builder always emits these CSA fields. Binary CFG
            # ignores them, but concrete values keep the generated Hydra CLI
            # valid instead of passing the string "None".
            "csa_bottom_quantile": 0.15,
            "csa_bottom_negative_prob": 0.5,
            "csa_positive_prompt_prob": 0.85,
            "csa_weight_lambda": 0.2,
            "positive_residual_alpha": 0.5,
            "max_steps": args.policy_max_steps,
            "save_interval": args.policy_save_interval,
            "lr_warmup_steps": args.policy_lr_warmup_steps,
            "global_batch_size": args.policy_global_batch_size,
            "micro_batch_size": args.policy_micro_batch_size,
            "extra_overrides": args.policy_extra_override,
        },
        "eval": {
            "guidance_type": args.eval_guidance_type,
            "positive_only_conditional": args.eval_positive_only_conditional,
            "guidance_scale": args.eval_guidance_scale,
            "negative_guidance_scale": args.eval_negative_guidance_scale,
            "eval_rollout_epoch": args.eval_rollout_epochs,
            "total_num_envs": args.eval_total_num_envs,
            "warmup_before_env": args.eval_warmup_before_env,
            "save_video": args.eval_save_video,
        },
        "results": {
            "enabled": not args.no_eval,
            "round_index": 1,
            "policy_label": "FixedValue-PureCFG",
            "critic_label": "FixedValue",
            "output_dir": _path_under_run_root(run_root, "results"),
            "output_name": "fixed_value_purecfg",
        },
    }


def build_plan(args: argparse.Namespace) -> tuple[dict, list[Step]]:
    """Build the ordered, non-ReCap training plan for ``args``.

    This intentionally runs collection sequentially. A single GPU fixed-Value
    baseline should not launch several OpenPI rollouts against the same device.
    """
    ctx = _build_ctx(_build_round_config(args))
    steps = [
        *_steps_collect_multitask(ctx),
        _step_merge_multitask(ctx),
        *_steps_score_critic_multitask(ctx),
        *_steps_export_multitask_raw(ctx),
        *_steps_train_policy(ctx),
    ]
    if not args.no_eval:
        steps.extend(_steps_eval_policy(ctx))
    return ctx, steps


def _validate_paths(args: argparse.Namespace) -> None:
    """Reject accidental reuse before any collection or training can start."""
    run_root = Path(args.run_root).absolute()
    if run_root.exists():
        raise FileExistsError(
            f"--run-root already exists: {run_root}. Choose a new baseline root; "
            "this script never overwrites an existing experiment."
        )
    checkpoint = Path(args.value_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"--value-checkpoint is not a file: {checkpoint}"
        )


def _print_plan(steps: Sequence[Step]) -> None:
    for index, step in enumerate(steps, start=1):
        print(f"\n=== {index:02d}: {step.name} ===")
        print(shlex.join(step.argv))
        for artifact in step.artifacts:
            print(f"# artifact: {artifact}")


def run(args: argparse.Namespace) -> None:
    """Run the requested baseline, or print it when ``--dry-run`` is set."""
    ctx, steps = build_plan(args)
    if args.dry_run:
        _print_plan(steps)
        return

    _validate_paths(args)
    env = _build_env(ctx)
    log_root = Path(args.run_root).absolute() / "logs"
    for index, step in enumerate(steps, start=1):
        _run_step(
            step,
            log_root / f"{index:02d}_{step.name}.log",
            env,
            ctx["repo_root"],
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the isolated baseline command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--value-checkpoint", required=True)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--siglip-path", default=DEFAULT_SIGLIP)
    parser.add_argument("--gemma3-path", default=DEFAULT_GEMMA)
    parser.add_argument("--tokenizer-path", default=DEFAULT_GEMMA)
    parser.add_argument(
        "--tasks",
        default="task0,task1,task2,task3,task4,task5,task6,task7,task8,task9",
        help="comma-separated LIBERO task names",
    )
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--episodes-per-task", type=int, default=40)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    collect = parser.add_argument_group("collection")
    collect.add_argument("--openpi-config-name", default="pi05_libero")
    collect.add_argument("--fps", type=int, default=10)
    collect.add_argument("--action-chunk", type=int, default=5)
    collect.add_argument("--num-steps", type=int, default=5)
    collect.add_argument("--num-steps-wait", type=int, default=10)
    collect.add_argument(
        "--warmup-before-env", action=argparse.BooleanOptionalAction, default=True
    )
    collect.add_argument("--collection-guidance-type", default="positive")
    collect.add_argument(
        "--collection-positive-only-conditional",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    collect.add_argument("--collection-guidance-scale", type=float, default=1.0)
    collect.add_argument(
        "--collection-negative-guidance-scale", type=float, default=0.0
    )

    score = parser.add_argument_group("fixed value scoring")
    score.add_argument("--returns-tag", default="fixed_value_returns")
    score.add_argument(
        "--fixed-value-steps",
        type=int,
        default=1200,
        help="metadata only; the external Value checkpoint is not trained",
    )
    score.add_argument("--base-advantage-tag", default="fixed_value_base")
    score.add_argument("--raw-advantage-tag", default="fixed_value_raw")
    score.add_argument("--merged-label-name", default="phase_progress_multitask")
    score.add_argument("--failure-reward", type=float, default=-300.0)
    score.add_argument("--gamma", type=float, default=1.0)
    score.add_argument("--return-global-min", type=float, default=-900.0)
    score.add_argument("--return-global-max", type=float, default=0.0)
    score.add_argument("--return-workers", type=int, default=64)
    score.add_argument("--action-dim", type=int, default=7)
    score.add_argument("--action-horizon", type=int, default=10)
    score.add_argument("--critic-expert-variant", default="gemma_1m")
    score.add_argument("--num-phases", type=int, default=4)
    score.add_argument("--success-phase", type=int, default=3)
    score.add_argument("--lookahead-step", type=int, default=10)
    score.add_argument("--positive-quantile", type=float, default=0.3)
    score.add_argument("--extract-batch-size", type=int, default=16)

    policy = parser.add_argument_group("binary pure-CFG policy")
    policy.add_argument("--policy-guidance-type", default="positive")
    policy.add_argument(
        "--policy-positive-only-conditional",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    policy.add_argument("--policy-negative-guidance-scale", type=float, default=0.0)
    policy.add_argument("--policy-unconditional-prob", type=float, default=0.1)
    policy.add_argument("--policy-max-steps", type=int, default=1500)
    policy.add_argument("--policy-save-interval", type=int, default=1000)
    policy.add_argument("--policy-lr-warmup-steps", type=int, default=50)
    policy.add_argument("--policy-global-batch-size", type=int, default=64)
    policy.add_argument("--policy-micro-batch-size", type=int, default=8)
    policy.add_argument(
        "--policy-extra-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="repeatable actor/FSDP Hydra override passed to cfg_train",
    )

    evaluation = parser.add_argument_group("evaluation")
    evaluation.add_argument("--no-eval", action="store_true")
    evaluation.add_argument("--eval-guidance-type", default="positive")
    evaluation.add_argument(
        "--eval-positive-only-conditional",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    evaluation.add_argument("--eval-guidance-scale", type=float, default=1.0)
    evaluation.add_argument("--eval-negative-guidance-scale", type=float, default=0.0)
    evaluation.add_argument("--eval-rollout-epochs", type=int, default=5)
    evaluation.add_argument("--eval-total-num-envs", type=int, default=10)
    evaluation.add_argument(
        "--eval-warmup-before-env",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    evaluation.add_argument(
        "--eval-save-video", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    """Run the fixed-Value Pure-CFG baseline CLI."""
    run(parse_args())


if __name__ == "__main__":
    main()
