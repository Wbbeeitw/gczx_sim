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

"""Round-level orchestrator for iterated ReCap training on LIBERO.

One round YAML pins every path, tag and hyperparameter for a single round
of a single task. The orchestrator expands it into the exact stage commands,
runs them in order with per-step logs and artifact checks, and writes
``stage_reports/*.json`` so interrupted rounds can resume where they stopped.

Stages (run in this order):
  collect            collect fresh rollouts with the parent policy, then STOP
                     for a mandatory human semantic audit
  fit_critic         merge datasets, compute returns, train Value, extract
                     features once, build base from cache, train z/p + fusion,
                     predict and compare on the merged dataset
  export_policy_data re-index the child episodes out of the merged critic
                     outputs onto the child dataset (no extra inference)
  train_policy       train the next policy (PR-CFG or Pure CFG) from the
                     parent policy weights via init_checkpoint_path
  eval_policy        evaluate the trained checkpoint and record round results

Usage:
  python examples/recap/rounds/run_round.py \
      --config examples/recap/rounds/config/task1_iter02_prcfg.yaml \
      --stage collect

  python examples/recap/rounds/run_round.py \
      --config examples/recap/rounds/config/task1_iter02_prcfg.yaml \
      --stage fit_critic --confirm-audit

``--dry-run`` prints every command without executing it. ``--force`` re-runs
compute steps, while dataset replacement additionally requires
``--overwrite-datasets``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

logger = logging.getLogger("run_round")

STAGE_ORDER = [
    "collect",
    "fit_critic",
    "export_policy_data",
    "train_policy",
    "eval_policy",
]
KNOWN_STAGES = [
    *STAGE_ORDER,
    "fit_value",
    "build_child_base",
    "export_bootstrap",
    "collect_multitask",
    "build_multitask_critic_pool",
    "fit_critic_multitask",
    "score_critic_multitask",
    "task_heads",
    "predict_multitask",
    "export_multitask",
    "export_multitask_raw",
]


def _stage_dependencies(ctx: dict, stage: str) -> list[str]:
    """Linear dependency: the stage right before `stage` in the pipeline."""
    pipeline = ctx["pipeline"]
    if stage not in pipeline:
        return []
    index = pipeline.index(stage)
    return [pipeline[index - 1]] if index > 0 else []


def _audit_gate_stage(ctx: dict) -> str | None:
    """First stage after collect; it requires the human audit confirmation."""
    pipeline = ctx["pipeline"]
    collect_stage = next(
        (stage for stage in ("collect", "collect_multitask") if stage in pipeline),
        None,
    )
    if collect_stage is None:
        return None
    index = pipeline.index(collect_stage)
    if index + 1 < len(pipeline):
        return pipeline[index + 1]
    return None


@dataclass
class Step:
    """One executable step inside a stage."""

    name: str
    argv: list[str]
    artifacts: list[str] = field(default_factory=list)
    dataset_path: str | None = None


def _load_yaml(path: Path) -> dict:
    try:
        from omegaconf import OmegaConf

        loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        if isinstance(loaded, dict):
            return loaded
        raise TypeError(f"round config must be a mapping, got {type(loaded)}")
    except ImportError:
        pass
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "run_round.py requires omegaconf or PyYAML to read the round config"
        ) from exc
    with open(path, "r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    if not isinstance(loaded, dict):
        raise TypeError(f"round config must be a mapping, got {type(loaded)}")
    return loaded


def _get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _require(cfg: dict, dotted: str) -> Any:
    value = _get(cfg, dotted)
    if value is None:
        raise ValueError(f"round config is missing required key: {dotted}")
    return value


def _normalize_demo_dataset(task: str, value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        path = value
        start = None
        end = None
    elif isinstance(value, dict):
        path = str(value.get("path") or "")
        start = value.get("start")
        end = value.get("end")
    else:
        raise ValueError(
            f"demo_datasets[{task}] must be a path or mapping, got {type(value)!r}"
        )
    if not path:
        raise ValueError(f"demo_datasets[{task}] is missing path")
    episodes_path = Path(path) / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise ValueError(
            f"demo_datasets[{task}] missing episodes.jsonl: {episodes_path}"
        )
    total_episodes = sum(
        1
        for line in open(episodes_path, "r", encoding="utf-8")
        if line.strip()
    )
    if start is None and end is None:
        return {
            "path": path,
            "input": path,
            "episodes": total_episodes,
        }
    if start is None or end is None:
        raise ValueError(
            f"demo_datasets[{task}] must provide both start and end"
        )
    start = int(start)
    end = int(end)
    if start < 0 or end <= start or end > total_episodes:
        raise ValueError(
            f"demo_datasets[{task}] invalid slice {start}:{end} for "
            f"total_episodes={total_episodes}"
        )
    return {
        "path": path,
        "input": f"{path}::{start}:{end}",
        "episodes": end - start,
        "start": start,
        "end": end,
    }


def _episode_count(dataset_path: str) -> int | None:
    episodes_path = Path(dataset_path) / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        return None
    with episodes_path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def _apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply ``--set key=value`` overrides onto the round config.

    Values are parsed as JSON first (so ``40``, ``true``, ``0.3`` become
    native types) and fall back to plain strings. Overrides are applied
    before the config hash is computed, so stage reports stay correct.
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(
                    f"cannot override {key!r}: {part!r} is not a mapping"
                )
            node = child
        node[parts[-1]] = value
        logger.info("config override: %s = %r", key, value)
    return cfg


def _config_hash(cfg: dict) -> str:
    payload = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _step_hash(step: Step) -> str:
    payload = json.dumps(
        {"name": step.name, "argv": step.argv, "artifacts": step.artifacts},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stage_hash(steps: list[Step]) -> str:
    payload = json.dumps([_step_hash(step) for step in steps])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_commit(repo_root: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - best-effort audit info
        return "unknown"


def _build_ctx(cfg: dict) -> dict[str, Any]:
    """Expand the round config into every derived path/tag used by stages."""
    round_id = int(_require(cfg, "round_id"))
    method = str(_require(cfg, "method"))
    task = str(_require(cfg, "task"))
    exp_root = str(_require(cfg, "paths.exp_root"))
    merged_ds = str(_require(cfg, "paths.merged_dataset"))
    child_ds = str(_get(cfg, "paths.child_dataset") or "")
    parent_ds = str(_require(cfg, "paths.parent_dataset"))
    bootstrap = bool(_get(cfg, "bootstrap", False))
    if bootstrap:
        merged_ds = child_ds
    tasks = [str(task) for task in (_get(cfg, "tasks") or [])]
    task_datasets: dict[str, str] = {}
    policy_task_datasets: dict[str, str] = {}
    parent_task_datasets: dict[str, str] = {}
    task_ranges: dict[str, tuple[int, int]] = {}
    policy_task_ranges: dict[str, tuple[int, int]] = {}
    task_episode_counts: dict[str, int] = {}
    policy_episode_counts: dict[str, int] = {}
    demo_counts: dict[str, int] = {}
    normalized_demo_cfg: dict[str, dict[str, Any]] = {}
    if not tasks and not child_ds:
        raise ValueError("paths.child_dataset is required for single-task rounds")
    if tasks:
        child_pattern = _get(cfg, "paths.child_pattern")
        if not child_pattern:
            raise ValueError("paths.child_pattern is required when tasks: is set")
        policy_pattern = _get(cfg, "paths.policy_pattern") or child_pattern
        critic_pattern = _get(cfg, "paths.critic_pattern")
        parent_pattern = _get(cfg, "paths.parent_pattern")
        if critic_pattern and not parent_pattern:
            raise ValueError(
                "paths.parent_pattern is required when paths.critic_pattern is set"
            )
        demo_cfg = _get(cfg, "demo_datasets") or {}
        if critic_pattern and demo_cfg:
            raise ValueError(
                "demo_datasets cannot be combined with paths.critic_pattern; "
                "materialize demos into the fixed per-task parent pool first"
            )
        start = 0
        configured_per_task = int(_get(cfg, "collect.num_episodes"))
        configured_parent_per_task = int(
            _get(cfg, "datasets.parent_episodes_per_task", 0)
        )
        for task in tasks:
            policy_task_datasets[task] = str(policy_pattern).format(task=task)
            if parent_pattern:
                parent_task_datasets[task] = str(parent_pattern).format(task=task)
            task_datasets[task] = str(
                critic_pattern or child_pattern
            ).format(task=task)

            policy_count = (
                _episode_count(policy_task_datasets[task]) or configured_per_task
            )
            parent_count = configured_parent_per_task
            if task in parent_task_datasets:
                parent_count = (
                    _episode_count(parent_task_datasets[task]) or parent_count
                )
            critic_count = _episode_count(task_datasets[task])
            if critic_count is None:
                critic_count = (
                    parent_count + policy_count if critic_pattern else policy_count
                )
            demo_path = demo_cfg.get(task)
            demo_count = 0
            if demo_path:
                normalized_demo = _normalize_demo_dataset(task, demo_path)
                normalized_demo_cfg[task] = normalized_demo
                demo_count = int(normalized_demo["episodes"])
            demo_counts[task] = demo_count
            critic_count += demo_count
            if parent_count + policy_count > critic_count:
                raise ValueError(
                    f"{task}: parent episodes ({parent_count}) + policy episodes "
                    f"({policy_count}) exceed critic episodes ({critic_count})"
                )
            task_episode_counts[task] = critic_count
            policy_episode_counts[task] = policy_count
            task_ranges[task] = (start, start + critic_count)
            policy_start = start + parent_count
            policy_task_ranges[task] = (
                policy_start,
                policy_start + policy_count,
            )
            start += critic_count
        merged_episodes = start
    parent_episodes = int(_require(cfg, "datasets.parent_episodes"))
    child_episodes = int(_get(cfg, "collect.num_episodes"))
    if tasks:
        merged_episodes = start
    else:
        merged_episodes = parent_episodes + child_episodes

    merged_name = os.path.basename(merged_ds.rstrip("/"))
    value_steps = int(_get(cfg, "value.steps"))
    value_exp = f"value_{merged_name}"
    policy_exp = f"policy{round_id}_{method}"
    eval_cfg = cfg.get("eval", {})
    eval_episodes = int(eval_cfg.get("eval_rollout_epoch", 0)) * int(
        eval_cfg.get("total_num_envs", 0)
    )
    # Heavy, re-trainable checkpoints live under results_root
    # (/workspace/results); small artifacts stay under exp_root
    # (/data/libero_long). Defaults to exp_root when unset.
    results_root = str(_get(cfg, "paths.results_root") or exp_root)

    ctx = {
        "cfg": cfg,
        "round_id": round_id,
        "method": method,
        "task": task,
        "task_id": int(_require(cfg, "task_id")),
        "task_suite_name": str(_get(cfg, "task_suite_name", "libero_10")),
        "repo_root": str(_get(cfg, "repo_root", "/workspace/RLinf")),
        "exp_root": exp_root,
        "results_root": results_root,
        "pipeline": [str(s) for s in (_get(cfg, "pipeline") or STAGE_ORDER)],
        "parent_ds": parent_ds,
        "child_ds": child_ds,
        "merged_ds": merged_ds,
        "parent_episodes": parent_episodes,
        "child_episodes": child_episodes,
        "merged_episodes": merged_episodes,
        "bootstrap": bootstrap,
        "tasks": tasks,
        "task_datasets": task_datasets,
        "policy_task_datasets": policy_task_datasets,
        "parent_task_datasets": parent_task_datasets,
        "task_ranges": task_ranges,
        "policy_task_ranges": policy_task_ranges,
        "task_episode_counts": task_episode_counts,
        "policy_episode_counts": policy_episode_counts,
        "demo_datasets": normalized_demo_cfg,
        "multitask": bool(tasks),
        "base_model": str(_require(cfg, "paths.base_model")),
        "parent_checkpoint": str(_get(cfg, "parent_policy.checkpoint") or ""),
        "parent_label": str(_get(cfg, "parent_policy.label", "parent")),
        "returns_tag": str(_require(cfg, "tags.returns")),
        "merged_base_tag": str(_require(cfg, "tags.merged_base")),
        "child_fused_tag": str(
            _get(cfg, "tags.policy_advantage")
            or _require(cfg, "tags.child_fused")
        ),
        "value_exp": value_exp,
        "value_ckpt": (
            str(_get(cfg, "paths.value_checkpoint") or "")
            or (
                f"{results_root}/value_sft/{value_exp}"
                f"/checkpoints/global_step_{value_steps}"
            )
        ),
        "value_ckpt_external": bool(_get(cfg, "paths.value_checkpoint")),
        "policy_exp": policy_exp,
        "policy_ckpt": (
            f"{results_root}/policy/{policy_exp}/checkpoints"
            f"/global_step_{int(_get(cfg, 'policy.max_steps') or 0)}"
            "/actor/model_state_dict/full_weights.pt"
        ),
        "eval_exp": f"{policy_exp}_eval{eval_episodes}",
        "revalue_root": f"{exp_root}/revalue",
        "policy_data_root": f"{exp_root}/policy_data",
        "advantage_source": str(
            _get(cfg, "policy.advantage_source")
            or (
                "raw"
                if "export_multitask_raw"
                in (_get(cfg, "pipeline") or STAGE_ORDER)
                else "fused"
            )
        ),
    }
    ctx["predictions"] = f"{ctx['revalue_root']}/predictions.parquet"
    ctx["comparison"] = str(
        _get(cfg, "results.comparison")
        or f"{ctx['revalue_root']}/return_compare.json"
    )
    ctx["merged_base_adv"] = (
        f"{merged_ds}/meta/advantages_{ctx['merged_base_tag']}.parquet"
    )
    ctx["child_fused_adv"] = (
        f"{child_ds}/meta/advantages_{ctx['child_fused_tag']}.parquet"
        if child_ds
        else ""
    )
    ctx["child_manifest"] = (
        f"{ctx['policy_data_root']}/multitask_episode_manifest.json"
        if tasks
        else f"{ctx['policy_data_root']}/episode_manifest.json"
    )
    if ctx["advantage_source"] not in {"raw", "fused"}:
        raise ValueError("policy.advantage_source must be 'raw' or 'fused'")
    return ctx


def _revalue_entry(stage: str, overrides: list[str]) -> list[str]:
    return [
        sys.executable,
        "examples/recap/revalue/revalue.py",
        "--config-name",
        "revalue_shared_mlp_fusion",
        f"stage={stage}",
        *overrides,
    ]


def _script_entry(relpath: str, args: list[str]) -> list[str]:
    return [sys.executable, relpath, *args]


def _data_overrides(ctx: dict, dataset: str) -> list[str]:
    cfg = ctx["cfg"]
    return [
        f"data.dataset_path={dataset}",
        "data.robot_type=libero",
        "data.env_type=libero",
        "data.model_type=pi05",
        f"data.action_dim={_get(cfg, 'value.action_dim', 7)}",
        f"data.label_name={_require(cfg, 'revalue.label_name')}",
        "data.val_episode_ratio=0.0",
        f"manifest.num_phases={_get(cfg, 'revalue.num_phases', 4)}",
    ]


def _value_model_overrides(ctx: dict, checkpoint: str | None) -> list[str]:
    cfg = ctx["cfg"]
    overrides = [
        f"value.siglip_path={_require(cfg, 'paths.siglip')}",
        f"value.gemma3_path={_require(cfg, 'paths.gemma3')}",
        f"value.tokenizer_path={_require(cfg, 'paths.tokenizer')}",
        f"value.critic_expert_variant={_get(cfg, 'value.critic_expert_variant')}",
    ]
    if checkpoint:
        overrides.insert(0, f"value.checkpoint={checkpoint}")
    return overrides


def _list_override(key: str, values: list[str]) -> str:
    return f"{key}=[{','.join(json.dumps(str(v)) for v in values)}]"


def _hydra_container_value(value: Any) -> str:
    if isinstance(value, dict):
        return "{" + ",".join(
            f"{key}:{_hydra_container_value(item)}"
            for key, item in value.items()
        ) + "}"
    if isinstance(value, list):
        return "[" + ",".join(_hydra_container_value(item) for item in value) + "]"
    if isinstance(value, str):
        return json.dumps(value)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _json_override(key: str, value: Any) -> str:
    return f"{key}={_hydra_container_value(value)}"


def _steps_collect(ctx: dict) -> list[Step]:
    cfg = ctx["cfg"]
    collect = cfg.get("collect", {})
    overrides = [
        "rollout_collect.enabled=true",
        f"rollout_collect.output_dir={ctx['child_ds']}",
        f"rollout_collect.model_path={ctx['base_model']}",
        f"rollout_collect.model_type={_get(cfg, 'collect.model_type', 'cfg_model')}",
        f"rollout_collect.openpi_config_name={_get(cfg, 'collect.openpi_config_name')}",
        f"rollout_collect.task_suite_name={ctx['task_suite_name']}",
        f"rollout_collect.task_id={ctx['task_id']}",
        f"rollout_collect.num_episodes={ctx['child_episodes']}",
        f"rollout_collect.seed={_get(cfg, 'collect.seed', 42)}",
        f"rollout_collect.gpu_id={_get(cfg, 'gpu.gpu_id', 0)}",
        f"rollout_collect.fps={_get(cfg, 'collect.fps', 10)}",
        f"rollout_collect.action_chunk={_get(cfg, 'collect.action_chunk', 5)}",
        f"rollout_collect.num_steps={_get(cfg, 'collect.num_steps', 5)}",
        f"rollout_collect.num_steps_wait={_get(cfg, 'collect.num_steps_wait', 10)}",
        "rollout_collect.warmup_before_env="
        f"{_get(cfg, 'collect.warmup_before_env', True)}",
        "rollout_collect.guidance_type="
        f"{_get(cfg, 'collect.guidance_type', 'positive')}",
        "rollout_collect.positive_only_conditional="
        f"{_get(cfg, 'collect.positive_only_conditional', True)}",
        "rollout_collect.guidance_scale="
        f"{_get(cfg, 'collect.guidance_scale', 1.0)}",
        "rollout_collect.negative_guidance_scale="
        f"{_get(cfg, 'collect.negative_guidance_scale', 0.0)}",
        "rollout_collect.semantic_trace=true",
        f"rollout_collect.semantic_trace_task={ctx['task']}",
        f"rollout_collect.semantic_trace_output_name=semantic_trace_{ctx['task']}",
        f"output.root={ctx['revalue_root']}",
    ]
    if ctx["parent_checkpoint"]:
        overrides.append(
            f"rollout_collect.checkpoint_path={ctx['parent_checkpoint']}"
        )
    steps = [
        Step(
            name="collect_rollouts",
            argv=_revalue_entry("collect_rollouts", overrides),
            artifacts=[
                f"{ctx['child_ds']}/meta/info.json",
                f"{ctx['child_ds']}/collection_summary.json",
            ],
        )
    ]
    if collect.get("visualize", True):
        steps.append(
            Step(
                name="visualize_phases",
                argv=_script_entry(
                    "phase_split_script/visualize_phases.py",
                    [
                        f"--dataset_path={ctx['child_ds']}",
                        f"--output_dir={ctx['exp_root']}/collection/visualizations",
                        "--annotation_name="
                        f"phase_progress_semantic_trace_{ctx['task']}",
                        f"--num_success={int(collect.get('viz_success', 1))}",
                        f"--num_failure={int(collect.get('viz_failure', 1))}",
                    ],
                ),
                artifacts=[f"{ctx['exp_root']}/collection/visualizations"],
            )
        )
    return steps


def _steps_fit_value(ctx: dict) -> list[Step]:
    """Merge parent+child datasets, compute returns, train the value model."""
    cfg = ctx["cfg"]
    returns = cfg.get("returns", {})
    value = cfg.get("value", {})
    steps = [
        Step(
            name="merge_datasets",
            argv=_script_entry(
                "examples/recap/process/merge_lerobot_rollout_datasets.py",
                [
                    f"--first_dataset={ctx['parent_ds']}",
                    f"--second_dataset={ctx['child_ds']}",
                    f"--output_dataset={ctx['merged_ds']}",
                ],
            ),
            artifacts=[f"{ctx['merged_ds']}/meta/info.json"],
        ),
        Step(
            name="compute_returns",
            argv=_script_entry(
                "examples/recap/process/compute_returns.py",
                [
                    "--config-name",
                    "compute_returns",
                    "data.train_data_paths=[{dataset_path: "
                    f"{ctx['merged_ds']}, type: rollout}}]",
                    "data.dataset_type=rollout",
                    f"data.gamma={returns['gamma']}",
                    f"data.failure_reward={returns['failure_reward']}",
                    f"data.tag={ctx['returns_tag']}",
                    f"data.num_workers={_get(cfg, 'returns.num_workers', 64)}",
                ],
            ),
            artifacts=[
                f"{ctx['merged_ds']}/meta/returns_{ctx['returns_tag']}.parquet"
            ],
        ),
        Step(
            name="value_sft",
            argv=[
                "bash",
                "examples/recap/value/run_value_sft.sh",
                "libero_sft_value",
                "data.train_data_paths=[{dataset_path: "
                f"{ctx['merged_ds']}, type: rollout, weight: 1.0, "
                "robot_type: libero, model_type: pi05}]",
                "data.eval_data_paths=[]",
                f"data.tag={ctx['returns_tag']}",
                "data.robot_type=libero",
                "data.model_type=pi05",
                f"data.action_dim={value['action_dim']}",
                f"data.action_horizon={value['action_horizon']}",
                f"data.gamma={returns['gamma']}",
                f"actor.model.siglip_path={_require(cfg, 'paths.siglip')}",
                f"actor.model.gemma3_path={_require(cfg, 'paths.gemma3')}",
                f"actor.model.tokenizer_path={_require(cfg, 'paths.tokenizer')}",
                f"actor.model.freeze_vlm={value.get('freeze_vlm', True)}",
                "actor.fsdp_config.use_orig_params=true",
                f"actor.micro_batch_size={value['micro_batch_size']}",
                f"actor.global_batch_size={value['global_batch_size']}",
                f"actor.optim.lr={value['lr']}",
                f"actor.optim.value_lr={value['value_lr']}",
                f"actor.optim.lr_warmup_steps={value['lr_warmup_steps']}",
                f"actor.optim.total_training_steps={value['steps']}",
                "runner.max_epochs=-1",
                f"runner.max_steps={value['steps']}",
                "runner.val_check_interval=-1",
                f"runner.save_interval={value['save_interval']}",
                f"runner.logger.log_path={ctx['results_root']}/value_sft",
                f"runner.logger.experiment_name={ctx['value_exp']}",
            ],
            artifacts=[ctx["value_ckpt"]],
        ),
    ]
    if ctx.get("bootstrap"):
        steps = [step for step in steps if step.name != "merge_datasets"]
    if ctx.get("multitask"):
        steps = [
            _step_merge_multitask(ctx) if step.name == "merge_datasets" else step
            for step in steps
        ]
    return steps


def _steps_fit_critic(ctx: dict) -> list[Step]:
    cfg = ctx["cfg"]
    revalue_root = ctx["revalue_root"]
    returns = cfg.get("returns", {})
    value = cfg.get("value", {})
    revalue = cfg.get("revalue", {})
    steps = _steps_fit_value(ctx) + [
        Step(
            name="prepare_data",
            argv=_revalue_entry(
                "prepare_data",
                [
                    f"data.dataset_path={ctx['merged_ds']}",
                    f"data.label_name={revalue['label_name']}",
                    f"data.seed={revalue.get('seed', 42)}",
                    f"manifest.num_episodes={ctx['merged_episodes']}",
                    "manifest.success_ratio=0.5",
                    "manifest.val_episode_ratio=0.0",
                    "manifest.test_episode_ratio=0.0",
                    f"manifest.success_phase={revalue.get('success_phase', 3)}",
                    f"manifest.num_phases={revalue.get('num_phases', 4)}",
                    "manifest.overwrite=true",
                    f"output.root={revalue_root}",
                ],
            ),
            artifacts=[f"{revalue_root}/episode_manifest.json"],
        ),
        Step(
            name="extract_features",
            argv=_revalue_entry(
                "extract_features",
                [
                    *_data_overrides(ctx, ctx["merged_ds"]),
                    *_value_model_overrides(ctx, ctx["value_ckpt"]),
                    f"output.root={revalue_root}",
                    f"output.features_dir={revalue_root}/features",
                    "train.device=cuda",
                    f"train.extract_batch_size={revalue.get('extract_batch_size', 16)}",
                    "train.num_workers=0",
                ],
            ),
            artifacts=[
                f"{revalue_root}/features/train.pt",
                f"{revalue_root}/features/val.pt",
            ],
        ),
        Step(
            name="build_base_from_cache",
            argv=_revalue_entry(
                "build_base_from_cache",
                [
                    *_data_overrides(ctx, ctx["merged_ds"]),
                    f"returns.global_min={returns['global_min']}",
                    f"returns.global_max={returns['global_max']}",
                    "returns.dataset_type=rollout",
                    f"returns.failure_reward={returns['failure_reward']}",
                    "returns.compute=false",
                    f"base.tag={ctx['merged_base_tag']}",
                    f"base.returns_tag={ctx['returns_tag']}",
                    "base.compute_returns=false",
                    "base.compute_advantages=true",
                    f"recap.lookahead_step={revalue.get('lookahead_step', 10)}",
                    f"recap.gamma={returns['gamma']}",
                    f"recap.positive_quantile={revalue.get('positive_quantile', 0.3)}",
                    "recap.discount_next_value=true",
                    f"output.root={revalue_root}",
                ],
            ),
            artifacts=[
                ctx["merged_base_adv"],
                f"{revalue_root}/build_base_from_cache_report.json",
            ],
        ),
        Step(
            name="train_zp",
            argv=_revalue_entry(
                "train_zp",
                [
                    *_data_overrides(ctx, ctx["merged_ds"]),
                    f"output.root={revalue_root}",
                    f"output.features_dir={revalue_root}/features",
                    f"output.zp_dir={revalue_root}/zp_head",
                    "train.device=cuda",
                    f"train.batch_size={revalue.get('train_batch_size', 256)}",
                    "train.num_workers=0",
                    f"zp.head_type={revalue.get('zp_head_type')}",
                    f"zp.use_class_weights={revalue.get('zp_use_class_weights')}",
                    f"zp.max_epochs={revalue.get('zp_max_epochs', 100)}",
                    "zp.early_stop_patience="
                    f"{revalue.get('zp_early_stop_patience', 10)}",
                ],
            ),
            artifacts=[
                f"{revalue_root}/zp_head/zp_head.pt",
                f"{revalue_root}/zp_head/metrics.json",
            ],
        ),
        Step(
            name="train_fusion",
            argv=_revalue_entry(
                "train_fusion",
                [
                    *_data_overrides(ctx, ctx["merged_ds"]),
                    *_value_model_overrides(ctx, None),
                    f"returns.global_min={returns['global_min']}",
                    f"returns.global_max={returns['global_max']}",
                    f"base.tag={ctx['merged_base_tag']}",
                    f"output.root={revalue_root}",
                    f"output.features_dir={revalue_root}/features",
                    f"output.zp_dir={revalue_root}/zp_head",
                    f"output.fusion_dir={revalue_root}/fusion",
                    "train.device=cuda",
                    f"train.batch_size={revalue.get('train_batch_size', 256)}",
                    "train.num_workers=0",
                    f"fusion.max_epochs={revalue.get('fusion_max_epochs', 100)}",
                    "fusion.early_stop_patience="
                    f"{revalue.get('fusion_early_stop_patience', 10)}",
                    f"fusion.alpha={revalue.get('fusion_alpha', 1.0)}",
                ],
            ),
            artifacts=[
                f"{revalue_root}/fusion/fusion.pt",
                f"{revalue_root}/fusion/metrics.json",
            ],
        ),
        Step(
            name="predict",
            argv=_revalue_entry(
                "predict",
                [
                    *_data_overrides(ctx, ctx["merged_ds"]),
                    f"base.tag={ctx['merged_base_tag']}",
                    f"recap.source_advantages_path={ctx['merged_base_adv']}",
                    f"output.root={revalue_root}",
                    f"output.features_dir={revalue_root}/features",
                    f"output.zp_dir={revalue_root}/zp_head",
                    f"output.fusion_dir={revalue_root}/fusion",
                    f"output.predictions_path={ctx['predictions']}",
                    "train.device=cuda",
                    f"train.batch_size={revalue.get('train_batch_size', 256)}",
                ],
            ),
            artifacts=[ctx["predictions"]],
        ),
        Step(
            name="compare_returns",
            argv=_revalue_entry(
                "compare_returns",
                [
                    *_data_overrides(ctx, ctx["merged_ds"]),
                    f"base.tag={ctx['merged_base_tag']}",
                    f"recap.source_advantages_path={ctx['merged_base_adv']}",
                    f"output.root={revalue_root}",
                    f"output.predictions_path={ctx['predictions']}",
                    f"output.comparison_path={ctx['comparison']}",
                    f"returns.global_min={returns['global_min']}",
                    f"returns.global_max={returns['global_max']}",
                    "value.v_min=-1.0",
                    "value.v_max=0.0",
                ],
            ),
            artifacts=[ctx["comparison"]],
        ),
    ]
    return steps


def _step_prepare_child_manifest(ctx: dict) -> Step:
    revalue = ctx["cfg"].get("revalue", {})
    policy_data_root = ctx["policy_data_root"]
    manifest_dataset = (
        ctx["merged_ds"] if ctx.get("multitask") else ctx["child_ds"]
    )
    return Step(
        name="prepare_child_manifest",
        argv=_revalue_entry(
            "prepare_data",
            [
                f"data.dataset_path={manifest_dataset}",
                f"data.label_name={revalue['label_name']}",
                f"data.seed={revalue.get('seed', 42)}",
                "manifest.num_episodes="
                f"{ctx['merged_episodes'] if ctx.get('multitask') else ctx['child_episodes']}",
                "manifest.success_ratio=0.5",
                "manifest.val_episode_ratio=0.0",
                "manifest.test_episode_ratio=0.0",
                f"manifest.success_phase={revalue.get('success_phase', 3)}",
                f"manifest.num_phases={revalue.get('num_phases', 4)}",
                "manifest.overwrite=true",
                f"output.root={policy_data_root}",
            ],
        ),
        artifacts=[ctx["child_manifest"]],
    )


def _steps_build_child_base(ctx: dict) -> list[Step]:
    """Returns + raw value base advantages + manifest for the child dataset.

    Used by the raw (Pure CFG) track: the shared (or separately trained)
    value model scores the child episodes once; no z/p head or fusion.
    """
    cfg = ctx["cfg"]
    revalue = cfg.get("revalue", {})
    returns = cfg.get("returns", {})
    child_root = f"{ctx['exp_root']}/revalue_child"
    return [
        _step_prepare_child_manifest(ctx),
        Step(
            name="compute_child_returns",
            argv=_script_entry(
                "examples/recap/process/compute_returns.py",
                [
                    "--config-name",
                    "compute_returns",
                    "data.train_data_paths=[{dataset_path: "
                    f"{ctx['child_ds']}, type: rollout}}]",
                    "data.dataset_type=rollout",
                    f"data.gamma={returns['gamma']}",
                    f"data.failure_reward={returns['failure_reward']}",
                    f"data.tag={ctx['returns_tag']}",
                    f"data.num_workers={_get(cfg, 'returns.num_workers', 64)}",
                ],
            ),
            artifacts=[
                f"{ctx['child_ds']}/meta/returns_{ctx['returns_tag']}.parquet"
            ],
        ),
        Step(
            name="build_child_base",
            argv=_revalue_entry(
                "build_base",
                [
                    *_data_overrides(ctx, ctx["child_ds"]),
                    *_value_model_overrides(ctx, ctx["value_ckpt"]),
                    f"data.episode_split_path={ctx['child_manifest']}",
                    f"returns.global_min={returns['global_min']}",
                    f"returns.global_max={returns['global_max']}",
                    "returns.dataset_type=rollout",
                    f"returns.failure_reward={returns['failure_reward']}",
                    "returns.compute=false",
                    f"base.tag={ctx['child_fused_tag']}",
                    f"base.returns_tag={ctx['returns_tag']}",
                    "base.compute_returns=false",
                    "base.compute_advantages=true",
                    f"recap.lookahead_step={revalue.get('lookahead_step', 10)}",
                    f"recap.gamma={returns['gamma']}",
                    "recap.positive_quantile="
                    f"{revalue.get('positive_quantile', 0.3)}",
                    "recap.discount_next_value=true",
                    f"output.root={child_root}",
                ],
            ),
            artifacts=[ctx["child_fused_adv"]],
        ),
    ]


def _steps_export_policy_data(ctx: dict) -> list[Step]:
    cfg = ctx["cfg"]
    revalue = cfg.get("revalue", {})
    policy_data_root = ctx["policy_data_root"]
    return [
        _step_prepare_child_manifest(ctx),
        Step(
            name="export_dataset_view",
            argv=_revalue_entry(
                "export_dataset_view",
                [
                    f"export_view.source_advantages_path={ctx['merged_base_adv']}",
                    f"export_view.predictions_path={ctx['predictions']}",
                    f"export_view.child_dataset_path={ctx['child_ds']}",
                    f"export_view.output_tag={ctx['child_fused_tag']}",
                    f"export_view.source_episode_start={ctx['parent_episodes']}",
                    f"export_view.source_episode_end={ctx['merged_episodes']}",
                    f"export_view.child_episode_offset=-{ctx['parent_episodes']}",
                    f"export_view.lookahead_step={revalue.get('lookahead_step', 10)}",
                    f"export_view.gamma={_get(cfg, 'returns.gamma')}",
                    "export_view.positive_quantile="
                    f"{revalue.get('positive_quantile', 0.3)}",
                    "export_view.discount_next_value=true",
                    f"export_view.expected_episodes={ctx['child_episodes']}",
                    f"export_view.report_path={policy_data_root}/export_report.json",
                    f"output.root={policy_data_root}",
                ],
            ),
            artifacts=[
                ctx["child_fused_adv"],
                f"{policy_data_root}/export_report.json",
            ],
        ),
    ]


def _steps_train_policy(ctx: dict) -> list[Step]:
    cfg = ctx["cfg"]
    policy = cfg.get("policy", {})
    policy_dataset = ctx["child_ds"]
    extra_overrides = list(policy.get("extra_overrides", []))
    if ctx.get("multitask"):
        policy_dataset = ctx["policy_task_datasets"][ctx["tasks"][0]]
        train_data_paths = [
            {
                "dataset_path": ctx["policy_task_datasets"][task],
                "type": "rollout",
                "weight": 1.0,
            }
            for task in ctx["tasks"]
        ]
        extra_overrides.extend(
            [
                _json_override("data.train_data_paths", train_data_paths),
                "data.balance_dataset_weights=false",
            ]
        )
    overrides = [
        "cfg_train.enabled=true",
        f"cfg_train.dataset_path={policy_dataset}",
        f"cfg_train.base_model_path={ctx['base_model']}",
        f"cfg_train.advantage_tag={ctx['child_fused_tag']}",
        f"cfg_train.episode_split_path={ctx['child_manifest']}",
        "cfg_train.episode_split_name=selected",
        f"cfg_train.experiment_name={ctx['policy_exp']}",
        f"cfg_train.log_dir={ctx['results_root']}/policy",
        "cfg_train.model_type=cfg_model",
        f"cfg_train.openpi_config_name={_get(cfg, 'collect.openpi_config_name')}",
        f"cfg_train.strategy={policy.get('strategy')}",
        f"cfg_train.guidance_type={policy.get('guidance_type', 'positive')}",
        "cfg_train.positive_only_conditional="
        f"{policy.get('positive_only_conditional')}",
        "cfg_train.negative_guidance_scale="
        f"{policy.get('negative_guidance_scale', 0.0)}",
        f"cfg_train.unconditional_prob={policy.get('unconditional_prob')}",
        f"cfg_train.csa_positive_quantile={policy.get('positive_quantile')}",
        f"cfg_train.csa_bottom_quantile={policy.get('csa_bottom_quantile', 0.15)}",
        f"cfg_train.csa_bottom_negative_prob={policy.get('csa_bottom_negative_prob', 0.5)}",
        f"cfg_train.csa_positive_prompt_prob={policy.get('csa_positive_prompt_prob', 0.85)}",
        f"cfg_train.csa_weight_lambda={policy.get('csa_weight_lambda', 0.2)}",
        f"cfg_train.positive_residual_alpha={policy.get('positive_residual_alpha', 0.5)}",
        "cfg_train.max_epochs=-1",
        f"cfg_train.max_steps={policy.get('max_steps')}",
        f"cfg_train.save_interval={policy.get('save_interval')}",
        "cfg_train.val_check_interval=-1",
        f"cfg_train.total_training_steps={policy.get('max_steps')}",
        f"cfg_train.lr_warmup_steps={policy.get('lr_warmup_steps')}",
        f"cfg_train.global_batch_size={policy.get('global_batch_size')}",
        f"cfg_train.micro_batch_size={policy.get('micro_batch_size')}",
        _list_override(
            "cfg_train.extra_overrides", extra_overrides
        ),
        f"output.root={ctx['revalue_root']}",
    ]
    if ctx["parent_checkpoint"]:
        overrides.insert(
            3, f"cfg_train.init_checkpoint_path={ctx['parent_checkpoint']}"
        )
    return [
        Step(
            name="train_cfg",
            argv=_revalue_entry("train_cfg", overrides),
            artifacts=[ctx["policy_ckpt"]],
        )
    ]


def _steps_eval_policy(ctx: dict) -> list[Step]:
    cfg = ctx["cfg"]
    policy = cfg.get("policy", {})
    eval_cfg = cfg.get("eval", {})
    default_guidance = policy.get("guidance_type", "positive")
    default_neg_scale = policy.get("negative_guidance_scale", 0.0)
    results = cfg.get("results", {})
    task_ids = (
        [(task, int(task.removeprefix("task"))) for task in ctx["tasks"]]
        if ctx["multitask"]
        else [(ctx["task"], ctx["task_id"])]
    )
    eval_summaries: dict[str, str] = {}
    steps: list[Step] = []
    for task, task_id in task_ids:
        log_dir = (
            f"{ctx['exp_root']}/eval/{task}"
            if ctx["multitask"]
            else f"{ctx['exp_root']}/eval"
        )
        experiment_name = (
            f"{ctx['eval_exp']}_{task}" if ctx["multitask"] else ctx["eval_exp"]
        )
        summary_path = f"{log_dir}/eval_policy_summary.json"
        eval_summaries[task] = summary_path
        steps.append(
            Step(
                name=f"eval_policy_{task}" if ctx["multitask"] else "eval_policy",
                argv=_revalue_entry(
                    "eval_policy",
                    [
                        "policy_eval.enabled=true",
                        f"policy_eval.model_path={ctx['base_model']}",
                        "policy_eval.model_type=cfg_model",
                        f"policy_eval.checkpoint_path={ctx['policy_ckpt']}",
                        f"policy_eval.experiment_name={experiment_name}",
                        f"policy_eval.log_dir={log_dir}",
                        "policy_eval.config_name=libero_10_pi05_sft_eval",
                        "policy_eval.openpi_config_name="
                        f"{_get(cfg, 'collect.openpi_config_name')}",
                        "policy_eval.guidance_type="
                        f"{eval_cfg.get('guidance_type', default_guidance)}",
                        "policy_eval.positive_only_conditional="
                        f"{eval_cfg.get('positive_only_conditional', True)}",
                        "policy_eval.negative_guidance_scale="
                        f"{eval_cfg.get('negative_guidance_scale', default_neg_scale)}",
                        "policy_eval.warmup_before_env="
                        f"{eval_cfg.get('warmup_before_env', True)}",
                        _list_override(
                            "policy_eval.extra_overrides",
                            [
                                "+actor.model.openpi.cfgrl_guidance_scale="
                                f"{eval_cfg.get('guidance_scale', 1.0)}",
                                "actor.model.add_value_head=false",
                                "env.eval.is_eval=true",
                                "env.eval.use_fixed_reset_state_ids=true",
                                "++env.eval.use_ordered_reset_state_ids=true",
                                "env.eval.video_cfg.save_video=false",
                            ],
                        ),
                        "policy_eval.eval_rollout_epoch="
                        f"{eval_cfg['eval_rollout_epoch']}",
                        f"policy_eval.total_num_envs={eval_cfg['total_num_envs']}",
                        f"policy_eval.task_suite_name={ctx['task_suite_name']}",
                        f"policy_eval.task_id_filter=[{task_id}]",
                        f"policy_eval.save_video={eval_cfg.get('save_video', False)}",
                        f"output.root={ctx['revalue_root']}",
                    ],
                ),
                artifacts=[summary_path],
            )
        )
    if results.get("enabled", True):
        if ctx["multitask"]:
            policy_data_report = (
                f"{ctx['policy_data_root']}/multitask_policy_data_report.json"
            )
            steps.append(
                Step(
                    name="record_multitask_round_results",
                    argv=_script_entry(
                        "examples/recap/process/record_multitask_round_results.py",
                        [
                            f"--round={results['round_index']}",
                            f"--policy-label={results['policy_label']}",
                            f"--critic-label={results['critic_label']}",
                            f"--checkpoint-path={ctx['policy_ckpt']}",
                            *[
                                argument
                                for task, (start, end) in ctx["task_ranges"].items()
                                for argument in (
                                    "--task-range",
                                    f"{task}={start}:{end}",
                                )
                            ],
                            *[
                                argument
                                for task in ctx["tasks"]
                                for argument in (
                                    "--eval-summary",
                                    f"{task}={eval_summaries[task]}",
                                )
                            ],
                            *(
                                [
                                    argument
                                    for task in ctx["tasks"]
                                    for argument in (
                                        "--zp-metrics",
                                        f"{task}={ctx['revalue_root']}/zp_head/"
                                        f"{task}/metrics.json",
                                        "--fusion-metrics",
                                        f"{task}={ctx['revalue_root']}/fusion/"
                                        f"{task}/metrics.json",
                                    )
                                ]
                                if ctx["advantage_source"] == "fused"
                                else []
                            ),
                            f"--comparison={ctx['comparison']}",
                            f"--policy-data-report={policy_data_report}",
                            f"--output-dir={results['output_dir']}",
                            f"--output-name={results['output_name']}",
                        ],
                    ),
                    artifacts=[
                        f"{results['output_dir']}/{results['output_name']}.json",
                        f"{results['output_dir']}/{results['output_name']}.csv",
                    ],
                )
            )
            return steps
        steps.append(
            Step(
                name="record_round_results",
                argv=_script_entry(
                    "examples/recap/process/record_round_results.py",
                    [
                        f"--task={ctx['task']}",
                        f"--round={results['round_index']}",
                        f"--policy-label={results['policy_label']}",
                        f"--critic-label={results['critic_label']}",
                        f"--train-dataset={ctx['merged_ds']}",
                        f"--critic-dataset={ctx['merged_ds']}",
                        "--critic-scope=merged_training_dataset",
                        "--policy-eval-summary="
                        f"{ctx['exp_root']}/eval/eval_policy_summary.json",
                        f"--comparison={ctx['comparison']}",
                        f"--zp-metrics={ctx['revalue_root']}/zp_head/metrics.json",
                        f"--fusion-metrics={ctx['revalue_root']}/fusion/metrics.json",
                        f"--output-dir={results['output_dir']}",
                        f"--output-name={results['output_name']}",
                    ],
                ),
                artifacts=[
                    f"{results['output_dir']}/{results['output_name']}.json",
                    f"{results['output_dir']}/{results['output_name']}.csv",
                ],
            )
        )
    return steps


def _steps_export_bootstrap(ctx: dict) -> list[Step]:
    """Export fused advantages onto the bootstrap dataset itself (no slicing).

    Bootstrap rounds train the critic and the policy on the same freshly
    collected dataset, so the export covers every episode instead of a
    re-indexed child slice of a merged dataset.
    """
    cfg = ctx["cfg"]
    revalue = cfg.get("revalue", {})
    returns = cfg.get("returns", {})
    return [
        _step_prepare_child_manifest(ctx),
        Step(
            name="export_fused_advantages",
            argv=_revalue_entry(
                "export",
                [
                    *_data_overrides(ctx, ctx["merged_ds"]),
                    f"recap.source_advantages_path={ctx['merged_base_adv']}",
                    f"recap.output_tag={ctx['child_fused_tag']}",
                    f"recap.lookahead_step={revalue.get('lookahead_step', 10)}",
                    f"recap.gamma={returns['gamma']}",
                    "recap.positive_quantile="
                    f"{revalue.get('positive_quantile', 0.3)}",
                    "recap.discount_next_value=true",
                    "recap.export_split=null",
                    f"output.root={ctx['revalue_root']}",
                    f"output.predictions_path={ctx['predictions']}",
                ],
            ),
            artifacts=[ctx["child_fused_adv"]],
        ),
    ]


def _step_merge_multitask(ctx: dict) -> Step:
    inputs: list[str] = []
    demo_cfg = ctx.get("demo_datasets") or {}
    for task in ctx["tasks"]:
        inputs.append(ctx["task_datasets"][task])
        if task in demo_cfg:
            inputs.append(demo_cfg[task]["input"])
    return Step(
        name="merge_datasets",
        argv=_script_entry(
            "examples/recap/process/merge_lerobot_multitask_datasets.py",
            [
                "--datasets",
                *inputs,
                f"--output_dataset={ctx['merged_ds']}",
            ],
        ),
        artifacts=[
            f"{ctx['merged_ds']}/meta/info.json",
            f"{ctx['merged_ds']}/meta/phase_progress_multitask.parquet",
        ],
        dataset_path=ctx["merged_ds"],
    )


def _steps_collect_multitask(ctx: dict) -> list[Step]:
    """Collect fresh rollouts for every task in the round (parallel-safe)."""
    cfg = ctx["cfg"]
    collect = cfg.get("collect", {})
    steps: list[Step] = []
    for task in ctx["tasks"]:
        dataset = ctx["policy_task_datasets"][task]
        task_id = int(task.replace("task", ""))
        overrides = [
            "rollout_collect.enabled=true",
            f"rollout_collect.output_dir={dataset}",
            f"rollout_collect.model_path={ctx['base_model']}",
            f"rollout_collect.model_type={_get(cfg, 'collect.model_type', 'openpi')}",
            "rollout_collect.openpi_config_name="
            f"{_get(cfg, 'collect.openpi_config_name')}",
            f"rollout_collect.task_suite_name={ctx['task_suite_name']}",
            f"rollout_collect.task_id={task_id}",
            f"rollout_collect.num_episodes={ctx['child_episodes']}",
            f"rollout_collect.seed={_get(cfg, 'collect.seed', 42)}",
            f"rollout_collect.gpu_id={_get(cfg, 'gpu.gpu_id', 0)}",
            f"rollout_collect.fps={_get(cfg, 'collect.fps', 10)}",
            f"rollout_collect.action_chunk={_get(cfg, 'collect.action_chunk', 5)}",
            f"rollout_collect.num_steps={_get(cfg, 'collect.num_steps', 5)}",
            f"rollout_collect.num_steps_wait={_get(cfg, 'collect.num_steps_wait', 10)}",
            "rollout_collect.warmup_before_env="
            f"{_get(cfg, 'collect.warmup_before_env', True)}",
            "rollout_collect.guidance_type="
            f"{_get(cfg, 'collect.guidance_type', 'positive')}",
            "rollout_collect.positive_only_conditional="
            f"{_get(cfg, 'collect.positive_only_conditional', True)}",
            "rollout_collect.guidance_scale="
            f"{_get(cfg, 'collect.guidance_scale', 1.0)}",
            "rollout_collect.negative_guidance_scale="
            f"{_get(cfg, 'collect.negative_guidance_scale', 0.0)}",
            "rollout_collect.semantic_trace=true",
            f"rollout_collect.semantic_trace_task={task}",
            f"rollout_collect.semantic_trace_output_name=semantic_trace_{task}",
            f"output.root={ctx['exp_root']}/revalue",
        ]
        if ctx["parent_checkpoint"]:
            overrides.append(
                f"rollout_collect.checkpoint_path={ctx['parent_checkpoint']}"
            )
        steps.append(
            Step(
                name=f"collect_rollouts_{task}",
                argv=_revalue_entry("collect_rollouts", overrides),
                artifacts=[
                    f"{dataset}/meta/info.json",
                    f"{dataset}/collection_summary.json",
                ],
                dataset_path=dataset,
            )
        )
        if collect.get("visualize", True):
            steps.append(
                Step(
                    name=f"visualize_{task}",
                    argv=_script_entry(
                        "phase_split_script/visualize_phases.py",
                        [
                            f"--dataset_path={dataset}",
                            "--output_dir="
                            f"{ctx['exp_root']}/collection/visualizations/{task}",
                            "--annotation_name="
                            f"phase_progress_semantic_trace_{task}",
                            f"--num_success={int(collect.get('viz_success', 1))}",
                            f"--num_failure={int(collect.get('viz_failure', 1))}",
                        ],
                    ),
                    artifacts=[f"{ctx['exp_root']}/collection/visualizations/{task}"],
                )
            )
    return steps


def _steps_build_multitask_critic_pool(ctx: dict) -> list[Step]:
    """Merge each task's prior critic pool with its fresh policy rollouts."""
    if not ctx["parent_task_datasets"]:
        raise ValueError(
            "build_multitask_critic_pool requires paths.parent_pattern"
        )
    steps: list[Step] = []
    for task in ctx["tasks"]:
        output_dataset = ctx["task_datasets"][task]
        steps.append(
            Step(
                name=f"build_critic_pool_{task}",
                argv=_script_entry(
                    "examples/recap/process/merge_lerobot_multitask_datasets.py",
                    [
                        "--datasets",
                        ctx["parent_task_datasets"][task],
                        ctx["policy_task_datasets"][task],
                        f"--output_dataset={output_dataset}",
                    ],
                ),
                artifacts=[f"{output_dataset}/meta/info.json"],
                dataset_path=output_dataset,
            )
        )
    return steps


def _steps_fit_critic_multitask(ctx: dict) -> list[Step]:
    """Joint critic on the merged multi-task dataset (no joint z/p/fusion)."""
    critic_steps = _steps_fit_value(ctx)
    critic_steps += _steps_fit_critic(ctx)[3:6]
    return critic_steps


def _steps_score_critic_multitask(ctx: dict) -> list[Step]:
    """Score an existing multi-task pool with an external Value model."""
    if not ctx["value_ckpt_external"]:
        raise ValueError(
            "score_critic_multitask requires paths.value_checkpoint; use "
            "fit_critic_multitask when Value training is part of this run"
        )
    return [
        step
        for step in _steps_fit_critic_multitask(ctx)
        if step.name not in {"merge_datasets", "value_sft"}
    ]


def _task_feature_split_args(ctx: dict) -> list[str]:
    cfg = ctx["cfg"]
    revalue = cfg.get("revalue", {})
    revalue_root = ctx["revalue_root"]
    split_args = [
        f"--features_dir={revalue_root}/features",
        "--ranges",
        *[
            f"{task}={start}-{end}"
            for task, (start, end) in ctx["task_ranges"].items()
        ],
        f"--out_dir={revalue_root}/features",
        "--val_episode_ratio="
        f"{revalue.get('task_val_episode_ratio', 0.2)}",
        f"--seed={revalue.get('seed', 42)}",
    ]
    if revalue.get("allow_single_episode_overlap", False):
        split_args.append("--allow_single_episode_overlap")
    return split_args


def _steps_task_heads(ctx: dict) -> list[Step]:
    """Split features by task and train one z/p head and one fusion per task."""
    cfg = ctx["cfg"]
    revalue = cfg.get("revalue", {})
    returns = cfg.get("returns", {})
    revalue_root = ctx["revalue_root"]
    first_task = ctx["tasks"][0]
    steps = [
        Step(
            name="split_features_by_task",
            argv=_script_entry(
                "examples/recap/process/split_feature_cache_by_task.py",
                _task_feature_split_args(ctx),
            ),
            artifacts=[f"{revalue_root}/features/{first_task}/train.pt"],
        )
    ]
    for task in ctx["tasks"]:
        steps.append(
            Step(
                name=f"train_zp_{task}",
                argv=_revalue_entry(
                    "train_zp",
                    [
                        *_data_overrides(ctx, ctx["merged_ds"]),
                        f"output.root={revalue_root}",
                        f"output.features_dir={revalue_root}/features/{task}",
                        f"output.zp_dir={revalue_root}/zp_head/{task}",
                        "train.device=cuda",
                        f"train.batch_size={revalue.get('train_batch_size', 256)}",
                        "train.num_workers=0",
                        f"zp.head_type={revalue.get('zp_head_type')}",
                        f"zp.use_class_weights={revalue.get('zp_use_class_weights')}",
                        f"zp.max_epochs={revalue.get('zp_max_epochs', 100)}",
                        "zp.early_stop_patience="
                        f"{revalue.get('zp_early_stop_patience', 10)}",
                    ],
                ),
                artifacts=[
                    f"{revalue_root}/zp_head/{task}/zp_head.pt",
                    f"{revalue_root}/zp_head/{task}/metrics.json",
                ],
            )
        )
    for task in ctx["tasks"]:
        steps.append(
            Step(
                name=f"train_fusion_{task}",
                argv=_revalue_entry(
                    "train_fusion",
                    [
                        *_data_overrides(ctx, ctx["merged_ds"]),
                        *_value_model_overrides(ctx, None),
                        f"returns.global_min={returns['global_min']}",
                        f"returns.global_max={returns['global_max']}",
                        f"base.tag={ctx['merged_base_tag']}",
                        f"output.root={revalue_root}",
                        f"output.features_dir={revalue_root}/features/{task}",
                        f"output.zp_dir={revalue_root}/zp_head/{task}",
                        f"output.fusion_dir={revalue_root}/fusion/{task}",
                        "train.device=cuda",
                        f"train.batch_size={revalue.get('train_batch_size', 256)}",
                        "train.num_workers=0",
                        f"fusion.max_epochs={revalue.get('fusion_max_epochs', 100)}",
                        "fusion.early_stop_patience="
                        f"{revalue.get('fusion_early_stop_patience', 10)}",
                        f"fusion.alpha={revalue.get('fusion_alpha', 1.0)}",
                    ],
                ),
                artifacts=[
                    f"{revalue_root}/fusion/{task}/fusion.pt",
                    f"{revalue_root}/fusion/{task}/metrics.json",
                ],
            )
        )
    return steps


def _steps_predict_multitask(ctx: dict) -> list[Step]:
    """Predict fused values per task, then concat and compare jointly."""
    cfg = ctx["cfg"]
    revalue = cfg.get("revalue", {})
    returns = cfg.get("returns", {})
    revalue_root = ctx["revalue_root"]
    steps: list[Step] = []
    prediction_files: list[str] = []
    for task in ctx["tasks"]:
        prediction_path = f"{revalue_root}/predictions_{task}.parquet"
        prediction_files.append(prediction_path)
        steps.append(
            Step(
                name=f"predict_{task}",
                argv=_revalue_entry(
                    "predict",
                    [
                        *_data_overrides(ctx, ctx["merged_ds"]),
                        f"base.tag={ctx['merged_base_tag']}",
                        f"recap.source_advantages_path={ctx['merged_base_adv']}",
                        f"output.root={revalue_root}",
                        f"output.features_dir={revalue_root}/features/{task}",
                        f"output.zp_dir={revalue_root}/zp_head/{task}",
                        f"output.fusion_dir={revalue_root}/fusion/{task}",
                        f"output.predictions_path={prediction_path}",
                        "train.device=cuda",
                        f"train.batch_size={revalue.get('train_batch_size', 256)}",
                    ],
                ),
                artifacts=[prediction_path],
            )
        )
    steps.append(
        Step(
            name="concat_predictions",
            argv=_script_entry(
                "examples/recap/process/concat_parquet.py",
                ["--inputs", *prediction_files, f"--output={ctx['predictions']}"],
            ),
            artifacts=[ctx["predictions"]],
        )
    )
    steps.append(
        Step(
            name="compare_returns",
            argv=_revalue_entry(
                "compare_returns",
                [
                    *_data_overrides(ctx, ctx["merged_ds"]),
                    f"base.tag={ctx['merged_base_tag']}",
                    f"recap.source_advantages_path={ctx['merged_base_adv']}",
                    f"output.root={revalue_root}",
                    f"output.predictions_path={ctx['predictions']}",
                    f"output.comparison_path={ctx['comparison']}",
                    f"returns.global_min={returns['global_min']}",
                    f"returns.global_max={returns['global_max']}",
                    "value.v_min=-1.0",
                    "value.v_max=0.0",
                ],
            ),
            artifacts=[ctx["comparison"]],
        )
    )
    return steps


def _steps_export_multitask(ctx: dict) -> list[Step]:
    """Export per-task fused advantages with per-task positive thresholds."""
    cfg = ctx["cfg"]
    revalue = cfg.get("revalue", {})
    policy_data_root = ctx["policy_data_root"]
    steps: list[Step] = []
    manifest_paths: dict[str, str] = {}
    for task in ctx["tasks"]:
        start, end = ctx["policy_task_ranges"][task]
        policy_dataset = ctx["policy_task_datasets"][task]
        policy_episodes = ctx["policy_episode_counts"][task]
        task_root = f"{policy_data_root}/{task}"
        manifest_paths[task] = f"{task_root}/episode_manifest.json"
        steps.append(
            Step(
                name=f"prepare_manifest_{task}",
                argv=_revalue_entry(
                    "prepare_data",
                    [
                        f"data.dataset_path={policy_dataset}",
                        "data.label_name="
                        f"phase_progress_semantic_trace_{task}",
                        f"data.seed={revalue.get('seed', 42)}",
                        f"manifest.num_episodes={policy_episodes}",
                        "manifest.success_ratio=0.5",
                        "manifest.val_episode_ratio=0.0",
                        "manifest.test_episode_ratio=0.0",
                        f"manifest.success_phase={revalue.get('success_phase', 3)}",
                        f"manifest.num_phases={revalue.get('num_phases', 4)}",
                        "manifest.overwrite=true",
                        f"output.root={task_root}",
                    ],
                ),
                artifacts=[f"{task_root}/episode_manifest.json"],
            )
        )
        steps.append(
            Step(
                name=f"export_dataset_view_{task}",
                argv=_revalue_entry(
                    "export_dataset_view",
                    [
                        "export_view.source_advantages_path="
                        f"{ctx['merged_base_adv']}",
                        f"export_view.predictions_path={ctx['predictions']}",
                        "export_view.child_dataset_path="
                        f"{policy_dataset}",
                        f"export_view.output_tag={ctx['child_fused_tag']}",
                        f"export_view.source_episode_start={start}",
                        f"export_view.source_episode_end={end}",
                        f"export_view.child_episode_offset=-{start}",
                        "export_view.lookahead_step="
                        f"{revalue.get('lookahead_step', 10)}",
                        f"export_view.gamma={_get(cfg, 'returns.gamma')}",
                        "export_view.positive_quantile="
                        f"{revalue.get('positive_quantile', 0.3)}",
                        "export_view.discount_next_value=true",
                        f"export_view.expected_episodes={policy_episodes}",
                        f"export_view.report_path={task_root}/export_report.json",
                        f"output.root={task_root}",
                    ],
                ),
                artifacts=[
                    f"{policy_dataset}/meta/advantages_"
                    f"{ctx['child_fused_tag']}.parquet",
                    f"{task_root}/export_report.json",
                ],
            )
        )
    report_path = f"{policy_data_root}/multitask_policy_data_report.json"
    steps.append(
        Step(
            name="summarize_multitask_policy_data",
            argv=_script_entry(
                "examples/recap/process/summarize_multitask_policy_data.py",
                [
                    *[
                        argument
                        for task in ctx["tasks"]
                        for argument in (
                            "--task-dataset",
                            f"{task}={ctx['policy_task_datasets'][task]}",
                        )
                    ],
                    *[
                        argument
                        for task in ctx["tasks"]
                        for argument in (
                            "--manifest",
                            f"{task}={manifest_paths[task]}",
                        )
                    ],
                    f"--advantage-tag={ctx['child_fused_tag']}",
                    f"--report-path={report_path}",
                    f"--combined-manifest-path={ctx['child_manifest']}",
                ],
            ),
            artifacts=[report_path, ctx["child_manifest"]],
        )
    )
    return steps


def _steps_export_multitask_raw(ctx: dict) -> list[Step]:
    """Export per-task raw Value advantages without z/p or fusion."""
    if ctx["advantage_source"] != "raw":
        raise ValueError(
            "export_multitask_raw requires policy.advantage_source=raw"
        )
    cfg = ctx["cfg"]
    revalue = cfg.get("revalue", {})
    returns = cfg.get("returns", {})
    policy_data_root = ctx["policy_data_root"]
    steps: list[Step] = []
    manifest_paths: dict[str, str] = {}
    for task in ctx["tasks"]:
        start, end = ctx["policy_task_ranges"][task]
        policy_dataset = ctx["policy_task_datasets"][task]
        policy_episodes = ctx["policy_episode_counts"][task]
        task_root = f"{policy_data_root}/{task}"
        manifest_paths[task] = f"{task_root}/episode_manifest.json"
        steps.append(
            Step(
                name=f"prepare_raw_manifest_{task}",
                argv=_revalue_entry(
                    "prepare_data",
                    [
                        f"data.dataset_path={policy_dataset}",
                        "data.label_name="
                        f"phase_progress_semantic_trace_{task}",
                        f"data.seed={revalue.get('seed', 42)}",
                        f"manifest.num_episodes={policy_episodes}",
                        "manifest.success_ratio=0.5",
                        "manifest.val_episode_ratio=0.0",
                        "manifest.test_episode_ratio=0.0",
                        f"manifest.success_phase={revalue.get('success_phase', 3)}",
                        f"manifest.num_phases={revalue.get('num_phases', 4)}",
                        "manifest.overwrite=true",
                        f"output.root={task_root}",
                    ],
                ),
                artifacts=[f"{task_root}/episode_manifest.json"],
            )
        )
        steps.append(
            Step(
                name=f"export_raw_dataset_view_{task}",
                argv=_revalue_entry(
                    "export_dataset_view",
                    [
                        "export_view.mode=raw",
                        "export_view.source_advantages_path="
                        f"{ctx['merged_base_adv']}",
                        "export_view.child_dataset_path="
                        f"{policy_dataset}",
                        f"export_view.output_tag={ctx['child_fused_tag']}",
                        f"export_view.source_episode_start={start}",
                        f"export_view.source_episode_end={end}",
                        f"export_view.child_episode_offset=-{start}",
                        "export_view.positive_quantile="
                        f"{revalue.get('positive_quantile', 0.3)}",
                        f"export_view.expected_episodes={policy_episodes}",
                        f"export_view.report_path={task_root}/raw_export_report.json",
                        f"output.root={task_root}",
                    ],
                ),
                artifacts=[
                    f"{policy_dataset}/meta/advantages_"
                    f"{ctx['child_fused_tag']}.parquet",
                    f"{task_root}/raw_export_report.json",
                ],
            )
        )
    steps.append(
        Step(
            name="write_raw_value_comparison",
            argv=_script_entry(
                "examples/recap/process/write_raw_value_comparison.py",
                [
                    f"--advantages-path={ctx['merged_base_adv']}",
                    f"--output-path={ctx['comparison']}",
                    f"--return-min={returns['global_min']}",
                    f"--return-max={returns['global_max']}",
                    "--value-min=-1.0",
                    "--value-max=0.0",
                ],
            ),
            artifacts=[ctx["comparison"]],
        )
    )
    report_path = f"{policy_data_root}/multitask_policy_data_report.json"
    steps.append(
        Step(
            name="summarize_multitask_raw_policy_data",
            argv=_script_entry(
                "examples/recap/process/summarize_multitask_policy_data.py",
                [
                    *[
                        argument
                        for task in ctx["tasks"]
                        for argument in (
                            "--task-dataset",
                            f"{task}={ctx['policy_task_datasets'][task]}",
                        )
                    ],
                    *[
                        argument
                        for task in ctx["tasks"]
                        for argument in (
                            "--manifest",
                            f"{task}={manifest_paths[task]}",
                        )
                    ],
                    f"--advantage-tag={ctx['child_fused_tag']}",
                    f"--report-path={report_path}",
                    f"--combined-manifest-path={ctx['child_manifest']}",
                ],
            ),
            artifacts=[report_path, ctx["child_manifest"]],
        )
    )
    return steps


_STAGE_BUILDERS = {
    "collect": _steps_collect,
    "fit_value": _steps_fit_value,
    "fit_critic": _steps_fit_critic,
    "build_child_base": _steps_build_child_base,
    "export_bootstrap": _steps_export_bootstrap,
    "collect_multitask": _steps_collect_multitask,
    "build_multitask_critic_pool": _steps_build_multitask_critic_pool,
    "fit_critic_multitask": _steps_fit_critic_multitask,
    "score_critic_multitask": _steps_score_critic_multitask,
    "task_heads": _steps_task_heads,
    "predict_multitask": _steps_predict_multitask,
    "export_multitask": _steps_export_multitask,
    "export_multitask_raw": _steps_export_multitask_raw,
    "export_policy_data": _steps_export_policy_data,
    "train_policy": _steps_train_policy,
    "eval_policy": _steps_eval_policy,
}


def _report_path(ctx: dict, stage: str) -> Path:
    return Path(ctx["exp_root"]) / "stage_reports" / f"{stage}.json"


def _step_report_path(ctx: dict, stage: str, step: Step) -> Path:
    return (
        Path(ctx["exp_root"])
        / "stage_reports"
        / "steps"
        / f"{stage}_{step.name}.json"
    )


def _load_report(ctx: dict, stage: str) -> dict | None:
    path = _report_path(ctx, stage)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _load_step_report(ctx: dict, stage: str, step: Step) -> dict | None:
    path = _step_report_path(ctx, stage, step)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _step_is_current(ctx: dict, stage: str, step: Step) -> bool:
    report = _load_step_report(ctx, stage, step)
    return bool(
        report
        and report.get("step_hash") == _step_hash(step)
        and all(Path(path).exists() for path in step.artifacts)
    )


def _write_step_report(
    ctx: dict,
    stage: str,
    step: Step,
    executed_argv: list[str],
    started_at: str,
    finished_at: str,
) -> None:
    path = _step_report_path(ctx, stage, step)
    payload = {
        "stage": stage,
        "step": step.name,
        "step_hash": _step_hash(step),
        "git_commit": _git_commit(ctx["repo_root"]),
        "argv": executed_argv,
        "artifacts": step.artifacts,
        "started_at": started_at,
        "finished_at": finished_at,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _execution_step(step: Step, overwrite_datasets: bool) -> Step:
    argv = list(step.argv)
    if overwrite_datasets and step.name.startswith("collect_rollouts"):
        argv.append("rollout_collect.overwrite=true")
    if overwrite_datasets and (
        step.name == "merge_datasets"
        or step.name.startswith("build_critic_pool_")
    ):
        argv.append("--overwrite")
    return Step(
        name=step.name,
        argv=argv,
        artifacts=list(step.artifacts),
        dataset_path=step.dataset_path,
    )


def _dataset_output_exists(ctx: dict, step: Step) -> bool:
    if step.dataset_path:
        return Path(step.dataset_path).exists()
    if step.name.startswith("collect_rollouts"):
        return Path(ctx["child_ds"]).exists()
    if step.name == "merge_datasets":
        return Path(ctx["merged_ds"]).exists()
    return False


def _stage_is_current(ctx: dict, stage: str, steps: list[Step]) -> bool:
    report = _load_report(ctx, stage)
    return bool(
        report
        and report.get("stage_hash") == _stage_hash(steps)
        and all(_step_is_current(ctx, stage, step) for step in steps)
    )


def _build_env(ctx: dict) -> dict[str, str]:
    cfg = ctx["cfg"]
    env = os.environ.copy()
    env.pop("RAY_ADDRESS", None)
    repo_root = ctx["repo_root"]
    env["REPO_PATH"] = repo_root
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = str(
        _get(cfg, "gpu.cuda_visible_devices", "0")
    )
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def _run_step(
    step: Step,
    log_path: Path,
    env: dict,
    cwd: str,
    stream: bool = True,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(">>> %s", step.name)
    logger.info("    %s", shlex.join(step.argv))
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            step.argv,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
            if stream:
                sys.stdout.write(line)
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"step {step.name!r} failed with exit code {proc.returncode}; "
            f"see {log_path}"
        )
    missing = [path for path in step.artifacts if not Path(path).exists()]
    if missing:
        raise RuntimeError(
            f"step {step.name!r} finished but artifacts are missing: {missing}"
        )


def _run_steps_parallel(
    ctx: dict,
    stage: str,
    steps: list[Step],
    env: dict,
    args: argparse.Namespace,
    workers: int,
) -> None:
    """Run independent steps concurrently (used for multi-task collection)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    pending: list[Step] = []
    for step in steps:
        if not args.force and _step_is_current(ctx, stage, step):
            logger.info("step %s has a matching report; skipping", step.name)
            continue
        if (
            step.dataset_path is not None
            and _dataset_output_exists(ctx, step)
            and not args.overwrite_datasets
        ):
            raise RuntimeError(
                f"step {step.name!r} needs to replace an existing dataset; "
                "re-run with --overwrite-datasets after verifying the target path"
            )
        pending.append(_execution_step(step, args.overwrite_datasets))
    if not pending:
        return

    logger.info("running %d steps with %d workers", len(pending), workers)

    def _job(step: Step) -> tuple[Step, str, str]:
        log_path = Path(ctx["exp_root"]) / "logs" / f"{stage}_{step.name}.log"
        started = datetime.now(timezone.utc).isoformat()
        _run_step(step, log_path, env, ctx["repo_root"], stream=False)
        finished = datetime.now(timezone.utc).isoformat()
        return step, started, finished

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_job, step): step for step in pending}
        for future in as_completed(futures):
            try:
                done_step, started, finished = future.result()
            except Exception:  # noqa: BLE001 - report and re-raise below
                failures.append(futures[future].name)
                logger.exception("step %s failed", futures[future].name)
                continue
            logger.info("step %s finished", done_step.name)
            _write_step_report(
                ctx,
                stage,
                _step_by_name(steps, done_step.name),
                done_step.argv,
                started,
                finished,
            )
    if failures:
        raise RuntimeError(f"parallel steps failed: {sorted(failures)}")


def _step_by_name(steps: list[Step], name: str) -> Step:
    for step in steps:
        if step.name == name:
            return step
    raise KeyError(name)


def _write_manifest(ctx: dict, cfg_hash: str) -> None:
    manifest_path = Path(ctx["exp_root"]) / "run_manifest.json"
    reports = {}
    for stage in ctx["pipeline"]:
        report = _load_report(ctx, stage)
        reports[stage] = (
            {"status": "done", "finished_at": report.get("finished_at")}
            if report
            else {"status": "pending"}
        )
    payload = {
        "round_config": ctx["cfg"],
        "config_hash": cfg_hash,
        "git_commit": _git_commit(ctx["repo_root"]),
        "cuda_visible_devices": _get(ctx["cfg"], "gpu.cuda_visible_devices", "0"),
        "derived": {
            key: ctx[key]
            for key in (
                "parent_ds",
                "child_ds",
                "merged_ds",
                "parent_episodes",
                "child_episodes",
                "merged_episodes",
                "value_ckpt",
                "policy_ckpt",
                "merged_base_adv",
                "child_fused_adv",
                "predictions",
                "comparison",
            )
        },
        "stages": reports,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_stage(stage: str, ctx: dict, args: argparse.Namespace) -> None:
    cfg_hash = _config_hash(ctx["cfg"])
    steps = _STAGE_BUILDERS[stage](ctx)
    stage_hash = _stage_hash(steps)
    if _stage_is_current(ctx, stage, steps) and not args.force:
        logger.info(
            "stage %s already completed with matching step reports; skipping "
            "(use --force to re-run)",
            stage,
        )
        return

    for dep in _stage_dependencies(ctx, stage):
        dep_steps = _STAGE_BUILDERS[dep](ctx)
        if not _stage_is_current(ctx, dep, dep_steps) and not args.dry_run:
            raise RuntimeError(
                f"stage {stage!r} requires current stage {dep!r}; "
                f"run --stage {dep} with the current round config first"
            )

    if args.dry_run:
        print(f"\n=== stage: {stage} (dry-run) ===")
        for step in steps:
            execution_step = _execution_step(step, args.overwrite_datasets)
            print(f"\n# step: {step.name}")
            print(shlex.join(execution_step.argv))
            for artifact in step.artifacts:
                print(f"#   artifact: {artifact}")
        return

    gate_stage = _audit_gate_stage(ctx)
    if stage == gate_stage and not args.confirm_audit:
        raise RuntimeError(
            f"stage {stage!r} requires --confirm-audit: review the "
            "collection summary, semantic trace and visualizations of the "
            "fresh rollouts before building critic data on top of them"
        )
    if (
        stage in {"build_child_base", "score_critic_multitask"}
        and ctx.get("value_ckpt_external")
        and not Path(ctx["value_ckpt"]).exists()
    ):
        raise RuntimeError(
            f"value checkpoint not found: {ctx['value_ckpt']}"
        )
    if stage == "score_critic_multitask" and not (
        Path(ctx["merged_ds"]) / "meta" / "info.json"
    ).exists():
        raise RuntimeError(
            "score_critic_multitask requires an existing merged dataset at "
            f"{ctx['merged_ds']}; run build_multitask_critic_pool first or "
            "use fit_critic_multitask to build the pool and train Value"
        )

    started = datetime.now(timezone.utc).isoformat()
    env = _build_env(ctx)
    parallel_workers = int(_get(ctx["cfg"], "collect.parallel", 1))
    if stage == "collect_multitask" and parallel_workers > 1:
        _run_steps_parallel(ctx, stage, steps, env, args, parallel_workers)
    for step in steps:
        if not args.force and _step_is_current(ctx, stage, step):
            logger.info(
                "step %s has a matching report and artifacts; skipping", step.name
            )
            continue
        if (
            (
                step.dataset_path is not None
                or step.name in {"collect_rollouts", "merge_datasets"}
            )
            and _dataset_output_exists(ctx, step)
            and not args.overwrite_datasets
        ):
            raise RuntimeError(
                f"step {step.name!r} needs to replace an existing dataset; "
                "re-run with --overwrite-datasets after verifying the target path"
            )
        execution_step = _execution_step(step, args.overwrite_datasets)
        log_path = (
            Path(ctx["exp_root"]) / "logs" / f"{stage}_{step.name}.log"
        )
        step_started = datetime.now(timezone.utc).isoformat()
        _run_step(execution_step, log_path, env, ctx["repo_root"])
        step_finished = datetime.now(timezone.utc).isoformat()
        _write_step_report(
            ctx,
            stage,
            step,
            execution_step.argv,
            step_started,
            step_finished,
        )

    finished = datetime.now(timezone.utc).isoformat()
    report = {
        "stage": stage,
        "config_hash": cfg_hash,
        "stage_hash": stage_hash,
        "git_commit": _git_commit(ctx["repo_root"]),
        "steps": [step.name for step in steps],
        "started_at": started,
        "finished_at": finished,
    }
    report_path = _report_path(ctx, stage)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_manifest(ctx, cfg_hash)
    logger.info("stage %s completed; report at %s", stage, report_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="round YAML path")
    parser.add_argument(
        "--stage",
        required=True,
        choices=[*KNOWN_STAGES, "all"],
        help="stage to run",
    )
    parser.add_argument(
        "--confirm-audit",
        action="store_true",
        help="confirm the human semantic audit of freshly collected rollouts",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-run compute steps even if their reports are current",
    )
    parser.add_argument(
        "--overwrite-datasets",
        action="store_true",
        help="allow collect or merge steps to replace their output datasets",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print stage commands without executing them",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "override a round config value (repeatable), e.g. "
            "--set paths.child_dataset=/data/libero_long/task1_r1b_40 "
            "--set collect.num_episodes=20"
        ),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    cfg = _load_yaml(Path(args.config))
    cfg = _apply_overrides(cfg, args.overrides)
    ctx = _build_ctx(cfg)
    if args.stage == "all" and args.confirm_audit:
        collect_stage = next(
            (
                stage
                for stage in ("collect", "collect_multitask")
                if stage in ctx["pipeline"]
            ),
            None,
        )
        if collect_stage is None:
            raise RuntimeError(
                "--confirm-audit requires collect or collect_multitask in pipeline"
            )
        collect_steps = _STAGE_BUILDERS[collect_stage](ctx)
        if not _stage_is_current(ctx, collect_stage, collect_steps):
            raise RuntimeError(
                "--stage all --confirm-audit requires a previously completed "
                f"{collect_stage} stage; run --stage all once, audit the new "
                "rollouts, "
                "then repeat the command with --confirm-audit"
            )
    stages = ctx["pipeline"] if args.stage == "all" else [args.stage]
    for stage in stages:
        run_stage(stage, ctx, args)
        if stage == "collect" and args.stage == "all" and not args.confirm_audit:
            logger.info(
                "collection finished; review the rollouts, then continue with "
                "--stage fit_critic --confirm-audit"
            )
            break


if __name__ == "__main__":
    main()
