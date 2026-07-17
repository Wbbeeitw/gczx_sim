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
    if "collect" not in pipeline:
        return None
    index = pipeline.index("collect")
    if index + 1 < len(pipeline):
        return pipeline[index + 1]
    return None


@dataclass
class Step:
    """One executable step inside a stage."""

    name: str
    argv: list[str]
    artifacts: list[str] = field(default_factory=list)


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
    child_ds = str(_require(cfg, "paths.child_dataset"))
    parent_ds = str(_require(cfg, "paths.parent_dataset"))
    bootstrap = bool(_get(cfg, "bootstrap", False))
    if bootstrap:
        merged_ds = child_ds
    parent_episodes = int(_require(cfg, "datasets.parent_episodes"))
    child_episodes = int(_get(cfg, "collect.num_episodes"))
    merged_episodes = parent_episodes + child_episodes

    merged_name = os.path.basename(merged_ds.rstrip("/"))
    value_steps = int(_get(cfg, "value.steps"))
    value_exp = f"value_{merged_name}"
    policy_exp = f"policy{round_id}_{method}"
    eval_cfg = cfg.get("eval", {})
    eval_episodes = int(eval_cfg["eval_rollout_epoch"]) * int(
        eval_cfg["total_num_envs"]
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
        "base_model": str(_require(cfg, "paths.base_model")),
        "parent_checkpoint": str(_get(cfg, "parent_policy.checkpoint") or ""),
        "parent_label": str(_get(cfg, "parent_policy.label", "parent")),
        "returns_tag": str(_require(cfg, "tags.returns")),
        "merged_base_tag": str(_require(cfg, "tags.merged_base")),
        "child_fused_tag": str(_require(cfg, "tags.child_fused")),
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
            f"/global_step_{int(_get(cfg, 'policy.max_steps'))}"
            "/actor/model_state_dict/full_weights.pt"
        ),
        "eval_exp": f"{policy_exp}_eval{eval_episodes}",
        "revalue_root": f"{exp_root}/revalue",
        "policy_data_root": f"{exp_root}/policy_data",
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
    )
    ctx["child_manifest"] = f"{ctx['policy_data_root']}/episode_manifest.json"
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
                "actor.model.freeze_vlm=true",
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
    return Step(
        name="prepare_child_manifest",
        argv=_revalue_entry(
            "prepare_data",
            [
                f"data.dataset_path={ctx['child_ds']}",
                f"data.label_name={revalue['label_name']}",
                f"data.seed={revalue.get('seed', 42)}",
                f"manifest.num_episodes={ctx['child_episodes']}",
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
    overrides = [
        "cfg_train.enabled=true",
        f"cfg_train.dataset_path={ctx['child_ds']}",
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
        f"cfg_train.csa_bottom_quantile={policy.get('csa_bottom_quantile')}",
        f"cfg_train.csa_bottom_negative_prob={policy.get('csa_bottom_negative_prob')}",
        f"cfg_train.csa_positive_prompt_prob={policy.get('csa_positive_prompt_prob')}",
        f"cfg_train.csa_weight_lambda={policy.get('csa_weight_lambda')}",
        f"cfg_train.positive_residual_alpha={policy.get('positive_residual_alpha')}",
        "cfg_train.max_epochs=-1",
        f"cfg_train.max_steps={policy.get('max_steps')}",
        f"cfg_train.save_interval={policy.get('save_interval')}",
        "cfg_train.val_check_interval=-1",
        f"cfg_train.total_training_steps={policy.get('max_steps')}",
        f"cfg_train.lr_warmup_steps={policy.get('lr_warmup_steps')}",
        f"cfg_train.global_batch_size={policy.get('global_batch_size')}",
        f"cfg_train.micro_batch_size={policy.get('micro_batch_size')}",
        _list_override(
            "cfg_train.extra_overrides", policy.get("extra_overrides", [])
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
    steps = [
        Step(
            name="eval_policy",
            argv=_revalue_entry(
                "eval_policy",
                [
                    "policy_eval.enabled=true",
                    f"policy_eval.model_path={ctx['base_model']}",
                    "policy_eval.model_type=cfg_model",
                    f"policy_eval.checkpoint_path={ctx['policy_ckpt']}",
                    f"policy_eval.experiment_name={ctx['eval_exp']}",
                    f"policy_eval.log_dir={ctx['exp_root']}/eval",
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
                            f"{eval_cfg.get('guidance_scale', 1.0)}"
                        ],
                    ),
                    f"policy_eval.eval_rollout_epoch={eval_cfg['eval_rollout_epoch']}",
                    f"policy_eval.total_num_envs={eval_cfg['total_num_envs']}",
                    f"policy_eval.task_suite_name={ctx['task_suite_name']}",
                    f"policy_eval.task_id_filter=[{ctx['task_id']}]",
                    f"policy_eval.save_video={eval_cfg.get('save_video', False)}",
                    f"output.root={ctx['revalue_root']}",
                ],
            ),
            artifacts=[f"{ctx['exp_root']}/eval/eval_policy_summary.json"],
        )
    ]
    if results.get("enabled", True):
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


_STAGE_BUILDERS = {
    "collect": _steps_collect,
    "fit_value": _steps_fit_value,
    "fit_critic": _steps_fit_critic,
    "build_child_base": _steps_build_child_base,
    "export_bootstrap": _steps_export_bootstrap,
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
    if overwrite_datasets and step.name == "collect_rollouts":
        argv.append("rollout_collect.overwrite=true")
    if overwrite_datasets and step.name == "merge_datasets":
        argv.append("--overwrite")
    return Step(name=step.name, argv=argv, artifacts=list(step.artifacts))


def _dataset_output_exists(ctx: dict, step: Step) -> bool:
    if step.name == "collect_rollouts":
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


def _run_step(step: Step, log_path: Path, env: dict, cwd: str) -> None:
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
        stage == "build_child_base"
        and ctx.get("value_ckpt_external")
        and not Path(ctx["value_ckpt"]).exists()
    ):
        raise RuntimeError(
            f"value checkpoint not found: {ctx['value_ckpt']}; run the "
            "fused track's fit_critic first, or switch to separate-critic "
            "mode (pipeline: [collect, fit_value, build_child_base, ...] "
            "with paths.value_checkpoint: null)"
        )

    started = datetime.now(timezone.utc).isoformat()
    env = _build_env(ctx)
    for step in steps:
        if not args.force and _step_is_current(ctx, stage, step):
            logger.info(
                "step %s has a matching report and artifacts; skipping", step.name
            )
            continue
        if (
            step.name in {"collect_rollouts", "merge_datasets"}
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
        collect_steps = _STAGE_BUILDERS["collect"](ctx)
        if not _stage_is_current(ctx, "collect", collect_steps):
            raise RuntimeError(
                "--stage all --confirm-audit requires a previously completed "
                "collect stage; run --stage all once, audit the new rollouts, "
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
