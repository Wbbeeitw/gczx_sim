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

"""Embodied downstream stages for Revalue."""

from __future__ import annotations

import ast
from collections import deque
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rlinf.revalue.io import load_json, save_json

logger = logging.getLogger(__name__)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclass
class DownstreamCFGTrainingConfig:
    """Config for downstream CFG training."""

    repo_root: str
    dataset_path: str
    base_model_path: str
    advantage_tag: str
    experiment_name: str
    log_dir: str
    config_name: str = "libero_cfg_openpi"
    episode_split_path: str | None = None
    episode_split_name: str = "train"
    model_type: str = "cfg_model"
    openpi_config_name: str = "pi05_libero"
    strategy: str = "binary"
    guidance_type: str = "positive"
    positive_only_conditional: bool = True
    unconditional_prob: float = 0.1
    negative_guidance_scale: float = 0.0
    csa_positive_quantile: float = 0.30
    csa_bottom_quantile: float = 0.15
    csa_positive_prompt_prob: float = 0.85
    csa_bottom_negative_prob: float = 0.50
    csa_weight_lambda: float = 0.20
    positive_residual_alpha: float = 0.50
    max_epochs: int = -1
    max_steps: int = 5000
    save_interval: int = 5000
    val_check_interval: int = -1
    total_training_steps: int | None = None
    lr_warmup_steps: int | None = None
    global_batch_size: int = 256
    micro_batch_size: int = 16
    data_type: str = "rollout"
    dataset_weight: float = 1.0
    python_bin: str | None = None
    extra_overrides: list[str] = field(default_factory=list)


@dataclass
class PolicyEvaluationConfig:
    """Config for embodied policy evaluation."""

    repo_root: str
    model_path: str
    checkpoint_path: str | None
    experiment_name: str
    log_dir: str
    config_name: str = "libero_10_pi05_sft_eval"
    model_type: str = "openpi"
    openpi_config_name: str = "pi05_libero"
    guidance_type: str = "positive"
    positive_only_conditional: bool = True
    eval_rollout_epoch: int = 10
    total_num_envs: int = 5
    save_video: bool = False
    task_suite_name: str | None = None
    task_id_filter: list[int] = field(default_factory=list)
    python_bin: str | None = None
    extra_overrides: list[str] = field(default_factory=list)
    negative_guidance_scale: float = 0.0


@dataclass
class LiberoRolloutCollectionConfig:
    """Config for collecting LIBERO rollouts into a LeRobot dataset."""

    output_dir: str
    model_path: str
    model_type: str = "openpi"
    checkpoint_path: str | None = None
    openpi_config_name: str = "pi05_libero"
    task_suite_name: str = "libero_10"
    task_id: int = 0
    num_episodes: int = 64
    noise_scale: float = 0.0
    noise_clip: float = 0.3
    action_chunk: int = 5
    num_steps: int = 5
    num_steps_wait: int = 10
    seed: int = 42
    gpu_id: int = 0
    fps: int = 10
    overwrite: bool = False
    failure_reward: float | None = None
    semantic_trace: bool = False
    semantic_trace_task: str = "task1"
    semantic_trace_output_name: str = "semantic_trace_task1"


def _python_bin(explicit_python: str | None) -> str:
    return explicit_python or sys.executable


def _base_env(repo_root: str) -> dict[str, str]:
    env = os.environ.copy()
    repo_path = Path(repo_root)
    env.setdefault("REPO_PATH", str(repo_path))
    env.setdefault("EMBODIED_PATH", str(repo_path / "examples" / "embodiment"))
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    python_path_entries = [str(repo_path)]
    existing_python_path = env.get("PYTHONPATH")
    if existing_python_path:
        python_path_entries.append(existing_python_path)
    env["PYTHONPATH"] = os.pathsep.join(python_path_entries)
    return env


def _quote_override(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, dict)):
        return _hydra_container_literal(value)
    text = str(value)
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


_HYDRA_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _hydra_key_literal(key: Any) -> str:
    text = str(key)
    if _HYDRA_KEY_RE.fullmatch(text):
        return text
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _hydra_container_literal(value: Any) -> str:
    if isinstance(value, dict):
        parts = [
            f"{_hydra_key_literal(key)}:{_quote_override(item)}"
            for key, item in value.items()
        ]
        return "{" + ",".join(parts) + "}"
    if isinstance(value, list):
        return "[" + ",".join(_quote_override(item) for item in value) + "]"
    raise TypeError(f"Unsupported container type: {type(value).__name__}")


def _run_python_entry(
    *,
    repo_root: str,
    script_relpath: str,
    config_dir_relpath: str,
    config_name: str,
    overrides: list[str],
    log_file: str | Path,
    python_bin: str | None = None,
) -> subprocess.CompletedProcess[str]:
    repo_path = Path(repo_root)
    script_path = repo_path / script_relpath
    config_dir = repo_path / config_dir_relpath
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _python_bin(python_bin),
        str(script_path),
        "--config-path",
        str(config_dir),
        "--config-name",
        config_name,
        *overrides,
    ]
    env = _base_env(repo_root)
    logger.info("running command: %s", " ".join(cmd))
    logger.info("streaming child output to terminal and log: %s", log_file)
    tail_lines: deque[str] = deque(maxlen=400)
    with subprocess.Popen(
        cmd,
        cwd=str(repo_path),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    ) as proc, open(log_file, "w", encoding="utf-8") as handle:
        assert proc.stdout is not None
        for line in proc.stdout:
            handle.write(line)
            handle.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
            tail_lines.append(line.rstrip("\n"))
        returncode = proc.wait()
    stdout_tail = "\n".join(tail_lines)
    proc = subprocess.CompletedProcess(
        args=cmd,
        returncode=returncode,
        stdout=stdout_tail,
        stderr="",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}\n"
            f"See log: {log_file}"
        )
    return proc


def _latest_checkpoint_path(log_dir: str | Path, max_steps: int) -> Path | None:
    experiment_root = Path(log_dir)
    candidate = (
        experiment_root
        / "checkpoints"
        / f"global_step_{max_steps}"
        / "actor"
        / "model_state_dict"
        / "full_weights.pt"
    )
    if candidate.exists():
        return candidate
    checkpoints_dir = experiment_root / "checkpoints"
    if not checkpoints_dir.exists():
        return None
    step_dirs = sorted(
        (
            path
            for path in checkpoints_dir.iterdir()
            if path.is_dir() and path.name.startswith("global_step_")
        ),
        key=lambda path: int(path.name.split("global_step_")[-1]),
    )
    for step_dir in reversed(step_dirs):
        full_weights = step_dir / "actor" / "model_state_dict" / "full_weights.pt"
        if full_weights.exists():
            return full_weights
    return None


def train_cfg_from_advantages(cfg: DownstreamCFGTrainingConfig) -> dict[str, Any]:
    """Launch downstream CFG training and return a stable run summary."""
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    total_training_steps = (
        cfg.total_training_steps if cfg.total_training_steps is not None else cfg.max_steps
    )
    lr_warmup_steps = (
        cfg.lr_warmup_steps
        if cfg.lr_warmup_steps is not None
        else min(5000, max(0, cfg.max_steps // 10))
    )
    train_data_paths = [
        {
            "dataset_path": cfg.dataset_path,
            "type": cfg.data_type,
            "weight": float(cfg.dataset_weight),
        }
    ]
    overrides = [
        f"runner.logger.log_path={_quote_override(str(log_dir))}",
        f"runner.logger.experiment_name={_quote_override(cfg.experiment_name)}",
        f"runner.max_epochs={cfg.max_epochs}",
        f"runner.max_steps={cfg.max_steps}",
        f"runner.save_interval={cfg.save_interval}",
        f"runner.val_check_interval={cfg.val_check_interval}",
        f"actor.optim.total_training_steps={total_training_steps}",
        f"actor.optim.lr_warmup_steps={lr_warmup_steps}",
        f"data.train_data_paths={_quote_override(train_data_paths)}",
        f"data.advantage_tag={_quote_override(cfg.advantage_tag)}",
        f"actor.model.model_path={_quote_override(cfg.base_model_path)}",
        f"actor.model.model_type={_quote_override(cfg.model_type)}",
        f"actor.model.openpi.config_name={_quote_override(cfg.openpi_config_name)}",
        f"actor.global_batch_size={cfg.global_batch_size}",
        f"actor.micro_batch_size={cfg.micro_batch_size}",
    ]
    if cfg.episode_split_path:
        overrides.extend(
            [
                f"data.episode_split_path={_quote_override(cfg.episode_split_path)}",
                f"data.episode_split_name={_quote_override(cfg.episode_split_name)}",
            ]
        )
    if cfg.model_type == "cfg_model":
        binary_style_strategies = {"binary", "binary_weighted", "acp_cfg"}
        effective_positive_only_conditional = (
            cfg.positive_only_conditional
            if cfg.strategy in binary_style_strategies
            else False
        )
        overrides.extend(
            [
                f"data.cfg_strategy={_quote_override(cfg.strategy)}",
                f"data.csa_positive_quantile={cfg.csa_positive_quantile}",
                f"data.csa_bottom_quantile={cfg.csa_bottom_quantile}",
                f"data.csa_bottom_negative_prob={cfg.csa_bottom_negative_prob}",
                f"data.csa_weight_lambda={cfg.csa_weight_lambda}",
                f"actor.model.openpi.guidance_type={_quote_override(cfg.guidance_type)}",
                f"actor.model.openpi.positive_only_conditional={_quote_override(effective_positive_only_conditional)}",
                f"actor.model.openpi.unconditional_prob={cfg.unconditional_prob}",
                f"actor.model.openpi.cfgrl_negative_guidance_scale={cfg.negative_guidance_scale}",
                f"actor.model.openpi.csa_positive_prompt_prob={cfg.csa_positive_prompt_prob}",
                f"actor.model.openpi.positive_residual_alpha={cfg.positive_residual_alpha}",
            ]
        )
    overrides.extend(cfg.extra_overrides)
    proc = _run_python_entry(
        repo_root=cfg.repo_root,
        script_relpath="examples/recap/cfg/train_cfg.py",
        config_dir_relpath="examples/recap/cfg/config",
        config_name=cfg.config_name,
        overrides=overrides,
        log_file=log_dir / "train_cfg.log",
        python_bin=cfg.python_bin,
    )
    checkpoint_path = _latest_checkpoint_path(log_dir / cfg.experiment_name, cfg.max_steps)
    summary = {
        "config_name": cfg.config_name,
        "experiment_name": cfg.experiment_name,
        "log_dir": str(log_dir / cfg.experiment_name),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "advantage_tag": cfg.advantage_tag,
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-4000:],
    }
    save_json(summary, log_dir / "train_cfg_summary.json")
    return summary


def _parse_eval_metrics(stdout: str) -> dict[str, Any] | None:
    marker = "[INFO"
    lines = stdout.splitlines()
    for line in reversed(lines):
        if marker in line and "RLinf" in line and "{" in line and "}" in line:
            payload = line[line.find("{") : line.rfind("}") + 1]
            cleaned = payload.replace("array(", "").replace(", dtype=float32)", "")
            try:
                parsed = ast.literal_eval(cleaned)
            except (ValueError, SyntaxError):
                continue
            result = {}
            for key, value in parsed.items():
                if isinstance(value, np.generic):
                    result[key] = value.item()
                else:
                    result[key] = value
            return result
    return None


def evaluate_policy_checkpoint(cfg: PolicyEvaluationConfig) -> dict[str, Any]:
    """Launch embodied evaluation and return parsed success metrics."""
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    overrides = [
        f"runner.logger.log_path={_quote_override(str(log_dir))}",
        f"runner.logger.experiment_name={_quote_override(cfg.experiment_name)}",
        f"actor.model.model_path={_quote_override(cfg.model_path)}",
        f"rollout.model.model_path={_quote_override(cfg.model_path)}",
        f"actor.model.model_type={_quote_override(cfg.model_type)}",
        f"algorithm.eval_rollout_epoch={cfg.eval_rollout_epoch}",
        f"env.eval.total_num_envs={cfg.total_num_envs}",
        f"env.eval.video_cfg.save_video={_quote_override(cfg.save_video)}",
        f"actor.model.openpi.config_name={_quote_override(cfg.openpi_config_name)}",
    ]
    if cfg.checkpoint_path:
        overrides.append(f"runner.ckpt_path={_quote_override(cfg.checkpoint_path)}")
    if cfg.task_suite_name:
        overrides.append(f"env.eval.task_suite_name={_quote_override(cfg.task_suite_name)}")
    if cfg.task_id_filter:
        overrides.append(f"+env.eval.task_id_filter={_quote_override(cfg.task_id_filter)}")
    if cfg.model_type == "cfg_model":
        overrides.extend(
            [
                f"+actor.model.openpi.guidance_type={_quote_override(cfg.guidance_type)}",
                f"+actor.model.openpi.positive_only_conditional={_quote_override(cfg.positive_only_conditional)}",
                f"+actor.model.openpi.cfgrl_negative_guidance_scale={cfg.negative_guidance_scale}",
            ]
        )
    overrides.extend(cfg.extra_overrides)
    proc = _run_python_entry(
        repo_root=cfg.repo_root,
        script_relpath="examples/embodiment/eval_embodied_agent.py",
        config_dir_relpath="examples/embodiment/config",
        config_name=cfg.config_name,
        overrides=overrides,
        log_file=log_dir / "eval_policy.log",
        python_bin=cfg.python_bin,
    )
    metrics = _parse_eval_metrics(proc.stdout or "")
    summary = {
        "config_name": cfg.config_name,
        "experiment_name": cfg.experiment_name,
        "log_dir": str(log_dir / cfg.experiment_name),
        "checkpoint_path": cfg.checkpoint_path,
        "metrics": metrics,
        "returncode": proc.returncode,
    }
    save_json(summary, log_dir / "eval_policy_summary.json")
    return summary


def _quat2axisangle(quat):
    """Convert quaternion to axis-angle using the robosuite convention."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = math.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_libero_env(task, resolution, seed, gpu_id=0):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

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


def _max_steps_for_suite(suite_name: str) -> int:
    return {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }.get(suite_name, 520)


def _build_features(image_shape, state_dim, action_dim):
    video_feature = {
        "dtype": "video",
        "shape": list(image_shape),
        "names": ["height", "width", "channel"],
    }
    return {
        "image": video_feature,
        "wrist_image": dict(video_feature),
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
        "done": {"dtype": "bool", "shape": (1,), "names": ["done"]},
        "is_success": {"dtype": "bool", "shape": (1,), "names": ["is_success"]},
        "reward": {"dtype": "float32", "shape": (1,), "names": ["reward"]},
        "return": {"dtype": "float32", "shape": (1,), "names": ["return"]},
    }


def _compute_returns(rewards, gamma=1.0):
    returns = np.zeros(len(rewards), dtype=np.float32)
    g = 0.0
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g
        returns[t] = g
    return returns


def _load_rollout_policy(cfg: LiberoRolloutCollectionConfig):
    import torch
    from omegaconf import OmegaConf

    if cfg.model_type == "cfg_model" or cfg.checkpoint_path is not None:
        if cfg.model_type == "cfg_model":
            from rlinf.models.embodiment.openpi_cfg import get_model

            model_cfg = OmegaConf.create(
                {
                    "model_type": "cfg_model",
                    "model_path": cfg.model_path,
                    "precision": None,
                    "openpi": {
                        "config_name": cfg.openpi_config_name,
                        "guidance_type": "positive",
                        "positive_only_conditional": True,
                    },
                }
            )
        else:
            from rlinf.models.embodiment.openpi import get_model

            model_cfg = OmegaConf.create(
                {
                    "model_type": cfg.model_type,
                    "model_path": cfg.model_path,
                    "precision": None,
                    "openpi": {
                        "config_name": cfg.openpi_config_name,
                    },
                }
            )
        model = get_model(model_cfg)
        if cfg.checkpoint_path:
            model_dict = torch.load(cfg.checkpoint_path, map_location="cpu")
            model.load_state_dict(model_dict, strict=False)
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()
        return model

    from toolkits.eval_scripts_openpi import setup_policy

    policy_args = type(
        "PolicyArgs",
        (),
        {
            "config_name": cfg.openpi_config_name,
            "pretrained_path": cfg.model_path,
            "num_steps": cfg.num_steps,
            "action_chunk": cfg.action_chunk,
            "num_steps_wait": cfg.num_steps_wait,
        },
    )()
    return setup_policy(policy_args)


def collect_libero_rollouts(cfg: LiberoRolloutCollectionConfig) -> dict[str, Any]:
    """Collect LIBERO rollouts and save them as a LeRobot dataset."""
    from libero.libero import benchmark

    from rlinf.data.lerobot_writer import LeRobotDatasetWriter

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    output_path = Path(cfg.output_dir)
    if output_path.exists():
        if cfg.overwrite:
            shutil.rmtree(output_path)
            logger.info("removed existing rollout output dir: %s", output_path)
        else:
            raise FileExistsError(
                f"Output dir already exists: {output_path}. Set overwrite=true to replace."
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    policy = _load_rollout_policy(cfg)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    task = task_suite.get_task(cfg.task_id)
    initial_states = task_suite.get_task_init_states(cfg.task_id)
    max_steps = _max_steps_for_suite(cfg.task_suite_name)
    env, task_description = _get_libero_env(
        task, LIBERO_ENV_RESOLUTION, cfg.seed, cfg.gpu_id
    )
    writer = LeRobotDatasetWriter()

    semantic_trace_recorder = None
    semantic_trace_writer = None
    semantic_trace_records: list[dict[str, Any]] = []
    if cfg.semantic_trace:
        from rlinf.revalue.semantic_trace import (
            Task1SemanticTraceRecorder,
            Task2SemanticTraceRecorder,
            Task3SemanticTraceRecorder,
            Task4SemanticTraceRecorder,
            Task5SemanticTraceRecorder,
            Task6SemanticTraceRecorder,
            write_task1_semantic_artifacts,
            write_task2_semantic_artifacts,
            write_task3_semantic_artifacts,
            write_task4_semantic_artifacts,
            write_task5_semantic_artifacts,
            write_task6_semantic_artifacts,
        )

        semantic_trace_specs = {
            "task1": (
                1,
                Task1SemanticTraceRecorder,
                write_task1_semantic_artifacts,
            ),
            "task2": (
                2,
                Task2SemanticTraceRecorder,
                write_task2_semantic_artifacts,
            ),
            "task3": (
                3,
                Task3SemanticTraceRecorder,
                write_task3_semantic_artifacts,
            ),
            "task4": (
                4,
                Task4SemanticTraceRecorder,
                write_task4_semantic_artifacts,
            ),
            "task5": (
                5,
                Task5SemanticTraceRecorder,
                write_task5_semantic_artifacts,
            ),
            "task6": (
                6,
                Task6SemanticTraceRecorder,
                write_task6_semantic_artifacts,
            ),
        }
        trace_spec = semantic_trace_specs.get(cfg.semantic_trace_task)
        if trace_spec is None:
            raise ValueError(
                "Supported semantic_trace_task values are "
                f"{sorted(semantic_trace_specs)}, got "
                f"{cfg.semantic_trace_task!r}."
            )
        expected_task_id, recorder_type, semantic_trace_writer = trace_spec
        if cfg.task_suite_name != "libero_10" or cfg.task_id != expected_task_id:
            raise ValueError(
                f"semantic_trace_task={cfg.semantic_trace_task!r} requires "
                f"task_suite_name='libero_10' and task_id={expected_task_id}, got "
                f"task_suite_name={cfg.task_suite_name!r}, task_id={cfg.task_id}."
            )
        if cfg.semantic_trace_task != "task1" and (
            cfg.semantic_trace_output_name == "semantic_trace_task1"
        ):
            raise ValueError(
                f"semantic_trace_task={cfg.semantic_trace_task!r} requires a "
                f"task-specific output name, such as "
                f"'semantic_trace_{cfg.semantic_trace_task}'."
            )

        semantic_trace_recorder = recorder_type(env)

    successes = 0
    all_returns = []
    episode_lengths = []

    for ep_idx in range(cfg.num_episodes):
        if hasattr(policy, "reset"):
            policy.reset()
        env.reset()
        obs = env.set_init_state(initial_states[ep_idx % len(initial_states)])

        for _ in range(cfg.num_steps_wait):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

        frames = []
        episode_trace_records: list[dict[str, Any]] = []
        rewards = []
        done = False
        action_plan = []

        for _ in range(max_steps):
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            state = np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            ).astype(np.float32)

            if not action_plan:
                if hasattr(policy, "predict_action_batch"):
                    import torch

                    env_obs = {
                        "main_images": torch.from_numpy(np.stack([img])),
                        "wrist_images": torch.from_numpy(np.stack([wrist_img])),
                        "states": torch.from_numpy(np.stack([state])),
                        "task_descriptions": [str(task_description)],
                    }
                    action_chunk_out, _ = policy.predict_action_batch(env_obs, mode="eval")
                    if hasattr(action_chunk_out, "detach"):
                        action_chunk_out = action_chunk_out.detach().cpu().numpy()
                    action_chunk = np.asarray(action_chunk_out, dtype=np.float32)
                    if action_chunk.ndim == 3:
                        action_chunk = action_chunk[0]
                else:
                    observation = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": state,
                        "prompt": str(task_description),
                    }
                    action_chunk = policy.infer(observation)["actions"]
                action_plan = list(action_chunk[: cfg.action_chunk])

            action = np.asarray(action_plan.pop(0), dtype=np.float32)
            if cfg.noise_scale > 0:
                noise = np.random.randn(*action.shape) * cfg.noise_scale
                noise = np.clip(noise, -cfg.noise_clip, cfg.noise_clip)
                action = action + noise

            obs, reward, done, _ = env.step(action.tolist())
            if semantic_trace_recorder is not None:
                episode_trace_records.append(
                    semantic_trace_recorder.capture(
                        env,
                        ep_idx,
                        len(frames),
                        observation=obs,
                    )
                )
            reward_value = float(reward)
            if cfg.failure_reward is not None and done is False:
                reward_value = reward_value
            rewards.append(reward_value)
            frames.append(
                {
                    "image": img,
                    "wrist_image": wrist_img,
                    "state": state,
                    "actions": action.astype(np.float32),
                    "reward": np.array([reward_value], dtype=np.float32),
                    "task": str(task_description),
                    "done": np.array([False], dtype=bool),
                    "is_success": np.array([False], dtype=bool),
                }
            )
            if done:
                successes += 1
                break

        ep_len = len(frames)
        if ep_len == 0:
            logger.warning("episode %d empty, skipping", ep_idx)
            continue
        is_success = bool(done)
        returns = _compute_returns(rewards, gamma=1.0)
        for frame_idx, frame in enumerate(frames):
            frame["is_success"] = np.array([is_success], dtype=bool)
            frame["return"] = np.array([returns[frame_idx]], dtype=np.float32)
        for trace_record in episode_trace_records:
            trace_record["is_success"] = is_success
        semantic_trace_records.extend(episode_trace_records)
        frames[-1]["done"] = np.array([True], dtype=bool)

        if writer.dataset is None:
            first = frames[0]
            writer.create(
                repo_id=str(output_path),
                robot_type="franka_panda",
                fps=cfg.fps,
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
        all_returns.append(float(returns[0]) if len(returns) > 0 else 0.0)
        episode_lengths.append(ep_len)
        logger.info(
            "episode %d: len=%d success=%s return=%.2f total_success=%d/%d",
            ep_idx,
            ep_len,
            is_success,
            float(returns[0]) if len(returns) > 0 else 0.0,
            successes,
            ep_idx + 1,
        )

    writer.finalize()
    env.close()
    semantic_trace_artifacts = None
    if semantic_trace_recorder is not None and semantic_trace_writer is not None:
        semantic_trace_artifacts = semantic_trace_writer(
            output_path,
            semantic_trace_records,
            semantic_trace_recorder.metadata,
            output_name=cfg.semantic_trace_output_name,
            stable_frames=semantic_trace_recorder.config.stable_frames,
        )
    summary = {
        "output_dir": str(output_path),
        "model_path": cfg.model_path,
        "checkpoint_path": cfg.checkpoint_path,
        "model_type": cfg.model_type,
        "task_suite_name": cfg.task_suite_name,
        "task_id": cfg.task_id,
        "num_episodes": cfg.num_episodes,
        "successes": successes,
        "success_rate": float(successes / cfg.num_episodes) if cfg.num_episodes else 0.0,
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "mean_initial_return": float(np.mean(all_returns)) if all_returns else 0.0,
        "semantic_trace": semantic_trace_artifacts,
    }
    save_json(summary, output_path / "collection_summary.json")
    return summary


def resolve_latest_trained_checkpoint(summary_path: str | Path) -> str | None:
    """Read a training summary file and return the checkpoint path if available."""
    summary = load_json(summary_path)
    checkpoint_path = summary.get("checkpoint_path")
    if checkpoint_path:
        return str(checkpoint_path)
    return None
