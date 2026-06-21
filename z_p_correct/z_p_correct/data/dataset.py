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

"""Dataset for training z/p predictors on frozen VLM features."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import openpi.models.model as _openpi_model
import openpi.transforms as _openpi_transforms
import pandas as pd
import torch
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from torch.utils.data import Dataset

from examples.recap.process.episode_subset_utils import (
    load_episode_subset_file,
    resolve_episode_split_for_dataset,
    resolve_episode_subset_for_dataset,
)
from rlinf.data.datasets.recap.utils import decode_image_struct_batch
from rlinf.models.embodiment.openpi.policies import libero_policy

logger = logging.getLogger(__name__)

_MODEL_TYPE_MAP = {
    "pi0": _openpi_model.ModelType.PI0,
    "pi05": _openpi_model.ModelType.PI05,
    "pi0_fast": _openpi_model.ModelType.PI0_FAST,
}

_REPACK_KEYS = {
    "libero": {
        "observation/image": "image",
        "observation/wrist_image": "wrist_image",
        "observation/state": "state",
        "actions": "actions",
        "prompt": "prompt",
    },
}


def _load_task_descriptions(dataset_path: Path) -> dict[int, str]:
    """Load task descriptions from ``meta/tasks.jsonl`` if present."""
    tasks: dict[int, str] = {}
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        with open(tasks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                task_idx = entry.get("task_index", len(tasks))
                tasks[task_idx] = entry.get("task", "")
    return tasks


def _load_phase_labels(
    dataset_path: Path, label_name: str = "phase_progress_semantic"
) -> dict[tuple[int, int], dict[str, Any]]:
    """Load semantic phase labels from ``meta/{label_name}.parquet``."""
    label_path = dataset_path / "meta" / f"{label_name}.parquet"
    if not label_path.exists():
        raise FileNotFoundError(f"Phase label file not found: {label_path}")

    df = pd.read_parquet(label_path)
    required = {"episode_index", "frame_index", "phase", "phase_progress", "global_progress"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Phase label file missing columns: {missing}")

    records = df[
        ["episode_index", "frame_index", "phase", "phase_progress", "global_progress"]
    ].to_dict(orient="records")
    return {
        (int(r["episode_index"]), int(r["frame_index"])): {
            "phase": int(r["phase"]),
            "phase_progress": float(r["phase_progress"]),
            "global_progress": float(r["global_progress"]),
        }
        for r in records
    }


class PhaseProbeDataset(Dataset):
    """Flat dataset for probing semantic phase / progress from VLM features.

    Loads a LeRobot dataset directly (no delta_timestamps) for fast single-row
    access, applies the same OpenPI transforms used by the value model, and
    attaches pre-computed semantic phase labels. Train/val split is performed at
    the episode level to avoid leakage.
    """

    def __init__(
        self,
        dataset_path: str,
        robot_type: str = "libero",
        model_type: str = "pi05",
        action_dim: int = 32,
        default_prompt: Optional[str] = None,
        split: str = "train",
        val_episode_ratio: float = 0.2,
        label_name: str = "phase_progress_semantic",
        seed: int = 42,
        max_episodes: Optional[int] = None,
        episode_subset_path: Optional[str] = None,
        episode_split_path: Optional[str] = None,
    ):
        super().__init__()
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split}")

        self.dataset_path = Path(dataset_path).absolute()
        self.split = split
        self.default_prompt = default_prompt
        self.action_dim = action_dim

        self.dataset_meta = LeRobotDatasetMetadata(
            self.dataset_path.name, root=self.dataset_path
        )
        total_episodes = self.dataset_meta.total_episodes
        rng = np.random.default_rng(seed)
        if episode_split_path is not None:
            raw_spec = load_episode_subset_file(episode_split_path)
            split_spec = resolve_episode_split_for_dataset(raw_spec, self.dataset_path)
            if split_spec is None:
                raise ValueError(
                    f"No episode split entry found for dataset '{self.dataset_path.name}' "
                    f"in {episode_split_path}"
                )
            selected_eps = (
                split_spec.train_episodes if split == "train" else split_spec.val_episodes
            )
        elif episode_subset_path is not None:
            raw_spec = load_episode_subset_file(episode_subset_path)
            subset_spec = resolve_episode_subset_for_dataset(raw_spec, self.dataset_path)
            if subset_spec is None:
                raise ValueError(
                    f"No episode subset entry found for dataset '{self.dataset_path.name}' "
                    f"in {episode_subset_path}"
                )
            selected_pool = subset_spec.episodes
            if not selected_pool:
                raise ValueError(
                    f"Episode subset for dataset '{self.dataset_path.name}' is empty"
                )
            n_pool = len(selected_pool)
            n_val = max(1, int(n_pool * val_episode_ratio))
            if n_val >= n_pool:
                n_val = n_pool - 1
            val_eps = set(selected_pool[:n_val])
            train_eps = set(selected_pool[n_val:])
            selected_eps = train_eps if split == "train" else val_eps
        else:
            shuffled_eps = rng.permutation(total_episodes).tolist()
            if max_episodes is not None:
                if max_episodes <= 1:
                    raise ValueError(f"max_episodes must be > 1, got {max_episodes}")
                shuffled_eps = shuffled_eps[: min(max_episodes, total_episodes)]
            selected_pool = shuffled_eps
            n_pool = len(selected_pool)
            n_val = max(1, int(n_pool * val_episode_ratio))
            if n_val >= n_pool:
                n_val = n_pool - 1
            val_eps = set(selected_pool[:n_val])
            train_eps = set(selected_pool[n_val:])
            selected_eps = train_eps if split == "train" else val_eps

        self._base = LeRobotDataset(
            self.dataset_path.name,
            root=self.dataset_path,
            episodes=list(range(total_episodes)),
            download_videos=False,
        )
        self._base.hf_dataset.set_transform(decode_image_struct_batch)

        ep_idx = self._base.episode_data_index
        self._indices = [
            i
            for ep in sorted(selected_eps)
            for i in range(int(ep_idx["from"][ep].item()), int(ep_idx["to"][ep].item()))
        ]

        self._tasks = _load_task_descriptions(self.dataset_path)
        self._phase_labels = _load_phase_labels(self.dataset_path, label_name=label_name)

        self._transform = self._build_transform(
            robot_type=robot_type,
            model_type=model_type,
            action_dim=action_dim,
            default_prompt=default_prompt,
        )

        logger.info(
            "PhaseProbeDataset[%s]: %d episodes, %d frames from %s",
            split,
            len(selected_eps),
            len(self._indices),
            self.dataset_path,
        )

    @staticmethod
    def _build_transform(
        robot_type: str,
        model_type: str,
        action_dim: int,
        default_prompt: Optional[str],
    ):
        model_type_lower = model_type.lower()
        if model_type_lower not in _MODEL_TYPE_MAP:
            raise ValueError(f"Unsupported model_type: {model_type}")
        model_type_enum = _MODEL_TYPE_MAP[model_type_lower]
        robot = robot_type.lower()

        repack_keys = _REPACK_KEYS.get(robot)
        if repack_keys is None:
            raise ValueError(f"Unsupported robot_type: {robot_type}")

        transforms_list = [
            _openpi_transforms.RepackTransform(repack_keys),
            libero_policy.LiberoInputs(model_type=model_type_enum),
            _openpi_transforms.InjectDefaultPrompt(default_prompt),
            _openpi_transforms.PadStatesAndActions(action_dim),
        ]
        return _openpi_transforms.compose(transforms_list)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        real_idx = self._indices[idx]
        sample = self._base[real_idx]

        ep = int(sample.get("episode_index", -1))
        fr = int(sample.get("frame_index", -1))
        if ep < 0 or fr < 0:
            raise KeyError(
                f"LeRobot sample missing episode_index ({ep}) or frame_index ({fr}) "
                f"at real_idx={real_idx}. Available keys: {sorted(sample.keys())}"
            )

        if self._tasks and "task_index" in sample:
            ti = sample["task_index"]
            ti = int(ti.item()) if isinstance(ti, torch.Tensor) else int(ti)
            if ti in self._tasks:
                sample = {**sample, "prompt": self._tasks[ti]}

        sample = self._transform(sample)

        key = (ep, fr)
        if key not in self._phase_labels:
            raise KeyError(f"Missing phase label for episode={ep}, frame={fr}")
        labels = self._phase_labels[key]

        return {
            "episode_index": ep,
            "frame_index": fr,
            "phase": labels["phase"],
            "phase_progress": labels["phase_progress"],
            "global_progress": labels["global_progress"],
            "transformed": sample,
        }


def build_datasets(
    dataset_path: str,
    robot_type: str = "libero",
    model_type: str = "pi05",
    action_dim: int = 32,
    default_prompt: Optional[str] = None,
    val_episode_ratio: float = 0.2,
    label_name: str = "phase_progress_semantic",
    seed: int = 42,
    max_episodes: Optional[int] = None,
    episode_subset_path: Optional[str] = None,
    episode_split_path: Optional[str] = None,
) -> tuple[PhaseProbeDataset, PhaseProbeDataset]:
    """Build episode-level train/val datasets."""
    common_kwargs = {
        "dataset_path": dataset_path,
        "robot_type": robot_type,
        "model_type": model_type,
        "action_dim": action_dim,
        "default_prompt": default_prompt,
        "val_episode_ratio": val_episode_ratio,
        "label_name": label_name,
        "seed": seed,
        "max_episodes": max_episodes,
        "episode_subset_path": episode_subset_path,
        "episode_split_path": episode_split_path,
    }
    train_ds = PhaseProbeDataset(split="train", **common_kwargs)
    val_ds = PhaseProbeDataset(split="val", **common_kwargs)
    return train_ds, val_ds
