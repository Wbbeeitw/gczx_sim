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

"""LeRobot dataset wrapper with semantic phase/progress labels."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_MODEL_TYPE_MAP = {
    "pi0": "PI0",
    "pi05": "PI05",
    "pi0_fast": "PI0_FAST",
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


def load_task_descriptions(dataset_path: str | Path) -> dict[int, str]:
    """Load task descriptions from ``meta/tasks.jsonl`` if present."""
    dataset_path = Path(dataset_path)
    tasks: dict[int, str] = {}
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        return tasks
    with open(tasks_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            task_idx = int(entry.get("task_index", len(tasks)))
            tasks[task_idx] = str(entry.get("task", ""))
    return tasks


def load_phase_labels(
    dataset_path: str | Path,
    label_name: str = "phase_progress_semantic",
) -> dict[tuple[int, int], dict[str, Any]]:
    """Load semantic phase/progress labels keyed by episode and frame."""
    dataset_path = Path(dataset_path)
    label_path = dataset_path / "meta" / f"{label_name}.parquet"
    if not label_path.exists():
        raise FileNotFoundError(
            f"Phase label file not found: {label_path}. "
            "Run gen_semantic_phase.py or provide the correct label_name."
        )
    df = pd.read_parquet(label_path)
    required = {
        "episode_index",
        "frame_index",
        "phase",
        "phase_progress",
        "global_progress",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Phase label file missing columns: {sorted(missing)}")
    records = df[list(required)].to_dict(orient="records")
    return {
        (int(row["episode_index"]), int(row["frame_index"])): {
            "phase": int(row["phase"]),
            "phase_progress": float(row["phase_progress"]),
            "global_progress": float(row["global_progress"]),
        }
        for row in records
    }


class RevaluePhaseDataset(Dataset):
    """Flat frame dataset with OpenPI transforms and phase/progress labels."""

    def __init__(
        self,
        dataset_path: str | Path,
        *,
        robot_type: str = "libero",
        model_type: str = "pi05",
        action_dim: int = 32,
        default_prompt: str | None = None,
        split: str = "train",
        val_episode_ratio: float = 0.2,
        label_name: str = "phase_progress_semantic",
        seed: int = 42,
        max_episodes: int | None = None,
        episode_subset_path: str | None = None,
        episode_split_path: str | None = None,
    ) -> None:
        super().__init__()
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        from lerobot.common.datasets.lerobot_dataset import (
            LeRobotDataset,
            LeRobotDatasetMetadata,
        )

        from examples.recap.process.episode_subset_utils import (
            load_episode_subset_file,
            resolve_episode_split_for_dataset,
            resolve_episode_subset_for_dataset,
        )
        from rlinf.data.datasets.recap.utils import decode_image_struct_batch

        self.dataset_path = Path(dataset_path).absolute()
        self.split = split
        self.dataset_meta = LeRobotDatasetMetadata(
            self.dataset_path.name,
            root=self.dataset_path,
        )
        total_episodes = int(self.dataset_meta.total_episodes)

        if episode_split_path:
            raw_spec = load_episode_subset_file(episode_split_path)
            split_spec = resolve_episode_split_for_dataset(raw_spec, self.dataset_path)
            if split_spec is None:
                raise ValueError(
                    f"No episode split entry found for {self.dataset_path.name} "
                    f"in {episode_split_path}"
                )
            selected_eps = (
                split_spec.train_episodes
                if split == "train"
                else split_spec.val_episodes
            )
        elif episode_subset_path:
            raw_spec = load_episode_subset_file(episode_subset_path)
            subset_spec = resolve_episode_subset_for_dataset(raw_spec, self.dataset_path)
            if subset_spec is None:
                raise ValueError(
                    f"No episode subset entry found for {self.dataset_path.name} "
                    f"in {episode_subset_path}"
                )
            selected_eps = self._split_episode_pool(
                subset_spec.episodes,
                split=split,
                val_episode_ratio=val_episode_ratio,
            )
        else:
            rng = np.random.default_rng(seed)
            episode_pool = rng.permutation(total_episodes).tolist()
            if max_episodes is not None:
                if max_episodes <= 1:
                    raise ValueError(f"max_episodes must be > 1, got {max_episodes}")
                episode_pool = episode_pool[: min(max_episodes, total_episodes)]
            selected_eps = self._split_episode_pool(
                episode_pool,
                split=split,
                val_episode_ratio=val_episode_ratio,
            )

        self._base = LeRobotDataset(
            self.dataset_path.name,
            root=self.dataset_path,
            episodes=list(range(total_episodes)),
            download_videos=False,
        )
        self._base.hf_dataset.set_transform(decode_image_struct_batch)

        ep_idx = self._base.episode_data_index
        self._indices = [
            frame
            for ep in sorted(set(int(ep) for ep in selected_eps))
            for frame in range(
                int(ep_idx["from"][ep].item()),
                int(ep_idx["to"][ep].item()),
            )
        ]
        self._tasks = load_task_descriptions(self.dataset_path)
        self._phase_labels = load_phase_labels(self.dataset_path, label_name=label_name)
        self._transform = self._build_transform(
            robot_type=robot_type,
            model_type=model_type,
            action_dim=action_dim,
            default_prompt=default_prompt,
        )
        logger.info(
            "RevaluePhaseDataset[%s]: episodes=%d frames=%d path=%s",
            split,
            len(set(selected_eps)),
            len(self._indices),
            self.dataset_path,
        )

    @staticmethod
    def _split_episode_pool(
        episode_pool: list[int],
        *,
        split: str,
        val_episode_ratio: float,
    ) -> list[int]:
        if len(episode_pool) <= 1:
            raise ValueError("Need at least two episodes for train/val split.")
        n_val = max(1, int(len(episode_pool) * val_episode_ratio))
        if n_val >= len(episode_pool):
            n_val = len(episode_pool) - 1
        val_eps = episode_pool[:n_val]
        train_eps = episode_pool[n_val:]
        return train_eps if split == "train" else val_eps

    @staticmethod
    def _build_transform(
        *,
        robot_type: str,
        model_type: str,
        action_dim: int,
        default_prompt: str | None,
    ):
        import openpi.models.model as openpi_model
        import openpi.transforms as openpi_transforms

        from rlinf.models.embodiment.openpi.policies import libero_policy

        robot = robot_type.lower()
        repack_keys = _REPACK_KEYS.get(robot)
        if repack_keys is None:
            raise ValueError(
                f"Unsupported robot_type {robot_type!r}. "
                f"Supported: {sorted(_REPACK_KEYS)}"
            )

        model_type_key = model_type.lower()
        model_type_attr = _MODEL_TYPE_MAP.get(model_type_key)
        if model_type_attr is None:
            raise ValueError(
                f"Unsupported model_type {model_type!r}. "
                f"Supported: {sorted(_MODEL_TYPE_MAP)}"
            )
        model_type_enum = getattr(openpi_model.ModelType, model_type_attr)
        return openpi_transforms.compose(
            [
                openpi_transforms.RepackTransform(repack_keys),
                libero_policy.LiberoInputs(model_type=model_type_enum),
                openpi_transforms.InjectDefaultPrompt(default_prompt),
                openpi_transforms.PadStatesAndActions(action_dim),
            ]
        )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        real_index = self._indices[index]
        sample = self._base[real_index]
        episode_index = int(sample.get("episode_index", -1))
        frame_index = int(sample.get("frame_index", -1))
        if episode_index < 0 or frame_index < 0:
            raise KeyError(
                f"Sample missing episode_index/frame_index at real_index={real_index}"
            )

        if self._tasks and "task_index" in sample:
            task_index = sample["task_index"]
            task_index = (
                int(task_index.item())
                if isinstance(task_index, torch.Tensor)
                else int(task_index)
            )
            if task_index in self._tasks:
                sample = {**sample, "prompt": self._tasks[task_index]}

        key = (episode_index, frame_index)
        if key not in self._phase_labels:
            raise KeyError(
                f"Missing phase label for episode={episode_index}, frame={frame_index}"
            )
        labels = self._phase_labels[key]
        return {
            "episode_index": episode_index,
            "frame_index": frame_index,
            "phase": labels["phase"],
            "phase_progress": labels["phase_progress"],
            "global_progress": labels["global_progress"],
            "transformed": self._transform(sample),
        }


def build_phase_datasets(
    dataset_path: str | Path,
    **kwargs: Any,
) -> tuple[RevaluePhaseDataset, RevaluePhaseDataset]:
    """Build train and validation phase datasets with identical options."""
    train_dataset = RevaluePhaseDataset(dataset_path, split="train", **kwargs)
    val_dataset = RevaluePhaseDataset(dataset_path, split="val", **kwargs)
    return train_dataset, val_dataset
