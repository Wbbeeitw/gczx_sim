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

"""Episode subset and split manifests for the Revalue workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rlinf.revalue.io import load_json, save_json


@dataclass(frozen=True)
class EpisodeSubsetSpec:
    """Episode subset selection for one dataset."""

    episodes: list[int]
    num_frames: int | None = None


@dataclass(frozen=True)
class EpisodeSplitSpec:
    """Explicit train/val/test split for one dataset."""

    selected_episodes: list[int]
    train_episodes: list[int]
    val_episodes: list[int]
    test_episodes: list[int]
    num_frames: int | None = None


@dataclass(frozen=True)
class EpisodeManifestConfig:
    """Options for building a Revalue episode manifest."""

    dataset_path: str
    output_path: str
    label_name: str = "phase_progress_semantic"
    num_episodes: int | None = None
    success_ratio: float = 0.5
    val_episode_ratio: float = 0.2
    test_episode_ratio: float = 0.0
    seed: int = 42
    success_phase: int = 4
    overwrite: bool = False


def load_episode_manifest(path: str | Path) -> Any:
    """Load a JSON episode manifest."""
    return load_json(path)


def _coerce_episode_spec(value: Any) -> EpisodeSubsetSpec:
    if isinstance(value, list):
        episodes = [int(ep) for ep in value]
        return EpisodeSubsetSpec(episodes=sorted(set(episodes)))

    if not isinstance(value, dict):
        raise ValueError(f"Unsupported episode subset payload: {type(value)!r}")

    raw_episodes = value.get("selected_episodes", value.get("episodes"))
    if raw_episodes is None:
        raise ValueError(
            "Episode subset payload must include 'selected_episodes' or 'episodes'"
        )

    episodes = [int(ep) for ep in raw_episodes]
    num_frames = value.get("num_frames")
    return EpisodeSubsetSpec(
        episodes=sorted(set(episodes)),
        num_frames=int(num_frames) if num_frames is not None else None,
    )


def _coerce_split_spec(value: Any) -> EpisodeSplitSpec:
    if not isinstance(value, dict):
        raise ValueError(f"Unsupported episode split payload: {type(value)!r}")

    raw_selected = value.get("selected_episodes", value.get("episodes"))
    raw_train = value.get("train_episodes")
    raw_val = value.get("val_episodes")
    raw_test = value.get("test_episodes", [])
    if raw_selected is None or raw_train is None or raw_val is None:
        raise ValueError(
            "Episode split payload must include 'selected_episodes', "
            "'train_episodes', and 'val_episodes'"
        )

    selected = sorted(set(int(ep) for ep in raw_selected))
    train = sorted(set(int(ep) for ep in raw_train))
    val = sorted(set(int(ep) for ep in raw_val))
    test = sorted(set(int(ep) for ep in raw_test))
    if (set(train) | set(val) | set(test)) != set(selected):
        raise ValueError(
            "Train/val/test episode union must exactly match selected_episodes"
        )
    if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
        raise ValueError("Train/val/test episode lists must be disjoint")

    num_frames = value.get("num_frames")
    return EpisodeSplitSpec(
        selected_episodes=selected,
        train_episodes=train,
        val_episodes=val,
        test_episodes=test,
        num_frames=int(num_frames) if num_frames is not None else None,
    )


def resolve_episode_subset_for_dataset(
    raw_spec: Any,
    dataset_path: str | Path,
) -> EpisodeSubsetSpec | None:
    """Resolve a subset spec for one dataset path/name."""
    dataset_path = Path(dataset_path)
    dataset_name = dataset_path.name

    if isinstance(raw_spec, list):
        return _coerce_episode_spec(raw_spec)

    if not isinstance(raw_spec, dict):
        raise ValueError(f"Unsupported episode subset spec root: {type(raw_spec)!r}")

    if "selected_episodes" in raw_spec or "episodes" in raw_spec:
        return _coerce_episode_spec(raw_spec)

    for key in (str(dataset_path), str(dataset_path.absolute()), dataset_name):
        if key in raw_spec:
            return _coerce_episode_spec(raw_spec[key])

    return None


def resolve_episode_split_for_dataset(
    raw_spec: Any,
    dataset_path: str | Path,
) -> EpisodeSplitSpec | None:
    """Resolve an explicit train/val/test split spec for one dataset path/name."""
    dataset_path = Path(dataset_path)
    dataset_name = dataset_path.name

    if not isinstance(raw_spec, dict):
        raise ValueError(f"Unsupported episode split spec root: {type(raw_spec)!r}")

    candidate: Any | None = None
    if all(
        key in raw_spec
        for key in ("selected_episodes", "train_episodes", "val_episodes")
    ):
        candidate = raw_spec
    else:
        for key in (str(dataset_path), str(dataset_path.absolute()), dataset_name):
            if key in raw_spec:
                candidate = raw_spec[key]
                break

    if candidate is None:
        return None
    return _coerce_split_spec(candidate)


def compute_frame_indices_for_episodes(
    episode_end_offsets: list[int] | np.ndarray,
    episode_ids: list[int],
) -> list[int]:
    """Expand episode ids into sorted global frame indices."""
    episode_to = [int(x) for x in episode_end_offsets]
    num_episodes = len(episode_to)
    indices: list[int] = []
    for ep_idx in sorted(set(int(ep) for ep in episode_ids)):
        if ep_idx < 0 or ep_idx >= num_episodes:
            raise IndexError(
                f"Episode index {ep_idx} out of range for dataset with "
                f"{num_episodes} episodes"
            )
        start = 0 if ep_idx == 0 else episode_to[ep_idx - 1]
        end = episode_to[ep_idx]
        indices.extend(range(start, end))
    return indices


def count_frames_for_episodes(
    episode_end_offsets: list[int] | np.ndarray,
    episode_ids: list[int],
) -> int:
    """Count total frames for selected episodes."""
    return len(compute_frame_indices_for_episodes(episode_end_offsets, episode_ids))


def sample_balanced_episode_ids(
    episode_success: dict[int, bool],
    num_episodes: int,
    seed: int,
    success_ratio: float = 0.5,
) -> list[int]:
    """Sample a near-balanced set of success/failure episode ids."""
    if num_episodes <= 0:
        raise ValueError(f"num_episodes must be positive, got {num_episodes}")
    if not 0.0 <= success_ratio <= 1.0:
        raise ValueError(f"success_ratio must be in [0, 1], got {success_ratio}")

    success_eps = sorted(ep for ep, ok in episode_success.items() if ok)
    failure_eps = sorted(ep for ep, ok in episode_success.items() if not ok)
    total_available = len(success_eps) + len(failure_eps)
    if num_episodes > total_available:
        raise ValueError(
            f"Requested {num_episodes} episodes but only "
            f"{total_available} are available"
        )

    rng = np.random.default_rng(seed)
    target_success = int(round(num_episodes * success_ratio))
    n_success = min(target_success, len(success_eps))
    n_failure = min(num_episodes - n_success, len(failure_eps))

    selected_success = (
        rng.choice(success_eps, size=n_success, replace=False).tolist()
        if n_success > 0
        else []
    )
    selected_failure = (
        rng.choice(failure_eps, size=n_failure, replace=False).tolist()
        if n_failure > 0
        else []
    )

    selected = set(selected_success + selected_failure)
    remaining = num_episodes - len(selected)
    if remaining > 0:
        leftovers = [ep for ep in sorted(episode_success) if ep not in selected]
        fill = rng.choice(leftovers, size=remaining, replace=False).tolist()
        selected.update(fill)

    return sorted(int(ep) for ep in selected)


def split_episode_ids(
    episode_ids: list[int],
    *,
    val_episode_ratio: float,
    test_episode_ratio: float = 0.0,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Split a fixed episode list into train/val/test ids."""
    if not episode_ids:
        raise ValueError("episode_ids must not be empty")
    if val_episode_ratio < 0.0 or test_episode_ratio < 0.0:
        raise ValueError("val/test ratios must be non-negative")
    if val_episode_ratio + test_episode_ratio >= 1.0:
        raise ValueError("val_episode_ratio + test_episode_ratio must be < 1")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(sorted(set(int(ep) for ep in episode_ids))).tolist()
    n_pool = len(shuffled)
    if n_pool <= 1:
        raise ValueError("Need at least two episodes for train/val split.")

    n_test = int(round(n_pool * test_episode_ratio))
    n_val = max(1, int(round(n_pool * val_episode_ratio)))
    if n_test + n_val >= n_pool:
        overflow = n_test + n_val - (n_pool - 1)
        n_test = max(0, n_test - overflow)
    if n_test + n_val >= n_pool:
        n_val = n_pool - n_test - 1

    test_eps = sorted(int(ep) for ep in shuffled[:n_test])
    val_eps = sorted(int(ep) for ep in shuffled[n_test : n_test + n_val])
    train_eps = sorted(int(ep) for ep in shuffled[n_test + n_val :])
    return train_eps, val_eps, test_eps


def build_episode_manifest(cfg: EpisodeManifestConfig) -> Path:
    """Build one manifest containing subset and train/val/test split."""
    dataset_path = Path(cfg.dataset_path)
    output_path = Path(cfg.output_path)
    if output_path.exists() and not cfg.overwrite:
        return output_path

    label_path = dataset_path / "meta" / f"{cfg.label_name}.parquet"
    if not label_path.exists():
        raise FileNotFoundError(f"Phase label file not found: {label_path}")
    df = pd.read_parquet(
        label_path,
        columns=["episode_index", "frame_index", "phase"],
    )
    ep_stats = (
        df.groupby("episode_index")
        .agg(
            num_frames=("frame_index", "size"),
            max_phase=("phase", "max"),
        )
        .reset_index()
    )
    ep_stats["is_success"] = ep_stats["max_phase"] >= int(cfg.success_phase)

    episode_success = {
        int(row.episode_index): bool(row.is_success)
        for row in ep_stats.itertuples(index=False)
    }
    if cfg.num_episodes is None:
        selected = sorted(episode_success)
    else:
        selected = sample_balanced_episode_ids(
            episode_success=episode_success,
            num_episodes=cfg.num_episodes,
            seed=cfg.seed,
            success_ratio=cfg.success_ratio,
        )

    train_eps, val_eps, test_eps = split_episode_ids(
        selected,
        val_episode_ratio=cfg.val_episode_ratio,
        test_episode_ratio=cfg.test_episode_ratio,
        seed=cfg.seed,
    )

    selected_df = ep_stats[ep_stats["episode_index"].isin(selected)].copy()
    selected_df = selected_df.sort_values("episode_index").reset_index(drop=True)
    num_frames = int(selected_df["num_frames"].sum())
    success_count = int(selected_df["is_success"].sum())
    failure_count = int(len(selected_df) - success_count)

    payload = {
        str(dataset_path): {
            "selected_episodes": [int(ep) for ep in selected],
            "train_episodes": [int(ep) for ep in train_eps],
            "val_episodes": [int(ep) for ep in val_eps],
            "test_episodes": [int(ep) for ep in test_eps],
            "num_frames": num_frames,
            "num_success": success_count,
            "num_failure": failure_count,
            "seed": int(cfg.seed),
            "success_ratio": float(cfg.success_ratio),
            "val_episode_ratio": float(cfg.val_episode_ratio),
            "test_episode_ratio": float(cfg.test_episode_ratio),
            "label_name": cfg.label_name,
        }
    }
    save_json(payload, output_path)
    return output_path
