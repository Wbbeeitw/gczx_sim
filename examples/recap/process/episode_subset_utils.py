"""Utilities for sampling and resolving episode subsets for recap experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EpisodeSubsetSpec:
    """Episode subset selection for one dataset."""

    episodes: list[int]
    num_frames: int | None = None


def load_episode_subset_file(path: str | Path) -> Any:
    """Load a JSON episode subset specification."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _coerce_episode_spec(value: Any) -> EpisodeSubsetSpec:
    """Normalize one subset payload into an EpisodeSubsetSpec."""
    if isinstance(value, list):
        episodes = [int(ep) for ep in value]
        return EpisodeSubsetSpec(episodes=sorted(set(episodes)))

    if not isinstance(value, dict):
        raise ValueError(f"Unsupported episode subset payload: {type(value)!r}")

    raw_episodes = value.get("selected_episodes", value.get("episodes"))
    if raw_episodes is None:
        raise ValueError("Episode subset payload must include 'selected_episodes' or 'episodes'")

    episodes = [int(ep) for ep in raw_episodes]
    num_frames = value.get("num_frames")
    return EpisodeSubsetSpec(
        episodes=sorted(set(episodes)),
        num_frames=int(num_frames) if num_frames is not None else None,
    )


def resolve_episode_subset_for_dataset(
    raw_spec: Any,
    dataset_path: str | Path,
) -> EpisodeSubsetSpec | None:
    """Resolve a subset spec for one dataset path/name.

    Supported JSON formats:
    1. `{"selected_episodes": [...], "num_frames": ...}`
    2. `{"episodes": [...]}`
    3. `{"/abs/path/to/dataset": {"selected_episodes": [...]}}`
    4. `{"dataset_name": {"selected_episodes": [...]}}`
    5. `[1, 2, 3]`
    """
    dataset_path = Path(dataset_path)
    dataset_name = dataset_path.name

    if isinstance(raw_spec, list):
        return _coerce_episode_spec(raw_spec)

    if not isinstance(raw_spec, dict):
        raise ValueError(f"Unsupported episode subset spec root: {type(raw_spec)!r}")

    if "selected_episodes" in raw_spec or "episodes" in raw_spec:
        return _coerce_episode_spec(raw_spec)

    for key in (str(dataset_path), dataset_name):
        if key in raw_spec:
            return _coerce_episode_spec(raw_spec[key])

    return None


def compute_frame_indices_for_episodes(
    episode_end_offsets: list[int] | np.ndarray,
    episode_ids: list[int],
) -> list[int]:
    """Expand episode ids into sorted global frame indices.

    Args:
        episode_end_offsets: Exclusive end offset for every episode.
        episode_ids: Episode ids to include.
    """
    episode_to = [int(x) for x in episode_end_offsets]
    num_episodes = len(episode_to)
    indices: list[int] = []
    for ep_idx in sorted(set(int(ep) for ep in episode_ids)):
        if ep_idx < 0 or ep_idx >= num_episodes:
            raise IndexError(
                f"Episode index {ep_idx} out of range for dataset with {num_episodes} episodes"
            )
        start = 0 if ep_idx == 0 else episode_to[ep_idx - 1]
        end = episode_to[ep_idx]
        indices.extend(range(start, end))
    return indices


def count_frames_for_episodes(
    episode_end_offsets: list[int] | np.ndarray,
    episode_ids: list[int],
) -> int:
    """Count total frames for the selected episodes."""
    return len(compute_frame_indices_for_episodes(episode_end_offsets, episode_ids))


def sample_balanced_episode_ids(
    episode_success: dict[int, bool],
    num_episodes: int,
    seed: int,
    success_ratio: float = 0.5,
) -> list[int]:
    """Sample a near-balanced set of success/failure episode ids.

    The sampler tries to honor the requested success ratio, then fills any shortfall
    from the other pool so the final selection size is exactly `num_episodes`.
    """
    if num_episodes <= 0:
        raise ValueError(f"num_episodes must be positive, got {num_episodes}")
    if not 0.0 <= success_ratio <= 1.0:
        raise ValueError(f"success_ratio must be in [0, 1], got {success_ratio}")

    success_eps = sorted(ep for ep, ok in episode_success.items() if ok)
    failure_eps = sorted(ep for ep, ok in episode_success.items() if not ok)
    total_available = len(success_eps) + len(failure_eps)
    if num_episodes > total_available:
        raise ValueError(
            f"Requested {num_episodes} episodes but only {total_available} are available"
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
        leftovers = [
            ep
            for ep in sorted(episode_success)
            if ep not in selected
        ]
        fill = rng.choice(leftovers, size=remaining, replace=False).tolist()
        selected.update(fill)

    return sorted(int(ep) for ep in selected)
