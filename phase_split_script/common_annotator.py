"""Common utilities for phase/progress annotation of LIBERO rollout datasets.

Each task should define an annotator callable that takes actions and episode_length
and returns (phase_array, num_phases).  This module handles progress computation
and parquet I/O.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm


def smooth_1d(x: np.ndarray, window: int = 11) -> np.ndarray:
    """Centered moving average."""
    if window <= 1:
        return x
    pad = window // 2
    padded = np.concatenate([x[:pad][::-1], x, x[-pad:][::-1]])
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(padded, kernel, mode="valid")


def detect_grasp_segments(
    gripper_action: np.ndarray,
    threshold: float = 0.0,
    smooth_window: int = 11,
    min_len: int = 15,
) -> list[tuple[int, int]]:
    """Detect sustained gripper-close segments, sorted by time."""
    g = smooth_1d(gripper_action.astype(np.float32), window=smooth_window)
    close = g > threshold
    return detect_segments(close, min_len=min_len)


def detect_segments(mask: np.ndarray, min_len: int = 10) -> list[tuple[int, int]]:
    """Detect sustained True segments in a boolean array, sorted by time."""
    n = len(mask)
    segments: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                segments.append((int(i), int(j)))
            i = j
        else:
            i += 1
    return sorted(segments, key=lambda s: s[0])


def compute_progress(
    phase: np.ndarray, num_phases: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute phase_progress, global_progress, and overall_progress."""
    phase_progress = np.zeros(len(phase), dtype=np.float32)
    for ph in np.unique(phase):
        mask = phase == ph
        idx = np.where(mask)[0]
        if len(idx) > 1:
            phase_progress[mask] = (idx - idx[0]) / (idx[-1] - idx[0])
        elif len(idx) == 1:
            phase_progress[mask] = 0.0

    global_progress = (phase.astype(np.float32) + phase_progress) / num_phases
    overall_progress = np.arange(len(phase), dtype=np.float32) / max(len(phase) - 1, 1)
    return phase_progress, global_progress, overall_progress


def annotate_episode(
    ep_file: Path,
    annotator: Callable[[np.ndarray, np.ndarray, int], tuple[np.ndarray, int]],
) -> pd.DataFrame | None:
    """Apply a task-specific annotator to one episode parquet file."""
    df = pd.read_parquet(ep_file)
    if len(df) == 0:
        return None

    episode_index = int(df["episode_index"].iloc[0])
    actions = np.stack(df["actions"].values)
    state = np.stack(df["state"].values) if "state" in df.columns else np.empty((len(df), 0))

    phase, num_phases = annotator(actions, state, len(df))
    phase_progress, global_progress, overall_progress = compute_progress(phase, num_phases)

    return pd.DataFrame({
        "episode_index": np.full(len(df), episode_index, dtype=np.int64),
        "frame_index": df["frame_index"].values.astype(np.int64),
        "phase": phase.astype(np.int64),
        "phase_progress": phase_progress,
        "global_progress": global_progress,
        "progress": overall_progress,
    })


def annotate_dataset(
    dataset_path: str | Path,
    annotator: Callable[[np.ndarray, np.ndarray, int], tuple[np.ndarray, int]],
    output_name: str = "phase_progress_semantic",
) -> Path:
    """Run annotation over all episodes in a LeRobot dataset."""
    dataset_path = Path(dataset_path)
    data_root = dataset_path / "data"
    ep_files = sorted(data_root.rglob("episode_*.parquet"))
    print(f"Found {len(ep_files)} episodes in {data_root}")

    records: list[pd.DataFrame] = []
    for ep_file in tqdm(ep_files, desc="Annotating"):
        ep_df = annotate_episode(ep_file, annotator)
        if ep_df is not None:
            records.append(ep_df)

    out_df = pd.concat(records, ignore_index=True)
    out_df = out_df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    meta_dir = dataset_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_path = meta_dir / f"{output_name}.parquet"
    out_df.to_parquet(out_path, index=False)

    print(f"Saved {len(out_df)} frame labels to {out_path}")
    print("Phase distribution:")
    print(out_df["phase"].value_counts().sort_index())
    return out_path
