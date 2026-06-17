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

"""
Generate semantic phase/progress labels for LIBERO episodes based on gripper
action.  Designed for "pick two objects and put them in a basket" tasks.

Phase semantics:
    0 = approaching / pre-first-grasp
    1 = grasping/holding first object
    2 = between first release and second grasp
    3 = grasping/holding second object
    4 = after second release / task completion

Usage:
    python gen_semantic_phase.py --dataset_path /path/to/libero_dataset
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def smooth_1d(x: np.ndarray, window: int = 11) -> np.ndarray:
    """Simple centered moving average."""
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
    max_segments: int = 2,
):
    """Detect sustained gripper-close segments from gripper action.

    Returns a list of (start, end) frame indices sorted by start time.
    """
    g = smooth_1d(gripper_action.astype(np.float32), window=smooth_window)
    close = g > threshold
    n = len(close)

    segments = []
    i = 0
    while i < n:
        if close[i]:
            j = i
            while j < n and close[j]:
                j += 1
            if j - i >= min_len:
                segments.append((int(i), int(j)))
            i = j
        else:
            i += 1

    # Keep the longest segments, but no more than max_segments
    segments = sorted(segments, key=lambda s: s[1] - s[0], reverse=True)[:max_segments]
    # Then sort by time
    segments = sorted(segments, key=lambda s: s[0])
    return segments


def assign_phases(length: int, grasp_segments: list[tuple[int, int]]) -> np.ndarray:
    """Assign phase labels based on grasp segments."""
    phase = np.zeros(length, dtype=int)
    if not grasp_segments:
        return phase

    # Build boundaries: start, seg1_start, seg1_end, seg2_start, seg2_end, end
    boundaries = [0]
    labels = [0]
    for seg in grasp_segments:
        boundaries.append(seg[0])
        labels.append(labels[-1] + 1)
        boundaries.append(seg[1])
        labels.append(labels[-1] + 1)
    boundaries.append(length)
    labels = labels[: len(boundaries) - 1]

    for li in range(len(boundaries) - 1):
        phase[boundaries[li] : boundaries[li + 1]] = labels[li]
    return phase


def process_episode(ep_file: Path, gripper_action_dim: int = -1) -> pd.DataFrame | None:
    df = pd.read_parquet(ep_file)
    if len(df) == 0:
        return None

    episode_index = int(df["episode_index"].iloc[0])
    actions = np.stack(df["actions"].values)
    gripper_action = actions[:, gripper_action_dim]

    segments = detect_grasp_segments(gripper_action)
    phase = assign_phases(len(df), segments)

    # phase_progress: progress within the current semantic phase [0, 1]
    phase_progress = np.zeros(len(df), dtype=np.float32)
    ep_len = len(df)
    for ph in np.unique(phase):
        mask = phase == ph
        indices = np.where(mask)[0]
        if len(indices) > 1:
            phase_progress[mask] = (indices - indices[0]) / (indices[-1] - indices[0])
        else:
            phase_progress[mask] = 0.0

    # global_progress: (phase + phase_progress) / total_phases, monotonic in [0, 1]
    num_phases = 5
    global_progress = (phase.astype(np.float32) + phase_progress) / num_phases

    # legacy overall progress for reference
    overall_progress = np.arange(ep_len, dtype=np.float32) / max(ep_len - 1, 1)

    return pd.DataFrame(
        {
            "episode_index": np.full(ep_len, episode_index, dtype=np.int64),
            "frame_index": df["frame_index"].values.astype(np.int64),
            "phase": phase,
            "phase_progress": phase_progress,
            "global_progress": global_progress,
            "progress": overall_progress,
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to LeRobot dataset root (containing data/ and meta/)",
    )
    parser.add_argument(
        "--gripper_action_dim",
        type=int,
        default=-1,
        help="Index of gripper dimension in action vector (default: last)",
    )
    parser.add_argument(
        "--output_name",
        default="phase_progress_semantic",
        help="Output parquet name (saved under meta/)",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    data_root = dataset_path / "data"
    ep_files = sorted(data_root.rglob("episode_*.parquet"))
    print(f"Found {len(ep_files)} episode files in {data_root}")

    records = []
    for ep_file in tqdm(ep_files, desc="Processing episodes"):
        ep_df = process_episode(ep_file, gripper_action_dim=args.gripper_action_dim)
        if ep_df is not None:
            records.append(ep_df)

    out_df = pd.concat(records, ignore_index=True)
    out_df = out_df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    meta_dir = dataset_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_path = meta_dir / f"{args.output_name}.parquet"
    out_df.to_parquet(out_path, index=False)

    print(f"\nSaved {len(out_df)} frame labels to {out_path}")
    print("Phase distribution:")
    print(out_df["phase"].value_counts().sort_index())


if __name__ == "__main__":
    main()
