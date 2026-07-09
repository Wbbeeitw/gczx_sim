"""Apply manual phase labels to other episodes via DTW alignment.

Dependencies (install manually):
    pip install fastdtw numpy pandas scipy

Usage:
    # 1. Manually annotate one or more episodes
    python manual_phase_annotator.py \
        --dataset_path /data/libero_long/task1 \
        --episode_index 3 \
        --output_path manual_labels/episode_003.json

    # 2. Propagate labels to all episodes via DTW
    python template_match_annotate.py \
        --dataset_path /data/libero_long/task1 \
        --template_dir manual_labels \
        --output_name phase_progress_semantic_template

Output:
    /data/libero_long/task1/meta/phase_progress_semantic_template.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

from common_annotator import compute_progress

try:
    from fastdtw import fastdtw
except ImportError as exc:
    raise ImportError(
        "fastdtw is required. Install it with: pip install fastdtw"
    ) from exc


def _load_episode_state(ep_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load state and compute eef_pos + gripper_width features."""
    df = pd.read_parquet(ep_file)
    if len(df) == 0:
        raise ValueError(f"Empty episode file: {ep_file}")

    states = np.stack(df["state"].values)
    eef_pos = states[:, :3].astype(np.float32)
    gripper_qpos = states[:, 6:]
    gripper_width = np.abs(gripper_qpos).sum(axis=-1, keepdims=True).astype(np.float32)
    features = np.concatenate([eef_pos, gripper_width], axis=-1)

    # Normalize each dimension across this episode for stable DTW distance.
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    features = (features - mean) / std

    return features, np.arange(len(df), dtype=np.int64)


def _build_phase_array(switches: list[dict], length: int) -> np.ndarray:
    """Convert phase switches to a dense phase array."""
    phase = np.zeros(length, dtype=int)
    if not switches:
        return phase
    switches = sorted(switches, key=lambda x: int(x["frame_index"]))
    for i, sw in enumerate(switches):
        start = int(sw["frame_index"])
        end = switches[i + 1]["frame_index"] if i + 1 < len(switches) else length
        phase[start:end] = int(sw["phase"])
    return phase


def _load_manual_template(template_path: Path, dataset_path: Path) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Load a manual template: (features, phase_array, episode_index)."""
    with open(template_path, encoding="utf-8") as f:
        data = json.load(f)

    episode_index = int(data["episode_index"])
    switches = data.get("phase_switches", [])

    ep_file = _find_episode_file(dataset_path, episode_index)
    if ep_file is None:
        print(f"[warn] episode file not found for template {template_path}")
        return None

    features, _ = _load_episode_state(ep_file)
    phase = _build_phase_array(switches, len(features))
    return features, phase, episode_index


def _find_episode_file(dataset_path: Path, episode_index: int) -> Path | None:
    """Find episode parquet file by index."""
    chunk = episode_index // 1000
    candidate = (
        dataset_path
        / "data"
        / f"chunk-{chunk:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )
    if candidate.exists():
        return candidate

    data_root = dataset_path / "data"
    if data_root.exists():
        for pattern in [
            f"episode_{episode_index:06d}.parquet",
            f"episode_{episode_index:03d}.parquet",
            f"episode_{episode_index}.parquet",
        ]:
            for candidate in data_root.rglob(pattern):
                return candidate
    return None


def _align_to_template(
    template_features: np.ndarray,
    template_phase: np.ndarray,
    new_features: np.ndarray,
    max_regress: int = 1,
) -> tuple[np.ndarray, float]:
    """Align a new episode to a template and propagate phase labels."""
    distance, path = fastdtw(template_features, new_features, dist=euclidean)

    new_phase = np.zeros(len(new_features), dtype=int)
    for t_template, t_new in path:
        new_phase[t_new] = template_phase[t_template]

    # Allow small regressions to recover from single-frame misalignments.
    for i in range(1, len(new_phase)):
        new_phase[i] = max(new_phase[i], new_phase[i - 1] - max_regress)

    return new_phase, float(distance)


def _annotate_episode(
    ep_file: Path,
    templates: list[tuple[np.ndarray, np.ndarray, int]],
    max_regress: int = 1,
) -> pd.DataFrame | None:
    """Annotate one episode by aligning to the nearest template."""
    df = pd.read_parquet(ep_file)
    if len(df) == 0:
        return None

    episode_index = int(df["episode_index"].iloc[0])
    new_features, frame_indices = _load_episode_state(ep_file)

    best_phase = None
    best_distance = float("inf")
    best_template_idx = -1

    for idx, (template_features, template_phase, template_ep_idx) in enumerate(templates):
        phase, distance = _align_to_template(
            template_features, template_phase, new_features, max_regress=max_regress
        )
        if distance < best_distance:
            best_distance = distance
            best_phase = phase
            best_template_idx = template_ep_idx

    if best_phase is None:
        return None

    phase_progress, global_progress, overall_progress = compute_progress(best_phase, 6)

    return pd.DataFrame({
        "episode_index": np.full(len(df), episode_index, dtype=np.int64),
        "frame_index": frame_indices,
        "phase": best_phase.astype(np.int64),
        "phase_progress": phase_progress,
        "global_progress": global_progress,
        "progress": overall_progress,
    })


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Propagate manual phase labels to other episodes via DTW."
    )
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--template_dir", type=str, required=True)
    parser.add_argument(
        "--output_name",
        type=str,
        default="phase_progress_semantic_template",
        help="Output parquet name (saved under dataset/meta/).",
    )
    parser.add_argument(
        "--max_regress",
        type=int,
        default=1,
        help="Maximum allowed phase regression between frames.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    template_dir = Path(args.template_dir)

    # Load all manual templates.
    templates: list[tuple[np.ndarray, np.ndarray, int]] = []
    for template_path in sorted(template_dir.glob("*.json")):
        result = _load_manual_template(template_path, dataset_path)
        if result is not None:
            templates.append(result)
            print(f"Loaded template: episode {result[2]} from {template_path.name}")

    if not templates:
        raise RuntimeError(f"No valid templates found in {template_dir}")

    # Annotate all episodes.
    data_root = dataset_path / "data"
    ep_files = sorted(data_root.rglob("episode_*.parquet"))
    print(f"Found {len(ep_files)} episodes to annotate")

    records: list[pd.DataFrame] = []
    for ep_file in ep_files:
        ep_df = _annotate_episode(ep_file, templates, max_regress=args.max_regress)
        if ep_df is not None:
            records.append(ep_df)

    if not records:
        raise RuntimeError("No episodes were successfully annotated.")

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
