"""Sample a fixed episode subset for recap experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from episode_subset_utils import sample_balanced_episode_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--phase_path", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--success_ratio", type=float, default=0.5)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    phase_path = (
        Path(args.phase_path)
        if args.phase_path is not None
        else dataset_path / "meta" / "phase_progress_semantic.parquet"
    )

    df = pd.read_parquet(
        phase_path,
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
    ep_stats["is_success"] = ep_stats["max_phase"] >= 4

    episode_success = {
        int(row.episode_index): bool(row.is_success)
        for row in ep_stats.itertuples(index=False)
    }
    selected_episodes = sample_balanced_episode_ids(
        episode_success=episode_success,
        num_episodes=args.num_episodes,
        seed=args.seed,
        success_ratio=args.success_ratio,
    )

    selected_df = ep_stats[ep_stats["episode_index"].isin(selected_episodes)].copy()
    selected_df = selected_df.sort_values("episode_index").reset_index(drop=True)

    success_count = int(selected_df["is_success"].sum())
    failure_count = int(len(selected_df) - success_count)
    num_frames = int(selected_df["num_frames"].sum())

    payload = {
        str(dataset_path): {
            "selected_episodes": [int(ep) for ep in selected_episodes],
            "num_frames": num_frames,
            "num_success": success_count,
            "num_failure": failure_count,
            "seed": int(args.seed),
            "success_ratio": float(args.success_ratio),
        }
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"dataset: {dataset_path}")
    print(f"phase_path: {phase_path}")
    print(f"selected_episodes: {len(selected_episodes)}")
    print(f"success_count: {success_count}")
    print(f"failure_count: {failure_count}")
    print(f"num_frames: {num_frames}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
