"""Build a deterministic train/val split manifest from a fixed episode subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from episode_subset_utils import (
    load_episode_subset_file,
    resolve_episode_subset_for_dataset,
    split_episode_ids,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--subset_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val_episode_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    raw_subset = load_episode_subset_file(args.subset_path)
    subset_spec = resolve_episode_subset_for_dataset(raw_subset, dataset_path)
    if subset_spec is None:
        raise ValueError(
            f"No episode subset entry found for dataset '{dataset_path.name}' in {args.subset_path}"
        )

    train_episodes, val_episodes = split_episode_ids(
        subset_spec.episodes,
        val_episode_ratio=args.val_episode_ratio,
        seed=args.seed,
    )

    payload = {
        str(dataset_path): {
            "selected_episodes": subset_spec.episodes,
            "train_episodes": train_episodes,
            "val_episodes": val_episodes,
            "num_frames": subset_spec.num_frames,
            "seed": int(args.seed),
            "val_episode_ratio": float(args.val_episode_ratio),
        }
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"dataset: {dataset_path}")
    print(f"selected_episodes: {len(subset_spec.episodes)}")
    print(f"train_episodes: {len(train_episodes)}")
    print(f"val_episodes: {len(val_episodes)}")
    print(f"num_frames: {subset_spec.num_frames}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
