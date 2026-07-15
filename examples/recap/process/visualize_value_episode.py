"""Plot Value model predictions against returns for one LeRobot episode.

The Value SFT target uses the ReCap convention
``normalized_return = raw_return / abs(return_min)``.  The model therefore
predicts values in ``[-1, 0]`` while the return sidecar contains raw returns.
This script converts predictions back to raw return units before plotting.

Example:

    CUDA_VISIBLE_DEVICES=1 python examples/recap/process/visualize_value_episode.py \
        --dataset /data/libero_long/task1_d0_40 \
        --checkpoint /workspace/results/value_sft/task1_d0_40/value_task1_d0_40/checkpoints/global_step_1200 \
        --episode 0 \
        --tag fail300_d0 \
        --output /workspace/results/value_sft/task1_d0_40/value_episode_0000.png \
        --siglip-path /workspace/models/siglip2-so400m-patch14-224 \
        --gemma3-path /workspace/models/gemma-3-270m \
        --tokenizer-path /workspace/models/gemma-3-270m
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)

from rlinf.data.datasets.recap.utils import (
    decode_image_struct_batch,
    load_returns_sidecar,
    load_task_descriptions,
)
from rlinf.models.embodiment.value_model.modeling_critic import ValueCriticModel


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _episode_indices(dataset: LeRobotDataset, episode: int) -> list[int]:
    episode_data_index = dataset.episode_data_index
    if episode < 0 or episode >= len(episode_data_index["from"]):
        raise IndexError(
            f"Episode {episode} is outside dataset range "
            f"[0, {len(episode_data_index['from']) - 1}]"
        )
    start = int(episode_data_index["from"][episode].item())
    end = int(episode_data_index["to"][episode].item())
    return list(range(start, end))


def _build_observation(
    sample: dict[str, Any],
    task_descriptions: dict[int, str],
    default_prompt: str | None,
) -> dict[str, Any]:
    """Map a LeRobot sample to the LIBERO Value model input format."""
    task_index = _to_int(sample["task_index"]) if "task_index" in sample else 0
    prompt = task_descriptions.get(task_index, default_prompt)
    if not prompt:
        prompt = "perform the task"

    observation: dict[str, Any] = {
        "observation/image": sample["image"],
        "observation/wrist_image": sample["wrist_image"],
        "observation/state": sample["state"],
        "prompt": prompt,
    }
    return observation


def _denormalize_values(
    values: np.ndarray,
    return_min: float,
    return_max: float,
    normalization: str,
) -> np.ndarray:
    if normalization == "minus_one_zero":
        return values * abs(return_min)
    if normalization == "zero_one":
        return values * (return_max - return_min) + return_min
    raise ValueError(f"Unknown normalization: {normalization}")


def _write_predictions(
    path: Path,
    frame_indices: np.ndarray,
    returns: np.ndarray,
    predicted_values: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["frame_index", "return", "predicted_value"])
        writer.writerows(
            zip(
                frame_indices.tolist(),
                returns.tolist(),
                predicted_values.tolist(),
                strict=True,
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Value predictions and raw returns for one episode."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--robot-type", default="libero")
    parser.add_argument("--model-type", default="pi05")
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--critic-expert-variant", default="gemma_1m")
    parser.add_argument("--value-min", type=float, default=-1.0)
    parser.add_argument("--value-max", type=float, default=0.0)
    parser.add_argument("--return-min", type=float, default=-900.0)
    parser.add_argument("--return-max", type=float, default=0.0)
    parser.add_argument(
        "--normalization",
        choices=("minus_one_zero", "zero_one"),
        default="minus_one_zero",
    )
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument(
        "--siglip-path",
        required=True,
        help="Path to the pretrained SigLIP checkpoint.",
    )
    parser.add_argument(
        "--gemma3-path",
        required=True,
        help="Path to the pretrained Gemma checkpoint.",
    )
    parser.add_argument(
        "--tokenizer-path",
        required=True,
        help="Path to the local Gemma tokenizer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")
    if args.return_min >= args.return_max:
        raise ValueError(
            f"--return-min must be smaller than --return-max, "
            f"got {args.return_min} and {args.return_max}"
        )

    dataset_path = args.dataset.expanduser().resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    metadata = LeRobotDatasetMetadata(dataset_path.name, root=dataset_path)
    dataset = LeRobotDataset(
        dataset_path.name,
        root=dataset_path,
        delta_timestamps=None,
        download_videos=False,
    )
    dataset.hf_dataset.set_transform(decode_image_struct_batch)

    returns = load_returns_sidecar(dataset_path, args.tag)
    if returns is None or args.episode not in returns:
        raise KeyError(
            f"Episode {args.episode} has no return sidecar entry in "
            f"{dataset_path / 'meta'}"
        )

    indices = _episode_indices(dataset, args.episode)
    task_descriptions = load_task_descriptions(dataset_path)
    observations = []
    frame_indices = []
    for index in indices:
        sample = dataset[index]
        observations.append(
            _build_observation(sample, task_descriptions, args.default_prompt)
        )
        frame_indices.append(_to_int(sample["frame_index"]))

    sidecar_return = returns[args.episode]["return"]
    if len(sidecar_return) != len(observations):
        raise ValueError(
            f"Episode length mismatch: dataset has {len(observations)} frames, "
            f"return sidecar has {len(sidecar_return)} frames"
        )

    model = ValueCriticModel.from_checkpoint(
        args.checkpoint,
        device=args.device,
        env_type=args.robot_type,
        model_type=args.model_type,
        critic_expert_variant=args.critic_expert_variant,
        tokenizer_path=args.tokenizer_path,
        siglip_path=args.siglip_path,
        gemma3_path=args.gemma3_path,
        num_return_bins=201,
        return_min=args.value_min,
        return_max=args.value_max,
    )
    predictions = model.infer_batch(observations, batch_size=args.batch_size)
    normalized_values = np.asarray([item["value"] for item in predictions])
    predicted_values = _denormalize_values(
        normalized_values,
        args.return_min,
        args.return_max,
        args.normalization,
    )
    raw_returns = np.asarray(sidecar_return, dtype=np.float32)
    frame_indices_array = np.asarray(frame_indices, dtype=np.int64)

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_path.with_suffix(".csv")
    _write_predictions(csv_path, frame_indices_array, raw_returns, predicted_values)

    success = bool(np.isclose(raw_returns[-1], args.return_max))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(frame_indices_array, raw_returns, label="Target return", linewidth=2.0)
    ax.plot(
        frame_indices_array,
        predicted_values,
        label="Predicted value",
        linewidth=1.8,
    )
    ax.set_title(
        f"{dataset_path.name} | episode={args.episode} | "
        f"{'success' if success else 'failure'}"
    )
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Return / predicted value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    mae = float(np.mean(np.abs(predicted_values - raw_returns)))
    print(f"Saved plot: {output_path}")
    print(f"Saved predictions: {csv_path}")
    print(
        f"episode={args.episode}, frames={len(raw_returns)}, "
        f"success={success}, raw_mae={mae:.4f}, "
        f"target_start={raw_returns[0]:.4f}, "
        f"prediction_start={predicted_values[0]:.4f}"
    )


if __name__ == "__main__":
    main()
