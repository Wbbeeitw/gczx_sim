# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export one simulator frame from a successful episode for every task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from visualize_critic_trajectory import (
    _detect_image_keys,
    _episode_indices,
    _task_description,
    _to_image_array,
)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "success"}
    if pd.isna(value):
        return False
    return bool(value)


def _success_trace_path(dataset_path: Path, task: str) -> Path:
    candidates = (
        dataset_path / "meta" / f"phase_progress_semantic_trace_{task}.parquet",
        dataset_path / "meta" / f"semantic_trace_{task}.parquet",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No semantic success trace found for {task} under {dataset_path / 'meta'}"
    )


def _successful_episodes(trace_path: Path) -> list[int]:
    trace = pd.read_parquet(trace_path)
    success_column = next(
        (column for column in ("is_success", "is_success_bool") if column in trace),
        None,
    )
    if success_column is None:
        raise ValueError(
            f"Success trace lacks is_success/is_success_bool: {trace_path}"
        )
    if "episode_index" not in trace:
        raise ValueError(f"Success trace lacks episode_index: {trace_path}")

    sort_columns = ["episode_index"]
    if "frame_index" in trace:
        sort_columns.append("frame_index")
    trace = trace.sort_values(sort_columns)
    successful: list[int] = []
    for episode_index, episode in trace.groupby("episode_index", sort=True):
        values = episode[success_column].dropna()
        if not values.empty and _as_bool(values.iloc[-1]):
            successful.append(int(episode_index))
    if not successful:
        raise ValueError(f"No successful episodes found in {trace_path}")
    return successful


def _to_pil_image(value: Any) -> Image.Image:
    image = _to_image_array(value)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = np.rint(image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray(image)


def _export_task_frame(
    *,
    task_index: int,
    dataset_path: Path,
    output_path: Path,
    frame_fraction: float,
    image_key: str | None,
    rng: np.random.Generator,
) -> dict[str, Any]:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from rlinf.data.datasets.recap.utils import decode_image_struct_batch

    task = f"task{task_index}"
    trace_path = _success_trace_path(dataset_path, task)
    successful = _successful_episodes(trace_path)
    episode_index = int(rng.choice(successful))

    dataset = LeRobotDataset(
        dataset_path.name,
        root=dataset_path,
        delta_timestamps=None,
        download_videos=False,
    )
    dataset.hf_dataset.set_transform(decode_image_struct_batch)
    episode_dataset_indices = _episode_indices(dataset, episode_index)
    if not episode_dataset_indices:
        raise ValueError(
            f"{task} successful episode {episode_index} has no dataset frames"
        )

    position = round(frame_fraction * (len(episode_dataset_indices) - 1))
    position = min(max(int(position), 0), len(episode_dataset_indices) - 1)
    dataset_index = episode_dataset_indices[position]
    sample = dataset[dataset_index]

    detected_keys = _detect_image_keys(sample)
    selected_image_key = image_key or (detected_keys[0] if detected_keys else None)
    if selected_image_key is None:
        raise ValueError(
            f"No image field detected for {task}; pass --image-key explicitly"
        )
    if selected_image_key not in sample:
        raise ValueError(
            f"Image key {selected_image_key!r} missing for {task}; "
            f"detected keys: {detected_keys}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = _to_pil_image(sample[selected_image_key])
    image.save(output_path, format="PNG")
    frame_index = int(np.asarray(sample["frame_index"]).item())
    description = _task_description(dataset_path, sample)
    return {
        "task": task,
        "task_index": task_index,
        "dataset": str(dataset_path),
        "trace": str(trace_path),
        "successful_episodes_available": len(successful),
        "episode_index": episode_index,
        "frame_index": frame_index,
        "episode_frames": len(episode_dataset_indices),
        "frame_fraction": frame_fraction,
        "image_key": selected_image_key,
        "task_description": description,
        "output": str(output_path),
        "width": image.width,
        "height": image.height,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-pattern",
        required=True,
        help=(
            "Child dataset pattern containing {task} and optionally {task_index}, "
            "for example /data/run/{task}_30ep."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--frame-fraction",
        type=float,
        default=0.65,
        help="Relative position inside the selected successful episode.",
    )
    parser.add_argument(
        "--image-key",
        help="Optional camera field; the preferred simulator camera is auto-detected.",
    )
    return parser


def main() -> None:
    """Export exactly one successful-episode PNG per task."""
    args = _build_parser().parse_args()
    if args.num_tasks < 1:
        raise ValueError("--num-tasks must be positive")
    if not 0.0 <= args.frame_fraction <= 1.0:
        raise ValueError("--frame-fraction must be between 0 and 1")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    records: list[dict[str, Any]] = []
    for task_index in range(args.num_tasks):
        task = f"task{task_index}"
        dataset_path = Path(
            args.dataset_pattern.format(task=task, task_index=task_index)
        ).expanduser()
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset not found for {task}: {dataset_path}")
        output_path = output_dir / f"{task}.png"
        record = _export_task_frame(
            task_index=task_index,
            dataset_path=dataset_path,
            output_path=output_path,
            frame_fraction=args.frame_fraction,
            image_key=args.image_key,
            rng=rng,
        )
        records.append(record)
        print(
            f"[SAVED] {task}: episode={record['episode_index']} "
            f"frame={record['frame_index']} camera={record['image_key']} "
            f"-> {output_path}"
        )

    manifest_csv = output_dir / "selected_frames.csv"
    manifest_json = output_dir / "selected_frames.json"
    pd.DataFrame(records).to_csv(manifest_csv, index=False)
    manifest_json.write_text(
        json.dumps(
            {
                "dataset_pattern": args.dataset_pattern,
                "seed": args.seed,
                "frame_fraction": args.frame_fraction,
                "frames": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Exported {len(records)} PNG files to {output_dir}")
    print(f"CSV: {manifest_csv}")
    print(f"JSON: {manifest_json}")


if __name__ == "__main__":
    main()
