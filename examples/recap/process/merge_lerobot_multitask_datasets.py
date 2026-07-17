#!/usr/bin/env python
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

"""Merge N single-task LeRobot rollout datasets into one multi-task dataset.

Unlike merge_lerobot_rollout_datasets.py (which requires identical task
definitions), this merger is built for multi-task mixtures: every input may
carry a different task. It remaps the ``task_index`` column of every data
row onto a merged ``tasks.jsonl``, offsets episode/frame indices, and copies
each task's semantic-trace sidecars (whose filenames are already
task-specific) with matching episode offsets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _offset_column(table: pa.Table, column: str, offset: int) -> pa.Table:
    index = table.schema.get_field_index(column)
    if index < 0:
        return table
    shifted = pc.add(table.column(index), pa.scalar(offset, type=table.column(index).type))
    return table.set_column(index, column, shifted)


def _remap_column(table: pa.Table, column: str, mapping: dict[int, int]) -> pa.Table:
    index = table.schema.get_field_index(column)
    if index < 0:
        return table
    values = table.column(index).to_pylist()
    remapped = pa.array([mapping.get(int(v), int(v)) for v in values], type=table.column(index).type)
    return table.set_column(index, column, remapped)


def _episode_parquet_path(dataset_path: Path, episode_index: int) -> Path:
    matches = sorted(dataset_path.glob(f"data/**/episode_{episode_index:06d}.parquet"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one parquet for episode {episode_index} in "
            f"{dataset_path}, found {len(matches)}"
        )
    return matches[0]


def _validate_compatible(infos: list[dict[str, Any]], names: list[str]) -> None:
    base = infos[0]
    for name, info in zip(names[1:], infos[1:], strict=True):
        if info["features"] != base["features"]:
            raise ValueError(f"{name}: feature schema differs from {names[0]}")
        for key in ("robot_type", "fps", "chunks_size", "data_path", "video_path"):
            if info.get(key) != base.get(key):
                raise ValueError(f"{name}: metadata field {key!r} differs")


def _sidecar_names(dataset_path: Path) -> list[str]:
    meta = dataset_path / "meta"
    names = []
    for path in sorted(meta.glob("semantic_trace_*.parquet")) + sorted(
        meta.glob("phase_progress_semantic_trace_*.parquet")
    ):
        names.append(path.name)
    return names


def _audit_names(dataset_path: Path) -> list[str]:
    return [
        path.name
        for path in sorted((dataset_path / "meta").glob("semantic_trace_*_audit.csv"))
    ]


def _trace_metadata_names(dataset_path: Path) -> list[str]:
    return [
        path.name
        for path in sorted((dataset_path / "meta").glob("semantic_trace_*_metadata.json"))
    ]


def merge_multitask_datasets(
    dataset_paths: list[Path],
    output_path: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Merge N rollout datasets, remapping task_index and offsetting indices."""
    if len(dataset_paths) < 2:
        raise ValueError("Need at least two input datasets")
    names = [path.name for path in dataset_paths]
    infos = [_read_json(path / "meta" / "info.json") for path in dataset_paths]
    _validate_compatible(infos, names)

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output dataset already exists: {output_path}; "
                "use --overwrite to replace it"
            )
        shutil.rmtree(output_path)
    (output_path / "meta").mkdir(parents=True, exist_ok=True)

    chunks_size = int(infos[0]["chunks_size"])
    merged_tasks: list[dict[str, Any]] = []
    task_index_maps: list[dict[int, int]] = []
    for path in dataset_paths:
        mapping: dict[int, int] = {}
        for record in _read_jsonl(path / "meta" / "tasks.jsonl"):
            local_index = int(record["task_index"])
            mapping[local_index] = len(merged_tasks)
            merged_tasks.append(
                {"task_index": len(merged_tasks), "task": record["task"]}
            )
        task_index_maps.append(mapping)

    merged_episodes: list[dict[str, Any]] = []
    merged_stats: list[dict[str, Any]] = []
    phase_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    total_frames = 0
    total_videos = 0
    total_successes = 0

    for path, info, task_map in zip(dataset_paths, infos, task_index_maps, strict=True):
        episode_offset = len(merged_episodes)
        frame_offset = total_frames
        episodes = _read_jsonl(path / "meta" / "episodes.jsonl")
        frames_in_dataset = 0
        for episode in episodes:
            local_index = int(episode["episode_index"])
            global_index = local_index + episode_offset
            table = pq.read_table(_episode_parquet_path(path, local_index))
            table = _offset_column(table, "episode_index", episode_offset)
            table = _offset_column(table, "index", frame_offset)
            table = _remap_column(table, "task_index", task_map)
            target_chunk = global_index // chunks_size
            target_dir = output_path / "data" / f"chunk-{target_chunk:03d}"
            target_dir.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                table,
                target_dir / f"episode_{global_index:06d}.parquet",
                compression="zstd",
            )
            frames_in_dataset += int(table.num_rows)

        video_root = path / "videos"
        if video_root.exists():
            for source_video in video_root.rglob("episode_*"):
                if not source_video.is_file():
                    continue
                local_index = int(source_video.stem.removeprefix("episode_"))
                global_index = local_index + episode_offset
                relative_parent = source_video.parent.relative_to(video_root)
                if relative_parent.parts and relative_parent.parts[0].startswith("chunk-"):
                    relative_parent = Path(*relative_parent.parts[1:])
                target_chunk = global_index // chunks_size
                target_dir = (
                    output_path
                    / "videos"
                    / f"chunk-{target_chunk:03d}"
                    / relative_parent
                )
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(
                    source_video,
                    target_dir / f"episode_{global_index:06d}{source_video.suffix}",
                )
                total_videos += 1

        if frames_in_dataset != int(info["total_frames"]):
            raise ValueError(
                f"{path.name}: meta reports {info['total_frames']} frames, "
                f"parquet contains {frames_in_dataset}"
            )
        total_frames += frames_in_dataset

        for record in episodes:
            shifted = dict(record)
            shifted["episode_index"] = int(record["episode_index"]) + episode_offset
            merged_episodes.append(shifted)
        merged_stats.extend(_read_jsonl(path / "meta" / "episodes_stats.jsonl"))

        for name in _sidecar_names(path):
            frame = pd.read_parquet(path / "meta" / name)
            if "episode_index" in frame.columns:
                frame["episode_index"] = (
                    frame["episode_index"].astype("int64") + episode_offset
                )
            frame.to_parquet(output_path / "meta" / name, index=False)
            if name.startswith("phase_progress_"):
                phase_frames.append(frame)
        for name in _audit_names(path):
            frame = pd.read_csv(path / "meta" / name)
            if "episode_index" in frame.columns:
                frame["episode_index"] = (
                    frame["episode_index"].astype("int64") + episode_offset
                )
            frame.to_csv(output_path / "meta" / name, index=False)
        for name in _trace_metadata_names(path):
            shutil.copy2(path / "meta" / name, output_path / "meta" / name)

        summary_path = path / "collection_summary.json"
        summary = _read_json(summary_path) if summary_path.exists() else {}
        successes = int(summary.get("successes", 0))
        total_successes += successes
        summary_rows.append(
            {
                "dataset": str(path),
                "episodes": len(episodes),
                "frames": frames_in_dataset,
                "successes": successes,
                "success_rate": summary.get("success_rate"),
                "task": [record["task"] for record in _read_jsonl(path / "meta" / "tasks.jsonl")],
            }
        )

    if phase_frames:
        combined = pd.concat(phase_frames, ignore_index=True)
        sort_columns = [
            column
            for column in ("episode_index", "frame_index")
            if column in combined.columns
        ]
        if sort_columns:
            combined = combined.sort_values(sort_columns).reset_index(drop=True)
        combined.to_parquet(
            output_path / "meta" / "phase_progress_multitask.parquet", index=False
        )

    _write_jsonl(output_path / "meta" / "episodes.jsonl", merged_episodes)
    _write_jsonl(output_path / "meta" / "episodes_stats.jsonl", merged_stats)
    _write_jsonl(output_path / "meta" / "tasks.jsonl", merged_tasks)

    merged_info = dict(infos[0])
    merged_info["total_episodes"] = len(merged_episodes)
    merged_info["total_frames"] = total_frames
    merged_info["total_tasks"] = len(merged_tasks)
    merged_info["total_videos"] = total_videos
    merged_info["total_chunks"] = (
        (len(merged_episodes) - 1) // chunks_size + 1 if merged_episodes else 0
    )
    merged_info["splits"] = {"train": f"0:{len(merged_episodes)}"}
    with (output_path / "meta" / "info.json").open("w", encoding="utf-8") as file:
        json.dump(merged_info, file, ensure_ascii=False, indent=4)
        file.write("\n")

    summary = {
        "output_dir": str(output_path),
        "num_episodes": len(merged_episodes),
        "total_frames": total_frames,
        "num_tasks": len(merged_tasks),
        "successes": total_successes,
        "success_rate": (
            float(total_successes / len(merged_episodes)) if merged_episodes else 0.0
        ),
        "per_dataset": summary_rows,
    }
    with (output_path / "collection_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return summary


def main() -> None:
    """Parse arguments and merge the input datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output_dataset", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = merge_multitask_datasets(
        [Path(value) for value in args.datasets],
        Path(args.output_dataset),
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
