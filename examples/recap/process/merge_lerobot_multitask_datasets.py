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


def _replace_column(table: pa.Table, column: str, values: list[int]) -> pa.Table:
    index = table.schema.get_field_index(column)
    if index < 0:
        return table
    replacement = pa.array(values, type=table.column(index).type)
    return table.set_column(index, column, replacement)


def _remap_column(table: pa.Table, column: str, mapping: dict[int, int]) -> pa.Table:
    index = table.schema.get_field_index(column)
    if index < 0:
        return table
    values = [int(value) for value in table.column(index).to_pylist()]
    missing = sorted(set(values).difference(mapping))
    if missing:
        raise ValueError(f"No merged task mapping for local task indices {missing}")
    remapped = pa.array(
        [mapping[value] for value in values],
        type=table.column(index).type,
    )
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


def _parse_dataset_arg(value: str) -> tuple[Path, tuple[int, int] | None]:
    if "::" not in value:
        return Path(value), None
    raw_path, raw_slice = value.rsplit("::", 1)
    try:
        raw_start, raw_end = raw_slice.split(":", 1)
        start = int(raw_start)
        end = int(raw_end)
    except ValueError as exc:
        raise ValueError(
            f"Invalid dataset slice {value!r}; expected PATH::START:END"
        ) from exc
    if start < 0 or end <= start:
        raise ValueError(
            f"Invalid dataset slice {value!r}; require 0 <= START < END"
        )
    return Path(raw_path), (start, end)


def _selected_episodes(
    path: Path,
    episode_range: tuple[int, int] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, int]]:
    all_episodes = _read_jsonl(path / "meta" / "episodes.jsonl")
    if episode_range is None:
        start, end = 0, len(all_episodes)
    else:
        start, end = episode_range
    if end > len(all_episodes):
        raise ValueError(
            f"{path}: episode slice {start}:{end} exceeds "
            f"total_episodes={len(all_episodes)}"
        )
    selected = all_episodes[start:end]
    local_indices = [int(record["episode_index"]) for record in selected]
    if len(set(local_indices)) != len(local_indices):
        raise ValueError(f"{path}: selected episodes contain duplicate indices")
    relative_indices = {
        local_index: offset for offset, local_index in enumerate(local_indices)
    }
    return all_episodes, selected, relative_indices


def _selected_task_indices(
    path: Path,
    episodes: list[dict[str, Any]],
) -> set[int]:
    task_records = _read_jsonl(path / "meta" / "tasks.jsonl")
    index_by_name = {
        str(record["task"]): int(record["task_index"])
        for record in task_records
    }
    task_names = {
        str(task)
        for episode in episodes
        for task in episode.get("tasks", [])
    }
    if task_names:
        unknown = sorted(task_names.difference(index_by_name))
        if unknown:
            raise ValueError(f"{path}: episodes reference unknown tasks {unknown}")
        return {index_by_name[name] for name in task_names}

    selected_indices: set[int] = set()
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        table = pq.read_table(
            _episode_parquet_path(path, episode_index),
            columns=["task_index"],
        )
        selected_indices.update(
            int(value) for value in table["task_index"].to_pylist()
        )
    return selected_indices


def merge_multitask_datasets(
    dataset_paths: list[Path],
    output_path: Path,
    overwrite: bool = False,
    episode_ranges: list[tuple[int, int] | None] | None = None,
) -> dict[str, Any]:
    """Merge N rollout datasets, remapping task_index and offsetting indices."""
    if len(dataset_paths) < 2:
        raise ValueError("Need at least two input datasets")
    if episode_ranges is None:
        episode_ranges = [None] * len(dataset_paths)
    if len(episode_ranges) != len(dataset_paths):
        raise ValueError("episode_ranges must match dataset_paths length")
    names = [path.name for path in dataset_paths]
    infos = [_read_json(path / "meta" / "info.json") for path in dataset_paths]
    _validate_compatible(infos, names)
    selected_inputs = [
        _selected_episodes(path, episode_range)
        for path, episode_range in zip(
            dataset_paths,
            episode_ranges,
            strict=True,
        )
    ]

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
    merged_task_indices: dict[str, int] = {}
    for path, (_, selected_episodes, _) in zip(
        dataset_paths,
        selected_inputs,
        strict=True,
    ):
        selected_task_indices = _selected_task_indices(path, selected_episodes)
        mapping: dict[int, int] = {}
        for record in _read_jsonl(path / "meta" / "tasks.jsonl"):
            local_index = int(record["task_index"])
            if local_index not in selected_task_indices:
                continue
            task_name = str(record["task"])
            if task_name not in merged_task_indices:
                merged_task_indices[task_name] = len(merged_tasks)
                merged_tasks.append(
                    {"task_index": len(merged_tasks), "task": task_name}
                )
            mapping[local_index] = merged_task_indices[task_name]
        task_index_maps.append(mapping)

    merged_episodes: list[dict[str, Any]] = []
    merged_stats: list[dict[str, Any]] = []
    phase_frames: list[pd.DataFrame] = []
    sidecar_frames: dict[str, list[pd.DataFrame]] = {}
    audit_frames: dict[str, list[pd.DataFrame]] = {}
    summary_rows: list[dict[str, Any]] = []
    total_frames = 0
    total_videos = 0
    total_successes = 0

    for path, _info, task_map, episode_range, selected_input in zip(
        dataset_paths,
        infos,
        task_index_maps,
        episode_ranges,
        selected_inputs,
        strict=True,
    ):
        episode_offset = len(merged_episodes)
        frame_offset = total_frames
        all_episodes, episodes, relative_index_map = selected_input
        episode_index_map = {
            local_index: episode_offset + relative_index
            for local_index, relative_index in relative_index_map.items()
        }
        frames_in_dataset = 0
        for episode in episodes:
            local_index = int(episode["episode_index"])
            global_index = episode_index_map[local_index]
            table = pq.read_table(_episode_parquet_path(path, local_index))
            table = _replace_column(
                table,
                "episode_index",
                [global_index] * table.num_rows,
            )
            table = _replace_column(
                table,
                "index",
                list(
                    range(
                        frame_offset + frames_in_dataset,
                        frame_offset + frames_in_dataset + table.num_rows,
                    )
                ),
            )
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
                if local_index not in episode_index_map:
                    continue
                global_index = episode_index_map[local_index]
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

        expected_frames = sum(int(record["length"]) for record in episodes)
        if frames_in_dataset != expected_frames:
            raise ValueError(
                f"{path.name}: selected episode metadata reports "
                f"{expected_frames} frames, "
                f"parquet contains {frames_in_dataset}"
            )
        total_frames += frames_in_dataset

        for record in episodes:
            shifted = dict(record)
            shifted["episode_index"] = episode_index_map[int(record["episode_index"])]
            merged_episodes.append(shifted)
        stats_path = path / "meta" / "episodes_stats.jsonl"
        if stats_path.exists():
            all_stats = _read_jsonl(stats_path)
            if len(all_stats) == len(all_episodes):
                selected_positions = [
                    position
                    for position, record in enumerate(all_episodes)
                    if int(record["episode_index"]) in episode_index_map
                ]
                selected_stats = [all_stats[index] for index in selected_positions]
            else:
                selected_stats = [
                    record
                    for record in all_stats
                    if int(record.get("episode_index", -1)) in episode_index_map
                ]
            for record in selected_stats:
                shifted = dict(record)
                if "episode_index" in shifted:
                    shifted["episode_index"] = episode_index_map[
                        int(shifted["episode_index"])
                    ]
                merged_stats.append(shifted)

        selected_successes: dict[int, bool] = {}
        for name in _sidecar_names(path):
            frame = pd.read_parquet(path / "meta" / name)
            if "episode_index" in frame.columns:
                frame = frame[
                    frame["episode_index"].astype("int64").isin(episode_index_map)
                ].copy()
                frame["episode_index"] = (
                    frame["episode_index"].astype("int64").map(episode_index_map)
                ).astype("int64")
            if frame.empty:
                continue
            sidecar_frames.setdefault(name, []).append(frame)
            if name.startswith("phase_progress_"):
                phase_frames.append(frame)
                if "is_success" in frame.columns:
                    episode_success = frame.groupby("episode_index")[
                        "is_success"
                    ].last()
                    selected_successes.update(
                        {
                            int(index): bool(value)
                            for index, value in episode_success.items()
                        }
                    )
        for name in _audit_names(path):
            frame = pd.read_csv(path / "meta" / name)
            if "episode_index" in frame.columns:
                frame = frame[
                    frame["episode_index"].astype("int64").isin(episode_index_map)
                ].copy()
                frame["episode_index"] = (
                    frame["episode_index"].astype("int64").map(episode_index_map)
                ).astype("int64")
            if not frame.empty:
                audit_frames.setdefault(name, []).append(frame)
        for name in _trace_metadata_names(path):
            target = output_path / "meta" / name
            if not target.exists():
                shutil.copy2(path / "meta" / name, target)

        summary_path = path / "collection_summary.json"
        summary = _read_json(summary_path) if summary_path.exists() else {}
        successes = sum(selected_successes.values())
        if episode_range is None and not selected_successes:
            successes = int(summary.get("successes", 0))
        total_successes += successes
        success_rate = successes / len(episodes) if episodes else 0.0
        slice_suffix = (
            f"::{episode_range[0]}:{episode_range[1]}" if episode_range else ""
        )
        selected_task_names = [
            record["task"]
            for record in _read_jsonl(path / "meta" / "tasks.jsonl")
            if int(record["task_index"]) in task_map
        ]
        summary_rows.append(
            {
                "dataset": f"{path}{slice_suffix}",
                "episodes": len(episodes),
                "frames": frames_in_dataset,
                "successes": successes,
                "success_rate": success_rate,
                "task": selected_task_names,
            }
        )

    for name, frames in sidecar_frames.items():
        combined_sidecar = pd.concat(frames, ignore_index=True)
        sort_columns = [
            column
            for column in ("episode_index", "frame_index")
            if column in combined_sidecar.columns
        ]
        if sort_columns:
            combined_sidecar = combined_sidecar.sort_values(sort_columns)
        combined_sidecar.to_parquet(output_path / "meta" / name, index=False)
    for name, frames in audit_frames.items():
        combined_audit = pd.concat(frames, ignore_index=True)
        if "episode_index" in combined_audit.columns:
            combined_audit = combined_audit.sort_values("episode_index")
        combined_audit.to_csv(output_path / "meta" / name, index=False)

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
    parsed_inputs = [_parse_dataset_arg(value) for value in args.datasets]
    result = merge_multitask_datasets(
        [path for path, _ in parsed_inputs],
        Path(args.output_dataset),
        overwrite=args.overwrite,
        episode_ranges=[episode_range for _, episode_range in parsed_inputs],
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
