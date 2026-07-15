#!/usr/bin/env python
"""Merge compatible LeRobot rollout datasets without changing the sources."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


TRACE_NAME = "semantic_trace_task1"
PHASE_LABEL_NAME = "phase_progress_semantic_trace_task1"
AUDIT_NAME = "semantic_trace_task1_audit.csv"
TRACE_METADATA_NAME = "semantic_trace_task1_metadata.json"


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


def _replace_offset_column(
    table: pa.Table,
    column_name: str,
    offset: int,
) -> pa.Table:
    column_index = table.schema.get_field_index(column_name)
    if column_index < 0:
        raise ValueError(f"Parquet file is missing {column_name!r}")
    column = table.column(column_name)
    offset_value = pa.scalar(offset, type=column.type)
    shifted = pc.add(column, offset_value)
    return table.set_column(column_index, column_name, shifted)


def _episode_parquet_path(dataset_path: Path, episode_index: int) -> Path:
    matches = sorted(
        dataset_path.glob(f"data/**/episode_{episode_index:06d}.parquet")
    )
    if len(matches) != 1:
        raise ValueError(
            f"Expected one parquet file for episode {episode_index} in "
            f"{dataset_path}, found {len(matches)}"
        )
    return matches[0]


def _copy_additional_data(
    source_path: Path,
    output_path: Path,
    episode_offset: int,
    frame_offset: int,
    chunks_size: int,
) -> int:
    source_episodes = _read_jsonl(source_path / "meta" / "episodes.jsonl")
    total_frames = 0
    for local_episode in source_episodes:
        local_index = int(local_episode["episode_index"])
        global_index = local_index + episode_offset
        source_file = _episode_parquet_path(source_path, local_index)
        table = pq.read_table(source_file)
        table = _replace_offset_column(table, "episode_index", episode_offset)
        table = _replace_offset_column(table, "index", frame_offset)

        target_chunk = global_index // chunks_size
        target_dir = output_path / "data" / f"chunk-{target_chunk:03d}"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / f"episode_{global_index:06d}.parquet"
        pq.write_table(table, target_file, compression="zstd")
        total_frames += int(table.num_rows)

    source_video_root = source_path / "videos"
    if source_video_root.exists():
        for source_video in source_video_root.rglob("episode_*"):
            if not source_video.is_file():
                continue
            stem = source_video.stem
            local_index = int(stem.removeprefix("episode_"))
            global_index = local_index + episode_offset
            relative_parent = source_video.parent.relative_to(source_video_root)
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
            target_file = target_dir / f"episode_{global_index:06d}{source_video.suffix}"
            shutil.copy2(source_video, target_file)

    return total_frames


def _offset_episode_stats(
    records: list[dict[str, Any]],
    episode_offset: int,
) -> list[dict[str, Any]]:
    shifted_records = []
    for record in records:
        shifted = dict(record)
        shifted["episode_index"] = int(record["episode_index"]) + episode_offset
        episode_stats = dict(shifted.get("stats", {}).get("episode_index", {}))
        for key in ("min", "max", "mean"):
            if key in episode_stats:
                episode_stats[key] = [
                    float(value) + episode_offset for value in episode_stats[key]
                ]
        if "stats" in shifted:
            shifted["stats"] = dict(shifted["stats"])
            shifted["stats"]["episode_index"] = episode_stats
        shifted_records.append(shifted)
    return shifted_records


def _merge_sidecar(
    first_path: Path,
    second_path: Path,
    output_path: Path,
    name: str,
    episode_offset: int,
) -> None:
    first = pd.read_parquet(first_path / "meta" / name)
    second = pd.read_parquet(second_path / "meta" / name).copy()
    if "episode_index" not in second.columns:
        raise ValueError(f"Sidecar {name} is missing episode_index")
    second["episode_index"] = second["episode_index"].astype("int64") + episode_offset
    merged = pd.concat([first, second], ignore_index=True)
    sort_columns = [column for column in ("episode_index", "frame_index") if column in merged]
    if sort_columns:
        merged = merged.sort_values(sort_columns).reset_index(drop=True)
    merged.to_parquet(output_path / "meta" / name, index=False)


def _merge_audit(
    first_path: Path,
    second_path: Path,
    output_path: Path,
    episode_offset: int,
) -> None:
    first = pd.read_csv(first_path / "meta" / AUDIT_NAME)
    second = pd.read_csv(second_path / "meta" / AUDIT_NAME).copy()
    second["episode_index"] = second["episode_index"].astype("int64") + episode_offset
    merged = pd.concat([first, second], ignore_index=True)
    merged = merged.sort_values("episode_index").reset_index(drop=True)
    merged.to_csv(output_path / "meta" / AUDIT_NAME, index=False)


def _merge_summary(
    first_path: Path,
    second_path: Path,
    output_path: Path,
) -> None:
    first = _read_json(first_path / "collection_summary.json")
    second = _read_json(second_path / "collection_summary.json")
    episode_count = int(first["num_episodes"]) + int(second["num_episodes"])
    successes = int(first["successes"]) + int(second["successes"])
    first_length = float(first.get("mean_episode_length", 0.0))
    second_length = float(second.get("mean_episode_length", 0.0))
    first_return = float(first.get("mean_initial_return", 0.0))
    second_return = float(second.get("mean_initial_return", 0.0))
    summary = {
        "output_dir": str(output_path),
        "model_path": first.get("model_path"),
        "model_type": first.get("model_type"),
        "task_suite_name": first.get("task_suite_name"),
        "task_id": first.get("task_id"),
        "num_episodes": episode_count,
        "successes": successes,
        "success_rate": float(successes / episode_count) if episode_count else 0.0,
        "mean_episode_length": (
            (first_length * int(first["num_episodes"]) + second_length * int(second["num_episodes"]))
            / episode_count
            if episode_count
            else 0.0
        ),
        "mean_initial_return": (
            (first_return * int(first["num_episodes"]) + second_return * int(second["num_episodes"]))
            / episode_count
            if episode_count
            else 0.0
        ),
        "source_summaries": [str(first_path / "collection_summary.json"), str(second_path / "collection_summary.json")],
    }
    with (output_path / "collection_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")


def merge_datasets(
    first_path: Path,
    second_path: Path,
    output_path: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Merge two compatible rollout datasets into a new dataset."""
    first_info = _read_json(first_path / "meta" / "info.json")
    second_info = _read_json(second_path / "meta" / "info.json")
    if first_info["features"] != second_info["features"]:
        raise ValueError("Input datasets have different feature schemas")
    for key in ("robot_type", "fps", "chunks_size", "data_path", "video_path"):
        if first_info.get(key) != second_info.get(key):
            raise ValueError(f"Input datasets differ in metadata field {key!r}")

    first_tasks = _read_jsonl(first_path / "meta" / "tasks.jsonl")
    second_tasks = _read_jsonl(second_path / "meta" / "tasks.jsonl")
    if first_tasks != second_tasks:
        raise ValueError("Input datasets have different task definitions")

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output dataset already exists: {output_path}; use --overwrite to replace it"
            )
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(first_path, output_path)

    first_episodes = _read_jsonl(first_path / "meta" / "episodes.jsonl")
    second_episodes = _read_jsonl(second_path / "meta" / "episodes.jsonl")
    episode_offset = len(first_episodes)
    frame_offset = int(first_info["total_frames"])
    chunks_size = int(first_info["chunks_size"])
    second_frames = _copy_additional_data(
        second_path,
        output_path,
        episode_offset,
        frame_offset,
        chunks_size,
    )
    if second_frames != int(second_info["total_frames"]):
        raise ValueError(
            f"Second dataset metadata reports {second_info['total_frames']} frames, "
            f"but parquet files contain {second_frames}"
        )

    merged_episodes = list(first_episodes)
    for record in second_episodes:
        shifted = dict(record)
        shifted["episode_index"] = int(record["episode_index"]) + episode_offset
        merged_episodes.append(shifted)
    _write_jsonl(output_path / "meta" / "episodes.jsonl", merged_episodes)

    first_stats = _read_jsonl(first_path / "meta" / "episodes_stats.jsonl")
    second_stats = _read_jsonl(second_path / "meta" / "episodes_stats.jsonl")
    _write_jsonl(
        output_path / "meta" / "episodes_stats.jsonl",
        first_stats + _offset_episode_stats(second_stats, episode_offset),
    )

    merged_info = dict(first_info)
    merged_info["total_episodes"] = len(merged_episodes)
    merged_info["total_frames"] = frame_offset + second_frames
    merged_info["total_videos"] = int(first_info["total_videos"]) + int(
        second_info["total_videos"]
    )
    merged_info["total_chunks"] = (
        (merged_info["total_episodes"] - 1) // chunks_size + 1
        if merged_info["total_episodes"]
        else 0
    )
    merged_info["splits"] = {"train": f"0:{merged_info['total_episodes']}"}
    with (output_path / "meta" / "info.json").open("w", encoding="utf-8") as file:
        json.dump(merged_info, file, ensure_ascii=False, indent=4)
        file.write("\n")

    _merge_sidecar(first_path, second_path, output_path, f"{TRACE_NAME}.parquet", episode_offset)
    _merge_sidecar(
        first_path,
        second_path,
        output_path,
        f"{PHASE_LABEL_NAME}.parquet",
        episode_offset,
    )
    _merge_audit(first_path, second_path, output_path, episode_offset)
    if _read_json(first_path / "meta" / TRACE_METADATA_NAME) != _read_json(
        second_path / "meta" / TRACE_METADATA_NAME
    ):
        raise ValueError("Input datasets have different semantic trace metadata")
    _merge_summary(first_path, second_path, output_path)
    first_summary = _read_json(first_path / "collection_summary.json")
    second_summary = _read_json(second_path / "collection_summary.json")

    return {
        "output_dir": str(output_path),
        "episodes": len(merged_episodes),
        "frames": int(merged_info["total_frames"]),
        "successes": int(first_summary["successes"]) + int(second_summary["successes"]),
    }


def main() -> None:
    """Parse arguments and merge two rollout datasets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--first_dataset", required=True)
    parser.add_argument("--second_dataset", required=True)
    parser.add_argument("--output_dataset", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = merge_datasets(
        Path(args.first_dataset),
        Path(args.second_dataset),
        Path(args.output_dataset),
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
