"""Export a deterministic, self-contained sample of complete task episodes."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import tarfile
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

if __package__:
    from examples.recap.process.merge_lerobot_multitask_datasets import (
        merge_multitask_datasets,
    )
else:
    from merge_lerobot_multitask_datasets import merge_multitask_datasets


@dataclass(frozen=True)
class EpisodeSelection:
    """One source episode selected for the reproducibility subset."""

    task: str
    source_dataset: str
    source_episode_position: int
    source_episode_index: int
    output_episode_index: int
    episode_length: int
    is_success: bool
    trainable: bool
    source_type: str | None
    source_video_files: int
    source_episode_provenance: dict[str, Any] | None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _as_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"{field}: expected a boolean value, got {value!r}")


def _audit_path(dataset: Path, task: str) -> Path:
    exact = dataset / "meta" / f"semantic_trace_{task}_audit.csv"
    if exact.is_file():
        return exact
    matches = sorted((dataset / "meta").glob("semantic_trace_*_audit.csv"))
    if len(matches) != 1:
        raise ValueError(
            f"{dataset}: expected one semantic audit for {task}, found {len(matches)}"
        )
    return matches[0]


def _video_count(dataset: Path, episode_index: int) -> int:
    return sum(
        1
        for path in (dataset / "videos").glob(
            f"**/episode_{episode_index:06d}.*"
        )
        if path.is_file()
    )


def _episode_candidates(
    dataset: Path,
    task: str,
    *,
    trainable_only: bool,
) -> list[dict[str, Any]]:
    info = _read_json(dataset / "meta" / "info.json")
    episodes = _read_jsonl(dataset / "meta" / "episodes.jsonl")
    if int(info["total_episodes"]) != len(episodes):
        raise ValueError(f"{dataset}: info and episodes.jsonl disagree")

    audit = pd.read_csv(_audit_path(dataset, task))
    required_audit_columns = {"episode_index", "is_success", "trainable"}
    missing = sorted(required_audit_columns.difference(audit.columns))
    if missing:
        raise ValueError(f"{dataset}: semantic audit lacks columns {missing}")
    audit_by_episode = {
        int(row["episode_index"]): row for row in audit.to_dict(orient="records")
    }

    provenance_path = dataset / "meta" / "episode_provenance.jsonl"
    provenance_by_episode = (
        {
            int(row["episode_index"]): row
            for row in _read_jsonl(provenance_path)
        }
        if provenance_path.is_file()
        else {}
    )

    candidates: list[dict[str, Any]] = []
    for position, episode in enumerate(episodes):
        episode_index = int(episode["episode_index"])
        if episode_index not in audit_by_episode:
            raise ValueError(
                f"{dataset}: audit has no row for episode {episode_index}"
            )
        audit_row = audit_by_episode[episode_index]
        trainable = _as_bool(
            audit_row["trainable"],
            field=f"{task} episode {episode_index} trainable",
        )
        if trainable_only and not trainable:
            continue
        provenance = provenance_by_episode.get(episode_index)
        candidates.append(
            {
                "position": position,
                "episode_index": episode_index,
                "length": int(episode["length"]),
                "is_success": _as_bool(
                    audit_row["is_success"],
                    field=f"{task} episode {episode_index} is_success",
                ),
                "trainable": trainable,
                "source_type": (
                    str(provenance["source_type"])
                    if provenance and provenance.get("source_type") is not None
                    else None
                ),
                "video_files": _video_count(dataset, episode_index),
                "provenance": provenance,
            }
        )
    return candidates


def select_episodes(
    *,
    dataset_pattern: str,
    tasks: list[str],
    episodes_per_task: int,
    seed: int,
    trainable_only: bool,
) -> list[EpisodeSelection]:
    """Select complete episodes deterministically and independently per task."""
    if episodes_per_task < 1:
        raise ValueError("episodes_per_task must be positive")

    selected: list[EpisodeSelection] = []
    for task_index, task in enumerate(tasks):
        dataset = Path(dataset_pattern.format(task=task))
        candidates = _episode_candidates(
            dataset,
            task,
            trainable_only=trainable_only,
        )
        if len(candidates) < episodes_per_task:
            raise ValueError(
                f"{task}: requested {episodes_per_task} episodes, "
                f"but only {len(candidates)} eligible episodes are available"
            )
        task_rng = random.Random(seed + task_index)
        chosen = sorted(
            task_rng.sample(candidates, episodes_per_task),
            key=lambda row: row["position"],
        )
        for row in chosen:
            selected.append(
                EpisodeSelection(
                    task=task,
                    source_dataset=str(dataset),
                    source_episode_position=int(row["position"]),
                    source_episode_index=int(row["episode_index"]),
                    output_episode_index=len(selected),
                    episode_length=int(row["length"]),
                    is_success=bool(row["is_success"]),
                    trainable=bool(row["trainable"]),
                    source_type=row["source_type"],
                    source_video_files=int(row["video_files"]),
                    source_episode_provenance=row["provenance"],
                )
            )
    return selected


def _write_manifests(
    output_dataset: Path,
    selections: list[EpisodeSelection],
    *,
    dataset_pattern: str,
    seed: int,
    trainable_only: bool,
) -> None:
    rows = [asdict(selection) for selection in selections]
    manifest = {
        "format_version": 1,
        "selection_policy": "fixed-seed uniform sampling without replacement",
        "selection_scope": (
            "trainable episodes only" if trainable_only else "all episodes"
        ),
        "seed": seed,
        "dataset_pattern": dataset_pattern,
        "episodes": rows,
        "aggregate": {
            "tasks": len({selection.task for selection in selections}),
            "episodes": len(selections),
            "frames": sum(selection.episode_length for selection in selections),
            "successes": sum(selection.is_success for selection in selections),
            "failures": sum(not selection.is_success for selection in selections),
            "trainable": sum(selection.trainable for selection in selections),
        },
    }
    (output_dataset / "reproducibility_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    csv_fields = [
        "task",
        "source_dataset",
        "source_episode_position",
        "source_episode_index",
        "output_episode_index",
        "episode_length",
        "is_success",
        "trainable",
        "source_type",
        "source_video_files",
    ]
    with (output_dataset / "reproducibility_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in csv_fields} for row in rows
        )

    provenance_rows = []
    for selection in selections:
        provenance = dict(selection.source_episode_provenance or {})
        provenance.update(
            {
                "episode_index": selection.output_episode_index,
                "sampled_task": selection.task,
                "sampled_from_dataset": selection.source_dataset,
                "sampled_from_episode_position": selection.source_episode_position,
                "sampled_from_episode_index": selection.source_episode_index,
                "is_success": selection.is_success,
                "trainable": selection.trainable,
            }
        )
        provenance_rows.append(provenance)
    _write_jsonl(
        output_dataset / "meta" / "episode_provenance.jsonl",
        provenance_rows,
    )
    full_positive = [
        selection.output_episode_index
        for selection in selections
        if selection.source_type == "expert"
    ]
    (output_dataset / "meta" / "full_positive_episodes.json").write_text(
        json.dumps(full_positive, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _export_episode_sidecars(
    output_dataset: Path,
    selections: list[EpisodeSelection],
) -> list[dict[str, Any]]:
    """Export additional episode-indexed parquet artifacts such as advantages."""
    grouped: dict[str, list[pd.DataFrame]] = {}
    inventory: list[dict[str, Any]] = []
    frame_offsets: dict[int, int] = {}
    frame_cursor = 0
    for selection in selections:
        frame_offsets[selection.output_episode_index] = frame_cursor
        frame_cursor += selection.episode_length
    task_indices = {
        task: index
        for index, task in enumerate(dict.fromkeys(item.task for item in selections))
    }

    for selection in selections:
        source_meta = Path(selection.source_dataset) / "meta"
        for source_path in sorted(source_meta.glob("*.parquet")):
            name = source_path.name
            if name.startswith(("semantic_trace_", "phase_progress_")):
                continue
            frame = pd.read_parquet(source_path)
            if "episode_index" not in frame.columns:
                continue
            selected = frame[
                frame["episode_index"].astype("int64")
                == selection.source_episode_index
            ].copy()
            if selected.empty:
                raise ValueError(
                    f"{source_path}: no rows for selected episode "
                    f"{selection.source_episode_index}"
                )
            if (
                name.startswith("advantages_")
                and len(selected) != selection.episode_length
            ):
                raise ValueError(
                    f"{source_path}: advantage rows do not cover every selected frame"
                )
            selected["episode_index"] = selection.output_episode_index
            if "task_index" in selected.columns:
                selected["task_index"] = task_indices[selection.task]
            if "index" in selected.columns and "frame_index" in selected.columns:
                selected["index"] = (
                    frame_offsets[selection.output_episode_index]
                    + selected["frame_index"].astype("int64")
                )
            grouped.setdefault(name, []).append(selected)
            inventory.append(
                {
                    "artifact": name,
                    "task": selection.task,
                    "source_dataset": selection.source_dataset,
                    "source_episode_index": selection.source_episode_index,
                    "output_episode_index": selection.output_episode_index,
                    "rows": len(selected),
                }
            )

    for name, frames in grouped.items():
        combined = pd.concat(frames, ignore_index=True)
        sort_columns = [
            column
            for column in ("episode_index", "frame_index")
            if column in combined.columns
        ]
        if sort_columns:
            combined = combined.sort_values(sort_columns).reset_index(drop=True)
        combined.to_parquet(output_dataset / "meta" / name, index=False)
    (output_dataset / "meta" / "exported_sidecar_inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return inventory


def validate_export(
    output_dataset: Path,
    selections: list[EpisodeSelection],
) -> dict[str, Any]:
    """Validate episode, frame, task, trace, audit, video, and provenance coverage."""
    info = _read_json(output_dataset / "meta" / "info.json")
    episodes = _read_jsonl(output_dataset / "meta" / "episodes.jsonl")
    tasks = _read_jsonl(output_dataset / "meta" / "tasks.jsonl")
    expected_episodes = len(selections)
    expected_frames = sum(selection.episode_length for selection in selections)
    expected_videos = sum(selection.source_video_files for selection in selections)
    expected_task_count = len({selection.task for selection in selections})

    if [int(row["episode_index"]) for row in episodes] != list(
        range(expected_episodes)
    ):
        raise ValueError("exported episode indices are not contiguous")
    if int(info["total_episodes"]) != expected_episodes:
        raise ValueError("exported info has the wrong episode count")
    if int(info["total_frames"]) != expected_frames:
        raise ValueError("exported info has the wrong frame count")
    if (
        int(info["total_tasks"]) != expected_task_count
        or len(tasks) != expected_task_count
    ):
        raise ValueError("exported task metadata has the wrong task count")

    frame_cursor = 0
    for selection in selections:
        matches = sorted(
            output_dataset.glob(
                f"data/**/episode_{selection.output_episode_index:06d}.parquet"
            )
        )
        if len(matches) != 1:
            raise ValueError(
                f"output episode {selection.output_episode_index}: "
                f"expected one parquet, found {len(matches)}"
            )
        table = pq.read_table(matches[0])
        if table.num_rows != selection.episode_length:
            raise ValueError(
                f"output episode {selection.output_episode_index}: frame count mismatch"
            )
        if "episode_index" in table.column_names and set(
            int(value) for value in table["episode_index"].to_pylist()
        ) != {selection.output_episode_index}:
            raise ValueError("data parquet contains a wrong episode index")
        if "index" in table.column_names:
            expected_indices = list(
                range(frame_cursor, frame_cursor + selection.episode_length)
            )
            if [int(value) for value in table["index"].to_pylist()] != expected_indices:
                raise ValueError("global frame indices are not contiguous")
        frame_cursor += selection.episode_length

    phase_path = output_dataset / "meta" / "phase_progress_multitask.parquet"
    if not phase_path.is_file():
        raise FileNotFoundError(phase_path)
    phase = pd.read_parquet(phase_path)
    if len(phase) != expected_frames:
        raise ValueError("phase-progress trace does not cover every frame")
    if phase.groupby("episode_index").size().to_dict() != {
        selection.output_episode_index: selection.episode_length
        for selection in selections
    }:
        raise ValueError("phase-progress trace has incomplete episode coverage")

    audit_rows = sum(
        len(pd.read_csv(path))
        for path in (output_dataset / "meta").glob("semantic_trace_*_audit.csv")
    )
    if audit_rows != expected_episodes:
        raise ValueError("semantic audits do not contain one row per episode")
    provenance = _read_jsonl(
        output_dataset / "meta" / "episode_provenance.jsonl"
    )
    if len(provenance) != expected_episodes:
        raise ValueError("provenance does not contain one row per episode")
    actual_videos = sum(
        1 for path in (output_dataset / "videos").glob("**/*") if path.is_file()
    )
    if actual_videos != expected_videos:
        raise ValueError(
            f"video copy mismatch: expected {expected_videos}, found {actual_videos}"
        )

    sidecar_inventory = _read_json(
        output_dataset / "meta" / "exported_sidecar_inventory.json"
    )
    expected_sidecar_rows: dict[str, int] = {}
    for row in sidecar_inventory:
        expected_sidecar_rows[row["artifact"]] = (
            expected_sidecar_rows.get(row["artifact"], 0) + int(row["rows"])
        )
    for name, row_count in expected_sidecar_rows.items():
        exported_sidecar = output_dataset / "meta" / name
        if not exported_sidecar.is_file():
            raise FileNotFoundError(exported_sidecar)
        if len(pd.read_parquet(exported_sidecar)) != row_count:
            raise ValueError(f"{name}: exported sidecar row count mismatch")

    return {
        "passed": True,
        "dataset": str(output_dataset),
        "tasks": expected_task_count,
        "episodes": expected_episodes,
        "frames": expected_frames,
        "successes": sum(selection.is_success for selection in selections),
        "failures": sum(not selection.is_success for selection in selections),
        "videos": actual_videos,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_checksums(output_dataset: Path) -> Path:
    """Write stable SHA-256 checksums for every exported dataset file."""
    checksum_path = output_dataset / "SHA256SUMS"
    files = sorted(
        path
        for path in output_dataset.rglob("*")
        if path.is_file() and path != checksum_path
    )
    with checksum_path.open("w", encoding="utf-8", newline="\n") as file:
        for path in files:
            relative = path.relative_to(output_dataset).as_posix()
            file.write(f"{_sha256(path)}  {relative}\n")
    return checksum_path


def create_archive(output_dataset: Path, archive_path: Path, *, overwrite: bool) -> str:
    """Create a gzip tarball and an adjacent checksum file."""
    if archive_path.exists() and not overwrite:
        raise FileExistsError(
            f"Archive already exists: {archive_path}; use --overwrite to replace it"
        )
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(output_dataset, arcname=output_dataset.name)
    digest = _sha256(archive_path)
    archive_path.with_suffix(archive_path.suffix + ".sha256").write_text(
        f"{digest}  {archive_path.name}\n",
        encoding="utf-8",
    )
    return digest


def export_reproducibility_sample(
    *,
    dataset_pattern: str,
    output_dataset: Path,
    tasks: list[str],
    episodes_per_task: int,
    seed: int,
    trainable_only: bool = True,
    overwrite: bool = False,
    archive_path: Path | None = None,
) -> dict[str, Any]:
    """Export, validate, checksum, and optionally archive the selected subset."""
    selections = select_episodes(
        dataset_pattern=dataset_pattern,
        tasks=tasks,
        episodes_per_task=episodes_per_task,
        seed=seed,
        trainable_only=trainable_only,
    )
    if len(selections) < 2:
        raise ValueError("at least two total episodes are required")
    output_resolved = output_dataset.resolve()
    sources = {
        Path(selection.source_dataset).resolve() for selection in selections
    }
    for source in sources:
        if output_resolved == source or source in output_resolved.parents:
            raise ValueError("output dataset must not be inside a source dataset")

    merge_multitask_datasets(
        [Path(selection.source_dataset) for selection in selections],
        output_dataset,
        overwrite=overwrite,
        episode_ranges=[
            (
                selection.source_episode_position,
                selection.source_episode_position + 1,
            )
            for selection in selections
        ],
    )
    _export_episode_sidecars(output_dataset, selections)
    _write_manifests(
        output_dataset,
        selections,
        dataset_pattern=dataset_pattern,
        seed=seed,
        trainable_only=trainable_only,
    )
    validation = validate_export(output_dataset, selections)
    validation_path = output_dataset / "reproducibility_validation.json"
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    checksum_path = write_checksums(output_dataset)
    result: dict[str, Any] = {
        **validation,
        "manifest": str(output_dataset / "reproducibility_manifest.json"),
        "checksums": str(checksum_path),
    }
    if archive_path is not None:
        result["archive"] = str(archive_path)
        result["archive_sha256"] = create_archive(
            output_dataset,
            archive_path,
            overwrite=overwrite,
        )
    return result


def main() -> None:
    """Parse command-line arguments and export the reproducibility subset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-pattern", required=True)
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-untrainable", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    tasks = args.tasks or [f"task{index}" for index in range(args.num_tasks)]
    result = export_reproducibility_sample(
        dataset_pattern=args.dataset_pattern,
        output_dataset=args.output_dataset,
        tasks=tasks,
        episodes_per_task=args.episodes_per_task,
        seed=args.seed,
        trainable_only=not args.include_untrainable,
        overwrite=args.overwrite,
        archive_path=args.archive,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
