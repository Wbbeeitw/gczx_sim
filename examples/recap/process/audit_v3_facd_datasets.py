#!/usr/bin/env python
"""Strict audits for round1 v3 FACD task pools, merged data, and labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TASKS = tuple(f"task{task_id}" for task_id in range(10))
FRAME_KEYS = ["episode_index", "frame_index"]


def _parse_mapping(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected TASK=PATH, got {value!r}")
    task, raw_path = value.split("=", 1)
    if not task or not raw_path:
        raise ValueError(f"Expected TASK=PATH, got {value!r}")
    return task, Path(raw_path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _bool_series(series: pd.Series, field: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        if not series.isin([0, 1]).all():
            raise ValueError(f"{field} contains values other than 0/1")
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin(["true", "false", "1", "0"]).all():
        raise ValueError(f"{field} contains non-boolean values")
    return normalized.isin(["true", "1"])


def _episode_lengths(dataset: Path) -> tuple[dict[int, int], dict[str, Any]]:
    info = _read_json(dataset / "meta" / "info.json")
    episodes = _read_jsonl(dataset / "meta" / "episodes.jsonl")
    lengths = {
        int(record["episode_index"]): int(record["length"]) for record in episodes
    }
    total_episodes = int(info["total_episodes"])
    total_frames = int(info["total_frames"])
    if len(episodes) != total_episodes or set(lengths) != set(range(total_episodes)):
        raise ValueError(f"{dataset}: episode metadata is not contiguous")
    if sum(lengths.values()) != total_frames:
        raise ValueError(f"{dataset}: episode lengths do not match total_frames")
    return lengths, info


def _data_frame(dataset: Path, *, include_task_index: bool = False) -> pd.DataFrame:
    paths = sorted(dataset.glob("data/**/episode_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"{dataset}: no episode parquet files")
    extra_columns = ["task_index"] if include_task_index else []
    columns = [*FRAME_KEYS, *extra_columns]
    frames = [pd.read_parquet(path, columns=columns) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    for column in columns:
        frame[column] = frame[column].astype("int64")
    return frame


def _frame_key_index(frame: pd.DataFrame, name: str) -> pd.MultiIndex:
    missing = set(FRAME_KEYS).difference(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing frame columns {sorted(missing)}")
    keys = frame[FRAME_KEYS].astype("int64")
    if keys.duplicated().any():
        raise ValueError(f"{name}: duplicate episode/frame keys")
    return pd.MultiIndex.from_frame(keys)


def _check_frame_coverage(
    expected: pd.DataFrame,
    observed: pd.DataFrame,
    name: str,
) -> None:
    expected_keys = _frame_key_index(expected, "dataset parquet")
    observed_keys = _frame_key_index(observed, name)
    if len(expected_keys) != len(observed_keys) or not expected_keys.equals(
        observed_keys
    ):
        missing = expected_keys.difference(observed_keys)
        extra = observed_keys.difference(expected_keys)
        raise ValueError(
            f"{name}: frame coverage mismatch missing={len(missing)} "
            f"extra={len(extra)}"
        )


def _check_contiguous_frames(
    frame: pd.DataFrame,
    lengths: dict[int, int],
    name: str,
) -> None:
    for episode_index, episode_frame in frame.groupby("episode_index", sort=True):
        episode_id = int(episode_index)
        expected_length = lengths.get(episode_id)
        indices = np.sort(episode_frame["frame_index"].to_numpy(dtype=np.int64))
        if expected_length is None or not np.array_equal(
            indices, np.arange(expected_length)
        ):
            raise ValueError(f"{name}: episode {episode_id} is not frame-complete")


def audit_task_pools(
    task_datasets: dict[str, Path],
    *,
    rollout_root: Path,
    expert_root: Path,
    expected_episodes: int = 30,
    rollout_episodes: int = 20,
    expert_episodes: int = 10,
) -> dict[str, Any]:
    """Audit task-pool provenance and every data/trace frame key."""

    if set(task_datasets) != set(TASKS):
        raise ValueError(f"expected task mappings {list(TASKS)}")
    expected_full_positive = list(range(rollout_episodes, expected_episodes))
    seen_sources: set[tuple[str, int]] = set()
    reports: dict[str, Any] = {}
    total_frames = 0
    total_failure_frames = 0
    total_successes = 0

    for task in TASKS:
        dataset = task_datasets[task]
        lengths, info = _episode_lengths(dataset)
        if int(info["total_episodes"]) != expected_episodes:
            raise ValueError(f"{task}: expected {expected_episodes} episodes")

        data = _data_frame(dataset)
        if len(data) != int(info["total_frames"]):
            raise ValueError(f"{task}: data parquet row count mismatch")
        _check_contiguous_frames(data, lengths, f"{task} data")
        data = data.sort_values(FRAME_KEYS).reset_index(drop=True)

        raw_path = dataset / "meta" / f"semantic_trace_{task}.parquet"
        labels_path = (
            dataset / "meta" / f"phase_progress_semantic_trace_{task}.parquet"
        )
        audit_path = dataset / "meta" / f"semantic_trace_{task}_audit.csv"
        metadata_path = dataset / "meta" / f"semantic_trace_{task}_metadata.json"
        for path in (raw_path, labels_path, audit_path, metadata_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        raw = pd.read_parquet(raw_path).sort_values(FRAME_KEYS).reset_index(drop=True)
        labels = (
            pd.read_parquet(labels_path)
            .sort_values(FRAME_KEYS)
            .reset_index(drop=True)
        )
        _check_frame_coverage(data, raw, f"{task} raw semantic trace")
        _check_frame_coverage(data, labels, f"{task} phase labels")
        _check_contiguous_frames(labels, lengths, f"{task} phase labels")

        provenance = _read_jsonl(dataset / "meta" / "episode_provenance.jsonl")
        if len(provenance) != expected_episodes:
            raise ValueError(f"{task}: provenance row count mismatch")
        provenance_by_episode = {
            int(record["episode_index"]): record for record in provenance
        }
        if set(provenance_by_episode) != set(range(expected_episodes)):
            raise ValueError(f"{task}: provenance episode ids are not contiguous")
        for episode_index in range(expected_episodes):
            record = provenance_by_episode[episode_index]
            expected_type = "rollout" if episode_index < rollout_episodes else "expert"
            if record.get("source_type") != expected_type:
                raise ValueError(
                    f"{task}: episode {episode_index} is not {expected_type}"
                )
            source_dataset = Path(str(record["source_dataset"])).resolve()
            expected_root = rollout_root if expected_type == "rollout" else expert_root
            if expected_root.resolve() not in source_dataset.parents:
                raise ValueError(
                    f"{task}: episode {episode_index} uses unexpected source "
                    f"{source_dataset}"
                )
            source_key = (str(source_dataset), int(record["source_episode_index"]))
            if source_key in seen_sources:
                raise ValueError(f"duplicate source episode: {source_key}")
            seen_sources.add(source_key)

        full_positive = json.loads(
            (dataset / "meta" / "full_positive_episodes.json").read_text(
                encoding="utf-8"
            )
        )
        if full_positive != expected_full_positive:
            raise ValueError(f"{task}: full_positive_episodes must be 20..29")

        audit = pd.read_csv(audit_path)
        if len(audit) != expected_episodes:
            raise ValueError(f"{task}: semantic audit row count mismatch")
        audit["episode_index"] = audit["episode_index"].astype("int64")
        audit_success = _bool_series(audit["is_success"], f"{task} audit success")
        audit_trainable = _bool_series(
            audit["trainable"], f"{task} audit trainable"
        )
        audit = audit.assign(
            is_success_bool=audit_success,
            trainable_bool=audit_trainable,
        ).set_index("episode_index")
        expert_audit = audit.loc[expected_full_positive]
        if not expert_audit["is_success_bool"].all():
            raise ValueError(f"{task}: an expert episode is not successful")
        if not expert_audit["trainable_bool"].all():
            raise ValueError(f"{task}: an expert episode is not trainable")

        label_success = _bool_series(labels["is_success"], f"{task} label success")
        labels = labels.assign(is_success_bool=label_success)
        success_by_episode = labels.groupby("episode_index")["is_success_bool"].last()
        for episode_index in expected_full_positive:
            if not bool(success_by_episode.loc[episode_index]):
                raise ValueError(f"{task}: expert label is not successful")
            if not bool(provenance_by_episode[episode_index].get("is_success")):
                raise ValueError(f"{task}: expert provenance is not successful")

        failure_frames = int((~labels["is_success_bool"]).sum())
        successes = int(success_by_episode.sum())
        reports[task] = {
            "dataset": str(dataset),
            "episodes": expected_episodes,
            "frames": int(info["total_frames"]),
            "rollout_episodes": rollout_episodes,
            "expert_episodes": expert_episodes,
            "successes": successes,
            "failures": expected_episodes - successes,
            "failure_frames": failure_frames,
            "failure_frame_ratio": failure_frames / len(labels),
            "trainable_episodes": int(audit["trainable_bool"].sum()),
            "full_positive_episodes": full_positive,
        }
        total_frames += int(info["total_frames"])
        total_failure_frames += failure_frames
        total_successes += successes

    return {
        "passed": True,
        "mode": "task_pools",
        "tasks": reports,
        "aggregate": {
            "tasks": len(reports),
            "episodes": expected_episodes * len(reports),
            "frames": total_frames,
            "successes": total_successes,
            "failures": expected_episodes * len(reports) - total_successes,
            "failure_frames": total_failure_frames,
            "failure_frame_ratio": total_failure_frames / total_frames,
        },
    }


def audit_merged_pool(
    task_datasets: dict[str, Path],
    merged_dataset: Path,
    *,
    episodes_per_task: int = 30,
) -> dict[str, Any]:
    """Audit the merged 300-episode dataset and task-specific ranges."""

    lengths, info = _episode_lengths(merged_dataset)
    expected_episodes = episodes_per_task * len(task_datasets)
    if int(info["total_episodes"]) != expected_episodes:
        raise ValueError(f"merged pool must contain {expected_episodes} episodes")
    if int(info.get("total_tasks", -1)) != len(task_datasets):
        raise ValueError("merged pool task count mismatch")

    data = _data_frame(merged_dataset, include_task_index=True)
    if len(data) != int(info["total_frames"]):
        raise ValueError("merged data parquet row count mismatch")
    _check_contiguous_frames(data, lengths, "merged data")
    data = data.sort_values(FRAME_KEYS).reset_index(drop=True)
    phase = pd.read_parquet(
        merged_dataset / "meta" / "phase_progress_multitask.parquet"
    ).sort_values(FRAME_KEYS).reset_index(drop=True)
    _check_frame_coverage(data, phase, "merged phase_progress_multitask")

    task_reports: dict[str, Any] = {}
    task_indices: set[int] = set()
    expected_total_frames = 0
    for task_number, task in enumerate(TASKS):
        source_info = _read_json(task_datasets[task] / "meta" / "info.json")
        expected_frames = int(source_info["total_frames"])
        start = task_number * episodes_per_task
        end = start + episodes_per_task
        task_data = data[data["episode_index"].between(start, end - 1)].copy()
        if len(task_data) != expected_frames:
            raise ValueError(f"{task}: merged frame range has wrong size")
        unique_task_indices = set(task_data["task_index"].astype(int))
        if len(unique_task_indices) != 1:
            raise ValueError(f"{task}: merged range has multiple task_index values")
        task_index = next(iter(unique_task_indices))
        if task_index in task_indices:
            raise ValueError(f"{task}: duplicate merged task_index={task_index}")
        task_indices.add(task_index)

        raw = pd.read_parquet(
            merged_dataset / "meta" / f"semantic_trace_{task}.parquet"
        ).sort_values(FRAME_KEYS).reset_index(drop=True)
        labels = pd.read_parquet(
            merged_dataset
            / "meta"
            / f"phase_progress_semantic_trace_{task}.parquet"
        ).sort_values(FRAME_KEYS).reset_index(drop=True)
        _check_frame_coverage(task_data, raw, f"merged {task} raw trace")
        _check_frame_coverage(task_data, labels, f"merged {task} phase labels")

        audit = pd.read_csv(
            merged_dataset / "meta" / f"semantic_trace_{task}_audit.csv"
        )
        audit_ids = set(audit["episode_index"].astype(int))
        if audit_ids != set(range(start, end)):
            raise ValueError(f"{task}: merged semantic audit range mismatch")
        task_reports[task] = {
            "episode_start": start,
            "episode_end": end,
            "episodes": episodes_per_task,
            "frames": expected_frames,
            "task_index": task_index,
        }
        expected_total_frames += expected_frames

    if expected_total_frames != int(info["total_frames"]):
        raise ValueError("merged total frames differ from task-pool totals")
    if len(task_indices) != len(task_datasets):
        raise ValueError("merged task indices are incomplete")

    return {
        "passed": True,
        "mode": "merged_pool",
        "dataset": str(merged_dataset),
        "episodes": expected_episodes,
        "frames": int(info["total_frames"]),
        "tasks": task_reports,
    }


def audit_facd_labels(
    task_datasets: dict[str, Path],
    export_reports: dict[str, Path],
    *,
    advantage_tag: str,
    positive_quantile: float = 0.3,
    failure_positive_cap: float = 0.2,
) -> dict[str, Any]:
    """Audit final binary FACD labels before downstream policy training."""

    if set(task_datasets) != set(export_reports):
        raise ValueError("task datasets and export reports do not match")
    reports: dict[str, Any] = {}
    total_frames = 0
    total_positive = 0

    for task in TASKS:
        dataset = task_datasets[task]
        data = _data_frame(dataset).sort_values(FRAME_KEYS).reset_index(drop=True)
        labels = pd.read_parquet(
            dataset / "meta" / f"advantages_{advantage_tag}.parquet"
        ).sort_values(FRAME_KEYS).reset_index(drop=True)
        _check_frame_coverage(data, labels, f"{task} FACD advantages")
        positive = _bool_series(labels["advantage"], f"{task} advantage")
        labels = labels.assign(positive=positive)
        phase_labels = pd.read_parquet(
            dataset / "meta" / f"phase_progress_semantic_trace_{task}.parquet",
            columns=[*FRAME_KEYS, "is_success"],
        ).sort_values(FRAME_KEYS).reset_index(drop=True)
        _check_frame_coverage(data, phase_labels, f"{task} FACD success labels")

        full_positive = set(
            json.loads(
                (dataset / "meta" / "full_positive_episodes.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        expert = labels["episode_index"].astype(int).isin(full_positive)
        rollout = ~expert
        if not labels.loc[expert, "positive"].all():
            raise ValueError(f"{task}: not every expert frame is positive")

        rollout_frames = int(rollout.sum())
        rollout_budget = int(round(positive_quantile * rollout_frames))
        rollout_positive = labels["positive"] & rollout
        if int(rollout_positive.sum()) != rollout_budget:
            raise ValueError(f"{task}: rollout top30 budget is not filled")

        success_frame = _bool_series(
            phase_labels["is_success"], f"{task} is_success"
        )
        failure_positive = rollout_positive & ~success_frame
        failure_ratio = (
            int(failure_positive.sum()) / int(rollout_positive.sum())
            if rollout_positive.any()
            else 0.0
        )
        if failure_ratio > failure_positive_cap + 1e-12:
            raise ValueError(f"{task}: failure-positive cap is violated")

        export_report = _read_json(export_reports[task])
        required_flags = {
            "success_gate": True,
            "demo_backstop": True,
            "rollout_budget_filled": True,
            "failure_cap_satisfied": True,
        }
        for field, expected in required_flags.items():
            if bool(export_report.get(field)) is not expected:
                raise ValueError(f"{task}: export report {field} is not {expected}")
        if int(export_report.get("num_full_positive_episodes", -1)) != len(
            full_positive
        ):
            raise ValueError(
                f"{task}: export report expert count does not match metadata"
            )

        positive_frames = int(labels["positive"].sum())
        reports[task] = {
            "frames": len(labels),
            "expert_frames": int(expert.sum()),
            "expert_positive_ratio": float(labels.loc[expert, "positive"].mean()),
            "rollout_frames": rollout_frames,
            "rollout_positive_budget": rollout_budget,
            "rollout_positive_frames": int(rollout_positive.sum()),
            "rollout_positive_ratio": int(rollout_positive.sum()) / rollout_frames,
            "rollout_failure_positive_frames": int(failure_positive.sum()),
            "rollout_failure_positive_ratio": failure_ratio,
            "total_positive_frames": positive_frames,
        }
        total_frames += len(labels)
        total_positive += positive_frames

    return {
        "passed": True,
        "mode": "facd_labels",
        "advantage_tag": advantage_tag,
        "positive_quantile": positive_quantile,
        "failure_positive_cap": failure_positive_cap,
        "tasks": reports,
        "aggregate": {
            "tasks": len(reports),
            "frames": total_frames,
            "positive_frames": total_positive,
            "positive_ratio": total_positive / total_frames,
        },
    }


def _write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _add_task_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task-dataset", action="append", required=True)
    parser.add_argument("--output-path", required=True, type=Path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    pools = subparsers.add_parser("task-pools")
    _add_task_arguments(pools)
    pools.add_argument("--rollout-root", required=True, type=Path)
    pools.add_argument("--expert-root", required=True, type=Path)

    merged = subparsers.add_parser("merged")
    _add_task_arguments(merged)
    merged.add_argument("--merged-dataset", required=True, type=Path)

    facd = subparsers.add_parser("facd")
    _add_task_arguments(facd)
    facd.add_argument("--export-report", action="append", required=True)
    facd.add_argument("--advantage-tag", required=True)
    facd.add_argument("--positive-quantile", type=float, default=0.3)
    facd.add_argument("--failure-positive-cap", type=float, default=0.2)
    args = parser.parse_args()

    task_datasets = dict(_parse_mapping(value) for value in args.task_dataset)
    if args.mode == "task-pools":
        report = audit_task_pools(
            task_datasets,
            rollout_root=args.rollout_root,
            expert_root=args.expert_root,
        )
    elif args.mode == "merged":
        report = audit_merged_pool(task_datasets, args.merged_dataset)
    else:
        export_reports = dict(
            _parse_mapping(value) for value in args.export_report
        )
        report = audit_facd_labels(
            task_datasets,
            export_reports,
            advantage_tag=args.advantage_tag,
            positive_quantile=args.positive_quantile,
            failure_positive_cap=args.failure_positive_cap,
        )
    _write_report(report, args.output_path)


if __name__ == "__main__":
    main()
