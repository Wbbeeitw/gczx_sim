#!/usr/bin/env python
"""Validate collected LIBERO datasets and their semantic trace artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"expected a JSON object at {path}:{line_number}")
            records.append(record)
    return records


def _bool_values(series: pd.Series, field: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(series):
        values = series.to_numpy()
        if not np.isin(values, [0, 1]).all():
            raise ValueError(f"{field} contains values other than 0/1")
        return values.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin(["true", "false", "1", "0"]).all():
        raise ValueError(f"{field} contains non-boolean values")
    return normalized.isin(["true", "1"]).to_numpy(dtype=bool)


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    result = int(value)
    if float(value) != result:
        raise ValueError(f"{field} must be an integer")
    return result


def _record_error(errors: list[str], context: str, exc: Exception) -> None:
    errors.append(f"{context}: {type(exc).__name__}: {exc}")


def validate_collection(
    dataset_path: str | Path,
    *,
    expected_task_id: int | None = None,
    expected_episodes: int | None = None,
    min_successes: int = 0,
    min_trainable: int = 0,
) -> dict[str, Any]:
    """Validate one LIBERO rollout collection without modifying it."""
    dataset = Path(dataset_path)
    meta = dataset / "meta"
    errors: list[str] = []
    warnings: list[str] = []
    result: dict[str, Any] = {
        "dataset": str(dataset),
        "valid": False,
        "task_id": expected_task_id,
        "episodes": 0,
        "frames": 0,
        "successes": 0,
        "trainable": 0,
        "errors": errors,
        "warnings": warnings,
    }

    base_paths = {
        "info": meta / "info.json",
        "episodes": meta / "episodes.jsonl",
        "summary": dataset / "collection_summary.json",
    }
    missing_base = [str(path) for path in base_paths.values() if not path.is_file()]
    if missing_base:
        errors.append(f"missing required files: {', '.join(missing_base)}")
        return result

    try:
        info = _read_json(base_paths["info"])
        episodes = _read_jsonl(base_paths["episodes"])
        summary = _read_json(base_paths["summary"])
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        _record_error(errors, "failed to read base metadata", exc)
        return result

    try:
        task_id = _integer(summary["task_id"], "collection_summary.task_id")
        total_episodes = _integer(info["total_episodes"], "info.total_episodes")
        total_frames = _integer(info["total_frames"], "info.total_frames")
        requested_episodes = _integer(
            summary["num_episodes"], "collection_summary.num_episodes"
        )
    except (KeyError, TypeError, ValueError) as exc:
        _record_error(errors, "invalid collection metadata", exc)
        return result

    result.update(
        task_id=task_id,
        episodes=total_episodes,
        frames=total_frames,
    )
    if task_id not in range(10):
        errors.append(f"task_id must be in [0, 9], got {task_id}")
    if expected_task_id is not None and task_id != expected_task_id:
        errors.append(f"expected task_id={expected_task_id}, got {task_id}")
    if expected_episodes is not None and total_episodes != expected_episodes:
        errors.append(
            f"expected {expected_episodes} episodes, metadata reports {total_episodes}"
        )
    if requested_episodes != total_episodes:
        errors.append(
            "collection_summary.num_episodes does not match "
            f"info.total_episodes: {requested_episodes} != {total_episodes}"
        )
    if len(episodes) != total_episodes:
        errors.append(
            f"episodes.jsonl has {len(episodes)} rows, expected {total_episodes}"
        )

    episode_lengths: dict[int, int] = {}
    for row_number, episode in enumerate(episodes, start=1):
        try:
            episode_index = _integer(episode["episode_index"], "episode_index")
            episode_length = _integer(episode["length"], "length")
        except (KeyError, TypeError, ValueError) as exc:
            _record_error(errors, f"invalid episodes.jsonl row {row_number}", exc)
            continue
        if episode_index in episode_lengths:
            errors.append(f"duplicate episode_index={episode_index} in episodes.jsonl")
        if episode_length <= 0:
            errors.append(
                f"episode {episode_index} has non-positive length={episode_length}"
            )
        episode_lengths[episode_index] = episode_length
    expected_indices = set(range(total_episodes))
    if set(episode_lengths) != expected_indices:
        errors.append(
            "episodes.jsonl indices do not equal the contiguous range "
            f"0..{max(total_episodes - 1, 0)}"
        )
    if sum(episode_lengths.values()) != total_frames:
        errors.append(
            "sum of episode lengths does not match info.total_frames: "
            f"{sum(episode_lengths.values())} != {total_frames}"
        )

    trace_name = f"semantic_trace_task{task_id}"
    trace_paths = {
        "raw": meta / f"{trace_name}.parquet",
        "labels": meta / f"phase_progress_{trace_name}.parquet",
        "audit": meta / f"{trace_name}_audit.csv",
        "metadata": meta / f"{trace_name}_metadata.json",
    }
    missing_trace = [str(path) for path in trace_paths.values() if not path.is_file()]
    if missing_trace:
        errors.append(f"missing semantic trace files: {', '.join(missing_trace)}")
        return result

    try:
        raw_trace = pd.read_parquet(trace_paths["raw"])
        labels = pd.read_parquet(trace_paths["labels"])
        audit = pd.read_csv(trace_paths["audit"])
        _read_json(trace_paths["metadata"])
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        _record_error(errors, "failed to read semantic trace artifacts", exc)
        return result

    required_label_columns = {
        "episode_index",
        "frame_index",
        "phase",
        "phase_progress",
        "global_progress",
        "is_success",
    }
    required_audit_columns = {
        "episode_index",
        "episode_length",
        "is_success",
        "b1_frame",
        "b2_frame",
        "b3_frame",
        "trainable",
    }
    missing_label_columns = required_label_columns - set(labels.columns)
    missing_audit_columns = required_audit_columns - set(audit.columns)
    if missing_label_columns:
        errors.append(f"phase labels missing columns: {sorted(missing_label_columns)}")
    if missing_audit_columns:
        errors.append(f"audit CSV missing columns: {sorted(missing_audit_columns)}")
    if errors:
        return result

    if len(raw_trace) != total_frames:
        errors.append(f"raw trace has {len(raw_trace)} rows, expected {total_frames}")
    if len(labels) != total_frames:
        errors.append(f"phase labels have {len(labels)} rows, expected {total_frames}")
    if len(audit) != total_episodes:
        errors.append(f"audit CSV has {len(audit)} rows, expected {total_episodes}")

    frame_key_columns = ["episode_index", "frame_index"]
    if not set(frame_key_columns).issubset(raw_trace.columns):
        errors.append("raw trace is missing episode_index/frame_index")
    else:
        raw_keys = raw_trace[frame_key_columns].astype("int64")
        label_keys = labels[frame_key_columns].astype("int64")
        if raw_keys.duplicated().any():
            errors.append("raw trace contains duplicate episode/frame keys")
        if label_keys.duplicated().any():
            errors.append("phase labels contain duplicate episode/frame keys")
        if set(map(tuple, raw_keys.to_numpy())) != set(
            map(tuple, label_keys.to_numpy())
        ):
            errors.append("raw trace and phase labels have different frame keys")

    label_successes: dict[int, bool] = {}
    for episode_index, frame in labels.groupby("episode_index", sort=True):
        episode_id = int(episode_index)
        ordered = frame.sort_values("frame_index")
        frame_indices = ordered["frame_index"].to_numpy(dtype=np.int64)
        phases = ordered["phase"].to_numpy(dtype=np.int64)
        try:
            success_values = _bool_values(
                ordered["is_success"], f"labels episode {episode_id} is_success"
            )
        except ValueError as exc:
            _record_error(errors, f"invalid episode {episode_id}", exc)
            continue
        expected_length = episode_lengths.get(episode_id)
        if expected_length is None:
            errors.append(f"phase labels contain unknown episode_index={episode_id}")
            continue
        if not np.array_equal(frame_indices, np.arange(expected_length)):
            errors.append(
                f"episode {episode_id} frame indices are not contiguous "
                f"0..{expected_length - 1}"
            )
        if not np.isin(phases, [0, 1, 2, 3]).all():
            errors.append(f"episode {episode_id} contains a phase outside [0, 3]")
        phase_steps = np.diff(phases)
        if (phase_steps < 0).any() or (phase_steps > 1).any():
            errors.append(f"episode {episode_id} has a non-monotonic or skipped phase")
        if not (success_values == success_values[0]).all():
            errors.append(f"episode {episode_id} has inconsistent is_success labels")
        is_success = bool(success_values[-1])
        label_successes[episode_id] = is_success
        if is_success and (len(phases) == 0 or int(phases.max()) != 3):
            warnings.append(
                f"successful episode {episode_id} does not reach phase 3"
            )
        if not is_success and len(phases) and int(phases.max()) >= 3:
            errors.append(f"failed episode {episode_id} reaches success phase 3")
        for field in ("phase_progress", "global_progress"):
            values = ordered[field].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                errors.append(f"episode {episode_id} has invalid {field} values")

    audit_indices: set[int] = set()
    audit_successes = 0
    trainable_count = 0
    try:
        audit_success_values = _bool_values(audit["is_success"], "audit.is_success")
        audit_trainable_values = _bool_values(audit["trainable"], "audit.trainable")
    except ValueError as exc:
        _record_error(errors, "invalid audit CSV", exc)
        audit_success_values = np.zeros(len(audit), dtype=bool)
        audit_trainable_values = np.zeros(len(audit), dtype=bool)
    for position, (_, row) in enumerate(audit.iterrows()):
        try:
            episode_id = _integer(row["episode_index"], "audit.episode_index")
            episode_length = _integer(row["episode_length"], "audit.episode_length")
        except (TypeError, ValueError) as exc:
            _record_error(errors, f"invalid audit row {position + 1}", exc)
            continue
        if episode_id in audit_indices:
            errors.append(f"audit CSV contains duplicate episode_index={episode_id}")
        audit_indices.add(episode_id)
        expected_length = episode_lengths.get(episode_id)
        if expected_length is None or episode_length != expected_length:
            errors.append(
                f"audit episode {episode_id} length={episode_length}, "
                f"expected {expected_length}"
            )
        is_success = bool(audit_success_values[position])
        if label_successes.get(episode_id) != is_success:
            errors.append(f"audit/label success mismatch for episode {episode_id}")
        boundaries: list[tuple[str, int]] = []
        for field in ("b1_frame", "b2_frame", "b3_frame"):
            value = row[field]
            if pd.isna(value):
                continue
            try:
                boundary = _integer(value, f"audit.{field}")
            except (TypeError, ValueError) as exc:
                _record_error(errors, f"invalid audit episode {episode_id}", exc)
                continue
            if not 0 <= boundary < episode_length:
                errors.append(
                    f"audit episode {episode_id} {field}={boundary} is out of bounds"
                )
            boundaries.append((field, boundary))
        if any(right[1] <= left[1] for left, right in zip(boundaries, boundaries[1:])):
            errors.append(f"audit episode {episode_id} boundaries are not ordered")
        has_b3 = not pd.isna(row["b3_frame"])
        if has_b3 and not is_success:
            errors.append(
                f"failed audit episode {episode_id} has a b3 success boundary"
            )
        if is_success and not has_b3:
            warnings.append(
                f"successful audit episode {episode_id} has no b3 boundary"
            )
        audit_successes += int(is_success)
        trainable_count += int(audit_trainable_values[position])

    if audit_indices != expected_indices:
        errors.append("audit episode indices do not match episodes.jsonl")
    result["successes"] = audit_successes
    result["trainable"] = trainable_count
    try:
        summary_successes = _integer(
            summary["successes"], "collection_summary.successes"
        )
        summary_success_rate = float(summary["success_rate"])
    except (KeyError, TypeError, ValueError) as exc:
        _record_error(errors, "invalid success summary", exc)
    else:
        expected_rate = audit_successes / total_episodes if total_episodes else 0.0
        if summary_successes != audit_successes:
            errors.append(
                f"collection_summary.successes={summary_successes}, "
                f"audit reports {audit_successes}"
            )
        if not math.isclose(summary_success_rate, expected_rate, abs_tol=1e-9):
            errors.append(
                "collection_summary.success_rate does not match audit: "
                f"{summary_success_rate} != {expected_rate}"
            )
    if audit_successes < min_successes:
        errors.append(
            f"successes={audit_successes} is below required minimum {min_successes}"
        )
    if trainable_count < min_trainable:
        errors.append(
            f"trainable episodes={trainable_count} is below required minimum "
            f"{min_trainable}"
        )
    if audit_successes == 0:
        warnings.append("collection contains no successful episode")
    untrainable_count = total_episodes - trainable_count
    if untrainable_count:
        warnings.append(
            f"collection contains {untrainable_count} untrainable trace rows"
        )

    result["valid"] = not errors
    return result


def _print_result(result: dict[str, Any]) -> None:
    status = "PASS" if result["valid"] else "FAIL"
    print(
        f"[{status}] task{result['task_id']} episodes={result['episodes']} "
        f"frames={result['frames']} successes={result['successes']} "
        f"trainable={result['trainable']} dataset={result['dataset']}"
    )
    for warning in result["warnings"]:
        print(f"  WARN: {warning}")
    for error in result["errors"]:
        print(f"  ERROR: {error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--expected-task-id", type=int)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--min-successes", type=int, default=0)
    parser.add_argument("--min-trainable", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="Print JSON results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_task_id is not None and len(args.datasets) != 1:
        raise ValueError("--expected-task-id can only validate one dataset")
    results = [
        validate_collection(
            dataset,
            expected_task_id=args.expected_task_id,
            expected_episodes=args.expected_episodes,
            min_successes=args.min_successes,
            min_trainable=args.min_trainable,
        )
        for dataset in args.datasets
    ]
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        for result in results:
            _print_result(result)
        passed = sum(bool(result["valid"]) for result in results)
        print(
            f"validated={len(results)} passed={passed} "
            f"failed={len(results) - passed}"
        )
    if not all(result["valid"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
