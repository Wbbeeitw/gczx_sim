"""Tests for LIBERO rollout collection validation."""

from __future__ import annotations

import json

import pandas as pd

from examples.recap.process.validate_libero_rollout_collection import (
    validate_collection,
)


def _write_collection(tmp_path):
    dataset = tmp_path / "task5"
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps({"total_episodes": 2, "total_frames": 7}), encoding="utf-8"
    )
    episodes = [
        {"episode_index": 0, "length": 4},
        {"episode_index": 1, "length": 3},
    ]
    (meta / "episodes.jsonl").write_text(
        "".join(f"{json.dumps(row)}\n" for row in episodes), encoding="utf-8"
    )
    (dataset / "collection_summary.json").write_text(
        json.dumps(
            {
                "task_id": 5,
                "num_episodes": 2,
                "successes": 1,
                "success_rate": 0.5,
                "semantic_trace": {},
            }
        ),
        encoding="utf-8",
    )
    labels = pd.DataFrame(
        {
            "episode_index": [0, 0, 0, 0, 1, 1, 1],
            "frame_index": [0, 1, 2, 3, 0, 1, 2],
            "phase": [0, 1, 2, 3, 0, 1, 1],
            "phase_progress": [0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 1.0],
            "global_progress": [0.0, 0.25, 0.5, 1.0, 0.0, 0.25, 0.5],
            "is_success": [True, True, True, True, False, False, False],
        }
    )
    labels.to_parquet(meta / "phase_progress_semantic_trace_task5.parquet")
    labels[["episode_index", "frame_index", "is_success"]].to_parquet(
        meta / "semantic_trace_task5.parquet"
    )
    pd.DataFrame(
        {
            "episode_index": [0, 1],
            "episode_length": [4, 3],
            "is_success": [True, False],
            "b1_frame": [1, 1],
            "b2_frame": [2, None],
            "b3_frame": [3, None],
            "trainable": [True, True],
        }
    ).to_csv(meta / "semantic_trace_task5_audit.csv", index=False)
    (meta / "semantic_trace_task5_metadata.json").write_text("{}", encoding="utf-8")
    return dataset


def test_validate_collection_accepts_aligned_semantic_trace(tmp_path) -> None:
    dataset = _write_collection(tmp_path)

    result = validate_collection(
        dataset,
        expected_task_id=5,
        expected_episodes=2,
        min_successes=1,
        min_trainable=2,
    )

    assert result["valid"]
    assert result["successes"] == 1
    assert result["trainable"] == 2
    assert result["errors"] == []


def test_validate_collection_rejects_success_phase_on_failed_episode(tmp_path) -> None:
    dataset = _write_collection(tmp_path)
    labels_path = dataset / "meta" / "phase_progress_semantic_trace_task5.parquet"
    labels = pd.read_parquet(labels_path)
    failed_terminal = (labels["episode_index"] == 1) & (labels["frame_index"] == 2)
    labels.loc[failed_terminal, "phase"] = 3
    labels.to_parquet(labels_path)

    result = validate_collection(dataset)

    assert not result["valid"]
    assert any(
        "failed episode 1 reaches success phase 3" in error
        for error in result["errors"]
    )
