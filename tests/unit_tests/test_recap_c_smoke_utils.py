"""Focused tests for configurable ReCap C-stage smoke utilities."""

from __future__ import annotations

import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from examples.recap.process.merge_lerobot_multitask_datasets import (
    _parse_dataset_arg,
    merge_multitask_datasets,
)
from examples.recap.process.split_feature_cache_by_task import (
    split_feature_caches_by_task,
)
from examples.recap.rounds.run_round import (
    _normalize_demo_dataset,
    _step_merge_multitask,
    _task_feature_split_args,
)


def _write_jsonl(path, rows) -> None:
    path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )


def _write_rollout_dataset(
    path,
    episode_task_indices: list[int] | None = None,
) -> None:
    if episode_task_indices is None:
        episode_task_indices = [0, 0, 0]
    episodes = len(episode_task_indices)
    meta = path / "meta"
    data = path / "data" / "chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    info = {
        "features": {"value": {"dtype": "int64", "shape": [1]}},
        "robot_type": "franka_panda",
        "fps": 10,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "total_episodes": episodes,
        "total_frames": episodes * 2,
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    task_names = {
        task_index: f"task{task_index}" for task_index in episode_task_indices
    }
    _write_jsonl(
        meta / "tasks.jsonl",
        [
            {"task_index": task_index, "task": task_name}
            for task_index, task_name in sorted(task_names.items())
        ],
    )
    episode_rows = [
        {
            "episode_index": episode_index,
            "length": 2,
            "tasks": [task_names[episode_task_indices[episode_index]]],
        }
        for episode_index in range(episodes)
    ]
    _write_jsonl(meta / "episodes.jsonl", episode_rows)
    _write_jsonl(
        meta / "episodes_stats.jsonl",
        [
            {"episode_index": episode_index, "stats": {}}
            for episode_index in range(episodes)
        ],
    )
    trace_rows = []
    for episode_index, task_index in enumerate(episode_task_indices):
        table = pa.table(
            {
                "episode_index": [episode_index, episode_index],
                "frame_index": [0, 1],
                "index": [episode_index * 2, episode_index * 2 + 1],
                "task_index": [task_index, task_index],
                "value": [episode_index * 10, episode_index * 10 + 1],
            }
        )
        pq.write_table(table, data / f"episode_{episode_index:06d}.parquet")
        trace_rows.extend(
            [
                {
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "phase": frame_index,
                    "phase_progress": float(frame_index),
                    "global_progress": float(frame_index) / 4,
                    "is_success": episode_index % 2 == 0,
                }
                for frame_index in range(2)
            ]
        )
    trace = pd.DataFrame(trace_rows)
    trace.to_parquet(meta / "semantic_trace_task0.parquet", index=False)
    trace.to_parquet(
        meta / "phase_progress_semantic_trace_task0.parquet",
        index=False,
    )
    pd.DataFrame(
        {
            "episode_index": list(range(episodes)),
            "is_success": [index % 2 == 0 for index in range(episodes)],
        }
    ).to_csv(meta / "semantic_trace_task0_audit.csv", index=False)
    (meta / "semantic_trace_task0_metadata.json").write_text("{}", encoding="utf-8")
    (path / "collection_summary.json").write_text(
        json.dumps({"successes": 2, "success_rate": 2 / 3}),
        encoding="utf-8",
    )


def _feature_cache(episode_ids: list[int]) -> dict:
    rows = len(episode_ids)
    return {
        "features": torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2),
        "episode_index": torch.tensor(episode_ids),
        "frame_index": torch.zeros(rows, dtype=torch.long),
        "phase": torch.zeros(rows, dtype=torch.long),
        "phase_progress": torch.zeros(rows),
        "global_progress": torch.zeros(rows),
        "raw_logits": torch.zeros((rows, 5)),
        "raw_value": torch.zeros(rows),
        "atoms": torch.linspace(-1, 0, 5),
    }


def test_demo_dataset_slice_normalization(tmp_path) -> None:
    dataset = tmp_path / "demo"
    (dataset / "meta").mkdir(parents=True)
    _write_jsonl(
        dataset / "meta" / "episodes.jsonl",
        [{"episode_index": index} for index in range(70)],
    )

    normalized = _normalize_demo_dataset(
        "task5",
        {"path": str(dataset), "start": 0, "end": 15},
    )

    assert normalized["episodes"] == 15
    assert normalized["input"] == f"{dataset}::0:15"
    assert _parse_dataset_arg(normalized["input"]) == (dataset, (0, 15))


def test_merge_multitask_dataset_slices_reindexes_all_sidecars(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "merged"
    _write_rollout_dataset(source, [0, 0, 1])

    summary = merge_multitask_datasets(
        [source, source],
        output,
        episode_ranges=[(0, 1), (2, 3)],
    )

    episodes = [
        json.loads(line)
        for line in (output / "meta" / "episodes.jsonl").read_text().splitlines()
    ]
    assert [row["episode_index"] for row in episodes] == [0, 1]
    first = pq.read_table(output / "data" / "chunk-000" / "episode_000000.parquet")
    second = pq.read_table(output / "data" / "chunk-000" / "episode_000001.parquet")
    assert first.column("index").to_pylist() == [0, 1]
    assert second.column("index").to_pylist() == [2, 3]
    assert second.column("value").to_pylist() == [20, 21]
    assert first.column("task_index").to_pylist() == [0, 0]
    assert second.column("task_index").to_pylist() == [1, 1]
    tasks = [
        json.loads(line)
        for line in (output / "meta" / "tasks.jsonl").read_text().splitlines()
    ]
    assert tasks == [
        {"task_index": 0, "task": "task0"},
        {"task_index": 1, "task": "task1"},
    ]
    labels = pd.read_parquet(
        output / "meta" / "phase_progress_semantic_trace_task0.parquet"
    )
    audit = pd.read_csv(output / "meta" / "semantic_trace_task0_audit.csv")
    assert labels.groupby("episode_index").size().to_dict() == {0: 2, 1: 2}
    assert audit["episode_index"].tolist() == [0, 1]
    assert summary["num_episodes"] == 2
    assert summary["total_frames"] == 4
    assert [row["task"] for row in summary["per_dataset"]] == [
        ["task0"],
        ["task1"],
    ]
    assert [row["success_rate"] for row in summary["per_dataset"]] == [1.0, 1.0]


def test_per_task_feature_split_is_independent_and_disjoint(tmp_path) -> None:
    features = tmp_path / "features"
    features.mkdir()
    torch.save(_feature_cache([0, 2]), features / "train.pt")
    torch.save(_feature_cache([1, 3]), features / "val.pt")

    reports = split_feature_caches_by_task(
        features,
        ["task0=0-2", "task1=2-4"],
        features,
        seed=7,
    )

    for task in ("task0", "task1"):
        train = torch.load(features / task / "train.pt", weights_only=False)
        val = torch.load(features / task / "val.pt", weights_only=False)
        train_episodes = set(train["episode_index"].tolist())
        val_episodes = set(val["episode_index"].tolist())
        assert train_episodes
        assert val_episodes
        assert train_episodes.isdisjoint(val_episodes)
        assert reports[task]["single_episode_overlap"] is False


def test_single_episode_overlap_requires_explicit_smoke_flag(tmp_path) -> None:
    features = tmp_path / "features"
    features.mkdir()
    torch.save(_feature_cache([0]), features / "train.pt")
    torch.save(_feature_cache([]), features / "val.pt")

    with pytest.raises(ValueError, match="one episode cannot form disjoint"):
        split_feature_caches_by_task(features, ["task0=0-1"], features)

    reports = split_feature_caches_by_task(
        features,
        ["task0=0-1"],
        features,
        allow_single_episode_overlap=True,
    )
    assert reports["task0"]["single_episode_overlap"] is True
    train = torch.load(features / "task0" / "train.pt", weights_only=False)
    val = torch.load(features / "task0" / "val.pt", weights_only=False)
    assert train["episode_index"].tolist() == [0]
    assert val["episode_index"].tolist() == [0]


def test_c_stage_commands_include_demo_slices_and_smoke_split_flags() -> None:
    ctx = {
        "tasks": ["task5", "task8"],
        "task_datasets": {
            "task5": "/data/smk_test/task5_1ep",
            "task8": "/data/smk_test/task8_1ep",
        },
        "task_ranges": {"task5": (0, 16), "task8": (16, 26)},
        "demo_datasets": {
            "task5": {"input": "/data/demo58::0:15"},
            "task8": {"input": "/data/demo58::41:50"},
        },
        "merged_ds": "/data/smk_test/merged",
        "revalue_root": "/data/smk_test/exp/revalue",
        "cfg": {
            "revalue": {
                "task_val_episode_ratio": 0.2,
                "seed": 7,
                "allow_single_episode_overlap": True,
            },
            "returns": {},
        },
    }

    merge_command = _step_merge_multitask(ctx).argv
    split_command = _task_feature_split_args(ctx)

    assert "/data/demo58::0:15" in merge_command
    assert "/data/demo58::41:50" in merge_command
    assert "--val_episode_ratio=0.2" in split_command
    assert "--seed=7" in split_command
    assert "--allow_single_episode_overlap" in split_command
