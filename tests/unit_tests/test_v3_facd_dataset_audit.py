"""Focused tests for round1 v3 FACD dataset quality gates."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from examples.recap.process.audit_v3_facd_datasets import (
    TASKS,
    audit_facd_labels,
    audit_merged_pool,
    audit_task_pools,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _build_task_pool(
    root: Path,
    rollout_root: Path,
    expert_root: Path,
    task: str,
) -> Path:
    dataset = root / f"{task}_30ep"
    meta = dataset / "meta"
    episodes = [
        {"episode_index": episode, "length": 2, "tasks": [task]}
        for episode in range(3)
    ]
    _write_json(
        meta / "info.json",
        {"total_episodes": 3, "total_frames": 6, "total_tasks": 1},
    )
    _write_jsonl(meta / "episodes.jsonl", episodes)

    frame_rows = []
    success_by_episode = {0: True, 1: False, 2: True}
    for episode in range(3):
        episode_rows = []
        for frame_index in range(2):
            row = {
                "episode_index": episode,
                "frame_index": frame_index,
                "is_success": success_by_episode[episode],
                "phase": frame_index,
                "phase_progress": float(frame_index),
                "global_progress": float(frame_index) / 3.0,
            }
            episode_rows.append(row)
            frame_rows.append(row)
        data_path = dataset / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(episode_rows)[["episode_index", "frame_index"]].to_parquet(
            data_path,
            index=False,
        )

    labels = pd.DataFrame(frame_rows)
    labels.to_parquet(
        meta / f"phase_progress_semantic_trace_{task}.parquet",
        index=False,
    )
    labels[["episode_index", "frame_index", "is_success"]].to_parquet(
        meta / f"semantic_trace_{task}.parquet",
        index=False,
    )
    pd.DataFrame(
        {
            "episode_index": [0, 1, 2],
            "is_success": [True, False, True],
            "trainable": [True, True, True],
        }
    ).to_csv(meta / f"semantic_trace_{task}_audit.csv", index=False)
    _write_json(meta / f"semantic_trace_{task}_metadata.json", {})

    rollout_source = rollout_root / f"{task}_20ep"
    expert_source = expert_root / f"{task}_10ep"
    provenance = []
    for episode in range(3):
        source_type = "rollout" if episode < 2 else "expert"
        source_dataset = rollout_source if source_type == "rollout" else expert_source
        provenance.append(
            {
                "episode_index": episode,
                "source_type": source_type,
                "source_dataset": str(source_dataset),
                "source_episode_index": episode,
                "is_success": success_by_episode[episode],
            }
        )
    _write_jsonl(meta / "episode_provenance.jsonl", provenance)
    _write_json(meta / "full_positive_episodes.json", [2])
    return dataset


def _build_merged_pool(root: Path, task_datasets: dict[str, Path]) -> Path:
    merged = root / "merged"
    meta = merged / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    merged_episodes = []
    merged_phase = []
    for task_number, task in enumerate(TASKS):
        offset = task_number * 3
        task_dataset = task_datasets[task]
        task_phase = pd.read_parquet(
            task_dataset
            / "meta"
            / f"phase_progress_semantic_trace_{task}.parquet"
        )
        task_phase["episode_index"] += offset
        task_phase.to_parquet(
            meta / f"phase_progress_semantic_trace_{task}.parquet",
            index=False,
        )
        merged_phase.append(task_phase)

        task_raw = pd.read_parquet(
            task_dataset / "meta" / f"semantic_trace_{task}.parquet"
        )
        task_raw["episode_index"] += offset
        task_raw.to_parquet(
            meta / f"semantic_trace_{task}.parquet",
            index=False,
        )
        task_audit = pd.read_csv(
            task_dataset / "meta" / f"semantic_trace_{task}_audit.csv"
        )
        task_audit["episode_index"] += offset
        task_audit.to_csv(
            meta / f"semantic_trace_{task}_audit.csv",
            index=False,
        )

        for local_episode in range(3):
            global_episode = offset + local_episode
            merged_episodes.append(
                {"episode_index": global_episode, "length": 2, "tasks": [task]}
            )
            data_path = (
                merged
                / "data"
                / "chunk-000"
                / f"episode_{global_episode:06d}.parquet"
            )
            data_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "episode_index": [global_episode, global_episode],
                    "frame_index": [0, 1],
                    "task_index": [task_number, task_number],
                }
            ).to_parquet(data_path, index=False)

    _write_json(
        meta / "info.json",
        {"total_episodes": 30, "total_frames": 60, "total_tasks": 10},
    )
    _write_jsonl(meta / "episodes.jsonl", merged_episodes)
    pd.concat(merged_phase, ignore_index=True).to_parquet(
        meta / "phase_progress_multitask.parquet",
        index=False,
    )
    return merged


def test_task_pool_and_facd_audits_accept_complete_data(tmp_path: Path) -> None:
    rollout_root = tmp_path / "rollout"
    expert_root = tmp_path / "expert"
    task_datasets = {
        task: _build_task_pool(
            tmp_path / "pools",
            rollout_root,
            expert_root,
            task,
        )
        for task in TASKS
    }

    pool_report = audit_task_pools(
        task_datasets,
        rollout_root=rollout_root,
        expert_root=expert_root,
        expected_episodes=3,
        rollout_episodes=2,
        expert_episodes=1,
    )

    assert pool_report["passed"] is True
    assert pool_report["aggregate"]["episodes"] == 30
    assert pool_report["aggregate"]["frames"] == 60

    merged_report = audit_merged_pool(
        task_datasets,
        _build_merged_pool(tmp_path, task_datasets),
        episodes_per_task=3,
    )
    assert merged_report["passed"] is True
    assert merged_report["episodes"] == 30
    assert merged_report["frames"] == 60

    export_reports = {}
    for task, dataset in task_datasets.items():
        advantages = pd.DataFrame(
            {
                "episode_index": [0, 0, 1, 1, 2, 2],
                "frame_index": [0, 1, 0, 1, 0, 1],
                "advantage": [True, False, False, False, True, True],
            }
        )
        advantages.to_parquet(
            dataset / "meta" / "advantages_facd_test.parquet",
            index=False,
        )
        report_path = tmp_path / "reports" / task / "export_report.json"
        _write_json(
            report_path,
            {
                "success_gate": True,
                "demo_backstop": True,
                "rollout_budget_filled": True,
                "failure_cap_satisfied": True,
                "num_full_positive_episodes": 1,
            },
        )
        export_reports[task] = report_path

    facd_report = audit_facd_labels(
        task_datasets,
        export_reports,
        advantage_tag="facd_test",
        positive_quantile=0.25,
        failure_positive_cap=0.2,
    )

    assert facd_report["passed"] is True
    assert facd_report["aggregate"]["positive_frames"] == 30
