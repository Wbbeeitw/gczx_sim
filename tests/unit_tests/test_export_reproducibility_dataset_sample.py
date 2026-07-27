from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from examples.recap.process.export_reproducibility_dataset_sample import (
    export_reproducibility_sample,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_dataset(path: Path, task: str) -> None:
    meta = path / "meta"
    data = path / "data" / "chunk-000"
    videos = path / "videos" / "chunk-000" / "observation.images.image"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    videos.mkdir(parents=True)
    info = {
        "features": {"value": {"dtype": "int64", "shape": [1]}},
        "robot_type": "franka_panda",
        "fps": 10,
        "chunks_size": 1000,
        "data_path": (
            "data/chunk-{episode_chunk:03d}/"
            "episode_{episode_index:06d}.parquet"
        ),
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "total_episodes": 3,
        "total_frames": 6,
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    _write_jsonl(meta / "tasks.jsonl", [{"task_index": 0, "task": task}])
    _write_jsonl(
        meta / "episodes.jsonl",
        [
            {"episode_index": index, "length": 2, "tasks": [task]}
            for index in range(3)
        ],
    )
    _write_jsonl(
        meta / "episodes_stats.jsonl",
        [{"episode_index": index, "stats": {}} for index in range(3)],
    )
    _write_jsonl(
        meta / "episode_provenance.jsonl",
        [
            {
                "episode_index": index,
                "source_type": "expert" if index == 2 else "rollout",
                "source_dataset": f"/original/{task}",
                "source_episode_index": 100 + index,
            }
            for index in range(3)
        ],
    )
    trace_rows = []
    for episode_index in range(3):
        pq.write_table(
            pa.table(
                {
                    "episode_index": [episode_index, episode_index],
                    "frame_index": [0, 1],
                    "index": [episode_index * 2, episode_index * 2 + 1],
                    "task_index": [0, 0],
                    "value": [episode_index, episode_index + 1],
                }
            ),
            data / f"episode_{episode_index:06d}.parquet",
        )
        (videos / f"episode_{episode_index:06d}.mp4").write_bytes(
            f"video-{task}-{episode_index}".encode()
        )
        trace_rows.extend(
            {
                "episode_index": episode_index,
                "frame_index": frame_index,
                "phase": frame_index,
                "phase_progress": float(frame_index),
                "global_progress": float(frame_index),
                "is_success": episode_index == 2,
            }
            for frame_index in range(2)
        )
    trace = pd.DataFrame(trace_rows)
    trace.to_parquet(meta / f"semantic_trace_{task}.parquet", index=False)
    trace.to_parquet(
        meta / f"phase_progress_semantic_trace_{task}.parquet",
        index=False,
    )
    pd.DataFrame(
        {
            "episode_index": [index for index in range(3) for _ in range(2)],
            "frame_index": [0, 1] * 3,
            "advantage": [True, False] * 3,
        }
    ).to_parquet(meta / "advantages_facd.parquet", index=False)
    pd.DataFrame(
        {
            "episode_index": [0, 1, 2],
            "is_success": [False, False, True],
            "trainable": [True, False, True],
        }
    ).to_csv(meta / f"semantic_trace_{task}_audit.csv", index=False)
    (meta / f"semantic_trace_{task}_metadata.json").write_text(
        "{}", encoding="utf-8"
    )


def test_exports_deterministic_complete_trainable_episodes(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    for task in ("task0", "task1"):
        _write_dataset(source_root / task, task)
    pattern = str(source_root / "{task}")
    output = tmp_path / "sample"
    archive = tmp_path / "sample.tar.gz"

    result = export_reproducibility_sample(
        dataset_pattern=pattern,
        output_dataset=output,
        tasks=["task0", "task1"],
        episodes_per_task=1,
        seed=42,
        archive_path=archive,
    )

    assert result["passed"] is True
    assert result["episodes"] == 2
    assert result["tasks"] == 2
    assert result["frames"] == 4
    assert archive.is_file()
    manifest = json.loads(
        (output / "reproducibility_manifest.json").read_text(encoding="utf-8")
    )
    selections = manifest["episodes"]
    assert [row["output_episode_index"] for row in selections] == [0, 1]
    assert all(row["trainable"] for row in selections)
    assert all(row["source_episode_index"] != 1 for row in selections)
    assert all(row["source_episode_provenance"] for row in selections)
    provenance = [
        json.loads(line)
        for line in (output / "meta" / "episode_provenance.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["episode_index"] for row in provenance] == [0, 1]
    assert all("sampled_from_dataset" in row for row in provenance)
    advantages = pd.read_parquet(output / "meta" / "advantages_facd.parquet")
    assert len(advantages) == 4
    assert advantages.groupby("episode_index").size().to_dict() == {0: 2, 1: 2}
    assert (output / "meta" / "full_positive_episodes.json").is_file()

    checksum_lines = (output / "SHA256SUMS").read_text().splitlines()
    manifest_line = next(
        line
        for line in checksum_lines
        if line.endswith("reproducibility_manifest.json")
    )
    expected_digest = hashlib.sha256(
        (output / "reproducibility_manifest.json").read_bytes()
    ).hexdigest()
    assert manifest_line.startswith(expected_digest)

    second_output = tmp_path / "sample_again"
    export_reproducibility_sample(
        dataset_pattern=pattern,
        output_dataset=second_output,
        tasks=["task0", "task1"],
        episodes_per_task=1,
        seed=42,
    )
    second_manifest = json.loads(
        (second_output / "reproducibility_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert [
        (row["task"], row["source_episode_index"]) for row in selections
    ] == [
        (row["task"], row["source_episode_index"])
        for row in second_manifest["episodes"]
    ]
