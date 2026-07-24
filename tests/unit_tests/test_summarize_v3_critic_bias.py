"""Tests for detailed ReCAP-vs-Ours critic bias summaries."""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).parents[2]
    / "examples"
    / "recap"
    / "process"
    / "summarize_v3_critic_bias.py"
)
SPEC = importlib.util.spec_from_file_location("summarize_v3_bias", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_summarize_bias_reports_task_and_aggregate_metrics(tmp_path: Path) -> None:
    advantages_path = tmp_path / "advantages.parquet"
    predictions_path = tmp_path / "predictions.parquet"
    comparison_path = tmp_path / "return_compare.json"
    pd.DataFrame(
        {
            "episode_index": [0, 0, 1, 1],
            "frame_index": [0, 1, 0, 1],
            "return": [-100.0, -50.0, -80.0, -40.0],
            "value_current": [-0.7, -0.3, -0.7, -0.3],
        }
    ).to_parquet(advantages_path)
    pd.DataFrame(
        {
            "episode_index": [0, 0, 1, 1],
            "frame_index": [0, 1, 0, 1],
            "split": ["train", "train", "val", "val"],
            "value_fused": [-0.9, -0.5, -0.8, -0.4],
        }
    ).to_parquet(predictions_path)
    comparison_path.write_text(
        json.dumps(
            {
                "advantages_path": str(advantages_path),
                "predictions_path": str(predictions_path),
                "return_min": -100.0,
                "return_max": 0.0,
                "value_min": -1.0,
                "value_max": 0.0,
            }
        ),
        encoding="utf-8",
    )

    tasks, task_splits, episodes, aggregate = MODULE.summarize_bias(
        comparison_path,
        comparison_path,
        num_tasks=2,
        episodes_per_task=1,
    )

    assert list(tasks["task"]) == ["task0", "task1"]
    assert tasks.loc[0, "recap_bias"] == pytest.approx(25.0)
    assert tasks.loc[0, "ours_bias"] == pytest.approx(5.0)
    assert tasks.loc[1, "recap_bias"] == pytest.approx(10.0)
    assert tasks.loc[1, "ours_bias"] == pytest.approx(0.0)
    assert set(task_splits["split"]) == {"all", "train", "val"}
    assert len(episodes) == 2
    assert aggregate["frame_micro"]["ours_bias"] == pytest.approx(2.5)
    assert aggregate["frame_micro"]["recap_bias"] == pytest.approx(17.5)
    assert "optimistic" in aggregate["bias_interpretation"]["positive"]
    assert aggregate["bias_definition"].startswith("mean(predicted_return")
