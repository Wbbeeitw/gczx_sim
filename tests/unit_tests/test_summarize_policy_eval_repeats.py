"""Tests for repeated policy-evaluation summaries."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).parents[2]
    / "examples"
    / "recap"
    / "process"
    / "summarize_policy_eval_repeats.py"
)
SPEC = importlib.util.spec_from_file_location("summarize_policy_repeats", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_summary(root: Path, task: int, success_rate: float) -> None:
    task_dir = root / f"task{task}"
    task_dir.mkdir(parents=True)
    (task_dir / "eval_policy_summary.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "eval/success_rate": success_rate,
                    "eval/num_trajectories": 20,
                    "eval/success_count": int(success_rate * 20),
                }
            }
        ),
        encoding="utf-8",
    )


def test_summarize_repeats_reports_task_and_repeat_means(tmp_path: Path) -> None:
    repeat1 = tmp_path / "repeat1"
    repeat2 = tmp_path / "repeat2"
    for task, rates in enumerate(((1.0, 0.8), (0.5, 0.7))):
        _write_summary(repeat1, task, rates[0])
        _write_summary(repeat2, task, rates[1])

    repeat_task, by_task, aggregate = MODULE.summarize_repeats(
        [("repeat1", repeat1), ("repeat2", repeat2)],
        num_tasks=2,
    )

    assert len(repeat_task) == 4
    assert by_task.loc[0, "mean_success_rate"] == pytest.approx(0.9)
    assert by_task.loc[1, "mean_success_rate"] == pytest.approx(0.6)
    assert aggregate["repeat_macro_success_rate"]["repeat1"] == pytest.approx(0.75)
    assert aggregate["repeat_macro_success_rate"]["repeat2"] == pytest.approx(0.75)
    assert aggregate["three_repeat_task_macro_mean"] == pytest.approx(0.75)
