"""Tests for the held-out upstream Critic metric sweep."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).parents[2]
    / "examples"
    / "recap"
    / "process"
    / "summarize_critic_upstream_quality.py"
)
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("critic_upstream", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _synthetic_frame() -> pd.DataFrame:
    rows = []
    for task_index in range(2):
        episode_index = task_index
        target = [-4.0, -3.0, -2.0, -1.0]
        raw = [-7.0, -1.0, -4.0, -0.5]
        fused = [-4.2, -2.8, -2.1, -1.0]
        phases = [0, 0, 1, 1]
        for frame_index in range(4):
            raw_error = raw[frame_index] - target[frame_index]
            fused_error = fused[frame_index] - target[frame_index]
            rows.append(
                {
                    "task": f"task{task_index}",
                    "task_index": task_index,
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "target": target[frame_index],
                    "recap_prediction": raw[frame_index],
                    "ours_prediction": fused[frame_index],
                    "recap_error": raw_error,
                    "ours_error": fused_error,
                    "raw_abs_error": abs(raw_error),
                    "fused_abs_error": abs(fused_error),
                    "fused_frame_win": abs(fused_error) < abs(raw_error),
                    "frame_error_tie": abs(fused_error) == abs(raw_error),
                    "phase_true": phases[frame_index],
                    "global_progress_true": frame_index / 3.0,
                    "is_success": task_index == 0,
                }
            )
    return MODULE._mark_phase_boundaries(pd.DataFrame(rows), boundary_window=0)


def test_upstream_sweep_reports_consistent_fused_improvements() -> None:
    slices, tasks, episodes, phases, candidates = (
        MODULE.summarize_upstream_quality(
            _synthetic_frame(),
            num_tasks=2,
        )
    )

    overall = slices.set_index("slice").loc["all"]
    assert overall["fused_mae"] < overall["raw_mae"]
    assert overall["fused_p90_abs_error"] < overall["raw_p90_abs_error"]
    assert overall["fused_return_spearman"] > overall["raw_return_spearman"]
    assert episodes["fused_wins"].all()
    assert (tasks["fused_mae"] < tasks["raw_mae"]).all()
    assert set(phases["phase"]) == {0, 1}
    assert set(candidates["metric"]) >= {
        "Frame MAE",
        "P90 absolute error",
        "Boundary MAE",
        "Failure-frame MAE",
        "Episode win rate",
    }


def test_phase_boundary_window_marks_transition_frame() -> None:
    frame = MODULE._mark_phase_boundaries(
        _synthetic_frame().drop(columns="is_phase_boundary"),
        boundary_window=0,
    )
    marked = frame[frame["is_phase_boundary"]]
    assert set(marked["frame_index"]) == {2}
    assert len(marked) == 2


def test_improvement_uses_lower_is_better_percentage() -> None:
    assert MODULE._improvement(100.0, 60.0) == pytest.approx(40.0)
