"""Tests for held-out credit-progress rank correlation."""

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
    / "summarize_credit_progress_correlation.py"
)
SPEC = importlib.util.spec_from_file_location("credit_progress", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_fused_credit_has_higher_task_macro_progress_correlation(
    tmp_path: Path,
) -> None:
    advantages_path = tmp_path / "advantages.parquet"
    predictions_path = tmp_path / "predictions.parquet"
    comparison_path = tmp_path / "return_compare.json"
    advantage_rows = []
    prediction_rows = []
    for episode_index in range(2):
        progress = [0.0, 0.1, 0.3, 0.6]
        phase = [0, 1, 1, 1]
        raw_value = [-1.0, -0.5, -0.6, -0.4]
        fused_value = [-1.0, -0.9, -0.7, -0.4]
        raw_credit = [0.5, -0.1, 0.2, 0.0]
        value_probabilities = [
            [1.0, 0.0, 0.0],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.8, 0.1, 0.1],
            [1.0, 0.0, 0.0],
        ]
        for frame_index in range(4):
            advantage_rows.append(
                {
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "reward_sum": 0.0,
                    "num_valid_rewards": 1,
                    "value_current": raw_value[frame_index],
                    "advantage_continuous": raw_credit[frame_index],
                    "return": fused_value[frame_index],
                    "value_probs_current": value_probabilities[frame_index],
                }
            )
            prediction_rows.append(
                {
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "split": "val",
                    "value_fused": fused_value[frame_index],
                    "global_progress_true": progress[frame_index],
                    "phase_true": phase[frame_index],
                }
            )
    pd.DataFrame(advantage_rows).to_parquet(advantages_path, index=False)
    pd.DataFrame(prediction_rows).to_parquet(predictions_path, index=False)
    comparison_path.write_text(
        json.dumps(
            {
                "advantages_path": str(advantages_path),
                "predictions_path": str(predictions_path),
                "return_min": -1.0,
                "return_max": 0.0,
                "value_min": -1.0,
                "value_max": 0.0,
            }
        ),
        encoding="utf-8",
    )

    aligned, metadata = MODULE.build_credit_progress_frame(
        comparison_path,
        split="val",
        lookahead_step=1,
        gamma=1.0,
        num_tasks=2,
        episodes_per_task=1,
    )
    by_task, aggregate = MODULE.summarize_credit_progress(
        aligned,
        num_tasks=2,
    )

    assert metadata["aligned_rows"] == 6
    assert metadata["raw_credit_source_max_abs_diff"] == pytest.approx(0.0)
    assert set(aligned["future_frame_index"]) == {1, 2, 3}
    assert list(by_task["raw_credit_progress_spearman"]) == pytest.approx(
        [-0.5, -0.5]
    )
    assert list(by_task["fused_credit_progress_spearman"]) == pytest.approx(
        [1.0, 1.0]
    )
    assert aggregate["task_macro"]["raw"] == pytest.approx(-0.5)
    assert aggregate["task_macro"]["fused"] == pytest.approx(1.0)
    assert aggregate["task_macro"]["tasks_improved"] == 2

    validation, _ = MODULE.build_validation_diagnostics_frame(
        comparison_path,
        split="val",
        num_tasks=2,
        episodes_per_task=1,
    )
    _, boundary = MODULE.summarize_boundary_value_diagnostics(
        validation,
        num_tasks=2,
        boundary_window=0,
    )
    _, credit_jitter = MODULE.summarize_boundary_credit_jitter(
        aligned,
        num_tasks=2,
        boundary_window=0,
    )
    assert boundary["fused_boundary_residual_jitter"] == pytest.approx(0.0)
    assert boundary["raw_boundary_residual_jitter"] > 0.0
    assert boundary["fused_boundary_mae"] == pytest.approx(0.0)
    assert boundary["raw_entropy_error_spearman"] == pytest.approx(1.0)
    assert boundary["raw_boundary_entropy_lift"] > 0.0
    assert credit_jitter["fused_boundary_credit_jitter"] < credit_jitter[
        "raw_boundary_credit_jitter"
    ]
    candidates = MODULE._build_candidate_metric_table(
        aggregate,
        boundary,
        credit_jitter,
    )
    assert list(candidates["direction"]) == [
        "higher",
        "lower",
        "lower",
        "lower",
    ]
    assert list(candidates["tasks_covered"]) == [2, 2, 2, 2]
    assert list(candidates["tasks_improved"]) == [2, 2, 2, 2]

    paper_table = MODULE._build_paper_table(
        {"value_mae_from_comparison": {"all": {"raw": 124.24, "fused": 59.53}}},
        aggregate,
        raw_policy_sr=75.8,
        fused_policy_sr=83.3,
    )
    assert list(paper_table["value_mae"]) == pytest.approx([124.24, 59.53])
    assert list(paper_table["policy_sr_percent"]) == pytest.approx([75.8, 83.3])


def test_spearman_uses_average_ranks_for_ties() -> None:
    result = MODULE._spearman_rank_correlation(
        pd.Series([1.0, 1.0, 2.0, 3.0]),
        pd.Series([0.0, 0.0, 1.0, 2.0]),
    )
    assert result == pytest.approx(1.0)
