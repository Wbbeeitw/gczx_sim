"""Tests for the fixed-data ReCap baseline policy-label audit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.recap.process.audit_recap_baseline_policy_data import (
    audit_recap_baseline_policy_data,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _audit_inputs(
    tmp_path: Path,
) -> tuple[Path, dict[str, Path], Path, Path, Path]:
    source = tmp_path / "merged" / "meta" / "advantages_shared_raw.parquet"
    source.parent.mkdir(parents=True)
    source.touch()
    value_checkpoint = tmp_path / "value" / "global_step_1200"
    value_checkpoint.mkdir(parents=True)
    feature_manifest = _write_json(
        tmp_path / "features" / "manifest.json",
        {"value_checkpoint": str(value_checkpoint.resolve())},
    )
    summary = {
        "advantage_tag": "recap_raw_top30",
        "tasks": {},
        "aggregate": {
            "tasks": 2,
            "episodes": 60,
            "frames": 2000,
            "positive_frames": 600,
            "positive_ratio": 0.3,
        },
    }
    reports: dict[str, Path] = {}
    for task in ("task0", "task1"):
        summary["tasks"][task] = {
            "episodes": 30,
            "frames": 1000,
            "positive_frames": 300,
            "positive_ratio": 0.3,
            "sources": {
                "rollout": {"episodes": 20},
                "expert": {"episodes": 10},
            },
        }
        reports[task] = _write_json(
            tmp_path / "reports" / task / "raw_export_report.json",
            {
                "mode": "raw",
                "source_advantages_path": str(source.resolve()),
                "predictions_path": None,
                "output_tag": "recap_raw_top30",
                "positive_quantile": 0.3,
                "rows_exported": 1000,
                "episodes_exported": 30,
            },
        )
    summary_path = _write_json(tmp_path / "summary.json", summary)
    return summary_path, reports, source, feature_manifest, value_checkpoint


def test_recap_baseline_audit_accepts_ungated_raw_labels(tmp_path: Path) -> None:
    summary_path, reports, source, feature_manifest, value_checkpoint = (
        _audit_inputs(tmp_path)
    )

    audit = audit_recap_baseline_policy_data(
        summary_path=summary_path,
        raw_reports=reports,
        source_advantages_path=source,
        source_feature_manifest=feature_manifest,
        value_checkpoint=value_checkpoint,
        advantage_tag="recap_raw_top30",
        expected_tasks=2,
    )

    assert audit["passed"] is True
    assert set(audit["tasks"]) == {"task0", "task1"}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("success_gate", True, "must not enable success_gate"),
        ("demo_backstop", True, "must not enable demo_backstop"),
        ("predictions_path", "/tmp/predictions.parquet", "unexpectedly uses"),
    ],
)
def test_recap_baseline_audit_rejects_nonbaseline_export(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    summary_path, reports, source, feature_manifest, value_checkpoint = (
        _audit_inputs(tmp_path)
    )
    task0_report = json.loads(reports["task0"].read_text(encoding="utf-8"))
    task0_report[field] = value
    _write_json(reports["task0"], task0_report)

    with pytest.raises(ValueError, match=message):
        audit_recap_baseline_policy_data(
            summary_path=summary_path,
            raw_reports=reports,
            source_advantages_path=source,
            source_feature_manifest=feature_manifest,
            value_checkpoint=value_checkpoint,
            advantage_tag="recap_raw_top30",
            expected_tasks=2,
        )


def test_recap_baseline_audit_rejects_wrong_positive_ratio(tmp_path: Path) -> None:
    summary_path, reports, source, feature_manifest, value_checkpoint = (
        _audit_inputs(tmp_path)
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["tasks"]["task1"]["positive_ratio"] = 0.45
    _write_json(summary_path, summary)

    with pytest.raises(ValueError, match="raw positive ratio"):
        audit_recap_baseline_policy_data(
            summary_path=summary_path,
            raw_reports=reports,
            source_advantages_path=source,
            source_feature_manifest=feature_manifest,
            value_checkpoint=value_checkpoint,
            advantage_tag="recap_raw_top30",
            expected_tasks=2,
        )


def test_recap_baseline_audit_rejects_different_value_checkpoint(
    tmp_path: Path,
) -> None:
    summary_path, reports, source, feature_manifest, _ = _audit_inputs(tmp_path)
    different_checkpoint = tmp_path / "value" / "global_step_600"
    different_checkpoint.mkdir()

    with pytest.raises(ValueError, match="raw features use Value checkpoint"):
        audit_recap_baseline_policy_data(
            summary_path=summary_path,
            raw_reports=reports,
            source_advantages_path=source,
            source_feature_manifest=feature_manifest,
            value_checkpoint=different_checkpoint,
            advantage_tag="recap_raw_top30",
            expected_tasks=2,
        )
