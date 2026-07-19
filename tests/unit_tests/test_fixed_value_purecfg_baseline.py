"""Tests for the isolated fixed-Value, pure-CFG baseline plan."""

from __future__ import annotations

from examples.recap.rounds.run_fixed_value_purecfg_baseline import (
    _parse_tasks,
    build_plan,
    parse_args,
)


def test_baseline_plan_uses_external_value_and_raw_binary_cfg() -> None:
    args = parse_args(
        [
            "--run-root",
            "/tmp/fixed_value_baseline",
            "--value-checkpoint",
            "/models/fixed_value.pt",
            "--tasks",
            "task0,task1",
        ]
    )

    ctx, steps = build_plan(args)
    names = [step.name for step in steps]

    assert names[:3] == [
        "collect_rollouts_task0",
        "collect_rollouts_task1",
        "merge_datasets",
    ]
    assert "value_sft" not in names
    assert "train_zp_task0" not in names
    assert "train_fusion_task0" not in names
    assert "export_raw_dataset_view_task0" in names
    assert "export_raw_dataset_view_task1" in names
    assert ctx["value_ckpt"] == "/models/fixed_value.pt"

    extract_command = next(step.argv for step in steps if step.name == "extract_features")
    assert "value.checkpoint=/models/fixed_value.pt" in extract_command
    train_command = next(step.argv for step in steps if step.name == "train_cfg")
    assert "cfg_train.strategy=binary" in train_command
    assert "cfg_train.advantage_tag=fixed_value_raw" in train_command
    assert not any(argument.endswith("=None") for argument in train_command)


def test_task_parser_rejects_duplicate_or_invalid_tasks() -> None:
    try:
        _parse_tasks("task0,task0")
    except ValueError as error:
        assert "duplicates" in str(error)
    else:
        raise AssertionError("duplicate tasks were accepted")

    try:
        _parse_tasks("kitchen")
    except ValueError as error:
        assert "invalid task" in str(error)
    else:
        raise AssertionError("invalid task was accepted")
