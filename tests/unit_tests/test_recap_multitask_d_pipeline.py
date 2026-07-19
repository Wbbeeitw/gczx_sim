"""Focused tests for the multitask ReCap policy iteration stages."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from examples.recap.process.build_fixed_multitask_train_pools import (
    EpisodeCandidate,
    _select_task_pool,
)
from examples.recap.process.record_multitask_round_results import (
    record_multitask_results,
)
from examples.recap.process.record_round_results import (
    _episode_error_metrics,
    _error_metrics,
    _load_comparison_frame,
)
from examples.recap.process.summarize_multitask_policy_data import (
    summarize_policy_data,
)
from examples.recap.process.write_raw_value_comparison import (
    write_raw_value_comparison,
)
from examples.recap.rounds.run_round import (
    _build_ctx,
    _steps_eval_policy,
    _steps_export_multitask,
    _steps_export_multitask_raw,
    _steps_score_critic_multitask,
    _steps_train_policy,
)


TASKS = ["task0", "task1"]


def _round_cfg(tmp_path: Path, *, round_two: bool = False) -> dict:
    root = tmp_path.as_posix()
    cfg = {
        "round_id": 2 if round_two else 1,
        "method": "prcfg",
        "task": "multitask",
        "task_id": 0,
        "task_suite_name": "libero_10",
        "repo_root": "/workspace/RLinf",
        "pipeline": [
            "fit_critic_multitask",
            "task_heads",
            "predict_multitask",
            "export_multitask",
            "train_policy",
            "eval_policy",
        ],
        "tasks": TASKS,
        "paths": {
            "base_model": "/workspace/models/RLinf-Pi05-LIBERO-SFT",
            "siglip": "/models/siglip",
            "gemma3": "/models/gemma",
            "tokenizer": "/models/gemma",
            "parent_dataset": f"{root}/unused_parent",
            "child_pattern": f"{root}/fresh/{{task}}",
            "merged_dataset": f"{root}/merged",
            "exp_root": f"{root}/exp",
            "results_root": f"{root}/persistent_results",
        },
        "datasets": {"parent_episodes": 0},
        "parent_policy": {"checkpoint": "", "label": "SFT"},
        "tags": {
            "returns": "returns",
            "merged_base": "base",
            "child_fused": "fused",
        },
        "collect": {"num_episodes": 40, "openpi_config_name": "pi05_libero"},
        "returns": {
            "gamma": 1.0,
            "failure_reward": -300.0,
            "global_min": -900.0,
            "global_max": 0.0,
        },
        "value": {
            "steps": 1200,
            "save_interval": 600,
            "micro_batch_size": 8,
            "global_batch_size": 64,
            "lr": 1.0e-5,
            "value_lr": 1.0e-4,
            "lr_warmup_steps": 50,
            "action_dim": 7,
            "action_horizon": 10,
            "critic_expert_variant": "gemma_1m",
        },
        "revalue": {
            "label_name": "phase_progress_multitask",
            "positive_quantile": 0.3,
        },
        "policy": {
            "strategy": "csa_residual",
            "guidance_type": "positive",
            "positive_only_conditional": False,
            "negative_guidance_scale": 0.0,
            "unconditional_prob": 0.1,
            "positive_quantile": 0.3,
            "csa_bottom_quantile": 0.15,
            "csa_bottom_negative_prob": 0.5,
            "csa_positive_prompt_prob": 0.85,
            "csa_weight_lambda": 0.2,
            "positive_residual_alpha": 0.5,
            "max_steps": 500,
            "save_interval": 500,
            "lr_warmup_steps": 50,
            "global_batch_size": 64,
            "micro_batch_size": 8,
            "extra_overrides": [],
        },
        "eval": {
            "eval_rollout_epoch": 5,
            "total_num_envs": 10,
            "save_video": False,
        },
        "results": {
            "enabled": True,
            "round_index": 1,
            "policy_label": "Policy1",
            "critic_label": "Value1",
            "output_dir": f"{root}/results",
            "output_name": "round1",
        },
    }
    if round_two:
        cfg["paths"].update(
            {
                "parent_pattern": f"{root}/d0/{{task}}",
                "critic_pattern": f"{root}/critic80/{{task}}",
                "policy_pattern": f"{root}/fresh/{{task}}",
            }
        )
        cfg["datasets"]["parent_episodes_per_task"] = 40
        cfg["parent_policy"] = {
            "checkpoint": "/checkpoints/policy1.pt",
            "label": "Policy1",
        }
    return cfg


def test_round_context_keeps_policy_ranges_separate_from_critic_pool(
    tmp_path: Path,
) -> None:
    round_one = _build_ctx(_round_cfg(tmp_path))
    assert round_one["task_ranges"] == {"task0": (0, 40), "task1": (40, 80)}
    assert round_one["policy_task_ranges"] == round_one["task_ranges"]

    round_two = _build_ctx(_round_cfg(tmp_path, round_two=True))
    assert round_two["task_ranges"] == {"task0": (0, 80), "task1": (80, 160)}
    assert round_two["policy_task_ranges"] == {
        "task0": (40, 80),
        "task1": (120, 160),
    }


def test_multitask_export_and_training_use_fresh_fixed_pools(tmp_path: Path) -> None:
    ctx = _build_ctx(_round_cfg(tmp_path, round_two=True))
    export_steps = _steps_export_multitask(ctx)
    export_commands = [
        step.argv for step in export_steps if step.name.startswith("export_dataset")
    ]
    assert len(export_commands) == 2
    assert all(
        "export_view.output_tag=fused" in command for command in export_commands
    )
    assert all(
        "export_view.expected_episodes=40" in command
        for command in export_commands
    )
    assert "export_view.source_episode_start=40" in export_commands[0]
    assert "export_view.source_episode_start=120" in export_commands[1]

    train_command = _steps_train_policy(ctx)[0].argv
    extra_override = next(
        value
        for value in train_command
        if value.startswith("cfg_train.extra_overrides=")
    )
    assert '"data.balance_dataset_weights=false"' in extra_override
    assert 'data.train_data_paths=[{dataset_path:' in extra_override
    assert f'{tmp_path.as_posix()}/fresh/task0' in extra_override
    assert f'{tmp_path.as_posix()}/fresh/task1' in extra_override


def test_external_value_scores_multitask_pool_without_value_training(
    tmp_path: Path,
) -> None:
    cfg = _round_cfg(tmp_path)
    cfg["paths"]["value_checkpoint"] = "/checkpoints/value1"
    ctx = _build_ctx(cfg)

    steps = _steps_score_critic_multitask(ctx)
    names = [step.name for step in steps]

    assert "value_sft" not in names
    assert names == [
        "merge_datasets",
        "compute_returns",
        "prepare_data",
        "extract_features",
        "build_base_from_cache",
    ]
    extract_command = next(
        step.argv for step in steps if step.name == "extract_features"
    )
    assert "value.checkpoint=/checkpoints/value1" in extract_command


def test_external_value_scoring_requires_checkpoint(tmp_path: Path) -> None:
    ctx = _build_ctx(_round_cfg(tmp_path))
    try:
        _steps_score_critic_multitask(ctx)
    except ValueError as error:
        assert "paths.value_checkpoint" in str(error)
    else:
        raise AssertionError("missing external Value checkpoint was accepted")


def test_multitask_raw_export_bypasses_predictions_and_trains_binary_cfg(
    tmp_path: Path,
) -> None:
    cfg = _round_cfg(tmp_path, round_two=True)
    cfg["pipeline"] = [
        "score_critic_multitask",
        "export_multitask_raw",
        "train_policy",
        "eval_policy",
    ]
    cfg["paths"]["value_checkpoint"] = "/checkpoints/value1"
    cfg["tags"]["policy_advantage"] = "raw_top30"
    cfg["policy"]["strategy"] = "binary"
    cfg["policy"]["advantage_source"] = "raw"
    ctx = _build_ctx(cfg)

    export_steps = _steps_export_multitask_raw(ctx)
    export_commands = [
        step.argv
        for step in export_steps
        if step.name.startswith("export_raw_dataset_view")
    ]
    assert len(export_commands) == 2
    assert all("export_view.mode=raw" in command for command in export_commands)
    assert all(
        not any("predictions_path" in argument for argument in command)
        for command in export_commands
    )
    assert "export_view.source_episode_start=40" in export_commands[0]
    assert "export_view.source_episode_start=120" in export_commands[1]
    assert any(
        step.name == "write_raw_value_comparison" for step in export_steps
    )

    train_command = _steps_train_policy(ctx)[0].argv
    assert "cfg_train.strategy=binary" in train_command
    assert "cfg_train.advantage_tag=raw_top30" in train_command

    record_command = _steps_eval_policy(ctx)[-1].argv
    assert not any("--zp-metrics" in argument for argument in record_command)
    assert not any("--fusion-metrics" in argument for argument in record_command)


def test_multitask_eval_is_sequential_per_task_and_disables_value_head(
    tmp_path: Path,
) -> None:
    ctx = _build_ctx(_round_cfg(tmp_path))
    steps = _steps_eval_policy(ctx)
    eval_steps = steps[:-1]
    assert [step.name for step in eval_steps] == [
        "eval_policy_task0",
        "eval_policy_task1",
    ]
    for task_id, step in enumerate(eval_steps):
        command = step.argv
        assert f"policy_eval.task_id_filter=[{task_id}]" in command
        assert "policy_eval.total_num_envs=10" in command
        assert "policy_eval.eval_rollout_epoch=5" in command
        extra_override = next(
            value
            for value in command
            if value.startswith("policy_eval.extra_overrides=")
        )
        assert '"actor.model.add_value_head=false"' in extra_override
        assert '"+env.eval.use_ordered_reset_state_ids=true"' in extra_override
    assert steps[-1].name == "record_multitask_round_results"


def _candidate(
    position: int,
    *,
    source: str = "rollout",
    success: bool = False,
    phase: int = 0,
    progress: float = 0.0,
) -> EpisodeCandidate:
    return EpisodeCandidate(
        dataset_path=Path(f"/{source}"),
        position=position,
        episode_index=position,
        source_type=source,
        is_success=success,
        max_phase=phase,
        max_global_progress=progress,
        max_phase_progress=progress,
    )


def test_fixed_pool_preserves_successes_and_drops_lowest_progress_failures() -> None:
    rollout = [
        _candidate(0, success=True, phase=3, progress=1.0),
        _candidate(1, phase=0, progress=0.1),
        _candidate(2, phase=2, progress=0.7),
        _candidate(3, phase=1, progress=0.8),
    ]
    expert = [_candidate(0, source="expert", success=True, phase=3, progress=1.0)]

    selected, report = _select_task_pool(rollout, expert, 4, True)

    rollout_positions = [
        item.position for item in selected if item.source_type == "rollout"
    ]
    assert rollout_positions == [0, 2, 3]
    assert report["rollout_successes_preserved"] == 1
    assert report["rollout_failures_dropped"] == 1
    assert report["expert_episodes"] == 1


def _write_eval_summary(path: Path, success_steps: float) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "metrics": {
                    "eval/num_trajectories": 2,
                    "eval/success_count": 1,
                    "eval/failure_count": 1,
                    "eval/success_rate": 0.5,
                    "eval/all_episode_act_mean": 400.0,
                    "eval/success_episode_act_mean": success_steps,
                    "eval/success_episode_act_std": 0.0,
                }
            }
        ),
        encoding="utf-8",
    )


def test_multitask_record_and_policy_summary_report_primary_metrics(
    tmp_path: Path,
) -> None:
    advantages_path = tmp_path / "advantages.parquet"
    predictions_path = tmp_path / "predictions.parquet"
    pd.DataFrame(
        {
            "episode_index": [0, 1],
            "frame_index": [0, 0],
            "return": [-10.0, -20.0],
            "value_current": [-0.5, -0.5],
        }
    ).to_parquet(advantages_path)
    pd.DataFrame(
        {
            "episode_index": [0, 1],
            "frame_index": [0, 0],
            "value_fused": [-0.8, -0.7],
        }
    ).to_parquet(predictions_path)
    comparison_path = tmp_path / "comparison.json"
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
    eval_paths = {task: tmp_path / "eval" / task / "summary.json" for task in TASKS}
    _write_eval_summary(eval_paths["task0"], 200.0)
    _write_eval_summary(eval_paths["task1"], 300.0)
    policy_report = tmp_path / "policy_report.json"
    policy_report.write_text(
        json.dumps(
            {
                "tasks": {
                    "task0": {"positive_ratio": 0.3},
                    "task1": {"positive_ratio": 0.3},
                }
            }
        ),
        encoding="utf-8",
    )

    result = record_multitask_results(
        round_index=1,
        policy_label="Policy1",
        critic_label="Value1",
        checkpoint_path="/checkpoints/policy1.pt",
        task_ranges={"task0": (0, 1), "task1": (1, 2)},
        eval_summaries=eval_paths,
        zp_metrics={},
        fusion_metrics={},
        comparison_path=comparison_path,
        policy_data_report_path=policy_report,
        output_dir=tmp_path / "results",
        output_name="round1",
    )

    assert result["aggregate"]["num_trajectories"] == 4
    assert result["aggregate"]["macro_success_rate"] == 0.5
    assert result["aggregate"]["micro_success_rate"] == 0.5
    assert result["aggregate"]["success_episode_act_mean"] == 250.0
    assert result["aggregate"]["success_episode_act_std"] == 50.0


def test_policy_data_summary_audits_expert_selection(tmp_path: Path) -> None:
    dataset = tmp_path / "task5"
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps({"total_frames": 4, "total_episodes": 2}), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "episode_index": [0, 0, 1, 1],
            "frame_index": [0, 1, 0, 1],
            "advantage": [False, True, True, True],
        }
    ).to_parquet(meta / "advantages_fused.parquet")
    (meta / "episode_provenance.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"episode_index": 0, "source_type": "rollout"}),
                json.dumps({"episode_index": 1, "source_type": "expert"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({str(dataset): {"selected_episodes": [0, 1]}}),
        encoding="utf-8",
    )

    report = summarize_policy_data(
        {"task5": dataset},
        {"task5": manifest},
        "fused",
        tmp_path / "report.json",
        tmp_path / "combined_manifest.json",
    )

    assert report["tasks"]["task5"]["positive_ratio"] == 0.75
    assert report["tasks"]["task5"]["sources"]["expert"]["positive_ratio"] == 1.0


def test_raw_comparison_reports_only_base_value_error(tmp_path: Path) -> None:
    advantages = tmp_path / "raw.parquet"
    pd.DataFrame(
        {
            "episode_index": [0, 0],
            "frame_index": [0, 1],
            "return": [-100.0, -50.0],
            "value_current": [-0.5, -0.25],
        }
    ).to_parquet(advantages)
    comparison_path = tmp_path / "raw_comparison.json"
    comparison = write_raw_value_comparison(
        advantages_path=advantages,
        output_path=comparison_path,
        return_min=-200.0,
        return_max=0.0,
    )

    frame = _load_comparison_frame(comparison)
    metrics = _error_metrics(frame)
    episode_metrics = _episode_error_metrics(frame)

    assert metrics["base_mae"] == 0.0
    assert "fused_mae" not in metrics
    assert episode_metrics["mean_base_mae"] == 0.0
    assert "mean_fused_mae" not in episode_metrics
