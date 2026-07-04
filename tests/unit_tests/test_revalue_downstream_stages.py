from revalue_test_utils import install_omegaconf_stub

install_omegaconf_stub()

from pathlib import Path

from rlinf.revalue.pipeline.embodied import (  # noqa: E402
    DownstreamCFGTrainingConfig,
    _quote_override,
    PolicyEvaluationConfig,
    evaluate_policy_checkpoint,
    train_cfg_from_advantages,
)
from rlinf.revalue.config import RevalueConfig  # noqa: E402
from rlinf.revalue.constants import (  # noqa: E402
    STAGE_BUILD_BASE_FROM_CACHE,
    STAGE_COLLECT_ROLLOUTS,
    STAGE_EVAL_POLICY,
    STAGE_TRAIN_CFG,
)
from rlinf.revalue.pipeline.run import _stages_for_method  # noqa: E402


def test_stage_all_appends_enabled_downstream_stages() -> None:
    cfg = RevalueConfig()
    cfg.cfg_train.enabled = True
    cfg.policy_eval.enabled = True
    cfg.rollout_collect.enabled = True

    stages = _stages_for_method(cfg)

    assert STAGE_TRAIN_CFG in stages
    assert STAGE_EVAL_POLICY in stages
    assert STAGE_COLLECT_ROLLOUTS in stages
    assert STAGE_BUILD_BASE_FROM_CACHE not in stages
    assert stages.index(STAGE_TRAIN_CFG) < stages.index(STAGE_EVAL_POLICY)
    assert stages.index(STAGE_EVAL_POLICY) < stages.index(STAGE_COLLECT_ROLLOUTS)


def test_quote_override_uses_hydra_container_syntax_for_train_data_paths() -> None:
    train_data_paths = [
        {
            "dataset_path": "/workspace/datasets/recap_libero10_task0/libero10_task0_train",
            "type": "rollout",
            "weight": 1.0,
        }
    ]

    result = _quote_override(train_data_paths)

    assert result == (
        '[{dataset_path:"/workspace/datasets/recap_libero10_task0/libero10_task0_train",'
        'type:"rollout",weight:1.0}]'
    )


def test_evaluate_policy_checkpoint_appends_task_id_filter_override(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class DummyProc:
        returncode = 0
        stdout = ""

    def fake_run_python_entry(**kwargs):
        captured.update(kwargs)
        return DummyProc()

    monkeypatch.setattr(
        "rlinf.revalue.pipeline.embodied._run_python_entry",
        fake_run_python_entry,
    )

    evaluate_policy_checkpoint(
        PolicyEvaluationConfig(
            repo_root=str(tmp_path),
            model_path="/workspace/models/RLinf-Pi05-LIBERO-SFT",
            checkpoint_path="/workspace/checkpoint/full_weights.pt",
            experiment_name="eval_cfg_base_first200_fixscale_4k_50traj",
            log_dir=str(tmp_path / "logs"),
            model_type="cfg_model",
            task_suite_name="libero_10",
            task_id_filter=[0],
        )
    )

    overrides = captured["overrides"]
    assert isinstance(overrides, list)
    assert '+env.eval.task_id_filter=[0]' in overrides


def test_train_cfg_from_advantages_appends_csa_overrides(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class DummyProc:
        returncode = 0
        stdout = ""

    def fake_run_python_entry(**kwargs):
        captured.update(kwargs)
        return DummyProc()

    monkeypatch.setattr(
        "rlinf.revalue.pipeline.embodied._run_python_entry",
        fake_run_python_entry,
    )
    monkeypatch.setattr(
        "rlinf.revalue.pipeline.embodied._latest_checkpoint_path",
        lambda *_args, **_kwargs: None,
    )

    train_cfg_from_advantages(
        DownstreamCFGTrainingConfig(
            repo_root=str(tmp_path),
            dataset_path="/workspace/datasets/recap_libero10_task0/libero10_task0_train",
            base_model_path="/workspace/models/RLinf-Pi05-LIBERO-SFT",
            advantage_tag="fused_test",
            experiment_name="cfg_csa_test",
            log_dir=str(tmp_path / "logs"),
            strategy="csa_soft",
            guidance_type="positive",
            negative_guidance_scale=0.25,
            csa_positive_quantile=0.30,
            csa_bottom_quantile=0.15,
            csa_positive_prompt_prob=0.85,
            csa_bottom_negative_prob=0.50,
            csa_weight_lambda=0.20,
        )
    )

    overrides = captured["overrides"]
    assert isinstance(overrides, list)
    assert 'data.cfg_strategy="csa_soft"' in overrides
    assert "data.csa_positive_quantile=0.3" in overrides
    assert "data.csa_bottom_quantile=0.15" in overrides
    assert "data.csa_bottom_negative_prob=0.5" in overrides
    assert "data.csa_weight_lambda=0.2" in overrides
    assert "actor.model.openpi.unconditional_prob=0.1" in overrides
    assert "actor.model.openpi.cfgrl_negative_guidance_scale=0.25" in overrides
    assert "actor.model.openpi.csa_positive_prompt_prob=0.85" in overrides
    assert "actor.model.openpi.positive_only_conditional=false" in overrides
