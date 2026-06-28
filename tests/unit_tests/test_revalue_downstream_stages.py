from revalue_test_utils import install_omegaconf_stub

install_omegaconf_stub()

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
