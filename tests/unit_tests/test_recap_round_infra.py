"""Focused tests for the iterated ReCap round infrastructure."""

from __future__ import annotations

import json

import torch

from examples.recap.process.record_round_results import _policy_summary
from examples.recap.rounds.run_round import Step, _step_hash
from rlinf.revalue.pipeline.embodied import warmup_libero_rollout_policy


class _RandomPolicy:
    def __init__(self) -> None:
        self.calls = 0
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def predict_action_batch(self, env_obs, mode: str):
        self.calls += 1
        torch.rand(4)
        return torch.zeros((1, 5, 7)), {"mode": mode, "keys": sorted(env_obs)}


def test_rollout_warmup_preserves_cpu_rng_state() -> None:
    torch.manual_seed(1234)
    rng_state = torch.random.get_rng_state().clone()
    policy = _RandomPolicy()

    warmup_libero_rollout_policy(policy, "put both objects in the basket")

    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert policy.calls == 1
    assert policy.resets == 2


def test_policy_summary_extracts_primary_eval_metrics(tmp_path) -> None:
    summary_path = tmp_path / "eval_policy_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "eval/num_trajectories": 50,
                    "eval/success_count": 37,
                    "eval/failure_count": 13,
                    "eval/success_rate": 0.74,
                    "eval/all_episode_act_mean": 351.08,
                    "eval/success_episode_act_mean": 305.78,
                    "eval/success_episode_act_std": 49.71,
                }
            }
        ),
        encoding="utf-8",
    )

    metrics = _policy_summary(summary_path)

    assert metrics["episodes"] == 50
    assert metrics["successes"] == 37
    assert metrics["failures"] == 13
    assert metrics["success_rate"] == 0.74
    assert metrics["success_only_act"] == 305.78


def test_step_hash_changes_with_command_semantics() -> None:
    base = Step(name="predict", argv=["python", "predict.py", "alpha=1.0"])
    changed = Step(name="predict", argv=["python", "predict.py", "alpha=0.5"])

    assert _step_hash(base) != _step_hash(changed)
