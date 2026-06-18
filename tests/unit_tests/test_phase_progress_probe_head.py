import importlib.util

import pytest
import torch

HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
pytestmark = pytest.mark.skipif(
    not HAS_TRANSFORMERS, reason="transformers is required for phase probe head tests"
)

if HAS_TRANSFORMERS:
    from examples.recap.phase_progress_probe.model import PhaseProgressHead


def test_phase_progress_head_default_trunk_depth_forward() -> None:
    head = PhaseProgressHead(
        feature_dim=640,
        num_phases=5,
        hidden_dim=128,
        dropout=0.0,
    )
    features = torch.randn(4, 640)

    out = head(features)

    assert out["phase_logits"].shape == (4, 5)
    assert out["phase_progress"].shape == (4,)
    assert out["global_progress"].shape == (4,)


def test_phase_progress_head_two_layer_trunk_forward() -> None:
    head = PhaseProgressHead(
        feature_dim=640,
        num_phases=5,
        hidden_dim=128,
        dropout=0.0,
        trunk_depth=2,
    )
    features = torch.randn(4, 640)

    out = head(features)

    assert out["phase_logits"].shape == (4, 5)
    assert out["phase_progress"].shape == (4,)
    assert out["global_progress"].shape == (4,)
    assert torch.all(out["phase_progress"] >= 0.0)
    assert torch.all(out["phase_progress"] <= 1.0)
