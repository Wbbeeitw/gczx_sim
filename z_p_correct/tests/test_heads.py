"""Unit tests for phase/progress heads."""

import pytest
import torch

from z_p_correct.models.heads import (
    PhaseProgressHead,
    TemporalPhasePriorHead,
    TemporalZMLPProgressHead,
)
from z_p_correct.registry import get_head_class, list_heads


@pytest.mark.parametrize("head_type", list_heads())
def test_head_forward_shapes(head_type: str):
    feature_dim = 128
    num_phases = 5
    batch_size = 8
    cls = get_head_class(head_type)

    if head_type == "shared_mlp":
        head = cls(feature_dim=feature_dim, num_phases=num_phases)
        features = torch.randn(batch_size, feature_dim)
        out = head(features)
    else:
        window_size = 5
        head = cls(feature_dim=feature_dim, num_phases=num_phases, window_size=window_size)
        feature_window = torch.randn(batch_size, window_size, feature_dim)
        out = head(feature_window)

    assert out["phase_logits"].shape == (batch_size, num_phases)
    assert out["phase_probs"].shape == (batch_size, num_phases)
    assert out["phase_pred"].shape == (batch_size,)
    assert out["phase_progress"].shape == (batch_size,)
    assert out["global_progress"].shape == (batch_size,)

    # Probabilities are valid.
    assert torch.allclose(out["phase_probs"].sum(dim=-1), torch.ones(batch_size), atol=1e-5)
    assert (out["phase_progress"] >= 0.0).all() and (out["phase_progress"] <= 1.0).all()
    assert (out["global_progress"] >= 0.0).all() and (out["global_progress"] <= 1.0).all()


def test_shared_mlp_is_registered():
    assert "shared_mlp" in list_heads()
    assert get_head_class("shared_mlp") is PhaseProgressHead
