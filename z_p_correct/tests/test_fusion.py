"""Unit tests for logit-space fusion."""

import pytest
import torch

from z_p_correct.models.fusion.logit_fusion import LogitFusionMLP, fuse_logits
from z_p_correct.utils.metrics import value_from_logits


@pytest.mark.parametrize("batch_size", [1, 8])
def test_fusion_output_shape(batch_size: int):
    num_bins = 201
    num_phases = 5
    fusion = LogitFusionMLP(num_bins=num_bins, num_phases=num_phases)

    raw_logits = torch.randn(batch_size, num_bins)
    phase_repr = torch.randn(batch_size, num_phases)
    phase_progress = torch.rand(batch_size)
    global_progress = torch.rand(batch_size)

    delta = fusion(raw_logits, phase_repr, phase_progress, global_progress)
    assert delta.shape == (batch_size, num_bins)


def test_fuse_logits_and_value():
    num_bins = 201
    batch_size = 4
    raw_logits = torch.randn(batch_size, num_bins)
    delta = torch.randn(batch_size, num_bins)
    atoms = torch.linspace(-1.0, 0.0, num_bins)

    fused_logits = fuse_logits(raw_logits, delta, alpha=1.0)
    value = value_from_logits(fused_logits, atoms)
    assert value.shape == (batch_size,)
    assert (value >= -1.0).all() and (value <= 0.0).all()
