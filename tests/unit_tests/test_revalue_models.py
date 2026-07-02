import torch

from revalue_test_utils import install_omegaconf_stub

install_omegaconf_stub()

from rlinf.revalue.models import (
    LogitFusionMLP,
    SharedMLPPhaseProgressHead,
    TemporalZMLPProgressHead,
)


def test_shared_mlp_phase_progress_head_forward_shape() -> None:
    head = SharedMLPPhaseProgressHead(
        feature_dim=32,
        num_phases=5,
        hidden_dim=16,
        dropout=0.0,
        trunk_depth=2,
    )
    out = head(torch.randn(4, 32))

    assert out["phase_logits"].shape == (4, 5)
    assert out["phase_probs"].shape == (4, 5)
    assert out["phase_pred"].shape == (4,)
    assert out["phase_progress"].shape == (4,)
    assert out["global_progress"].shape == (4,)
    assert torch.all(out["phase_progress"] >= 0.0)
    assert torch.all(out["phase_progress"] <= 1.0)


def test_logit_fusion_mlp_forward_shape() -> None:
    fusion = LogitFusionMLP(
        num_bins=201,
        num_phases=5,
        hidden_dim=32,
        dropout=0.0,
        depth=2,
    )
    delta = fusion(
        raw_logits=torch.randn(3, 201),
        phase_repr=torch.rand(3, 5),
        phase_progress=torch.rand(3),
        global_progress=torch.rand(3),
    )

    assert delta.shape == (3, 201)


def test_temporal_z_mlp_progress_head_forward_shape() -> None:
    head = TemporalZMLPProgressHead(
        feature_dim=32,
        num_phases=5,
        hidden_dim=16,
        dropout=0.0,
        window_size=5,
        num_layers=1,
        num_heads=4,
        ffn_dim=32,
        stage_embedding_dim=8,
        progress_hidden_dim=16,
        progress_depth=1,
    )
    out = head(torch.randn(4, 5, 32))

    assert out["phase_logits"].shape == (4, 5)
    assert out["phase_probs"].shape == (4, 5)
    assert out["phase_pred"].shape == (4,)
    assert out["phase_progress"].shape == (4,)
    assert out["global_progress"].shape == (4,)
    assert out["center_hidden"].shape == (4, 16)
    assert torch.all(out["phase_progress"] >= 0.0)
    assert torch.all(out["phase_progress"] <= 1.0)
