import torch

from examples.recap.phase_progress_probe.logit_fusion_model import LogitFusionMLP


def test_logit_fusion_mlp_forward_shape() -> None:
    model = LogitFusionMLP(
        num_bins=201,
        num_phases=5,
        hidden_dim=128,
        dropout=0.0,
        depth=2,
    )
    out = model(
        raw_logits=torch.randn(4, 201),
        phase_repr=torch.randn(4, 5),
        phase_progress=torch.rand(4),
        global_progress=torch.rand(4),
    )
    assert out.shape == (4, 201)
