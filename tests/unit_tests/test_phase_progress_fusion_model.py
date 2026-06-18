import torch

from examples.recap.phase_progress_probe.fusion_model import FusionMLP


def test_fusion_mlp_forward_shape() -> None:
    model = FusionMLP(
        feature_dim=640,
        num_phases=5,
        hidden_dim=128,
        dropout=0.0,
        depth=2,
    )
    out = model(
        features=torch.randn(4, 640),
        phase_repr=torch.randn(4, 5),
        phase_progress=torch.rand(4),
        global_progress=torch.rand(4),
        raw_value=torch.randn(4),
    )
    assert out.shape == (4,)
