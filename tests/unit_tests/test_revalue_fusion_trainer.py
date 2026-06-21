import torch

from revalue_test_utils import install_omegaconf_stub

install_omegaconf_stub()

from rlinf.revalue.models import LogitFusionMLP, SharedMLPPhaseProgressHead
from rlinf.revalue.training import FusionTrainer, FusionTrainerConfig


def test_fusion_trainer_freezes_zp_head() -> None:
    head = SharedMLPPhaseProgressHead(
        feature_dim=8,
        num_phases=3,
        hidden_dim=12,
        dropout=0.0,
    )
    fusion = LogitFusionMLP(
        num_bins=7,
        num_phases=3,
        hidden_dim=12,
        dropout=0.0,
    )

    trainer = FusionTrainer(
        zp_head=head,
        fusion=fusion,
        atoms=torch.linspace(-1.0, 0.0, 7),
        cfg=FusionTrainerConfig(device="cpu"),
    )

    assert all(not param.requires_grad for param in trainer.zp_head.parameters())
    assert any(param.requires_grad for param in trainer.fusion.parameters())
