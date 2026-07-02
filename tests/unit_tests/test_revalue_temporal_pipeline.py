from pathlib import Path

import torch

from revalue_test_utils import install_omegaconf_stub

install_omegaconf_stub()

from rlinf.revalue.data.feature_cache import (
    HEAD_TYPE_TEMPORAL_Z_MLP_P,
    build_feature_loaders,
)
from rlinf.revalue.models import TemporalZMLPProgressHead
from rlinf.revalue.pipeline.train import load_zp_head


def _write_feature_cache(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": torch.randn(6, 8),
            "episode_index": torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
            "frame_index": torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long),
            "phase": torch.tensor([0, 1, 1, 2, 3, 4], dtype=torch.long),
            "phase_progress": torch.rand(6),
            "global_progress": torch.rand(6),
        },
        path,
    )


def test_build_feature_loaders_wraps_temporal_windows(tmp_path: Path) -> None:
    features_dir = tmp_path / "features"
    _write_feature_cache(features_dir / "train.pt")
    _write_feature_cache(features_dir / "val.pt")

    train_loader, _ = build_feature_loaders(
        features_dir,
        batch_size=2,
        num_workers=0,
        head_type=HEAD_TYPE_TEMPORAL_Z_MLP_P,
        window_size=3,
    )
    batch = next(iter(train_loader))

    assert batch["feature_window"].shape == (2, 3, 8)
    assert batch["features"].shape == (2, 8)
    assert batch["phase_center"].shape == (2,)
    assert batch["valid_mask"].shape == (2, 3)


def test_load_zp_head_temporal_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "zp_head.pt"
    head = TemporalZMLPProgressHead(
        feature_dim=8,
        num_phases=5,
        hidden_dim=16,
        dropout=0.0,
        window_size=3,
        num_layers=1,
        num_heads=4,
        ffn_dim=32,
        stage_embedding_dim=8,
        progress_hidden_dim=16,
        progress_depth=1,
        phase_span_priors=[0.1, 0.2, 0.2, 0.2, 0.3],
    )
    torch.save(
        {
            "state_dict": head.state_dict(),
            "head_type": "temporal_z_mlp_p",
            "feature_dim": 8,
            "num_phases": 5,
            "hidden_dim": 16,
            "dropout": 0.0,
            "window_size": 3,
            "num_layers": 1,
            "num_heads": 4,
            "ffn_dim": 32,
            "stage_embedding_dim": 8,
            "progress_hidden_dim": 16,
            "progress_depth": 1,
            "phase_span_priors": [0.1, 0.2, 0.2, 0.2, 0.3],
        },
        checkpoint_path,
    )

    loaded = load_zp_head(checkpoint_path, device="cpu")
    out = loaded(torch.randn(2, 3, 8), stage_prior=None)

    assert out["phase_logits"].shape == (2, 5)
