from pathlib import Path

import torch

from revalue_test_utils import install_omegaconf_stub

install_omegaconf_stub()

from rlinf.revalue.data.feature_cache import (
    HEAD_TYPE_TEMPORAL_LOCAL_STAGE_GATED,
    HEAD_TYPE_TEMPORAL_STAGE_PRIOR,
    HEAD_TYPE_TEMPORAL_STAGE_EXPERTS,
    HEAD_TYPE_TEMPORAL_Z_MLP_P,
    build_feature_loaders,
)
from rlinf.revalue.models import (
    TemporalLocalStageGatedProgressHead,
    TemporalStagePriorProgressHead,
    TemporalStageExpertsProgressHead,
    TemporalZMLPProgressHead,
)
from rlinf.revalue.pipeline.train import load_zp_head
from rlinf.revalue.training.zp_trainer import (
    TemporalZPHeadTrainerConfig,
    _is_better_temporal_checkpoint,
    _should_track_temporal_checkpoint,
)


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


def test_build_feature_loaders_wraps_stage_expert_temporal_windows(tmp_path: Path) -> None:
    features_dir = tmp_path / "features"
    _write_feature_cache(features_dir / "train.pt")
    _write_feature_cache(features_dir / "val.pt")

    train_loader, _ = build_feature_loaders(
        features_dir,
        batch_size=2,
        num_workers=0,
        head_type=HEAD_TYPE_TEMPORAL_STAGE_EXPERTS,
        window_size=3,
    )
    batch = next(iter(train_loader))

    assert batch["feature_window"].shape == (2, 3, 8)
    assert batch["phase_center"].shape == (2,)
    assert batch["valid_mask"].shape == (2, 3)


def test_build_feature_loaders_wraps_local_stage_gated_temporal_windows(tmp_path: Path) -> None:
    features_dir = tmp_path / "features"
    _write_feature_cache(features_dir / "train.pt")
    _write_feature_cache(features_dir / "val.pt")

    train_loader, _ = build_feature_loaders(
        features_dir,
        batch_size=2,
        num_workers=0,
        head_type=HEAD_TYPE_TEMPORAL_LOCAL_STAGE_GATED,
        window_size=3,
    )
    batch = next(iter(train_loader))

    assert batch["feature_window"].shape == (2, 3, 8)
    assert batch["phase_center"].shape == (2,)
    assert batch["valid_mask"].shape == (2, 3)


def test_build_feature_loaders_wraps_stage_prior_temporal_windows(tmp_path: Path) -> None:
    features_dir = tmp_path / "features"
    _write_feature_cache(features_dir / "train.pt")
    _write_feature_cache(features_dir / "val.pt")

    train_loader, _ = build_feature_loaders(
        features_dir,
        batch_size=2,
        num_workers=0,
        head_type=HEAD_TYPE_TEMPORAL_STAGE_PRIOR,
        window_size=3,
    )
    batch = next(iter(train_loader))

    assert batch["feature_window"].shape == (2, 3, 8)
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


def test_load_zp_head_local_stage_gated_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "zp_head.pt"
    head = TemporalLocalStageGatedProgressHead(
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
        trunk_depth=2,
        phase_span_priors=[0.1, 0.2, 0.2, 0.2, 0.3],
    )
    torch.save(
        {
            "state_dict": head.state_dict(),
            "head_type": "temporal_local_stage_gated",
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
            "trunk_depth": 2,
            "phase_span_priors": [0.1, 0.2, 0.2, 0.2, 0.3],
        },
        checkpoint_path,
    )

    loaded = load_zp_head(checkpoint_path, device="cpu")
    out = loaded(
        torch.randn(2, 3, 8),
        stage_prior=None,
        valid_mask=torch.tensor([[0.0, 1.0, 1.0], [1.0, 1.0, 0.0]]),
    )

    assert out["phase_logits"].shape == (2, 5)
    assert out["shared_hidden"].shape == (2, 16)


def test_load_zp_head_stage_prior_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "zp_head.pt"
    head = TemporalStagePriorProgressHead(
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
        trunk_depth=1,
        phase_span_priors=[0.1, 0.2, 0.2, 0.2, 0.3],
    )
    torch.save(
        {
            "state_dict": head.state_dict(),
            "head_type": "temporal_stage_prior",
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
            "trunk_depth": 1,
            "phase_span_priors": [0.1, 0.2, 0.2, 0.2, 0.3],
        },
        checkpoint_path,
    )

    loaded = load_zp_head(checkpoint_path, device="cpu")
    out = loaded(torch.randn(2, 3, 8), stage_prior=None)

    assert out["phase_logits"].shape == (2, 5)
    assert out["shared_hidden"].shape == (2, 16)


def test_load_zp_head_stage_expert_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "zp_head.pt"
    head = TemporalStageExpertsProgressHead(
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
        trunk_depth=1,
        phase_span_priors=[0.1, 0.2, 0.2, 0.2, 0.3],
    )
    torch.save(
        {
            "state_dict": head.state_dict(),
            "head_type": "temporal_stage_experts",
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
            "trunk_depth": 1,
            "phase_span_priors": [0.1, 0.2, 0.2, 0.2, 0.3],
        },
        checkpoint_path,
    )

    loaded = load_zp_head(checkpoint_path, device="cpu")
    out = loaded(torch.randn(2, 3, 8), stage_prior=None)

    assert out["phase_logits"].shape == (2, 5)
    assert out["phase_progress_all"].shape == (2, 5)


def test_temporal_checkpoint_selection_prioritizes_progress_quality() -> None:
    best_metrics = {
        "global_progress_mae": 0.1073,
        "progress_mae": 0.1956,
        "late_phase_acc": 0.80,
        "macro_phase_acc": 0.79,
        "loss": 1.13,
    }
    current_metrics = {
        "global_progress_mae": 0.0927,
        "progress_mae": 0.1762,
        "late_phase_acc": 0.78,
        "macro_phase_acc": 0.77,
        "loss": 1.29,
    }

    assert _is_better_temporal_checkpoint(
        current_metrics,
        best_metrics,
        min_delta=1.0e-5,
    )


def test_temporal_checkpoint_tracking_starts_after_stage_only_phase() -> None:
    cfg = TemporalZPHeadTrainerConfig(stage_only_epochs=4, device="cpu")

    assert not _should_track_temporal_checkpoint(epoch=4, cfg=cfg)
    assert _should_track_temporal_checkpoint(epoch=5, cfg=cfg)
