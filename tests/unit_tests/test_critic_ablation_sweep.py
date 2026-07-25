from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from revalue_test_utils import install_omegaconf_stub

install_omegaconf_stub()

from examples.recap.process.run_critic_ablation_sweep import (
    SweepConfig,
    VARIANT_FLAGS,
    _prediction_is_complete,
    evaluate_predictions,
    train_variant_task,
)
from rlinf.revalue.models import TemporalLocalStageGatedProgressHead
from rlinf.revalue.pipeline.train import load_zp_head


def _head(**overrides) -> TemporalLocalStageGatedProgressHead:
    kwargs = {
        "feature_dim": 8,
        "num_phases": 4,
        "hidden_dim": 16,
        "dropout": 0.0,
        "window_size": 3,
        "num_layers": 1,
        "num_heads": 4,
        "ffn_dim": 32,
        "stage_embedding_dim": 4,
        "progress_hidden_dim": 16,
        "progress_depth": 1,
        "trunk_depth": 1,
        "phase_span_priors": [0.1, 0.2, 0.3, 0.4],
    }
    kwargs.update(overrides)
    return TemporalLocalStageGatedProgressHead(**kwargs)


def _sweep_config() -> SweepConfig:
    return SweepConfig(
        num_tasks=1,
        num_phases=4,
        return_min=-900.0,
        return_max=0.0,
        value_min=-1.0,
        value_max=0.0,
        num_bins=201,
        boundary_window=0,
        seed=42,
        batch_size=2,
        prediction_batch_size=2,
        max_epochs=1,
        early_stop_patience=1,
        hidden_dim=16,
        dropout=0.0,
        trunk_depth=1,
        window_size=3,
        num_layers=1,
        num_heads=4,
        ffn_dim=32,
        stage_embedding_dim=4,
        progress_hidden_dim=16,
        progress_depth=1,
        use_class_weights=False,
        fusion_hidden_dim=16,
        fusion_depth=1,
        fusion_dropout=0.0,
        fusion_alpha=1.0,
        device="cpu",
    )


def test_structural_ablation_removes_local_and_motion_modules() -> None:
    without_local = _head(use_local_temporal=False)
    without_motion = _head(use_motion_difference=False)

    assert without_local.temporal_encoder is None
    assert without_local.position_embedding is None
    assert without_motion.motion_proj is None
    assert without_motion.shared_input_proj[0].in_features == 16

    window = torch.randn(2, 3, 8)
    valid = torch.ones(2, 3)
    assert without_local(window, valid_mask=valid)["phase_logits"].shape == (2, 4)
    assert without_motion(window, valid_mask=valid)["phase_logits"].shape == (2, 4)


def test_phase_and_progress_ablations_use_neutral_inputs_without_label_leakage(
) -> None:
    without_phase = _head(use_phase_head=False).eval()
    without_progress = _head(use_progress_head=False).eval()
    window = torch.randn(2, 3, 8)
    stage_prior = torch.nn.functional.one_hot(
        torch.tensor([1, 3]), num_classes=4
    ).float()

    phase_default = without_phase(window)
    phase_with_gt_prior = without_phase(window, stage_prior=stage_prior)
    torch.testing.assert_close(
        phase_default["phase_progress"], phase_with_gt_prior["phase_progress"]
    )
    torch.testing.assert_close(
        phase_default["phase_probs"], torch.full((2, 4), 0.25)
    )
    torch.testing.assert_close(
        phase_default["global_progress"], torch.full((2,), 0.5)
    )

    progress_out = without_progress(window)
    torch.testing.assert_close(
        progress_out["phase_progress"], torch.full((2,), 0.5)
    )
    assert without_progress.progress_head is None


def test_ablation_flags_survive_checkpoint_loading(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "zp_head.pt"
    head = _head(use_motion_difference=False, use_progress_head=False)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "head_type": "temporal_local_stage_gated",
            "feature_dim": 8,
            "num_phases": 4,
            "hidden_dim": 16,
            "dropout": 0.0,
            "window_size": 3,
            "num_layers": 1,
            "num_heads": 4,
            "ffn_dim": 32,
            "stage_embedding_dim": 4,
            "progress_hidden_dim": 16,
            "progress_depth": 1,
            "trunk_depth": 1,
            "phase_span_priors": [0.1, 0.2, 0.3, 0.4],
            "use_local_temporal": True,
            "use_motion_difference": False,
            "use_phase_head": True,
            "use_progress_head": False,
        },
        checkpoint_path,
    )

    loaded = load_zp_head(checkpoint_path, device="cpu")

    assert loaded.use_motion_difference is False
    assert loaded.use_progress_head is False
    assert loaded.motion_proj is None
    assert loaded.progress_head is None


def test_evaluation_inverse_maps_values_and_marks_exact_boundary(
    tmp_path: Path,
) -> None:
    val_index = pd.DataFrame(
        {
            "task": ["task0"] * 4,
            "episode_index": [0] * 4,
            "frame_index": [0, 1, 2, 3],
            "phase_true": [0, 0, 1, 1],
        }
    )
    target = np.asarray([-900.0, -600.0, -300.0, 0.0])
    normalized = target / 900.0
    advantages = val_index[["episode_index", "frame_index"]].copy()
    advantages["return"] = target
    advantages["value_current"] = normalized
    predictions = val_index[["episode_index", "frame_index"]].copy()
    predictions["value_fused"] = normalized + 0.1
    predictions_path = tmp_path / "predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)

    raw, _ = evaluate_predictions(
        variant="raw",
        predictions_path=predictions_path,
        advantages=advantages,
        val_index=val_index,
        cfg=_sweep_config(),
    )
    fused, by_task = evaluate_predictions(
        variant="full",
        predictions_path=predictions_path,
        advantages=advantages,
        val_index=val_index,
        cfg=_sweep_config(),
    )

    assert raw["held_out_mae"] == pytest.approx(0.0)
    assert fused["held_out_mae"] == pytest.approx(90.0)
    assert fused["boundary_mae"] == pytest.approx(90.0)
    assert fused["boundary_frames"] == 1
    assert by_task[0]["boundary_frames"] == 1


def test_prediction_resume_rejects_duplicate_or_incomplete_artifacts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predictions.parquet"
    complete = pd.DataFrame(
        {
            "split": ["val", "val"],
            "episode_index": [0, 0],
            "frame_index": [0, 1],
            "value_fused": [-0.5, -0.4],
        }
    )
    complete.to_parquet(path, index=False)
    assert _prediction_is_complete(path)

    pd.concat([complete, complete.iloc[[0]]], ignore_index=True).to_parquet(
        path, index=False
    )
    assert not _prediction_is_complete(path)


def test_one_task_ablation_training_prediction_smoke(tmp_path: Path) -> None:
    revalue_root = tmp_path / "revalue"
    features_dir = revalue_root / "features" / "task0"
    features_dir.mkdir(parents=True)
    atoms = torch.linspace(-1.0, 0.0, 201)
    advantage_rows = []

    def write_cache(path: Path, episode: int) -> None:
        frames = torch.arange(6)
        returns = torch.linspace(-900.0, 0.0, 6)
        values = returns / 900.0
        logits = -100.0 * (atoms[None, :] - values[:, None]) ** 2
        torch.save(
            {
                "features": torch.randn(6, 8),
                "episode_index": torch.full((6,), episode, dtype=torch.long),
                "frame_index": frames,
                "phase": torch.tensor([0, 0, 1, 1, 2, 3]),
                "phase_progress": torch.tensor(
                    [0.0, 1.0, 0.0, 1.0, 0.5, 1.0]
                ),
                "global_progress": torch.linspace(0.0, 1.0, 6),
                "atoms": atoms,
            },
            path,
        )
        for frame, target, value, row_logits in zip(
            frames, returns, values, logits, strict=True
        ):
            advantage_rows.append(
                {
                    "episode_index": episode,
                    "frame_index": int(frame),
                    "return": float(target),
                    "value_current": float(value),
                    "value_logits_current": row_logits.numpy(),
                }
            )

    write_cache(features_dir / "train.pt", 0)
    write_cache(features_dir / "val.pt", 1)
    advantages_path = tmp_path / "advantages.parquet"
    pd.DataFrame(advantage_rows).to_parquet(advantages_path, index=False)

    predictions = train_variant_task(
        variant="without_phase_z",
        task="task0",
        flags=VARIANT_FLAGS["without_phase_z"],
        revalue_root=revalue_root,
        advantages_path=advantages_path,
        output_root=tmp_path / "output",
        cfg=_sweep_config(),
        force=False,
    )

    assert _prediction_is_complete(predictions)
    result = pd.read_parquet(predictions)
    assert len(result) == 12
    assert set(result["split"]) == {"train", "val"}
