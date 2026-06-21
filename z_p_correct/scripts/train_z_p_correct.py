#!/usr/bin/env python
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic two-stage trainer for one phase/progress head + fusion MLP.

Usage:
    python scripts/train_z_p_correct.py \
        --head_type our_phase_progress \
        --features_dir /path/to/features \
        --advantages_path /path/to/advantages.parquet \
        --output_dir /path/to/output

The script performs:
    1. Train the selected z/p head on cached features.
    2. Freeze the head.
    3. Train the fusion MLP on train-split predictions + raw critic logits.
    4. Evaluate on the val split and write a report.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

# Ensure package root (z_p_correct/) is on path so that `import z_p_correct`
# resolves to the inner regular package z_p_correct/z_p_correct/.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from z_p_correct.analysis.evaluator import FusionEvaluator
from z_p_correct.analysis.predictor import PhaseProgressPredictor
from z_p_correct.data import build_feature_loaders, build_fusion_loaders
from z_p_correct.models import heads  # noqa: F401  # registers heads
from z_p_correct.models.fusion.logit_fusion import LogitFusionMLP
from z_p_correct.registry import get_head_class, list_heads
from z_p_correct.training.fusion_trainer import FusionTrainer, FusionTrainerConfig
from z_p_correct.training.head_trainer import HeadTrainer, HeadTrainerConfig
from z_p_correct.utils import make_output_dir

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a z/p head + fusion MLP.")
    parser.add_argument(
        "--head_type",
        required=True,
        choices=list_heads(),
        help="Which phase/progress head architecture to train.",
    )
    parser.add_argument(
        "--features_dir",
        required=True,
        help="Directory containing train.pt and val.pt feature caches.",
    )
    parser.add_argument(
        "--advantages_path",
        required=True,
        help="Path to source advantages parquet with value_logits_current.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Root output directory. A head_type subdirectory is created.",
    )
    parser.add_argument("--num_phases", type=int, default=5)
    parser.add_argument("--num_bins", type=int, default=201)
    parser.add_argument("--return_min", type=float, default=-700.0)
    parser.add_argument("--return_max", type=float, default=0.0)

    # Head architecture.
    parser.add_argument("--head_hidden_dim", type=int, default=256)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--head_trunk_depth", type=int, default=1)
    parser.add_argument("--head_window_size", type=int, default=5)
    parser.add_argument("--head_num_layers", type=int, default=2)
    parser.add_argument("--head_num_heads", type=int, default=4)
    parser.add_argument("--head_ffn_dim", type=int, default=512)

    # Fusion architecture.
    parser.add_argument("--fusion_hidden_dim", type=int, default=256)
    parser.add_argument("--fusion_depth", type=int, default=2)
    parser.add_argument("--fusion_dropout", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0)

    # Training.
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--head_max_epochs", type=int, default=200)
    parser.add_argument("--head_early_stop_patience", type=int, default=20)
    parser.add_argument("--fusion_lr", type=float, default=1e-3)
    parser.add_argument("--fusion_max_epochs", type=int, default=200)
    parser.add_argument("--fusion_early_stop_patience", type=int, default=20)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip_head",
        action="store_true",
        help="Skip head training and load from --head_checkpoint.",
    )
    parser.add_argument("--head_checkpoint", default=None)
    parser.add_argument(
        "--skip_fusion",
        action="store_true",
        help="Skip fusion training.",
    )
    parser.add_argument("--fusion_checkpoint", default=None)
    parser.add_argument(
        "--window_size",
        type=int,
        default=1,
        help="Temporal window size. Must be odd. Used by temporal heads.",
    )
    return parser.parse_args()


def _save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _build_head(head_type: str, feature_dim: int, args: argparse.Namespace) -> torch.nn.Module:
    cls = get_head_class(head_type)
    kwargs: dict[str, Any] = {
        "feature_dim": feature_dim,
        "num_phases": args.num_phases,
        "hidden_dim": args.head_hidden_dim,
        "dropout": args.head_dropout,
    }
    if head_type == "shared_mlp":
        kwargs["trunk_depth"] = args.head_trunk_depth
    elif head_type == "temporal_phase_prior":
        kwargs.update(
            {
                "window_size": args.head_window_size,
                "stage_layers": args.head_num_layers,
                "progress_layers": args.head_num_layers,
                "num_heads": args.head_num_heads,
                "ffn_dim": args.head_ffn_dim,
            }
        )
    elif head_type == "temporal_z_mlp_p":
        kwargs.update(
            {
                "window_size": args.head_window_size,
                "num_layers": args.head_num_layers,
                "num_heads": args.head_num_heads,
                "ffn_dim": args.head_ffn_dim,
            }
        )
    else:
        raise ValueError(f"Unsupported head_type: {head_type}")
    return cls(**kwargs)


def _add_fused_value_to_predictions(
    predictions_df: pd.DataFrame,
    advantages_df: pd.DataFrame,
    head: torch.nn.Module,
    fusion: LogitFusionMLP,
    atoms: torch.Tensor,
    alpha: float,
    device: str,
    batch_size: int = 512,
) -> pd.DataFrame:
    """Add a ``value_fused`` column to predictions using the trained fusion MLP."""
    import numpy as np

    required = {"episode_index", "frame_index", "phase_probs_pred", "phase_progress_pred", "global_progress_pred"}
    missing = required - set(predictions_df.columns)
    if missing:
        raise ValueError(f"Predictions missing columns: {sorted(missing)}")

    adv_required = {"episode_index", "frame_index", "value_logits_current"}
    missing_adv = adv_required - set(advantages_df.columns)
    if missing_adv:
        raise ValueError(f"Advantages missing columns: {sorted(missing_adv)}")

    merged = predictions_df.merge(
        advantages_df[["episode_index", "frame_index", "value_logits_current"]],
        on=["episode_index", "frame_index"],
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlap between predictions and advantages.")

    merged = merged.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    raw_logits = np.stack(merged["value_logits_current"].to_numpy()).astype(np.float32)
    phase_probs = np.stack(merged["phase_probs_pred"].to_numpy()).astype(np.float32)
    phase_progress = merged["phase_progress_pred"].to_numpy(dtype=np.float32)
    global_progress = merged["global_progress_pred"].to_numpy(dtype=np.float32)

    device_obj = torch.device(device)
    head.to(device_obj).eval()
    fusion.to(device_obj).eval()
    atoms_t = atoms.to(device_obj)

    fused_values = []
    with torch.no_grad():
        for start in range(0, len(merged), batch_size):
            end = start + batch_size
            logits_t = torch.tensor(raw_logits[start:end], device=device_obj)
            probs_t = torch.tensor(phase_probs[start:end], device=device_obj)
            prog_t = torch.tensor(phase_progress[start:end], device=device_obj)
            glob_t = torch.tensor(global_progress[start:end], device=device_obj)

            delta = fusion(logits_t, probs_t, prog_t, glob_t)
            fused_logits = logits_t + alpha * delta
            probs = torch.softmax(fused_logits, dim=-1)
            values = (probs * atoms_t.unsqueeze(0)).sum(dim=-1)
            fused_values.append(values.cpu().numpy())

    merged["value_fused"] = np.concatenate(fused_values).astype(np.float64)

    # Merge value_fused back into original predictions (preserving row order).
    out_df = predictions_df.merge(
        merged[["episode_index", "frame_index", "value_fused"]],
        on=["episode_index", "frame_index"],
        how="left",
    )
    return out_df


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    torch.manual_seed(args.seed)

    output_dir = make_output_dir(args.output_dir, args.head_type)
    logger.info("Output directory: %s", output_dir)

    # ------------------------------------------------------------------
    # Stage 1: train z/p head.
    # ------------------------------------------------------------------
    head_path = output_dir / "head.pt"
    head_metrics_path = output_dir / "head_metrics.json"

    # Infer window size from head type.
    if args.head_type == "shared_mlp":
        window_size = 1
    else:
        window_size = args.head_window_size
    if window_size > 1 and window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}")

    train_loader, val_loader = build_feature_loaders(
        args.features_dir,
        window_size=window_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    feature_dim = train_loader.dataset.feature_dim

    if args.skip_head and args.head_checkpoint:
        logger.info("Loading head from %s", args.head_checkpoint)
        head = _build_head(args.head_type, feature_dim, args)
        head.load_state_dict(torch.load(args.head_checkpoint, map_location="cpu", weights_only=False))
        head = head.to(args.device)
    else:
        logger.info("Training head: %s", args.head_type)
        head = _build_head(args.head_type, feature_dim, args).to(args.device)
        head_cfg = HeadTrainerConfig(
            lr=args.head_lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            max_epochs=args.head_max_epochs,
            early_stop_patience=args.head_early_stop_patience,
            device=args.device,
        )
        head_trainer = HeadTrainer(head, head_cfg)
        head_result = head_trainer.fit(train_loader, val_loader)
        torch.save(head.state_dict(), head_path)
        _save_json(head_result, head_metrics_path)
        logger.info(
            "Head best epoch=%d best_val_loss=%.6f",
            head_result["best_epoch"],
            head_result["best_val_loss"],
        )

    # ------------------------------------------------------------------
    # Predict z/p for both splits and save.
    # ------------------------------------------------------------------
    from z_p_correct.data.feature_cache import FeatureCache

    predictor = PhaseProgressPredictor(head, device=args.device, batch_size=args.batch_size)
    train_cache = FeatureCache(str(Path(args.features_dir) / "train.pt"), window_size=window_size)
    val_cache = FeatureCache(str(Path(args.features_dir) / "val.pt"), window_size=window_size)

    train_pred_df = predictor.predict(train_cache, split="train")
    val_pred_df = predictor.predict(val_cache, split="val")
    predictions_df = pd.concat([train_pred_df, val_pred_df], ignore_index=True)
    predictions_path = output_dir / "predictions.parquet"
    predictions_df.to_parquet(predictions_path, index=False)
    logger.info("Saved predictions to %s", predictions_path)

    if args.skip_fusion:
        logger.info("Fusion training skipped.")
        return

    # ------------------------------------------------------------------
    # Stage 2: freeze head, train fusion MLP.
    # ------------------------------------------------------------------
    logger.info("Training fusion MLP (head frozen).")
    advantages_df = pd.read_parquet(args.advantages_path)

    fusion_train_loader, fusion_val_loader = build_fusion_loaders(
        predictions_df,
        advantages_df,
        return_min=args.return_min,
        return_max=args.return_max,
        batch_size=args.batch_size,
    )

    atoms = torch.linspace(args.return_min, args.return_max, args.num_bins)
    fusion = LogitFusionMLP(
        num_bins=args.num_bins,
        num_phases=args.num_phases,
        hidden_dim=args.fusion_hidden_dim,
        depth=args.fusion_depth,
        dropout=args.fusion_dropout,
    )

    if args.fusion_checkpoint:
        logger.info("Loading fusion from %s", args.fusion_checkpoint)
        fusion.load_state_dict(torch.load(args.fusion_checkpoint, map_location="cpu", weights_only=False))
        fusion = fusion.to(args.device)

    fusion_cfg = FusionTrainerConfig(
        lr=args.fusion_lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.fusion_max_epochs,
        early_stop_patience=args.fusion_early_stop_patience,
        alpha=args.alpha,
        device=args.device,
    )
    fusion_trainer = FusionTrainer(head, fusion, atoms, fusion_cfg)
    fusion_result = fusion_trainer.fit(fusion_train_loader, fusion_val_loader)

    fusion_path = output_dir / "fusion.pt"
    torch.save(fusion_trainer.fusion.state_dict(), fusion_path)
    fusion_metrics_path = output_dir / "fusion_metrics.json"
    _save_json(fusion_result, fusion_metrics_path)
    logger.info(
        "Fusion best epoch=%d best_val_loss=%.6f",
        fusion_result["best_epoch"],
        fusion_result["best_val_loss"],
    )

    # ------------------------------------------------------------------
    # Final strict evaluation.
    # ------------------------------------------------------------------
    evaluator = FusionEvaluator(
        head=head,
        fusion=fusion_trainer.fusion,
        atoms=atoms,
        alpha=args.alpha,
        device=args.device,
        batch_size=args.batch_size,
    )
    final_metrics = evaluator.evaluate_predictions(
        predictions_df=predictions_df[predictions_df["split"] == "val"],
        advantages_df=advantages_df,
        return_min=args.return_min,
        return_max=args.return_max,
    )
    _save_json(final_metrics, output_dir / "eval_report.json")
    logger.info(
        "Val | raw_mse=%.6f fused_mse=%.6f improvement=%.2f%%",
        final_metrics["raw_value_mse"],
        final_metrics["value_mse"],
        final_metrics["improvement_pct"],
    )

    # ------------------------------------------------------------------
    # Export fused value predictions for downstream ReCap tag export.
    # ------------------------------------------------------------------
    logger.info("Exporting fused value predictions.")
    predictions_with_value_df = _add_fused_value_to_predictions(
        predictions_df=predictions_df,
        advantages_df=advantages_df,
        head=head,
        fusion=fusion_trainer.fusion,
        atoms=atoms,
        alpha=args.alpha,
        device=args.device,
        batch_size=args.batch_size,
    )
    predictions_with_value_df.to_parquet(predictions_path, index=False)
    logger.info(
        "Updated predictions saved to %s (with value_fused column)", predictions_path
    )


if __name__ == "__main__":
    main()
