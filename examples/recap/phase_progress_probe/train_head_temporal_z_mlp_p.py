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

"""Train a temporal z->MLP->p head on cached frozen features."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_temporal_z_mlp_p import TemporalZMLPProgressHead
from train_head_temporal_phase_prior import (
    TemporalWindowDataset,
    _build_plateau_scheduler,
    _build_stage_prior,
    _compute_metrics,
    _compute_phase_span_priors,
    _is_better_checkpoint,
)

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(
    head: TemporalZMLPProgressHead,
    data_loader: DataLoader,
    phase_criterion: nn.Module,
    progress_criterion: nn.Module,
    global_progress_criterion: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate the temporal z->MLP->p head on one split."""
    head.eval()
    total_loss = 0.0
    total_samples = 0
    all_logits: list[torch.Tensor] = []
    all_progress: list[torch.Tensor] = []
    all_global: list[torch.Tensor] = []
    all_phase_true: list[torch.Tensor] = []
    all_progress_true: list[torch.Tensor] = []
    all_global_true: list[torch.Tensor] = []

    for batch in data_loader:
        feature_window = batch["feature_window"].to(device)
        phase_center = batch["phase_center"].to(device)
        phase_progress_center = batch["phase_progress_center"].to(device)
        global_progress_center = batch["global_progress_center"].to(device)

        gt_stage_prior = F.one_hot(
            phase_center.long(), num_classes=args.num_phases
        ).to(feature_window.dtype)
        out = head(feature_window, stage_prior=gt_stage_prior)
        loss = (
            args.phase_loss_weight * phase_criterion(out["phase_logits"], phase_center)
            + args.progress_loss_weight
            * progress_criterion(out["phase_progress"], phase_progress_center)
        )
        if args.global_progress_loss_weight > 0.0:
            loss = loss + args.global_progress_loss_weight * global_progress_criterion(
                out["global_progress"], global_progress_center
            )

        batch_size = feature_window.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_logits.append(out["phase_logits"].cpu())
        all_progress.append(out["phase_progress"].cpu())
        all_global.append(out["global_progress"].cpu())
        all_phase_true.append(phase_center.cpu())
        all_progress_true.append(phase_progress_center.cpu())
        all_global_true.append(global_progress_center.cpu())

    avg_loss = total_loss / total_samples
    metrics = _compute_metrics(
        phase_logits=torch.cat(all_logits),
        phase_progress_pred=torch.cat(all_progress),
        global_progress_pred=torch.cat(all_global),
        phase_true=torch.cat(all_phase_true),
        phase_progress_true=torch.cat(all_progress_true),
        global_progress_true=torch.cat(all_global_true),
        num_phases=head.num_phases,
    )
    metrics["loss"] = avg_loss
    return metrics


def train(
    head: TemporalZMLPProgressHead,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    """Train the temporal z->MLP->p head and return best validation info."""
    if args.use_class_weights:
        phase_counts = torch.zeros(head.num_phases, dtype=torch.float32)
        for batch in train_loader:
            phase_center = batch["phase_center"]
            for ph in range(head.num_phases):
                phase_counts[ph] += (phase_center == ph).sum().item()
        weights = 1.0 / (phase_counts + 1.0)
        weights = weights / weights.sum() * head.num_phases
        logger.info("Class weights: %s", weights.tolist())
        phase_criterion = nn.CrossEntropyLoss(
            weight=weights.to(device),
            label_smoothing=args.label_smoothing,
        )
    else:
        phase_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    progress_criterion = nn.SmoothL1Loss(beta=args.progress_beta)
    global_progress_criterion = nn.SmoothL1Loss(beta=args.progress_beta)

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = _build_plateau_scheduler(optimizer, args.lr_patience)

    best_val_loss = float("inf")
    best_epoch = -1
    best_val_metrics: dict[str, float] | None = None
    best_state_dict: dict[str, torch.Tensor] | None = None
    patience_counter = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.max_epochs + 1):
        head.train()
        epoch_loss = 0.0
        epoch_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.max_epochs}")
        for batch in pbar:
            feature_window = batch["feature_window"].to(device)
            phase_center = batch["phase_center"].to(device)
            phase_progress_center = batch["phase_progress_center"].to(device)
            global_progress_center = batch["global_progress_center"].to(device)

            optimizer.zero_grad()

            if epoch <= args.stage_only_epochs:
                out = head(feature_window, stage_prior=None)
                loss = args.phase_loss_weight * phase_criterion(
                    out["phase_logits"], phase_center
                )
            else:
                stage_out = head(feature_window, stage_prior=None)
                stage_prior = _build_stage_prior(
                    phase_window=phase_center.unsqueeze(1),
                    phase_logits_all=stage_out["phase_logits"].unsqueeze(1),
                    valid_mask=torch.ones_like(phase_center, dtype=feature_window.dtype).unsqueeze(1),
                    epoch=epoch,
                    args=args,
                ).squeeze(1)
                out = head(feature_window, stage_prior=stage_prior.to(feature_window.dtype))
                loss = (
                    args.phase_loss_weight * phase_criterion(out["phase_logits"], phase_center)
                    + args.progress_loss_weight
                    * progress_criterion(out["phase_progress"], phase_progress_center)
                )
                if args.global_progress_loss_weight > 0.0:
                    loss = loss + args.global_progress_loss_weight * global_progress_criterion(
                        out["global_progress"], global_progress_center
                    )

            loss.backward()
            if args.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
            optimizer.step()

            batch_size = feature_window.shape[0]
            epoch_loss += loss.item() * batch_size
            epoch_samples += batch_size
            pbar.set_postfix({"loss": epoch_loss / epoch_samples})

        train_loss = epoch_loss / epoch_samples
        val_metrics = evaluate(
            head,
            val_loader,
            phase_criterion,
            progress_criterion,
            global_progress_criterion,
            args,
            device,
        )
        scheduler.step(val_metrics["loss"])

        logger.info(
            "Epoch %d | train_loss=%.4f | val_loss=%.4f | val_phase_acc=%.4f | "
            "val_macro_phase_acc=%.4f | val_late_phase_acc=%.4f | "
            "val_progress_mae=%.4f | val_global_mae=%.4f",
            epoch,
            train_loss,
            val_metrics["loss"],
            val_metrics["phase_acc"],
            val_metrics["macro_phase_acc"],
            val_metrics["late_phase_acc"],
            val_metrics["progress_mae"],
            val_metrics["global_progress_mae"],
        )

        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                **{f"val_{k}": float(v) for k, v in val_metrics.items()},
            }
        )

        if _is_better_checkpoint(
            metrics=val_metrics,
            best_metrics=best_val_metrics,
            min_delta=args.early_stop_delta,
        ):
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_val_metrics = {k: float(v) for k, v in val_metrics.items()}
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in head.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.early_stop_patience:
            logger.info("Early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
            break

    if best_state_dict is not None:
        head.load_state_dict(best_state_dict)

    return {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_metrics": best_val_metrics,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features_dir",
        default="/workspace/results/phase_progress_probe/features",
    )
    parser.add_argument(
        "--output_dir",
        default="/workspace/results/phase_progress_probe/head_temporal_z_mlp_p",
    )
    parser.add_argument("--num_phases", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ffn_dim", type=int, default=512)
    parser.add_argument("--stage_embedding_dim", type=int, default=32)
    parser.add_argument("--progress_hidden_dim", type=int, default=256)
    parser.add_argument("--progress_depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument("--early_stop_delta", type=float, default=1e-5)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--phase_loss_weight", type=float, default=1.0)
    parser.add_argument("--progress_loss_weight", type=float, default=1.0)
    parser.add_argument("--global_progress_loss_weight", type=float, default=0.25)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--use_class_weights", action="store_true", default=True)
    parser.add_argument("--no_class_weights", dest="use_class_weights", action="store_false")
    parser.add_argument("--progress_beta", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--stage_only_epochs", type=int, default=4)
    parser.add_argument("--gt_stage_prior_epochs", type=int, default=6)
    parser.add_argument("--stage_prior_ramp_epochs", type=int, default=10)
    parser.add_argument("--max_pred_stage_prior_weight", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "train_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=float)

    logger.info("Loading cached features from %s", features_dir)
    train_data = torch.load(features_dir / "train.pt", weights_only=True)
    val_data = torch.load(features_dir / "val.pt", weights_only=True)

    feature_dim = train_data["features"].shape[1]
    logger.info("Feature dim: %d", feature_dim)

    phase_span_priors = _compute_phase_span_priors(train_data, args.num_phases)
    logger.info("Phase span priors: %s", phase_span_priors)

    train_dataset = TemporalWindowDataset(train_data, window_size=args.window_size)
    val_dataset = TemporalWindowDataset(val_data, window_size=args.window_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    head = TemporalZMLPProgressHead(
        feature_dim=feature_dim,
        num_phases=args.num_phases,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        window_size=args.window_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        stage_embedding_dim=args.stage_embedding_dim,
        progress_hidden_dim=args.progress_hidden_dim,
        progress_depth=args.progress_depth,
        phase_span_priors=phase_span_priors,
    ).to(device)

    logger.info("Head parameters: %d", sum(p.numel() for p in head.parameters()))

    train_info = train(head, train_loader, val_loader, args, device)

    torch.save(
        {
            "state_dict": head.state_dict(),
            "config": {
                "feature_dim": feature_dim,
                "num_phases": args.num_phases,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "window_size": args.window_size,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "ffn_dim": args.ffn_dim,
                "stage_embedding_dim": args.stage_embedding_dim,
                "progress_hidden_dim": args.progress_hidden_dim,
                "progress_depth": args.progress_depth,
                "phase_span_priors": phase_span_priors,
            },
            "args": vars(args),
            "train_info": train_info,
            "head_type": "temporal_z_mlp_p",
        },
        output_dir / "head.pt",
    )

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": train_info["best_epoch"],
                "best_val_loss": train_info["best_val_loss"],
                "best_val_metrics": train_info["best_val_metrics"],
                "history": train_info["history"],
                "phase_span_priors": phase_span_priors,
            },
            f,
            indent=2,
            default=float,
        )

    logger.info(
        "Training complete. Best epoch=%d, best val_loss=%.6f. Outputs in %s",
        train_info["best_epoch"],
        train_info["best_val_loss"],
        output_dir,
    )


if __name__ == "__main__":
    main()
