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

"""Train a phase-specific z/p head on pre-extracted VLM prefix features."""

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
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model_phase_specific import PhaseSpecificProgressHead

logger = logging.getLogger(__name__)


def _build_plateau_scheduler(
    optimizer: torch.optim.Optimizer,
    lr_patience: int,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """Build a ReduceLROnPlateau scheduler compatible with older torch."""
    kwargs: dict[str, Any] = {
        "mode": "min",
        "factor": 0.5,
        "patience": lr_patience,
    }
    if "verbose" in inspect.signature(
        torch.optim.lr_scheduler.ReduceLROnPlateau.__init__
    ).parameters:
        kwargs["verbose"] = True
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)


def _gather_phase_progress(
    progress_all: torch.Tensor,
    phase_true: torch.Tensor,
) -> torch.Tensor:
    """Select the phase-specific progress expert for the ground-truth phase."""
    return progress_all.gather(1, phase_true.unsqueeze(1)).squeeze(1)


def _is_better_checkpoint(
    metrics: dict[str, float],
    best_metrics: dict[str, float] | None,
    min_delta: float,
) -> bool:
    """Select the best checkpoint using late-phase-aware metrics."""
    if best_metrics is None:
        return True

    comparisons = [
        ("late_phase_acc", True),
        ("macro_phase_acc", True),
        ("progress_mae", False),
        ("loss", False),
    ]
    for key, higher_is_better in comparisons:
        current = float(metrics[key])
        best = float(best_metrics[key])
        if higher_is_better:
            if current > best + min_delta:
                return True
            if current < best - min_delta:
                return False
        else:
            if current < best - min_delta:
                return True
            if current > best + min_delta:
                return False
    return False


def _compute_metrics(
    phase_logits: torch.Tensor,
    phase_progress_pred: torch.Tensor,
    global_progress_pred: torch.Tensor,
    phase_true: torch.Tensor,
    phase_progress_true: torch.Tensor,
    global_progress_true: torch.Tensor,
    num_phases: int,
) -> dict[str, float]:
    """Compute validation metrics for the phase-specific head."""
    phase_pred = phase_logits.argmax(dim=-1)
    phase_acc = (phase_pred == phase_true).float().mean().item()

    per_phase_acc: dict[str, float] = {}
    per_phase_progress_mae: dict[str, float] = {}
    for ph in range(num_phases):
        mask = phase_true == ph
        if mask.any():
            per_phase_acc[f"phase_{ph}_acc"] = (
                (phase_pred[mask] == phase_true[mask]).float().mean().item()
            )
            per_phase_progress_mae[f"phase_{ph}_progress_mae"] = (
                phase_progress_pred[mask] - phase_progress_true[mask]
            ).abs().mean().item()
        else:
            per_phase_acc[f"phase_{ph}_acc"] = float("nan")
            per_phase_progress_mae[f"phase_{ph}_progress_mae"] = float("nan")

    macro_phase_acc = float(
        np.nanmean([per_phase_acc[f"phase_{ph}_acc"] for ph in range(num_phases)])
    )
    late_phase_acc = float(
        np.nanmean(
            [
                per_phase_acc.get("phase_3_acc", float("nan")),
                per_phase_acc.get("phase_4_acc", float("nan")),
            ]
        )
    )

    progress_mae = (phase_progress_pred - phase_progress_true).abs().mean().item()
    progress_mse = ((phase_progress_pred - phase_progress_true) ** 2).mean().item()
    global_mae = (global_progress_pred - global_progress_true).abs().mean().item()
    global_mse = ((global_progress_pred - global_progress_true) ** 2).mean().item()

    return {
        "phase_acc": phase_acc,
        "macro_phase_acc": macro_phase_acc,
        "late_phase_acc": late_phase_acc,
        "progress_mae": progress_mae,
        "progress_mse": progress_mse,
        "global_progress_mae": global_mae,
        "global_progress_mse": global_mse,
        **per_phase_acc,
        **per_phase_progress_mae,
    }


@torch.no_grad()
def evaluate(
    head: PhaseSpecificProgressHead,
    data_loader: DataLoader,
    phase_criterion: nn.Module,
    progress_criterion: nn.Module,
    global_progress_criterion: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate the phase-specific head on one split."""
    head.eval()
    total_loss = 0.0
    total_samples = 0
    all_logits: list[torch.Tensor] = []
    all_progress: list[torch.Tensor] = []
    all_global: list[torch.Tensor] = []
    all_phase_true: list[torch.Tensor] = []
    all_progress_true: list[torch.Tensor] = []
    all_global_true: list[torch.Tensor] = []

    for features, phase, phase_progress, global_progress in data_loader:
        features = features.to(device)
        phase = phase.to(device)
        phase_progress = phase_progress.to(device)
        global_progress = global_progress.to(device)

        out = head(features)
        selected_progress = _gather_phase_progress(out["phase_progress_all"], phase)
        loss = (
            args.phase_loss_weight * phase_criterion(out["phase_logits"], phase)
            + args.progress_loss_weight
            * progress_criterion(selected_progress, phase_progress)
        )
        if args.global_progress_loss_weight > 0.0:
            loss = loss + args.global_progress_loss_weight * global_progress_criterion(
                out["global_progress"], global_progress
            )

        batch_size = features.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_logits.append(out["phase_logits"].cpu())
        all_progress.append(out["phase_progress"].cpu())
        all_global.append(out["global_progress"].cpu())
        all_phase_true.append(phase.cpu())
        all_progress_true.append(phase_progress.cpu())
        all_global_true.append(global_progress.cpu())

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
    head: PhaseSpecificProgressHead,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    """Train the phase-specific head and return best validation info."""
    if args.use_class_weights:
        phase_counts = torch.zeros(head.num_phases, dtype=torch.float32)
        for _, phase, _, _ in train_loader:
            for ph in range(head.num_phases):
                phase_counts[ph] += (phase == ph).sum().item()
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
        for features, phase, phase_progress, global_progress in pbar:
            features = features.to(device)
            phase = phase.to(device)
            phase_progress = phase_progress.to(device)
            global_progress = global_progress.to(device)

            optimizer.zero_grad()
            out = head(features)
            selected_progress = _gather_phase_progress(out["phase_progress_all"], phase)
            loss = (
                args.phase_loss_weight * phase_criterion(out["phase_logits"], phase)
                + args.progress_loss_weight
                * progress_criterion(selected_progress, phase_progress)
            )
            if args.global_progress_loss_weight > 0.0:
                loss = loss + args.global_progress_loss_weight * global_progress_criterion(
                    out["global_progress"], global_progress
                )

            loss.backward()
            if args.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
            optimizer.step()

            batch_size = features.shape[0]
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
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
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
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
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
        default="/workspace/results/phase_progress_probe/head_phase_specific",
    )
    parser.add_argument("--num_phases", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--trunk_depth", type=int, default=1)
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
        json.dump(vars(args), f, indent=2)

    logger.info("Loading cached features from %s", features_dir)
    train_data = torch.load(features_dir / "train.pt", weights_only=True)
    val_data = torch.load(features_dir / "val.pt", weights_only=True)

    feature_dim = train_data["features"].shape[1]
    logger.info("Feature dim: %d", feature_dim)

    train_dataset = TensorDataset(
        train_data["features"],
        train_data["phase"],
        train_data["phase_progress"],
        train_data["global_progress"],
    )
    val_dataset = TensorDataset(
        val_data["features"],
        val_data["phase"],
        val_data["phase_progress"],
        val_data["global_progress"],
    )

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

    head = PhaseSpecificProgressHead(
        feature_dim=feature_dim,
        num_phases=args.num_phases,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        trunk_depth=args.trunk_depth,
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
                "trunk_depth": args.trunk_depth,
            },
            "args": vars(args),
            "train_info": train_info,
            "head_type": "phase_specific_progress",
        },
        output_dir / "head.pt",
    )

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": train_info["best_epoch"],
                "best_val_loss": train_info["best_val_loss"],
                "history": train_info["history"],
            },
            f,
            indent=2,
        )

    logger.info(
        "Training complete. Best epoch=%d, best val_loss=%.6f. Outputs in %s",
        train_info["best_epoch"],
        train_info["best_val_loss"],
        output_dir,
    )


if __name__ == "__main__":
    main()
