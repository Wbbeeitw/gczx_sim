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

"""Train a temporal phase-prior z/p head on cached frozen features."""

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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model_temporal_phase_prior import TemporalPhasePriorProgressHead

logger = logging.getLogger(__name__)


class TemporalWindowDataset(Dataset):
    """Episode-aware temporal window dataset with dense token supervision."""

    def __init__(self, split_data: dict[str, torch.Tensor], window_size: int):
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError(
                f"window_size must be a positive odd integer, got {window_size}"
            )

        self.features = split_data["features"].float()
        self.phase = split_data["phase"].long()
        self.phase_progress = split_data["phase_progress"].float()
        self.global_progress = split_data["global_progress"].float()
        self.episode_index = split_data["episode_index"].long()
        self.frame_index = split_data["frame_index"].long()
        self.window_size = window_size
        self.radius = window_size // 2

        self._episode_to_rows: dict[int, torch.Tensor] = {}
        for ep in torch.unique(self.episode_index).tolist():
            row_idx = torch.nonzero(self.episode_index == int(ep), as_tuple=False).squeeze(1)
            order = torch.argsort(self.frame_index.index_select(0, row_idx))
            self._episode_to_rows[int(ep)] = row_idx.index_select(0, order)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep = int(self.episode_index[idx].item())
        rows = self._episode_to_rows[ep]
        pos = int(torch.nonzero(rows == idx, as_tuple=False).item())

        tokens = []
        phase_tokens = []
        progress_tokens = []
        global_tokens = []
        valid_mask = []
        center_row = rows[pos]

        for offset in range(-self.radius, self.radius + 1):
            tok_pos = pos + offset
            if tok_pos < 0:
                row = rows[0]
                valid = 0.0
            elif tok_pos >= len(rows):
                row = rows[-1]
                valid = 0.0
            else:
                row = rows[tok_pos]
                valid = 1.0

            tokens.append(self.features[row].unsqueeze(0))
            phase_tokens.append(self.phase[row].unsqueeze(0))
            progress_tokens.append(self.phase_progress[row].unsqueeze(0))
            global_tokens.append(self.global_progress[row].unsqueeze(0))
            valid_mask.append(torch.tensor([valid], dtype=torch.float32))

        return {
            "feature_window": torch.cat(tokens, dim=0),
            "phase_window": torch.cat(phase_tokens, dim=0),
            "phase_progress_window": torch.cat(progress_tokens, dim=0),
            "global_progress_window": torch.cat(global_tokens, dim=0),
            "valid_mask": torch.cat(valid_mask, dim=0),
            "phase_center": self.phase[center_row],
            "phase_progress_center": self.phase_progress[center_row],
            "global_progress_center": self.global_progress[center_row],
        }


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
    """Compute validation metrics for center-token predictions."""
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


def _compute_phase_span_priors(split_data: dict[str, torch.Tensor], num_phases: int) -> list[float]:
    """Estimate empirical phase span priors from frame counts."""
    phase = split_data["phase"].long()
    counts = torch.bincount(phase, minlength=num_phases).float()
    counts = counts / counts.sum().clamp(min=1.0)
    return counts.tolist()


def _build_stage_prior(
    phase_window: torch.Tensor,
    phase_logits_all: torch.Tensor | None,
    valid_mask: torch.Tensor,
    epoch: int,
    args: argparse.Namespace,
) -> torch.Tensor:
    """Build stage prior schedule for progress conditioning."""
    gt_onehot = F.one_hot(
        phase_window.long(), num_classes=args.num_phases
    ).to(valid_mask.dtype)
    gt_onehot = gt_onehot * valid_mask.unsqueeze(-1)

    if epoch <= args.stage_only_epochs + args.gt_stage_prior_epochs or phase_logits_all is None:
        return gt_onehot

    pred_probs = F.softmax(phase_logits_all.detach(), dim=-1)
    pred_probs = pred_probs * valid_mask.unsqueeze(-1)
    mix_progress = min(
        1.0,
        max(
            0.0,
            (epoch - args.stage_only_epochs - args.gt_stage_prior_epochs)
            / max(1, args.stage_prior_ramp_epochs),
        ),
    )
    pred_weight = args.max_pred_stage_prior_weight * mix_progress
    gt_weight = 1.0 - pred_weight
    return gt_weight * gt_onehot + pred_weight * pred_probs


def _masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """Masked token-wise cross entropy."""
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_target = target.reshape(-1)
    flat_mask = valid_mask.reshape(-1)
    losses = criterion(flat_logits, flat_target)
    losses = losses * flat_mask
    return losses.sum() / flat_mask.sum().clamp(min=1.0)


def _masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Masked SmoothL1 loss."""
    losses = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    losses = losses * valid_mask
    return losses.sum() / valid_mask.sum().clamp(min=1.0)


@torch.no_grad()
def evaluate(
    head: TemporalPhasePriorProgressHead,
    data_loader: DataLoader,
    phase_criterion: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate the temporal phase-prior head on one split."""
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
        phase_window = batch["phase_window"].to(device)
        phase_progress_window = batch["phase_progress_window"].to(device)
        global_progress_window = batch["global_progress_window"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        phase_center = batch["phase_center"].to(device)
        phase_progress_center = batch["phase_progress_center"].to(device)
        global_progress_center = batch["global_progress_center"].to(device)

        gt_stage_prior = F.one_hot(
            phase_window.long(), num_classes=args.num_phases
        ).to(feature_window.dtype)
        out = head(feature_window, stage_prior=gt_stage_prior)

        loss_stage = _masked_cross_entropy(
            out["phase_logits_all"],
            phase_window,
            valid_mask,
            phase_criterion,
        )
        loss_progress = _masked_smooth_l1(
            out["phase_progress_all_tokens"],
            phase_progress_window,
            valid_mask,
            beta=args.progress_beta,
        )
        loss_global = F.smooth_l1_loss(
            out["global_progress"],
            global_progress_center,
            beta=args.progress_beta,
        )
        loss = (
            args.phase_loss_weight * loss_stage
            + args.progress_loss_weight * loss_progress
            + args.global_progress_loss_weight * loss_global
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
    head: TemporalPhasePriorProgressHead,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    """Train the temporal phase-prior head and return best validation info."""
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
            reduction="none",
        )
    else:
        phase_criterion = nn.CrossEntropyLoss(
            label_smoothing=args.label_smoothing,
            reduction="none",
        )

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
            phase_window = batch["phase_window"].to(device)
            phase_progress_window = batch["phase_progress_window"].to(device)
            global_progress_center = batch["global_progress_center"].to(device)
            valid_mask = batch["valid_mask"].to(device)

            optimizer.zero_grad()

            if epoch <= args.stage_only_epochs:
                out = head(feature_window, stage_prior=None)
                loss_stage = _masked_cross_entropy(
                    out["phase_logits_all"],
                    phase_window,
                    valid_mask,
                    phase_criterion,
                )
                loss = args.phase_loss_weight * loss_stage
            else:
                stage_out = head(feature_window, stage_prior=None)
                stage_prior = _build_stage_prior(
                    phase_window=phase_window,
                    phase_logits_all=stage_out["phase_logits_all"],
                    valid_mask=valid_mask,
                    epoch=epoch,
                    args=args,
                )
                out = head(feature_window, stage_prior=stage_prior)
                loss_stage = _masked_cross_entropy(
                    out["phase_logits_all"],
                    phase_window,
                    valid_mask,
                    phase_criterion,
                )
                loss_progress = _masked_smooth_l1(
                    out["phase_progress_all_tokens"],
                    phase_progress_window,
                    valid_mask,
                    beta=args.progress_beta,
                )
                loss_global = F.smooth_l1_loss(
                    out["global_progress"],
                    global_progress_center,
                    beta=args.progress_beta,
                )
                loss = (
                    args.phase_loss_weight * loss_stage
                    + args.progress_loss_weight * loss_progress
                    + args.global_progress_loss_weight * loss_global
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
        val_metrics = evaluate(head, val_loader, phase_criterion, args, device)
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
        default="/workspace/results/phase_progress_probe/head_temporal_phase_prior",
    )
    parser.add_argument("--num_phases", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--stage_layers", type=int, default=2)
    parser.add_argument("--progress_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ffn_dim", type=int, default=512)
    parser.add_argument("--stage_embedding_dim", type=int, default=32)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
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

    head = TemporalPhasePriorProgressHead(
        feature_dim=feature_dim,
        num_phases=args.num_phases,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        window_size=args.window_size,
        stage_layers=args.stage_layers,
        progress_layers=args.progress_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        stage_embedding_dim=args.stage_embedding_dim,
        attention_dropout=args.attention_dropout,
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
                "stage_layers": args.stage_layers,
                "progress_layers": args.progress_layers,
                "num_heads": args.num_heads,
                "ffn_dim": args.ffn_dim,
                "stage_embedding_dim": args.stage_embedding_dim,
                "attention_dropout": args.attention_dropout,
                "phase_span_priors": phase_span_priors,
            },
            "args": vars(args),
            "train_info": train_info,
            "head_type": "temporal_phase_prior_progress",
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
