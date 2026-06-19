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

"""Jointly train z/p heads and 201-bin logit-space fusion on cached features."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from logit_fusion_model import LogitFusionMLP
from model import PhaseProgressHead
from predict_and_analyze import _compute_prediction_metrics
from train_logit_fusion_mlp import _normalize_return

logger = logging.getLogger(__name__)


class JointLogitFusionModel(nn.Module):
    """Joint model: shared frozen features -> z/p heads -> logit-space fusion."""

    def __init__(
        self,
        feature_dim: int,
        num_bins: int = 201,
        num_phases: int = 5,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.1,
        head_trunk_depth: int = 1,
        fusion_hidden_dim: int = 256,
        fusion_dropout: float = 0.1,
        fusion_depth: int = 2,
    ):
        super().__init__()
        self.head = PhaseProgressHead(
            feature_dim=feature_dim,
            num_phases=num_phases,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
            trunk_depth=head_trunk_depth,
        )
        self.fusion = LogitFusionMLP(
            num_bins=num_bins,
            num_phases=num_phases,
            hidden_dim=fusion_hidden_dim,
            dropout=fusion_dropout,
            depth=fusion_depth,
        )

    def forward(
        self,
        features: torch.Tensor,
        raw_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict z/p and fused 201-bin logits from frozen features."""
        head_out = self.head(features)
        phase_probs = F.softmax(head_out["phase_logits"], dim=-1)
        delta_logits = self.fusion(
            raw_logits,
            phase_probs,
            head_out["phase_progress"],
            head_out["global_progress"],
        )
        return {
            **head_out,
            "phase_probs": phase_probs,
            "delta_logits": delta_logits,
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


def _load_split_dataframe(
    features_path: Path,
    adv_df: pd.DataFrame,
    return_min: float,
    return_max: float,
) -> pd.DataFrame:
    """Align cached split metadata with exported logits/value rows."""
    feature_data = torch.load(features_path, weights_only=True)
    rows = pd.DataFrame(
        {
            "row_index": np.arange(len(feature_data["episode_index"]), dtype=np.int64),
            "episode_index": feature_data["episode_index"].numpy(),
            "frame_index": feature_data["frame_index"].numpy(),
            "phase_true": feature_data["phase"].numpy(),
            "phase_progress_true": feature_data["phase_progress"].numpy(),
            "global_progress_true": feature_data["global_progress"].numpy(),
        }
    )
    merged = rows.merge(
        adv_df,
        on=["episode_index", "frame_index"],
        how="inner",
    )
    merged["return_norm"] = _normalize_return(
        merged["return"], return_min=return_min, return_max=return_max
    )
    return merged.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)


def _build_joint_dataset(
    features_path: Path,
    advantages_path: Path,
    return_min: float,
    return_max: float,
) -> tuple[TensorDataset, dict[str, Any]]:
    """Create split-aligned tensors for joint training/evaluation."""
    feature_data = torch.load(features_path, weights_only=True)
    adv_df = pd.read_parquet(advantages_path)
    merged = _load_split_dataframe(
        features_path=features_path,
        adv_df=adv_df,
        return_min=return_min,
        return_max=return_max,
    )
    row_index = torch.tensor(merged["row_index"].values, dtype=torch.long)
    raw_logits = np.stack(merged["value_logits_current"].to_numpy()).astype(np.float32)
    raw_value = torch.tensor(merged["value_current"].values, dtype=torch.float32)
    return_norm = torch.tensor(merged["return_norm"].values, dtype=torch.float32)

    dataset = TensorDataset(
        feature_data["features"].float().index_select(0, row_index),
        torch.tensor(raw_logits, dtype=torch.float32),
        torch.tensor(merged["phase_true"].values, dtype=torch.long),
        torch.tensor(merged["phase_progress_true"].values, dtype=torch.float32),
        torch.tensor(merged["global_progress_true"].values, dtype=torch.float32),
        raw_value,
        return_norm,
    )
    payload = {
        "merged": merged,
        "feature_data": feature_data,
    }
    return dataset, payload


def _fused_value_from_logits(
    raw_logits: torch.Tensor,
    delta_logits: torch.Tensor,
    atoms: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Convert fused logits into scalar value."""
    fused_logits = raw_logits + alpha * delta_logits
    probs = torch.softmax(fused_logits, dim=-1)
    return (probs * atoms.unsqueeze(0)).sum(dim=-1)


def _build_phase_criterion(
    train_loader: DataLoader,
    num_phases: int,
    use_class_weights: bool,
    label_smoothing: float,
    device: torch.device,
) -> nn.Module:
    """Construct the phase classification criterion."""
    if not use_class_weights:
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    phase_counts = torch.zeros(num_phases, dtype=torch.float32)
    for _, _, phase, _, _, _, _ in train_loader:
        for ph in range(num_phases):
            phase_counts[ph] += (phase == ph).sum().item()
    weights = 1.0 / (phase_counts + 1.0)
    weights = weights / weights.sum() * num_phases
    logger.info("Class weights: %s", weights.tolist())
    return nn.CrossEntropyLoss(
        weight=weights.to(device),
        label_smoothing=label_smoothing,
    )


@torch.no_grad()
def evaluate(
    model: JointLogitFusionModel,
    data_loader: DataLoader,
    phase_criterion: nn.Module,
    progress_criterion: nn.Module,
    global_progress_criterion: nn.Module,
    atoms: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate joint z/p + fusion training on one split."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_phase_logits: list[torch.Tensor] = []
    all_phase_progress: list[torch.Tensor] = []
    all_global_progress: list[torch.Tensor] = []
    all_phase_true: list[torch.Tensor] = []
    all_phase_progress_true: list[torch.Tensor] = []
    all_global_progress_true: list[torch.Tensor] = []
    all_fused_value: list[torch.Tensor] = []
    all_raw_value: list[torch.Tensor] = []
    all_return_norm: list[torch.Tensor] = []

    for (
        features,
        raw_logits,
        phase_true,
        phase_progress_true,
        global_progress_true,
        raw_value,
        return_norm,
    ) in data_loader:
        features = features.to(device)
        raw_logits = raw_logits.to(device)
        phase_true = phase_true.to(device)
        phase_progress_true = phase_progress_true.to(device)
        global_progress_true = global_progress_true.to(device)
        raw_value = raw_value.to(device)
        return_norm = return_norm.to(device)

        out = model(features, raw_logits)
        fused_value = _fused_value_from_logits(
            raw_logits,
            out["delta_logits"],
            atoms.to(device),
            alpha=args.alpha,
        )
        fusion_loss = F.mse_loss(fused_value, return_norm)
        loss = (
            args.phase_loss_weight * phase_criterion(out["phase_logits"], phase_true)
            + args.progress_loss_weight
            * progress_criterion(out["phase_progress"], phase_progress_true)
            + args.fusion_loss_weight * fusion_loss
        )
        if args.global_progress_loss_weight > 0.0:
            loss = loss + args.global_progress_loss_weight * global_progress_criterion(
                out["global_progress"], global_progress_true
            )

        batch_size = features.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_phase_logits.append(out["phase_logits"].cpu())
        all_phase_progress.append(out["phase_progress"].cpu())
        all_global_progress.append(out["global_progress"].cpu())
        all_phase_true.append(phase_true.cpu())
        all_phase_progress_true.append(phase_progress_true.cpu())
        all_global_progress_true.append(global_progress_true.cpu())
        all_fused_value.append(fused_value.cpu())
        all_raw_value.append(raw_value.cpu())
        all_return_norm.append(return_norm.cpu())

    metrics = _compute_prediction_metrics(
        pd.DataFrame(
            {
                "phase_true": torch.cat(all_phase_true).numpy(),
                "phase_pred": torch.cat(all_phase_logits).argmax(dim=-1).numpy(),
                "phase_progress_true": torch.cat(all_phase_progress_true).numpy(),
                "phase_progress_pred": torch.cat(all_phase_progress).numpy(),
                "global_progress_true": torch.cat(all_global_progress_true).numpy(),
                "global_progress_pred": torch.cat(all_global_progress).numpy(),
            }
        ),
        num_phases=model.head.num_phases,
    )
    fused_all = torch.cat(all_fused_value)
    raw_all = torch.cat(all_raw_value)
    return_all = torch.cat(all_return_norm)
    metrics.update(
        {
            "loss": total_loss / total_samples,
            "fusion_value_mse": ((fused_all - return_all) ** 2).mean().item(),
            "raw_value_mse": ((raw_all - return_all) ** 2).mean().item(),
            "fusion_improvement_pct": (
                (1 - ((fused_all - return_all) ** 2).mean() / ((raw_all - return_all) ** 2).mean())
                * 100
            ).item()
            if ((raw_all - return_all) ** 2).mean().item() > 0
            else 0.0,
        }
    )
    return metrics


def train(
    model: JointLogitFusionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    atoms: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    """Jointly train phase/progress heads and logit-space fusion."""
    phase_criterion = _build_phase_criterion(
        train_loader=train_loader,
        num_phases=model.head.num_phases,
        use_class_weights=args.use_class_weights,
        label_smoothing=args.label_smoothing,
        device=device,
    )
    progress_criterion = nn.MSELoss()
    global_progress_criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = _build_plateau_scheduler(optimizer, args.lr_patience)

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    patience_counter = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.max_epochs}")
        for (
            features,
            raw_logits,
            phase_true,
            phase_progress_true,
            global_progress_true,
            _raw_value,
            return_norm,
        ) in pbar:
            features = features.to(device)
            raw_logits = raw_logits.to(device)
            phase_true = phase_true.to(device)
            phase_progress_true = phase_progress_true.to(device)
            global_progress_true = global_progress_true.to(device)
            return_norm = return_norm.to(device)

            optimizer.zero_grad()
            out = model(features, raw_logits)
            fused_value = _fused_value_from_logits(
                raw_logits,
                out["delta_logits"],
                atoms.to(device),
                alpha=args.alpha,
            )
            fusion_loss = F.mse_loss(fused_value, return_norm)
            loss = (
                args.phase_loss_weight * phase_criterion(out["phase_logits"], phase_true)
                + args.progress_loss_weight
                * progress_criterion(out["phase_progress"], phase_progress_true)
                + args.fusion_loss_weight * fusion_loss
            )
            if args.global_progress_loss_weight > 0.0:
                loss = loss + args.global_progress_loss_weight * global_progress_criterion(
                    out["global_progress"], global_progress_true
                )

            loss.backward()
            if args.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            batch_size = features.shape[0]
            epoch_loss += loss.item() * batch_size
            epoch_samples += batch_size
            pbar.set_postfix({"loss": epoch_loss / epoch_samples})

        train_loss = epoch_loss / epoch_samples
        val_metrics = evaluate(
            model=model,
            data_loader=val_loader,
            phase_criterion=phase_criterion,
            progress_criterion=progress_criterion,
            global_progress_criterion=global_progress_criterion,
            atoms=atoms,
            args=args,
            device=device,
        )
        scheduler.step(val_metrics["loss"])

        logger.info(
            "Epoch %d | train_loss=%.4f | val_loss=%.4f | val_phase_acc=%.4f | "
            "val_progress_mae=%.4f | val_fusion_mse=%.4f | val_improvement=%.2f%%",
            epoch,
            train_loss,
            val_metrics["loss"],
            val_metrics["phase_acc"],
            val_metrics["phase_progress_mae"],
            val_metrics["fusion_value_mse"],
            val_metrics["fusion_improvement_pct"],
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )

        if val_metrics["loss"] < best_val_loss - args.early_stop_delta:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.early_stop_patience:
            logger.info("Early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", required=True)
    parser.add_argument("--advantages_path", required=True)
    parser.add_argument(
        "--output_dir",
        default="/workspace/results/phase_progress_probe/joint_logit_fusion",
    )
    parser.add_argument("--return_min", type=float, default=-700.0)
    parser.add_argument("--return_max", type=float, default=0.0)
    parser.add_argument("--num_bins", type=int, default=201)
    parser.add_argument("--num_phases", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--head_hidden_dim", type=int, default=256)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--head_trunk_depth", type=int, default=1)
    parser.add_argument("--fusion_hidden_dim", type=int, default=256)
    parser.add_argument("--fusion_dropout", type=float, default=0.1)
    parser.add_argument("--fusion_depth", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--early_stop_delta", type=float, default=1e-5)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--phase_loss_weight", type=float, default=1.0)
    parser.add_argument("--progress_loss_weight", type=float, default=1.0)
    parser.add_argument("--global_progress_loss_weight", type=float, default=1.0)
    parser.add_argument("--fusion_loss_weight", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--use_class_weights", action="store_true", default=True)
    parser.add_argument("--no_class_weights", dest="use_class_weights", action="store_false")
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

    train_dataset, train_payload = _build_joint_dataset(
        features_path=features_dir / "train.pt",
        advantages_path=Path(args.advantages_path),
        return_min=args.return_min,
        return_max=args.return_max,
    )
    val_dataset, val_payload = _build_joint_dataset(
        features_path=features_dir / "val.pt",
        advantages_path=Path(args.advantages_path),
        return_min=args.return_min,
        return_max=args.return_max,
    )
    logger.info(
        "Joint rows: train=%d val=%d",
        len(train_payload["merged"]),
        len(val_payload["merged"]),
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

    feature_dim = train_payload["feature_data"]["features"].shape[1]
    model = JointLogitFusionModel(
        feature_dim=feature_dim,
        num_bins=args.num_bins,
        num_phases=args.num_phases,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
        head_trunk_depth=args.head_trunk_depth,
        fusion_hidden_dim=args.fusion_hidden_dim,
        fusion_dropout=args.fusion_dropout,
        fusion_depth=args.fusion_depth,
    ).to(device)
    logger.info("Joint model parameters: %d", sum(p.numel() for p in model.parameters()))

    atoms = torch.linspace(args.return_min, args.return_max, args.num_bins)
    atoms = (atoms - args.return_min) / (args.return_max - args.return_min) - 1.0

    train_info = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        atoms=atoms,
        args=args,
        device=device,
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "feature_dim": feature_dim,
                "num_bins": args.num_bins,
                "num_phases": args.num_phases,
                "head_hidden_dim": args.head_hidden_dim,
                "head_dropout": args.head_dropout,
                "head_trunk_depth": args.head_trunk_depth,
                "fusion_hidden_dim": args.fusion_hidden_dim,
                "fusion_dropout": args.fusion_dropout,
                "fusion_depth": args.fusion_depth,
            },
            "args": vars(args),
            "train_info": train_info,
        },
        output_dir / "joint_logit_fusion.pt",
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
        "Joint training complete. Best epoch=%d, best val_loss=%.6f. Outputs in %s",
        train_info["best_epoch"],
        train_info["best_val_loss"],
        output_dir,
    )


if __name__ == "__main__":
    main()
