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

"""Train a 201-bin logit-space fusion MLP using oracle or predicted z/p."""

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


def _normalize_return(
    values: pd.Series,
    return_min: float,
    return_max: float,
) -> pd.Series:
    ret_range = return_max - return_min
    if ret_range <= 0:
        return pd.Series(np.full(len(values), -0.5, dtype=np.float32), index=values.index)
    return (values - return_min) / ret_range - 1.0


def _load_head(head_path: str, device: torch.device) -> PhaseProgressHead:
    """Load a trained z/p head."""
    ckpt = torch.load(head_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    head = PhaseProgressHead(**config)
    head.load_state_dict(ckpt["state_dict"])
    head.to(device)
    head.eval()
    return head


@torch.no_grad()
def _predict_zp(
    head: PhaseProgressHead,
    features: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Run the trained z/p head over cached features."""
    loader = DataLoader(TensorDataset(features), batch_size=batch_size, shuffle=False)
    phase_probs: list[torch.Tensor] = []
    phase_pred: list[torch.Tensor] = []
    phase_progress: list[torch.Tensor] = []
    global_progress: list[torch.Tensor] = []
    for (batch,) in loader:
        out = head(batch.to(device))
        probs = F.softmax(out["phase_logits"], dim=-1).cpu()
        phase_probs.append(probs)
        phase_pred.append(probs.argmax(dim=-1))
        phase_progress.append(out["phase_progress"].cpu())
        global_progress.append(out["global_progress"].cpu())
    return {
        "phase_probs": torch.cat(phase_probs, dim=0),
        "phase_pred": torch.cat(phase_pred, dim=0),
        "phase_progress_pred": torch.cat(phase_progress, dim=0),
        "global_progress_pred": torch.cat(global_progress, dim=0),
    }


def _split_by_episode(
    df: pd.DataFrame,
    val_episode_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Episode-level train/val split."""
    episodes = sorted(df["episode_index"].unique().tolist())
    rng = np.random.default_rng(seed)
    episodes = rng.permutation(episodes).tolist()
    n_val = max(1, int(len(episodes) * val_episode_ratio))
    if n_val >= len(episodes):
        n_val = len(episodes) - 1
    val_eps = set(episodes[:n_val])
    train_df = df[~df["episode_index"].isin(val_eps)].copy()
    val_df = df[df["episode_index"].isin(val_eps)].copy()
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def _prepare_legacy_dataframe(
    parquet_path: Path,
    return_min: float,
    return_max: float,
    zp_source: str,
) -> pd.DataFrame:
    """Load exported logits parquet and prepare training targets."""
    df = pd.read_parquet(parquet_path)
    required = {
        "episode_index",
        "frame_index",
        "return",
        "value_current",
        "value_logits_current",
        "phase",
        "phase_progress",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for logit fusion: {sorted(missing)}")

    df = df.copy()
    df["return_norm"] = _normalize_return(df["return"], return_min, return_max)
    if "global_progress" not in df.columns:
        df["global_progress"] = (df["phase"].astype(np.float32) + df["phase_progress"]) / 5.0

    if zp_source != "oracle":
        raise ValueError(
            "Parquet-only logit fusion requires oracle z/p. "
            "Use --features_dir and --head_checkpoint for predicted mode."
        )

    df["phase_input"] = df["phase"].astype(int)
    df["phase_progress_input"] = df["phase_progress"].astype(np.float32)
    df["global_progress_input"] = df["global_progress"].astype(np.float32)
    return df


def _build_legacy_dataset(df: pd.DataFrame, num_phases: int) -> TensorDataset:
    """Convert a parquet-only oracle dataframe into a TensorDataset."""
    logits = np.stack(df["value_logits_current"].to_numpy()).astype(np.float32)
    phase = torch.tensor(df["phase_input"].values, dtype=torch.long)
    phase_onehot = F.one_hot(phase, num_classes=num_phases).to(torch.float32)
    phase_progress = torch.tensor(df["phase_progress_input"].values, dtype=torch.float32)
    global_progress = torch.tensor(df["global_progress_input"].values, dtype=torch.float32)
    return_norm = torch.tensor(df["return_norm"].values, dtype=torch.float32)
    raw_value = torch.tensor(df["value_current"].values, dtype=torch.float32)

    return TensorDataset(
        torch.tensor(logits, dtype=torch.float32),
        phase_onehot,
        phase_progress,
        global_progress,
        raw_value,
        return_norm,
    )


def _build_rows(
    split_name: str,
    feature_data: dict[str, torch.Tensor],
    zpred: dict[str, torch.Tensor] | None,
    adv_df: pd.DataFrame,
    return_min: float,
    return_max: float,
    zp_source: str,
) -> pd.DataFrame:
    """Merge cached split metadata, optional predictions, and exported logits."""
    rows = pd.DataFrame(
        {
            "row_index": np.arange(len(feature_data["episode_index"]), dtype=np.int64),
            "episode_index": feature_data["episode_index"].numpy(),
            "frame_index": feature_data["frame_index"].numpy(),
            "phase_true": feature_data["phase"].numpy(),
            "phase_progress_true": feature_data["phase_progress"].numpy(),
            "global_progress_true": feature_data["global_progress"].numpy(),
            "split": split_name,
        }
    )
    if zpred is not None:
        rows["phase_pred"] = zpred["phase_pred"].numpy()
        rows["phase_progress_pred"] = zpred["phase_progress_pred"].numpy()
        rows["global_progress_pred"] = zpred["global_progress_pred"].numpy()

    merged = rows.merge(
        adv_df,
        on=["episode_index", "frame_index"],
        how="inner",
    )
    merged["return_norm"] = _normalize_return(
        merged["return"], return_min=return_min, return_max=return_max
    )
    if zp_source == "oracle":
        merged["phase_progress_input"] = merged["phase_progress_true"]
        merged["global_progress_input"] = merged["global_progress_true"]
    elif zp_source == "predicted":
        if zpred is None:
            raise ValueError("Predicted z/p mode requires a trained head and cached features.")
        merged["phase_progress_input"] = merged["phase_progress_pred"]
        merged["global_progress_input"] = merged["global_progress_pred"]
    else:
        raise ValueError(f"Unsupported zp_source: {zp_source}")
    return merged.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)


def _gather_feature_payload(
    features_dir: Path,
    head: PhaseProgressHead | None,
    batch_size: int,
    device: torch.device,
    advantages_path: Path,
    return_min: float,
    return_max: float,
    zp_source: str,
    num_phases: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prepare train/val payloads aligned to cached feature splits."""
    adv_df = pd.read_parquet(advantages_path)
    payloads: dict[str, dict[str, Any]] = {}

    for split in ("train", "val"):
        feature_data = torch.load(features_dir / f"{split}.pt", weights_only=True)
        zpred = None
        if head is not None:
            zpred = _predict_zp(head, feature_data["features"], batch_size=batch_size, device=device)

        merged = _build_rows(
            split_name=split,
            feature_data=feature_data,
            zpred=zpred,
            adv_df=adv_df,
            return_min=return_min,
            return_max=return_max,
            zp_source=zp_source,
        )
        payloads[split] = {
            "merged": merged,
            "all_phase_probs": zpred["phase_probs"].float() if zpred is not None else None,
            "all_phase_true_onehot": F.one_hot(
                feature_data["phase"].long(), num_classes=num_phases
            ).float(),
            "all_phase_progress_pred": (
                zpred["phase_progress_pred"].float() if zpred is not None else None
            ),
            "all_global_progress_pred": (
                zpred["global_progress_pred"].float() if zpred is not None else None
            ),
            "all_phase_progress_true": feature_data["phase_progress"].float(),
            "all_global_progress_true": feature_data["global_progress"].float(),
        }

    return payloads["train"], payloads["val"]


def _build_dataset(payload: dict[str, Any], zp_source: str) -> TensorDataset:
    """Create logit-fusion tensors aligned with merged feature rows."""
    merged = payload["merged"]
    row_index = torch.tensor(merged["row_index"].values, dtype=torch.long)
    logits = np.stack(merged["value_logits_current"].to_numpy()).astype(np.float32)
    raw_value = torch.tensor(merged["value_current"].values, dtype=torch.float32)
    return_norm = torch.tensor(merged["return_norm"].values, dtype=torch.float32)

    if zp_source == "oracle":
        phase_repr = payload["all_phase_true_onehot"].index_select(0, row_index)
        phase_progress = payload["all_phase_progress_true"].index_select(0, row_index)
        global_progress = payload["all_global_progress_true"].index_select(0, row_index)
    elif zp_source == "predicted":
        if payload["all_phase_probs"] is None:
            raise ValueError("Predicted z/p dataset requires predicted phase probabilities.")
        phase_repr = payload["all_phase_probs"].index_select(0, row_index)
        phase_progress = payload["all_phase_progress_pred"].index_select(0, row_index)
        global_progress = payload["all_global_progress_pred"].index_select(0, row_index)
    else:
        raise ValueError(f"Unsupported zp_source: {zp_source}")

    return TensorDataset(
        torch.tensor(logits, dtype=torch.float32),
        phase_repr,
        phase_progress,
        global_progress,
        raw_value,
        return_norm,
    )


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


@torch.no_grad()
def evaluate(
    model: LogitFusionMLP,
    data_loader: DataLoader,
    atoms: torch.Tensor,
    alpha: float,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate logit-space fusion on one split."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_pred: list[torch.Tensor] = []
    all_true: list[torch.Tensor] = []
    all_raw: list[torch.Tensor] = []

    for raw_logits, phase_repr, phase_progress, global_progress, raw_value, return_norm in data_loader:
        delta = model(
            raw_logits.to(device),
            phase_repr.to(device),
            phase_progress.to(device),
            global_progress.to(device),
        )
        fused_value = _fused_value_from_logits(
            raw_logits.to(device),
            delta,
            atoms.to(device),
            alpha=alpha,
        )
        loss = F.mse_loss(fused_value, return_norm.to(device))

        batch_size = raw_logits.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_pred.append(fused_value.cpu())
        all_true.append(return_norm.cpu())
        all_raw.append(raw_value.cpu())

    pred_all = torch.cat(all_pred)
    true_all = torch.cat(all_true)
    raw_all = torch.cat(all_raw)
    mse = ((pred_all - true_all) ** 2).mean().item()
    raw_mse = ((raw_all - true_all) ** 2).mean().item()
    return {
        "loss": total_loss / total_samples,
        "value_mse": mse,
        "raw_value_mse": raw_mse,
        "improvement_pct": (1 - mse / raw_mse) * 100 if raw_mse > 0 else 0.0,
    }


def train(
    model: LogitFusionMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    atoms: torch.Tensor,
    alpha: float,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    """Train the logit fusion model."""
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
        for raw_logits, phase_repr, phase_progress, global_progress, raw_value, return_norm in pbar:
            optimizer.zero_grad()
            delta = model(
                raw_logits.to(device),
                phase_repr.to(device),
                phase_progress.to(device),
                global_progress.to(device),
            )
            fused_value = _fused_value_from_logits(
                raw_logits.to(device),
                delta,
                atoms.to(device),
                alpha=alpha,
            )
            loss = F.mse_loss(fused_value, return_norm.to(device))
            loss.backward()
            if args.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            batch_size = raw_logits.shape[0]
            epoch_loss += loss.item() * batch_size
            epoch_samples += batch_size
            pbar.set_postfix({"loss": epoch_loss / epoch_samples})

        train_loss = epoch_loss / epoch_samples
        val_metrics = evaluate(model, val_loader, atoms, alpha, device)
        scheduler.step(val_metrics["loss"])

        logger.info(
            "Epoch %d | train_loss=%.4f | val_loss=%.4f | val_value_mse=%.4f | val_improvement=%.2f%%",
            epoch,
            train_loss,
            val_metrics["loss"],
            val_metrics["value_mse"],
            val_metrics["improvement_pct"],
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
    parser.add_argument("--advantages_path", required=True)
    parser.add_argument("--features_dir", default=None)
    parser.add_argument("--head_checkpoint", default=None)
    parser.add_argument(
        "--output_dir",
        default="/workspace/results/phase_progress_probe/logit_fusion",
    )
    parser.add_argument("--return_min", type=float, default=-700.0)
    parser.add_argument("--return_max", type=float, default=0.0)
    parser.add_argument("--num_bins", type=int, default=201)
    parser.add_argument("--num_phases", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--zp_source", choices=["oracle", "predicted"], default="oracle")
    parser.add_argument("--val_episode_ratio", type=float, default=0.2)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--predict_batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--early_stop_delta", type=float, default=1e-5)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "train_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    train_loader: DataLoader
    val_loader: DataLoader
    if args.zp_source == "predicted" and not args.head_checkpoint:
        raise ValueError("Predicted z/p mode requires --head_checkpoint.")
    if args.zp_source == "predicted" and not args.features_dir:
        raise ValueError("Predicted z/p mode requires --features_dir.")

    if args.features_dir is not None:
        features_dir = Path(args.features_dir)
        head = None
        if args.head_checkpoint is not None:
            logger.info("Loading z/p head from %s", args.head_checkpoint)
            head = _load_head(args.head_checkpoint, device)
        logger.info("Preparing logit fusion payloads from %s", features_dir)
        train_payload, val_payload = _gather_feature_payload(
            features_dir=features_dir,
            head=head,
            batch_size=args.predict_batch_size,
            device=device,
            advantages_path=Path(args.advantages_path),
            return_min=args.return_min,
            return_max=args.return_max,
            zp_source=args.zp_source,
            num_phases=args.num_phases,
        )
        logger.info(
            "Logit fusion rows: train=%d val=%d",
            len(train_payload["merged"]),
            len(val_payload["merged"]),
        )
        train_loader = DataLoader(
            _build_dataset(train_payload, zp_source=args.zp_source),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            _build_dataset(val_payload, zp_source=args.zp_source),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
    else:
        df = _prepare_legacy_dataframe(
            parquet_path=Path(args.advantages_path),
            return_min=args.return_min,
            return_max=args.return_max,
            zp_source=args.zp_source,
        )
        train_df, val_df = _split_by_episode(
            df=df,
            val_episode_ratio=args.val_episode_ratio,
            seed=args.seed,
        )
        logger.info("Logit fusion rows: train=%d val=%d", len(train_df), len(val_df))
        train_loader = DataLoader(
            _build_legacy_dataset(train_df, num_phases=args.num_phases),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            _build_legacy_dataset(val_df, num_phases=args.num_phases),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    atoms = torch.linspace(args.return_min, args.return_max, args.num_bins)
    atoms = (atoms - args.return_min) / (args.return_max - args.return_min) - 1.0
    model = LogitFusionMLP(
        num_bins=args.num_bins,
        num_phases=args.num_phases,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        depth=args.depth,
    ).to(device)
    logger.info("Logit fusion parameters: %d", sum(p.numel() for p in model.parameters()))

    train_info = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        atoms=atoms,
        alpha=args.alpha,
        args=args,
        device=device,
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "num_bins": args.num_bins,
                "num_phases": args.num_phases,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "depth": args.depth,
            },
            "args": vars(args),
            "train_info": train_info,
        },
        output_dir / "logit_fusion.pt",
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
        "Logit fusion training complete. Best epoch=%d, best val_loss=%.6f. Outputs in %s",
        train_info["best_epoch"],
        train_info["best_val_loss"],
        output_dir,
    )


if __name__ == "__main__":
    main()
