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

"""Analyze a trained 201-bin logit-space fusion MLP."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from logit_fusion_model import LogitFusionMLP
from predict_and_analyze import (
    _add_next_frame_predictions,
    _apply_correction,
    _compute_prediction_metrics,
)
from train_logit_fusion_mlp import _load_head, _normalize_return, _predict_zp

logger = logging.getLogger(__name__)


def _load_model(path: str, device: torch.device) -> LogitFusionMLP:
    """Load trained logit fusion model."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = LogitFusionMLP(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


def _prepare_dataframe(
    parquet_path: Path,
    return_min: float,
    return_max: float,
) -> pd.DataFrame:
    """Load logits parquet and build oracle z/p input columns."""
    df = pd.read_parquet(parquet_path).copy()
    df["return_norm"] = _normalize_return(df["return"], return_min, return_max)
    if "global_progress" not in df.columns:
        df["global_progress"] = (df["phase"].astype(np.float32) + df["phase_progress"]) / 5.0
    return df


def _collect_rows_from_features(
    features_dir: Path,
    head_checkpoint: str | None,
    batch_size: int,
    device: torch.device,
    advantages_path: Path,
    return_min: float,
    return_max: float,
    zp_source: str,
) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    """Collect train+val rows aligned to cached features."""
    adv_df = pd.read_parquet(advantages_path)
    pred_parts: list[pd.DataFrame] = []
    merged_parts: list[pd.DataFrame] = []
    head = _load_head(head_checkpoint, device) if head_checkpoint is not None else None

    if zp_source == "predicted" and head is None:
        raise ValueError("Predicted z/p mode requires --head_checkpoint.")

    for split in ("train", "val"):
        feature_data = torch.load(features_dir / f"{split}.pt", weights_only=True)
        zpred = _predict_zp(head, feature_data["features"], batch_size=batch_size, device=device) if head is not None else None

        rows = pd.DataFrame(
            {
                "row_index": np.arange(len(feature_data["episode_index"]), dtype=np.int64),
                "episode_index": feature_data["episode_index"].numpy(),
                "frame_index": feature_data["frame_index"].numpy(),
                "phase_true": feature_data["phase"].numpy(),
                "phase_progress_true": feature_data["phase_progress"].numpy(),
                "global_progress_true": feature_data["global_progress"].numpy(),
                "split": split,
            }
        )
        if zpred is not None:
            rows["phase_pred"] = zpred["phase_pred"].numpy()
            rows["phase_progress_pred"] = zpred["phase_progress_pred"].numpy()
            rows["global_progress_pred"] = zpred["global_progress_pred"].numpy()
            rows["phase_probs_pred"] = list(zpred["phase_probs"].numpy())
            pred_parts.append(
                rows[
                    [
                        "episode_index",
                        "frame_index",
                        "phase_true",
                        "phase_progress_true",
                        "global_progress_true",
                        "phase_pred",
                        "phase_progress_pred",
                        "global_progress_pred",
                        "split",
                    ]
                ].copy()
            )

        merged = rows.merge(
            adv_df,
            on=["episode_index", "frame_index"],
            how="inner",
        )
        merged["return_norm"] = _normalize_return(
            merged["return"], return_min=return_min, return_max=return_max
        )
        merged_parts.append(merged)

    pred_df = pd.concat(pred_parts, ignore_index=True) if pred_parts else None
    merged_df = pd.concat(merged_parts, ignore_index=True)
    merged_df = merged_df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    return pred_df, merged_df


def _run_fusion(
    model: LogitFusionMLP,
    df: pd.DataFrame,
    batch_size: int,
    atoms: torch.Tensor,
    alpha: float,
    num_phases: int,
    device: torch.device,
    zp_source: str,
) -> pd.DataFrame:
    """Predict fused value for every row."""
    logits = np.stack(df["value_logits_current"].to_numpy()).astype(np.float32)
    phase_true = torch.tensor(df["phase_true"].values, dtype=torch.long)
    phase_true_onehot = torch.nn.functional.one_hot(phase_true, num_classes=num_phases).float()

    if zp_source == "oracle":
        phase_repr = phase_true_onehot
        phase_progress = torch.tensor(df["phase_progress_true"].values, dtype=torch.float32)
        global_progress = torch.tensor(df["global_progress_true"].values, dtype=torch.float32)
    elif zp_source == "predicted":
        phase_repr = torch.tensor(
            np.stack(df["phase_probs_pred"].to_numpy()).astype(np.float32),
            dtype=torch.float32,
        )
        phase_progress = torch.tensor(df["phase_progress_pred"].values, dtype=torch.float32)
        global_progress = torch.tensor(df["global_progress_pred"].values, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported zp_source: {zp_source}")

    loader = DataLoader(
        TensorDataset(
            torch.tensor(logits, dtype=torch.float32),
            phase_repr,
            phase_progress,
            global_progress,
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    fused_values: list[torch.Tensor] = []
    with torch.no_grad():
        for raw_logits, phase_repr, prog, glob in loader:
            delta = model(
                raw_logits.to(device),
                phase_repr.to(device),
                prog.to(device),
                glob.to(device),
            )
            fused_logits = raw_logits.to(device) + alpha * delta
            probs = torch.softmax(fused_logits, dim=-1)
            fused_value = (probs * atoms.to(device).unsqueeze(0)).sum(dim=-1)
            fused_values.append(fused_value.cpu())

    out = df.copy()
    out["value_fused"] = torch.cat(fused_values).numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--advantages_path", required=True)
    parser.add_argument("--features_dir", default=None)
    parser.add_argument("--head_checkpoint", default=None)
    parser.add_argument("--fusion_checkpoint", required=True)
    parser.add_argument(
        "--output_dir",
        default="/workspace/results/phase_progress_probe/logit_fusion_analysis",
    )
    parser.add_argument("--return_min", type=float, default=-700.0)
    parser.add_argument("--return_max", type=float, default=0.0)
    parser.add_argument("--num_bins", type=int, default=201)
    parser.add_argument("--num_phases", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--zp_source", choices=["oracle", "predicted"], default="oracle")
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "analyze_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading logit fusion model from %s", args.fusion_checkpoint)
    model = _load_model(args.fusion_checkpoint, device)

    atoms = torch.linspace(args.return_min, args.return_max, args.num_bins)
    atoms = (atoms - args.return_min) / (args.return_max - args.return_min) - 1.0

    pred_metrics: dict[str, float] | None = None
    pred_df: pd.DataFrame | None = None
    if args.features_dir is not None:
        pred_df, df = _collect_rows_from_features(
            features_dir=Path(args.features_dir),
            head_checkpoint=args.head_checkpoint,
            batch_size=args.batch_size,
            device=device,
            advantages_path=Path(args.advantages_path),
            return_min=args.return_min,
            return_max=args.return_max,
            zp_source=args.zp_source,
        )
        if pred_df is not None:
            pred_metrics = _compute_prediction_metrics(pred_df, args.num_phases)
    else:
        if args.zp_source != "oracle":
            raise ValueError("Parquet-only logit fusion analysis only supports oracle z/p.")
        df = _prepare_dataframe(
            parquet_path=Path(args.advantages_path),
            return_min=args.return_min,
            return_max=args.return_max,
        )
        df = df.rename(
            columns={
                "phase": "phase_true",
                "phase_progress": "phase_progress_true",
                "global_progress": "global_progress_true",
            }
        )

    fused_df = _run_fusion(
        model=model,
        df=df,
        batch_size=args.batch_size,
        atoms=atoms,
        alpha=args.alpha,
        num_phases=args.num_phases,
        device=device,
        zp_source=args.zp_source,
    )

    linear_predicted_mse: float | None = None
    oracle_mse: float | None = None
    if {"phase_pred", "phase_progress_pred"}.issubset(fused_df.columns):
        merged_with_next = _add_next_frame_predictions(fused_df.copy())
        linear_df = _apply_correction(
            merged_with_next,
            phase_col="phase_pred",
            progress_col="phase_progress_pred",
            phase_next_col="phase_pred_next",
            progress_next_col="phase_progress_pred_next",
        )
        linear_predicted_mse = (
            (linear_df["value_current_corr"] - linear_df["return_norm"]) ** 2
        ).mean()
        oracle_df = _apply_correction(
            merged_with_next,
            phase_col="phase_true",
            progress_col="phase_progress_true",
            phase_next_col="phase_true_next",
            progress_next_col="phase_progress_true_next",
        )
        oracle_mse = ((oracle_df["value_current_corr"] - oracle_df["return_norm"]) ** 2).mean()
    elif {"phase_true", "phase_progress_true"}.issubset(fused_df.columns):
        oracle_linear_df = _apply_correction(
            _add_next_frame_predictions(fused_df.copy()),
            phase_col="phase_true",
            progress_col="phase_progress_true",
            phase_next_col="phase_true_next",
            progress_next_col="phase_progress_true_next",
        )
        oracle_mse = (
            (oracle_linear_df["value_current_corr"] - oracle_linear_df["return_norm"]) ** 2
        ).mean()

    mse_raw = ((fused_df["value_current"] - fused_df["return_norm"]) ** 2).mean()
    mse_fused = ((fused_df["value_fused"] - fused_df["return_norm"]) ** 2).mean()
    improvement = (1 - mse_fused / mse_raw) * 100 if mse_raw > 0 else 0.0

    report: dict[str, Any] = {"correction": {"mse_raw": float(mse_raw), "mse_fused": float(mse_fused), "improvement_pct": float(improvement)}}
    if pred_metrics is not None:
        report["prediction_metrics"] = pred_metrics
    if linear_predicted_mse is not None:
        report["correction"]["mse_linear_predicted"] = float(linear_predicted_mse)
        report["correction"]["linear_improvement_pct"] = float(
            (1 - linear_predicted_mse / mse_raw) * 100 if mse_raw > 0 else 0.0
        )
    if oracle_mse is not None:
        report["correction"]["mse_oracle"] = float(oracle_mse)
        report["correction"]["oracle_improvement_pct"] = float(
            (1 - oracle_mse / mse_raw) * 100 if mse_raw > 0 else 0.0
        )

    if pred_metrics is not None:
        logger.info("Prediction metrics:")
        for k, v in pred_metrics.items():
            logger.info("  %s: %.4f", k, v)
    logger.info("=== Logit Fusion Value MSE comparison ===")
    logger.info("  raw:        %.6f", mse_raw)
    if linear_predicted_mse is not None:
        logger.info(
            "  linear:     %.6f (%.1f%% improvement)",
            linear_predicted_mse,
            report["correction"]["linear_improvement_pct"],
        )
    if oracle_mse is not None:
        logger.info(
            "  oracle:     %.6f (%.1f%% improvement)",
            oracle_mse,
            report["correction"]["oracle_improvement_pct"],
        )
    logger.info("  fusion:     %.6f (%.1f%% improvement)", mse_fused, improvement)

    fused_path = output_dir / "logit_fusion_predictions.parquet"
    fused_df.to_parquet(fused_path, index=False)
    if pred_df is not None:
        pred_path = output_dir / "phase_predictions.parquet"
        pred_df.to_parquet(pred_path, index=False)
        logger.info("Saved phase predictions to %s", pred_path)
    with open(output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    logger.info("Saved predictions to %s", fused_path)
    logger.info("Report saved to %s", output_dir / "report.json")


if __name__ == "__main__":
    main()
