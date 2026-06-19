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

"""Analyze a jointly trained phase/progress + logit-fusion model."""

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

from predict_and_analyze import (
    _add_next_frame_predictions,
    _apply_correction,
    _compute_prediction_metrics,
)
from train_joint_logit_fusion import JointLogitFusionModel
from train_logit_fusion_mlp import _normalize_return

logger = logging.getLogger(__name__)


def _load_model(path: str, device: torch.device) -> JointLogitFusionModel:
    """Load a trained joint model checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = JointLogitFusionModel(**config)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


def _collect_split_rows(
    features_dir: Path,
    advantages_path: Path,
    splits: list[str],
    return_min: float,
    return_max: float,
) -> pd.DataFrame:
    """Collect merged rows for requested splits."""
    adv_df = pd.read_parquet(advantages_path)
    merged_parts: list[pd.DataFrame] = []
    for split in splits:
        feature_data = torch.load(features_dir / f"{split}.pt", weights_only=True)
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
        merged = rows.merge(
            adv_df,
            on=["episode_index", "frame_index"],
            how="inner",
        )
        merged["return_norm"] = _normalize_return(
            merged["return"], return_min=return_min, return_max=return_max
        )
        merged_parts.append(merged)
    return pd.concat(merged_parts, ignore_index=True).sort_values(
        ["episode_index", "frame_index"]
    ).reset_index(drop=True)


def _run_joint_predictions(
    model: JointLogitFusionModel,
    df: pd.DataFrame,
    features_dir: Path,
    batch_size: int,
    atoms: torch.Tensor,
    alpha: float,
    device: torch.device,
) -> pd.DataFrame:
    """Run the joint model over merged split rows."""
    features: torch.Tensor | None = None
    for split in df["split"].unique().tolist():
        split_data = torch.load(features_dir / f"{split}.pt", weights_only=True)
        split_df = df[df["split"] == split]
        row_index = torch.tensor(split_df["row_index"].values, dtype=torch.long)
        split_features = split_data["features"].float().index_select(0, row_index)
        if features is None:
            features = torch.empty(
                (len(df), split_features.shape[1]),
                dtype=split_features.dtype,
            )
        features[split_df.index.to_numpy()] = split_features
    if features is None:
        raise ValueError("No rows available for joint prediction.")
    raw_logits = torch.tensor(
        np.stack(df["value_logits_current"].to_numpy()).astype(np.float32),
        dtype=torch.float32,
    )

    loader = DataLoader(
        TensorDataset(features, raw_logits),
        batch_size=batch_size,
        shuffle=False,
    )

    phase_logits_parts: list[torch.Tensor] = []
    phase_progress_parts: list[torch.Tensor] = []
    global_progress_parts: list[torch.Tensor] = []
    fused_value_parts: list[torch.Tensor] = []

    with torch.no_grad():
        for feat, logits in loader:
            out = model(feat.to(device), logits.to(device))
            fused_logits = logits.to(device) + alpha * out["delta_logits"]
            probs = torch.softmax(fused_logits, dim=-1)
            fused_value = (probs * atoms.to(device).unsqueeze(0)).sum(dim=-1)

            phase_logits_parts.append(out["phase_logits"].cpu())
            phase_progress_parts.append(out["phase_progress"].cpu())
            global_progress_parts.append(out["global_progress"].cpu())
            fused_value_parts.append(fused_value.cpu())

    out_df = df.copy()
    phase_logits = torch.cat(phase_logits_parts)
    out_df["phase_pred"] = phase_logits.argmax(dim=-1).numpy()
    out_df["phase_progress_pred"] = torch.cat(phase_progress_parts).numpy()
    out_df["global_progress_pred"] = torch.cat(global_progress_parts).numpy()
    out_df["value_fused"] = torch.cat(fused_value_parts).numpy()
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", required=True)
    parser.add_argument("--advantages_path", required=True)
    parser.add_argument("--joint_checkpoint", required=True)
    parser.add_argument(
        "--output_dir",
        default="/workspace/results/phase_progress_probe/joint_logit_fusion_analysis",
    )
    parser.add_argument("--return_min", type=float, default=-700.0)
    parser.add_argument("--return_max", type=float, default=0.0)
    parser.add_argument("--num_bins", type=int, default=201)
    parser.add_argument("--num_phases", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val"],
        default=["train", "val"],
        help="Which cached feature splits to analyze.",
    )
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
    logger.info("Loading joint model from %s", args.joint_checkpoint)
    model = _load_model(args.joint_checkpoint, device)

    merged = _collect_split_rows(
        features_dir=Path(args.features_dir),
        advantages_path=Path(args.advantages_path),
        splits=args.splits,
        return_min=args.return_min,
        return_max=args.return_max,
    )

    atoms = torch.linspace(args.return_min, args.return_max, args.num_bins)
    atoms = (atoms - args.return_min) / (args.return_max - args.return_min) - 1.0
    pred_df = _run_joint_predictions(
        model=model,
        df=merged,
        features_dir=Path(args.features_dir),
        batch_size=args.batch_size,
        atoms=atoms,
        alpha=args.alpha,
        device=device,
    )

    pred_metrics = _compute_prediction_metrics(
        pred_df[
            [
                "phase_true",
                "phase_pred",
                "phase_progress_true",
                "phase_progress_pred",
                "global_progress_true",
                "global_progress_pred",
            ]
        ],
        args.num_phases,
    )

    merged_with_next = _add_next_frame_predictions(pred_df.copy())
    linear_df = _apply_correction(
        merged_with_next,
        phase_col="phase_pred",
        progress_col="phase_progress_pred",
        phase_next_col="phase_pred_next",
        progress_next_col="phase_progress_pred_next",
    )
    oracle_df = _apply_correction(
        merged_with_next,
        phase_col="phase_true",
        progress_col="phase_progress_true",
        phase_next_col="phase_true_next",
        progress_next_col="phase_progress_true_next",
    )

    mse_raw = ((pred_df["value_current"] - pred_df["return_norm"]) ** 2).mean()
    mse_linear = ((linear_df["value_current_corr"] - linear_df["return_norm"]) ** 2).mean()
    mse_oracle = ((oracle_df["value_current_corr"] - oracle_df["return_norm"]) ** 2).mean()
    mse_joint = ((pred_df["value_fused"] - pred_df["return_norm"]) ** 2).mean()

    report: dict[str, Any] = {
        "prediction_metrics": pred_metrics,
        "correction": {
            "mse_raw": float(mse_raw),
            "mse_linear_predicted": float(mse_linear),
            "mse_oracle": float(mse_oracle),
            "mse_joint_fusion": float(mse_joint),
            "linear_improvement_pct": float((1 - mse_linear / mse_raw) * 100),
            "oracle_improvement_pct": float((1 - mse_oracle / mse_raw) * 100),
            "joint_improvement_pct": float((1 - mse_joint / mse_raw) * 100),
        },
    }

    logger.info("Prediction metrics:")
    for k, v in pred_metrics.items():
        logger.info("  %s: %.4f", k, v)
    logger.info("=== Joint Logit Fusion Value MSE comparison ===")
    logger.info("  raw:        %.6f", mse_raw)
    logger.info(
        "  linear:     %.6f (%.1f%% improvement)",
        mse_linear,
        report["correction"]["linear_improvement_pct"],
    )
    logger.info(
        "  oracle:     %.6f (%.1f%% improvement)",
        mse_oracle,
        report["correction"]["oracle_improvement_pct"],
    )
    logger.info(
        "  joint:      %.6f (%.1f%% improvement)",
        mse_joint,
        report["correction"]["joint_improvement_pct"],
    )

    pred_path = output_dir / "joint_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)
    with open(output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    logger.info("Saved joint predictions to %s", pred_path)
    logger.info("Report saved to %s", output_dir / "report.json")


if __name__ == "__main__":
    main()
