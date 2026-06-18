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

"""Analyze a trained fusion MLP against raw and linear z/p correction baselines."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from fusion_model import FusionMLP
from model import PhaseProgressHead
from predict_and_analyze import (
    _add_next_frame_predictions,
    _apply_correction,
    _compute_prediction_metrics,
)
from train_fusion_mlp import _build_rows, _load_head, _normalize_return, _predict_zp

logger = logging.getLogger(__name__)


def _load_fusion_model(path: str, device: torch.device) -> FusionMLP:
    """Load a trained fusion MLP checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = FusionMLP(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


def _collect_predictions(
    features_dir: Path,
    head: PhaseProgressHead,
    fusion_model: FusionMLP,
    batch_size: int,
    device: torch.device,
    advantages_path: Path,
    return_min: float,
    return_max: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare per-frame z/p and fusion predictions for train+val rows."""
    adv_df = pd.read_parquet(advantages_path)
    pred_parts: list[pd.DataFrame] = []
    merged_parts: list[pd.DataFrame] = []

    for split in ("train", "val"):
        feature_data = torch.load(features_dir / f"{split}.pt", weights_only=True)
        zpred = _predict_zp(head, feature_data["features"], batch_size=batch_size, device=device)
        pred_df = _build_rows(
            split_name=split,
            feature_data=feature_data,
            zpred=zpred,
            adv_df=adv_df,
            return_min=return_min,
            return_max=return_max,
        )
        pred_parts.append(pred_df[[
            "episode_index",
            "frame_index",
            "phase_true",
            "phase_progress_true",
            "global_progress_true",
            "phase_pred",
            "phase_progress_pred",
            "global_progress_pred",
            "split",
        ]].copy())

        row_index = torch.tensor(pred_df["row_index"].values, dtype=torch.long)
        features = feature_data["features"].float().index_select(0, row_index)
        phase_probs = zpred["phase_probs"].float().index_select(0, row_index)
        phase_progress = zpred["phase_progress_pred"].float().index_select(0, row_index)
        global_progress = zpred["global_progress_pred"].float().index_select(0, row_index)
        raw_value = torch.tensor(pred_df["value_current"].values, dtype=torch.float32)

        loader = DataLoader(
            TensorDataset(features, phase_probs, phase_progress, global_progress, raw_value),
            batch_size=batch_size,
            shuffle=False,
        )
        bias_pred: list[torch.Tensor] = []
        with torch.no_grad():
            for feat, probs, prog, glob, raw in loader:
                out = fusion_model(
                    feat.to(device),
                    probs.to(device),
                    prog.to(device),
                    glob.to(device),
                    raw.to(device),
                )
                bias_pred.append(out.cpu())

        merged = pred_df.copy()
        merged["fusion_bias_pred"] = torch.cat(bias_pred).numpy()
        merged["value_current_fusion"] = merged["value_current"] - merged["fusion_bias_pred"]
        merged_parts.append(merged)

    pred_all = pd.concat(pred_parts, ignore_index=True)
    merged_all = pd.concat(merged_parts, ignore_index=True)
    merged_all = merged_all.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    merged_all["return_norm"] = _normalize_return(
        merged_all["return"], return_min=return_min, return_max=return_max
    )
    return pred_all, merged_all


def _apply_fusion_next(df: pd.DataFrame) -> pd.DataFrame:
    """Add next-frame fusion-corrected values and corrected bias columns."""
    df = _add_next_frame_predictions(df)
    next_df = df[[
        "episode_index",
        "frame_index",
        "value_current_fusion",
    ]].rename(
        columns={
            "frame_index": "next_frame_index",
            "value_current_fusion": "value_next_fusion",
        }
    )
    df = df.merge(next_df, on=["episode_index", "next_frame_index"], how="left")
    df["value_next_fusion"] = df["value_next_fusion"].fillna(df["value_current_fusion"])
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", required=True)
    parser.add_argument("--head_checkpoint", required=True)
    parser.add_argument("--fusion_checkpoint", required=True)
    parser.add_argument("--advantages_path", required=True)
    parser.add_argument(
        "--output_dir",
        default="/workspace/results/phase_progress_probe/fusion_analysis",
    )
    parser.add_argument("--return_min", type=float, default=-700.0)
    parser.add_argument("--return_max", type=float, default=0.0)
    parser.add_argument("--num_phases", type=int, default=5)
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
    logger.info("Loading z/p head from %s", args.head_checkpoint)
    head = _load_head(args.head_checkpoint, device)
    logger.info("Loading fusion MLP from %s", args.fusion_checkpoint)
    fusion_model = _load_fusion_model(args.fusion_checkpoint, device)

    logger.info("Collecting predictions and fusion corrections")
    pred_df, merged = _collect_predictions(
        features_dir=Path(args.features_dir),
        head=head,
        fusion_model=fusion_model,
        batch_size=args.batch_size,
        device=device,
        advantages_path=Path(args.advantages_path),
        return_min=args.return_min,
        return_max=args.return_max,
    )
    pred_metrics = _compute_prediction_metrics(pred_df, args.num_phases)

    merged_with_next = _add_next_frame_predictions(merged.copy())

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
    fusion_df = _apply_fusion_next(merged.copy())

    mse_raw = ((merged["value_current"] - merged["return_norm"]) ** 2).mean()
    mse_linear = ((linear_df["value_current_corr"] - linear_df["return_norm"]) ** 2).mean()
    mse_oracle = ((oracle_df["value_current_corr"] - oracle_df["return_norm"]) ** 2).mean()
    mse_fusion = ((fusion_df["value_current_fusion"] - fusion_df["return_norm"]) ** 2).mean()

    report: dict[str, Any] = {
        "prediction_metrics": pred_metrics,
        "correction": {
            "mse_raw": float(mse_raw),
            "mse_linear_predicted": float(mse_linear),
            "mse_oracle": float(mse_oracle),
            "mse_fusion_predicted": float(mse_fusion),
            "linear_improvement_pct": float((1 - mse_linear / mse_raw) * 100),
            "oracle_improvement_pct": float((1 - mse_oracle / mse_raw) * 100),
            "fusion_improvement_pct": float((1 - mse_fusion / mse_raw) * 100),
        },
    }

    pred_path = output_dir / "phase_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)
    fusion_path = output_dir / "fusion_predictions.parquet"
    fusion_df.to_parquet(fusion_path, index=False)

    logger.info("Prediction metrics:")
    for k, v in pred_metrics.items():
        logger.info("  %s: %.4f", k, v)
    logger.info("=== Value MSE comparison ===")
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
        "  fusion:     %.6f (%.1f%% improvement)",
        mse_fusion,
        report["correction"]["fusion_improvement_pct"],
    )

    with open(output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    logger.info("Report saved to %s", output_dir / "report.json")


if __name__ == "__main__":
    main()
