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

"""Predict z/p for all frames and analyze value-bias correction."""

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

from model import PhaseProgressHead

logger = logging.getLogger(__name__)


def _load_head(head_path: str, device: torch.device) -> PhaseProgressHead:
    """Load a trained PhaseProgressHead from checkpoint."""
    ckpt = torch.load(head_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    head = PhaseProgressHead(**config)
    head.load_state_dict(ckpt["state_dict"])
    head.to(device)
    head.eval()
    return head


@torch.no_grad()
def _predict_all(
    head: PhaseProgressHead,
    features_dir: Path,
    device: torch.device,
    batch_size: int,
    splits: list[str],
) -> pd.DataFrame:
    """Load cached features for the requested splits and run the head over them."""
    parts = []
    split_labels: list[np.ndarray] = []
    for split in splits:
        path = features_dir / f"{split}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing cached features: {path}")
        data = torch.load(path, weights_only=True)
        parts.append(data)
        split_labels.append(np.full(len(data["episode_index"]), split, dtype=object))

    all_features = torch.cat([p["features"] for p in parts], dim=0)
    all_episode = torch.cat([p["episode_index"] for p in parts], dim=0)
    all_frame = torch.cat([p["frame_index"] for p in parts], dim=0)
    all_phase = torch.cat([p["phase"] for p in parts], dim=0)
    all_phase_progress = torch.cat([p["phase_progress"] for p in parts], dim=0)
    all_global_progress = torch.cat([p["global_progress"] for p in parts], dim=0)
    all_split = np.concatenate(split_labels, axis=0)

    dataset = TensorDataset(all_features)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    phase_preds: list[torch.Tensor] = []
    progress_preds: list[torch.Tensor] = []
    global_preds: list[torch.Tensor] = []
    for (batch,) in loader:
        batch = batch.to(device)
        out = head(batch)
        phase_preds.append(out["phase_logits"].argmax(dim=-1).cpu())
        progress_preds.append(out["phase_progress"].cpu())
        global_preds.append(out["global_progress"].cpu())

    df = pd.DataFrame(
        {
            "split": all_split,
            "episode_index": all_episode.numpy(),
            "frame_index": all_frame.numpy(),
            "phase_true": all_phase.numpy(),
            "phase_progress_true": all_phase_progress.numpy(),
            "global_progress_true": all_global_progress.numpy(),
            "phase_pred": torch.cat(phase_preds).numpy(),
            "phase_progress_pred": torch.cat(progress_preds).numpy(),
            "global_progress_pred": torch.cat(global_preds).numpy(),
        }
    )
    return df


def _compute_prediction_metrics(df: pd.DataFrame, num_phases: int) -> dict[str, float]:
    """Compute prediction quality metrics."""
    metrics: dict[str, float] = {
        "phase_acc": (df["phase_pred"] == df["phase_true"]).mean(),
        "phase_progress_mae": (df["phase_progress_pred"] - df["phase_progress_true"]).abs().mean(),
        "phase_progress_mse": ((df["phase_progress_pred"] - df["phase_progress_true"]) ** 2).mean(),
        "global_progress_mae": (df["global_progress_pred"] - df["global_progress_true"]).abs().mean(),
        "global_progress_mse": ((df["global_progress_pred"] - df["global_progress_true"]) ** 2).mean(),
    }
    for ph in range(num_phases):
        sub = df[df["phase_true"] == ph]
        if len(sub) > 0:
            metrics[f"phase_{ph}_acc"] = (sub["phase_pred"] == sub["phase_true"]).mean()
        else:
            metrics[f"phase_{ph}_acc"] = float("nan")
    return metrics


def _add_next_frame_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Add next-frame predicted/true phase & progress for value_next correction."""
    ep_lengths = df.groupby("episode_index")["frame_index"].max() + 1
    df = df.copy()
    df["episode_length"] = df["episode_index"].map(ep_lengths)
    df["next_frame_index"] = (df["frame_index"] + 1).clip(upper=df["episode_length"] - 1)

    next_cols = [
        "episode_index",
        "frame_index",
        "phase_pred",
        "phase_progress_pred",
        "phase_true",
        "phase_progress_true",
    ]
    next_df = df[next_cols].rename(
        columns={
            "frame_index": "next_frame_index",
            "phase_pred": "phase_pred_next",
            "phase_progress_pred": "phase_progress_pred_next",
            "phase_true": "phase_true_next",
            "phase_progress_true": "phase_progress_true_next",
        }
    )
    df = df.merge(
        next_df,
        on=["episode_index", "next_frame_index"],
        how="left",
    )
    # Fill terminal frames with current frame labels.
    for col in ["phase_pred_next", "phase_progress_pred_next", "phase_true_next", "phase_progress_true_next"]:
        fill_map = {
            "phase_pred_next": "phase_pred",
            "phase_progress_pred_next": "phase_progress_pred",
            "phase_true_next": "phase_true",
            "phase_progress_true_next": "phase_progress_true",
        }
        df[col] = df[col].fillna(df[fill_map[col]])
    df["phase_pred_next"] = df["phase_pred_next"].astype(int)
    df["phase_true_next"] = df["phase_true_next"].astype(int)
    return df


def _fit_phase_progress_bias(
    df: pd.DataFrame,
    phase_col: str,
    progress_col: str,
    value_error_col: str = "value_error",
) -> dict[int, tuple[float, float]]:
    """Fit per-phase linear bias: bias = a * progress + b."""
    coeffs: dict[int, tuple[float, float]] = {}
    for ph in sorted(df[phase_col].dropna().unique().astype(int)):
        sub = df[df[phase_col] == ph]
        x = sub[progress_col].values.astype(np.float64)
        y = sub[value_error_col].values.astype(np.float64)
        if len(x) >= 2 and np.std(x) > 1e-6:
            a, b = np.polyfit(x, y, 1)
        else:
            a, b = 0.0, float(y.mean()) if len(y) > 0 else 0.0
        coeffs[int(ph)] = (float(a), float(b))
    return coeffs


def _apply_correction(
    df: pd.DataFrame,
    phase_col: str,
    progress_col: str,
    phase_next_col: str,
    progress_next_col: str,
    value_current_col: str = "value_current",
    value_next_col: str = "value_next",
    return_norm_col: str = "return_norm",
) -> pd.DataFrame:
    """Apply phase+progress correction using the given phase/progress columns."""
    df = df.copy()
    df["value_error"] = df[value_current_col] - df[return_norm_col]

    coeffs = _fit_phase_progress_bias(df, phase_col, progress_col)

    def bias(row):
        a, b = coeffs[int(row[phase_col])]
        return a * row[progress_col] + b

    def bias_next(row):
        a, b = coeffs[int(row[phase_next_col])]
        return a * row[progress_next_col] + b

    df["bias_current"] = df.apply(bias, axis=1)
    df["bias_next"] = df.apply(bias_next, axis=1)
    df["value_current_corr"] = df[value_current_col] - df["bias_current"]
    df["value_next_corr"] = df[value_next_col] - df["bias_next"]
    return df


def _evaluate_correction(
    df: pd.DataFrame,
    return_min: float,
    return_max: float,
) -> dict[str, Any]:
    """Evaluate raw, predicted, and oracle z/p corrections."""
    ret_range = return_max - return_min

    def normalize_return(x: pd.Series) -> pd.Series:
        return -0.5 if ret_range <= 0 else (x - return_min) / ret_range - 1.0

    df["return_norm"] = normalize_return(df["return"])

    # Need next-frame predictions for value_next correction.
    df = _add_next_frame_predictions(df)

    results: dict[str, Any] = {}

    # Raw.
    mse_raw = ((df["value_current"] - df["return_norm"]) ** 2).mean()
    results["mse_raw"] = float(mse_raw)

    # Oracle (true z/p).
    df_oracle = _apply_correction(
        df,
        phase_col="phase_true",
        progress_col="phase_progress_true",
        phase_next_col="phase_true_next",
        progress_next_col="phase_progress_true_next",
    )
    mse_oracle = ((df_oracle["value_current_corr"] - df_oracle["return_norm"]) ** 2).mean()
    results["mse_oracle"] = float(mse_oracle)
    results["oracle_improvement_pct"] = float((1 - mse_oracle / mse_raw) * 100)

    # Predicted z/p.
    df_pred = _apply_correction(
        df,
        phase_col="phase_pred",
        progress_col="phase_progress_pred",
        phase_next_col="phase_pred_next",
        progress_next_col="phase_progress_pred_next",
    )
    mse_pred = ((df_pred["value_current_corr"] - df_pred["return_norm"]) ** 2).mean()
    results["mse_predicted"] = float(mse_pred)
    results["predicted_improvement_pct"] = float((1 - mse_pred / mse_raw) * 100)

    # Per-phase breakdown for predicted correction.
    per_phase: dict[int, dict[str, float]] = {}
    for ph in sorted(df["phase_true"].unique().astype(int)):
        sub = df[df["phase_true"] == ph]
        if len(sub) == 0:
            continue
        raw_ph = ((sub["value_current"] - sub["return_norm"]) ** 2).mean()
        corr_ph = ((df_pred.loc[sub.index, "value_current_corr"] - sub["return_norm"]) ** 2).mean()
        per_phase[int(ph)] = {
            "n": int(len(sub)),
            "mse_raw": float(raw_ph),
            "mse_predicted": float(corr_ph),
            "improvement_pct": float((1 - corr_ph / raw_ph) * 100) if raw_ph > 0 else 0.0,
        }
    results["per_phase"] = per_phase

    return results, df_pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features_dir",
        default="/workspace/results/phase_progress_probe/features",
    )
    parser.add_argument(
        "--head_checkpoint",
        default="/workspace/results/phase_progress_probe/head/head.pt",
    )
    parser.add_argument(
        "--output_dir",
        default="/workspace/results/phase_progress_probe/analysis",
    )
    parser.add_argument("--advantages_path", default=None)
    parser.add_argument("--return_min", type=float, default=-700.0)
    parser.add_argument("--return_max", type=float, default=0.0)
    parser.add_argument("--num_phases", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=512)
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
    logger.info("Loading head from %s", args.head_checkpoint)
    head = _load_head(args.head_checkpoint, device)

    logger.info("Predicting z/p for splits: %s", ",".join(args.splits))
    pred_df = _predict_all(
        head,
        Path(args.features_dir),
        device,
        args.batch_size,
        args.splits,
    )
    pred_df = pred_df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    pred_path = output_dir / "phase_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)
    logger.info("Saved predictions to %s (%d rows)", pred_path, len(pred_df))

    pred_metrics = _compute_prediction_metrics(pred_df, args.num_phases)
    logger.info("Prediction metrics:")
    for k, v in pred_metrics.items():
        logger.info("  %s: %.4f", k, v)

    report: dict[str, Any] = {"prediction_metrics": pred_metrics}

    if args.advantages_path:
        logger.info("Loading raw advantages from %s", args.advantages_path)
        adv_df = pd.read_parquet(args.advantages_path)
        merged = adv_df.merge(
            pred_df,
            on=["episode_index", "frame_index"],
            how="inner",
        )
        logger.info("Merged %d rows with advantages", len(merged))

        correction_results, corrected_df = _evaluate_correction(
            merged,
            return_min=args.return_min,
            return_max=args.return_max,
        )
        report["correction"] = correction_results

        logger.info("=== Value MSE comparison ===")
        logger.info("  raw:        %.6f", correction_results["mse_raw"])
        logger.info("  oracle:     %.6f (%.1f%% improvement)",
                    correction_results["mse_oracle"],
                    correction_results["oracle_improvement_pct"])
        logger.info("  predicted:  %.6f (%.1f%% improvement)",
                    correction_results["mse_predicted"],
                    correction_results["predicted_improvement_pct"])

        corrected_path = output_dir / "advantages_predicted_corrected.parquet"
        corrected_df.to_parquet(corrected_path, index=False)
        logger.info("Saved corrected advantages to %s", corrected_path)

    with open(output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    logger.info("Report saved to %s", output_dir / "report.json")


if __name__ == "__main__":
    main()
