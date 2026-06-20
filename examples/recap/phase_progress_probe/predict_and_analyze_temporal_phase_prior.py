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

"""Predict z/p with the temporal phase-prior head and analyze value correction."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from model_temporal_phase_prior import TemporalPhasePriorProgressHead
from predict_and_analyze import (
    _apply_correction,
    _compute_prediction_metrics,
)
from train_head_temporal_phase_prior import TemporalWindowDataset

logger = logging.getLogger(__name__)


def _load_head(
    head_path: str,
    device: torch.device,
) -> TemporalPhasePriorProgressHead:
    """Load a trained temporal phase-prior head checkpoint."""
    ckpt = torch.load(head_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    head = TemporalPhasePriorProgressHead(**config)
    head.load_state_dict(ckpt["state_dict"])
    head.to(device)
    head.eval()
    return head


@torch.no_grad()
def _predict_all(
    head: TemporalPhasePriorProgressHead,
    features_dir: Path,
    device: torch.device,
    batch_size: int,
    splits: list[str],
) -> pd.DataFrame:
    """Run the temporal phase-prior head over cached feature windows."""
    import numpy as np

    split_frames = []
    split_labels: list[np.ndarray] = []
    split_windows = []
    for split in splits:
        path = features_dir / f"{split}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing cached features: {path}")
        data = torch.load(path, weights_only=True)
        split_frames.append(data)
        split_labels.append(np.full(len(data["episode_index"]), split, dtype=object))
        split_windows.append(TemporalWindowDataset(data, window_size=head.window_size))

    all_split = np.concatenate(split_labels, axis=0)
    all_episode = torch.cat([d["episode_index"] for d in split_frames], dim=0)
    all_frame = torch.cat([d["frame_index"] for d in split_frames], dim=0)
    all_phase = torch.cat([d["phase"] for d in split_frames], dim=0)
    all_phase_progress = torch.cat([d["phase_progress"] for d in split_frames], dim=0)
    all_global_progress = torch.cat([d["global_progress"] for d in split_frames], dim=0)

    class _ConcatWindowDataset(torch.utils.data.Dataset):
        def __init__(self, datasets: list[TemporalWindowDataset]):
            self.datasets = datasets
            self.offsets = []
            total = 0
            for ds in datasets:
                self.offsets.append(total)
                total += len(ds)
            self.total = total

        def __len__(self) -> int:
            return self.total

        def __getitem__(self, idx: int):
            for ds, offset in zip(self.datasets, self.offsets):
                if idx < offset + len(ds):
                    return ds[idx - offset]
            raise IndexError(idx)

    concat_ds = _ConcatWindowDataset(split_windows)
    loader = DataLoader(
        concat_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    phase_preds: list[torch.Tensor] = []
    progress_preds_soft: list[torch.Tensor] = []
    progress_preds_hard: list[torch.Tensor] = []
    global_preds_soft: list[torch.Tensor] = []
    global_preds_hard: list[torch.Tensor] = []
    for batch in loader:
        out = head(batch["feature_window"].to(device))
        phase_preds.append(out["phase_pred"].cpu())
        progress_preds_soft.append(out["phase_progress_soft"].cpu())
        progress_preds_hard.append(out["phase_progress_hard"].cpu())
        global_preds_soft.append(out["global_progress_soft"].cpu())
        global_preds_hard.append(out["global_progress_hard"].cpu())

    return pd.DataFrame(
        {
            "split": all_split,
            "episode_index": all_episode.numpy(),
            "frame_index": all_frame.numpy(),
            "phase_true": all_phase.numpy(),
            "phase_progress_true": all_phase_progress.numpy(),
            "global_progress_true": all_global_progress.numpy(),
            "phase_pred": torch.cat(phase_preds).numpy(),
            "phase_progress_pred": torch.cat(progress_preds_soft).numpy(),
            "phase_progress_pred_soft": torch.cat(progress_preds_soft).numpy(),
            "phase_progress_pred_hard": torch.cat(progress_preds_hard).numpy(),
            "global_progress_pred": torch.cat(global_preds_soft).numpy(),
            "global_progress_pred_soft": torch.cat(global_preds_soft).numpy(),
            "global_progress_pred_hard": torch.cat(global_preds_hard).numpy(),
        }
    )


def _add_next_frame_predictions_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """Add next-frame columns for both soft and hard progress variants."""
    ep_lengths = df.groupby("episode_index")["frame_index"].max() + 1
    df = df.copy()
    df["episode_length"] = df["episode_index"].map(ep_lengths)
    df["next_frame_index"] = (df["frame_index"] + 1).clip(
        upper=df["episode_length"] - 1
    )

    next_cols = [
        "episode_index",
        "frame_index",
        "phase_pred",
        "phase_progress_pred",
        "phase_progress_pred_soft",
        "phase_progress_pred_hard",
        "phase_true",
        "phase_progress_true",
    ]
    next_df = df[next_cols].rename(
        columns={
            "frame_index": "next_frame_index",
            "phase_pred": "phase_pred_next",
            "phase_progress_pred": "phase_progress_pred_next",
            "phase_progress_pred_soft": "phase_progress_pred_soft_next",
            "phase_progress_pred_hard": "phase_progress_pred_hard_next",
            "phase_true": "phase_true_next",
            "phase_progress_true": "phase_progress_true_next",
        }
    )
    df = df.merge(
        next_df,
        on=["episode_index", "next_frame_index"],
        how="left",
    )

    fill_map = {
        "phase_pred_next": "phase_pred",
        "phase_progress_pred_next": "phase_progress_pred",
        "phase_progress_pred_soft_next": "phase_progress_pred_soft",
        "phase_progress_pred_hard_next": "phase_progress_pred_hard",
        "phase_true_next": "phase_true",
        "phase_progress_true_next": "phase_progress_true",
    }
    for col, base_col in fill_map.items():
        df[col] = df[col].fillna(df[base_col])

    df["phase_pred_next"] = df["phase_pred_next"].astype(int)
    df["phase_true_next"] = df["phase_true_next"].astype(int)
    return df


def _evaluate_correction(
    df: pd.DataFrame,
    return_min: float,
    return_max: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Evaluate raw, oracle, soft, and hard z/p corrections."""
    ret_range = return_max - return_min

    def normalize_return(x: pd.Series) -> pd.Series:
        return -0.5 if ret_range <= 0 else (x - return_min) / ret_range - 1.0

    df = df.copy()
    df["return_norm"] = normalize_return(df["return"])
    df = _add_next_frame_predictions_temporal(df)

    results: dict[str, Any] = {}
    mse_raw = ((df["value_current"] - df["return_norm"]) ** 2).mean()
    results["mse_raw"] = float(mse_raw)

    df_oracle = _apply_correction(
        df,
        phase_col="phase_true",
        progress_col="phase_progress_true",
        phase_next_col="phase_true_next",
        progress_next_col="phase_progress_true_next",
    )
    mse_oracle = (
        (df_oracle["value_current_corr"] - df_oracle["return_norm"]) ** 2
    ).mean()
    results["mse_oracle"] = float(mse_oracle)
    results["oracle_improvement_pct"] = float((1 - mse_oracle / mse_raw) * 100)

    df_pred_soft = _apply_correction(
        df,
        phase_col="phase_pred",
        progress_col="phase_progress_pred_soft",
        phase_next_col="phase_pred_next",
        progress_next_col="phase_progress_pred_soft_next",
    )
    mse_pred_soft = (
        (df_pred_soft["value_current_corr"] - df_pred_soft["return_norm"]) ** 2
    ).mean()
    results["mse_predicted"] = float(mse_pred_soft)
    results["predicted_improvement_pct"] = float((1 - mse_pred_soft / mse_raw) * 100)
    results["mse_predicted_soft"] = float(mse_pred_soft)
    results["predicted_improvement_pct_soft"] = float(
        (1 - mse_pred_soft / mse_raw) * 100
    )

    df_pred_hard = _apply_correction(
        df,
        phase_col="phase_pred",
        progress_col="phase_progress_pred_hard",
        phase_next_col="phase_pred_next",
        progress_next_col="phase_progress_pred_hard_next",
    )
    mse_pred_hard = (
        (df_pred_hard["value_current_corr"] - df_pred_hard["return_norm"]) ** 2
    ).mean()
    results["mse_predicted_hard"] = float(mse_pred_hard)
    results["predicted_improvement_pct_hard"] = float(
        (1 - mse_pred_hard / mse_raw) * 100
    )

    return results, df_pred_soft, df_pred_hard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features_dir",
        default="/workspace/results/phase_progress_probe/features",
    )
    parser.add_argument(
        "--head_checkpoint",
        default="/workspace/results/phase_progress_probe/head_temporal_phase_prior/head.pt",
    )
    parser.add_argument(
        "--output_dir",
        default="/workspace/results/phase_progress_probe/analysis_temporal_phase_prior",
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
        json.dump(vars(args), f, indent=2, default=float)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading temporal phase-prior head from %s", args.head_checkpoint)
    head = _load_head(args.head_checkpoint, device)

    logger.info("Predicting z/p for splits: %s", ",".join(args.splits))
    pred_df = _predict_all(
        head=head,
        features_dir=Path(args.features_dir),
        device=device,
        batch_size=args.batch_size,
        splits=args.splits,
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

        correction_results, corrected_df_soft, corrected_df_hard = _evaluate_correction(
            merged,
            return_min=args.return_min,
            return_max=args.return_max,
        )
        report["correction"] = correction_results

        logger.info("=== Value MSE comparison ===")
        logger.info("  raw:        %.6f", correction_results["mse_raw"])
        logger.info(
            "  oracle:     %.6f (%.1f%% improvement)",
            correction_results["mse_oracle"],
            correction_results["oracle_improvement_pct"],
        )
        logger.info(
            "  predicted_soft: %.6f (%.1f%% improvement)",
            correction_results["mse_predicted"],
            correction_results["predicted_improvement_pct"],
        )
        logger.info(
            "  predicted_hard: %.6f (%.1f%% improvement)",
            correction_results["mse_predicted_hard"],
            correction_results["predicted_improvement_pct_hard"],
        )

        corrected_path_soft = output_dir / "advantages_predicted_corrected_soft.parquet"
        corrected_df_soft.to_parquet(corrected_path_soft, index=False)
        logger.info("Saved soft corrected advantages to %s", corrected_path_soft)

        corrected_path_hard = output_dir / "advantages_predicted_corrected_hard.parquet"
        corrected_df_hard.to_parquet(corrected_path_hard, index=False)
        logger.info("Saved hard corrected advantages to %s", corrected_path_hard)

    with open(output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    logger.info("Report saved to %s", output_dir / "report.json")


if __name__ == "__main__":
    main()
