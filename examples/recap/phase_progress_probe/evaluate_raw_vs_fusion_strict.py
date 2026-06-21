# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Strict raw-vs-fused evaluation on val using frozen z/p predictions and fusion MLP."""

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

logger = logging.getLogger(__name__)


def _normalize_return(
    values: pd.Series,
    return_min: float,
    return_max: float,
) -> pd.Series:
    ret_range = return_max - return_min
    if ret_range <= 0:
        return pd.Series(
            np.full(len(values), -0.5, dtype=np.float32),
            index=values.index,
        )
    return (values - return_min) / ret_range - 1.0


def _load_model(path: Path, device: torch.device) -> LogitFusionMLP:
    """Load trained fusion model."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = LogitFusionMLP(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


def _load_and_merge(
    predictions_path: Path,
    advantages_path: Path,
    return_min: float,
    return_max: float,
) -> pd.DataFrame:
    """Load merged prediction/value rows for evaluation."""
    pred_df = pd.read_parquet(predictions_path)
    adv_df = pd.read_parquet(advantages_path)

    required_pred_cols = {
        "split",
        "episode_index",
        "frame_index",
        "phase_probs_pred",
        "phase_progress_pred",
        "global_progress_pred",
    }
    missing = required_pred_cols - set(pred_df.columns)
    if missing:
        raise ValueError(
            f"Prediction parquet missing columns {sorted(missing)}: {predictions_path}"
        )

    merged = adv_df.merge(
        pred_df,
        on=["episode_index", "frame_index"],
        how="inner",
    )
    merged = merged[merged["split"] == "val"].copy()
    merged = merged.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    merged["return_norm"] = _normalize_return(
        merged["return"],
        return_min=return_min,
        return_max=return_max,
    )
    return merged


def _run_fusion(
    model: LogitFusionMLP,
    df: pd.DataFrame,
    batch_size: int,
    atoms: torch.Tensor,
    alpha: float,
    device: torch.device,
) -> pd.DataFrame:
    """Predict fused scalar value for val rows."""
    logits = np.stack(df["value_logits_current"].to_numpy()).astype(np.float32)
    phase_probs = np.stack(df["phase_probs_pred"].to_numpy()).astype(np.float32)
    phase_progress = df["phase_progress_pred"].to_numpy(dtype=np.float32)
    global_progress = df["global_progress_pred"].to_numpy(dtype=np.float32)

    loader = DataLoader(
        TensorDataset(
            torch.tensor(logits, dtype=torch.float32),
            torch.tensor(phase_probs, dtype=torch.float32),
            torch.tensor(phase_progress, dtype=torch.float32),
            torch.tensor(global_progress, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    fused_values: list[torch.Tensor] = []
    with torch.no_grad():
        for raw_logits, phase_probs, phase_progress, global_progress in loader:
            delta = model(
                raw_logits.to(device),
                phase_probs.to(device),
                phase_progress.to(device),
                global_progress.to(device),
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
    parser.add_argument("--predictions_path", required=True)
    parser.add_argument("--advantages_path", required=True)
    parser.add_argument("--fusion_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--return_min", type=float, default=-700.0)
    parser.add_argument("--return_max", type=float, default=0.0)
    parser.add_argument("--num_bins", type=int, default=201)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=float)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(Path(args.fusion_checkpoint), device)
    merged = _load_and_merge(
        predictions_path=Path(args.predictions_path),
        advantages_path=Path(args.advantages_path),
        return_min=args.return_min,
        return_max=args.return_max,
    )

    atoms = torch.linspace(args.return_min, args.return_max, args.num_bins)
    atoms = (atoms - args.return_min) / (args.return_max - args.return_min) - 1.0
    fused_df = _run_fusion(
        model=model,
        df=merged,
        batch_size=args.batch_size,
        atoms=atoms,
        alpha=args.alpha,
        device=device,
    )

    mse_raw = ((fused_df["value_current"] - fused_df["return_norm"]) ** 2).mean()
    mse_fused = ((fused_df["value_fused"] - fused_df["return_norm"]) ** 2).mean()
    improvement = (1 - mse_fused / mse_raw) * 100 if mse_raw > 0 else 0.0

    logger.info("=== Strict Raw vs Fusion on val ===")
    logger.info("  raw:    %.6f", mse_raw)
    logger.info("  fused:  %.6f (%.1f%% improvement)", mse_fused, improvement)

    fused_df.to_parquet(output_dir / "val_fusion_predictions.parquet", index=False)
    report: dict[str, Any] = {
        "eval_split": {
            "rows": int(len(fused_df)),
            "episodes": int(fused_df["episode_index"].nunique()),
        },
        "metrics": {
            "mse_raw": float(mse_raw),
            "mse_fused": float(mse_fused),
            "fusion_improvement_pct": float(improvement),
        },
    }
    with open(output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)

    logger.info("Saved strict fusion report to %s", output_dir / "report.json")


if __name__ == "__main__":
    main()
