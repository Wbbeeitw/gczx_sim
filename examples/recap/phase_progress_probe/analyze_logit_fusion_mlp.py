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
from train_logit_fusion_mlp import _normalize_return

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


def _run_fusion(
    model: LogitFusionMLP,
    df: pd.DataFrame,
    batch_size: int,
    atoms: torch.Tensor,
    alpha: float,
    num_phases: int,
    device: torch.device,
) -> pd.DataFrame:
    """Predict fused value for every row."""
    logits = np.stack(df["value_logits_current"].to_numpy()).astype(np.float32)
    phase = torch.tensor(df["phase"].values, dtype=torch.long)
    phase_onehot = torch.nn.functional.one_hot(phase, num_classes=num_phases).float()
    phase_progress = torch.tensor(df["phase_progress"].values, dtype=torch.float32)
    global_progress = torch.tensor(df["global_progress"].values, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(
            torch.tensor(logits, dtype=torch.float32),
            phase_onehot,
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

    df = _prepare_dataframe(
        parquet_path=Path(args.advantages_path),
        return_min=args.return_min,
        return_max=args.return_max,
    )
    fused_df = _run_fusion(
        model=model,
        df=df,
        batch_size=args.batch_size,
        atoms=atoms,
        alpha=args.alpha,
        num_phases=args.num_phases,
        device=device,
    )

    mse_raw = ((fused_df["value_current"] - fused_df["return_norm"]) ** 2).mean()
    mse_fused = ((fused_df["value_fused"] - fused_df["return_norm"]) ** 2).mean()
    improvement = (1 - mse_fused / mse_raw) * 100 if mse_raw > 0 else 0.0

    report: dict[str, Any] = {
        "correction": {
            "mse_raw": float(mse_raw),
            "mse_fused": float(mse_fused),
            "improvement_pct": float(improvement),
        }
    }

    logger.info("=== Logit Fusion Value MSE comparison ===")
    logger.info("  raw:    %.6f", mse_raw)
    logger.info("  fused:  %.6f (%.1f%% improvement)", mse_fused, improvement)

    fused_path = output_dir / "logit_fusion_predictions.parquet"
    fused_df.to_parquet(fused_path, index=False)
    with open(output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    logger.info("Saved predictions to %s", fused_path)
    logger.info("Report saved to %s", output_dir / "report.json")


if __name__ == "__main__":
    main()
