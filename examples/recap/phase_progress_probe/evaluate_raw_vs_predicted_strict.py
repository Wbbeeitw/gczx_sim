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

"""Strict raw-vs-predicted evaluation with train-fit / val-eval bias correction.

This script fixes the optimistic validation issue in the older analysis scripts:
it fits per-phase linear bias coefficients on the train split only, then applies
those frozen coefficients on the val split for evaluation.

It reports only:
1. raw critic value MSE on val
2. predicted corrected value MSE on val

No oracle metric is computed here.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _normalize_return(
    x: pd.Series,
    return_min: float,
    return_max: float,
) -> pd.Series:
    ret_range = return_max - return_min
    if ret_range <= 0:
        return pd.Series(np.full(len(x), -0.5, dtype=np.float64), index=x.index)
    return (x - return_min) / ret_range - 1.0


def _add_next_frame_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Add next-frame predicted phase/progress lookup columns."""
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
    ]
    next_df = df[next_cols].rename(
        columns={
            "frame_index": "next_frame_index",
            "phase_pred": "phase_pred_next",
            "phase_progress_pred": "phase_progress_pred_next",
        }
    )
    df = df.merge(
        next_df,
        on=["episode_index", "next_frame_index"],
        how="left",
    )

    df["phase_pred_next"] = df["phase_pred_next"].fillna(df["phase_pred"]).astype(int)
    df["phase_progress_pred_next"] = df["phase_progress_pred_next"].fillna(
        df["phase_progress_pred"]
    )
    return df


def _fit_phase_progress_bias(
    df: pd.DataFrame,
    phase_col: str,
    progress_col: str,
    value_error_col: str = "value_error",
) -> dict[int, tuple[float, float]]:
    """Fit per-phase linear bias coefficients on the fit split."""
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


def _apply_correction_with_coeffs(
    df: pd.DataFrame,
    coeffs: dict[int, tuple[float, float]],
) -> pd.DataFrame:
    """Apply frozen per-phase bias coefficients."""
    df = _add_next_frame_predictions(df)
    df = df.copy()

    def predict_bias(row: pd.Series) -> float:
        a, b = coeffs[int(row["phase_pred"])]
        return a * float(row["phase_progress_pred"]) + b

    def predict_bias_next(row: pd.Series) -> float:
        a, b = coeffs[int(row["phase_pred_next"])]
        return a * float(row["phase_progress_pred_next"]) + b

    df["bias_current"] = df.apply(predict_bias, axis=1)
    df["bias_next"] = df.apply(predict_bias_next, axis=1)
    df["value_current_corr"] = df["value_current"] - df["bias_current"]
    df["value_next_corr"] = df["value_next"] - df["bias_next"]
    return df


def _summarize_metrics(df_eval: pd.DataFrame) -> dict[str, Any]:
    """Compute strict val metrics for raw vs predicted correction."""
    mse_raw = ((df_eval["value_current"] - df_eval["return_norm"]) ** 2).mean()
    mse_pred = ((df_eval["value_current_corr"] - df_eval["return_norm"]) ** 2).mean()
    return {
        "rows_eval": int(len(df_eval)),
        "episodes_eval": int(df_eval["episode_index"].nunique()),
        "mse_raw": float(mse_raw),
        "mse_predicted": float(mse_pred),
        "predicted_improvement_pct": float((1 - mse_pred / mse_raw) * 100)
        if mse_raw > 0
        else 0.0,
    }


def _load_and_merge(
    advantages_path: Path,
    predictions_path: Path,
    splits: set[str],
    return_min: float,
    return_max: float,
) -> pd.DataFrame:
    """Load advantages and predictions, then merge the requested splits."""
    adv_df = pd.read_parquet(advantages_path)
    pred_df = pd.read_parquet(predictions_path)

    required_pred_cols = {
        "split",
        "episode_index",
        "frame_index",
        "phase_pred",
        "phase_progress_pred",
    }
    missing = required_pred_cols - set(pred_df.columns)
    if missing:
        raise ValueError(
            f"Prediction parquet missing columns {sorted(missing)}: {predictions_path}"
        )

    pred_df = pred_df[pred_df["split"].isin(splits)].copy()
    merged = adv_df.merge(
        pred_df,
        on=["episode_index", "frame_index"],
        how="inner",
    )
    merged = merged.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    merged["return_norm"] = _normalize_return(
        merged["return"],
        return_min=return_min,
        return_max=return_max,
    )
    merged["value_error"] = merged["value_current"] - merged["return_norm"]
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--advantages_path", required=True)
    parser.add_argument("--predictions_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--return_min", type=float, default=-700.0)
    parser.add_argument("--return_max", type=float, default=0.0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=float)

    logger.info("Loading merged train split for coefficient fitting")
    train_df = _load_and_merge(
        advantages_path=Path(args.advantages_path),
        predictions_path=Path(args.predictions_path),
        splits={"train"},
        return_min=args.return_min,
        return_max=args.return_max,
    )
    logger.info(
        "Fit split rows=%d episodes=%d",
        len(train_df),
        train_df["episode_index"].nunique(),
    )

    coeffs = _fit_phase_progress_bias(
        train_df,
        phase_col="phase_pred",
        progress_col="phase_progress_pred",
    )
    logger.info("Fitted coeffs for %d phases", len(coeffs))
    for ph in sorted(coeffs):
        a, b = coeffs[ph]
        logger.info("  phase=%d: a=%.6f b=%.6f", ph, a, b)

    logger.info("Loading merged val split for strict evaluation")
    val_df = _load_and_merge(
        advantages_path=Path(args.advantages_path),
        predictions_path=Path(args.predictions_path),
        splits={"val"},
        return_min=args.return_min,
        return_max=args.return_max,
    )
    logger.info(
        "Eval split rows=%d episodes=%d",
        len(val_df),
        val_df["episode_index"].nunique(),
    )

    val_corr_df = _apply_correction_with_coeffs(val_df, coeffs)
    metrics = _summarize_metrics(val_corr_df)

    logger.info("=== Strict Raw vs Predicted on val ===")
    logger.info("  raw:       %.6f", metrics["mse_raw"])
    logger.info(
        "  predicted: %.6f (%.1f%% improvement)",
        metrics["mse_predicted"],
        metrics["predicted_improvement_pct"],
    )

    save_cols = [
        "split",
        "episode_index",
        "frame_index",
        "phase_pred",
        "phase_progress_pred",
        "phase_pred_next",
        "phase_progress_pred_next",
        "return",
        "return_norm",
        "value_current",
        "value_next",
        "value_current_corr",
        "value_next_corr",
        "bias_current",
        "bias_next",
    ]
    save_cols = [c for c in save_cols if c in val_corr_df.columns]
    val_corr_df[save_cols].to_parquet(output_dir / "val_predictions_corrected.parquet", index=False)

    with open(output_dir / "bias_coeffs.json", "w", encoding="utf-8") as f:
        json.dump({int(k): {"a": v[0], "b": v[1]} for k, v in coeffs.items()}, f, indent=2)

    report = {
        "fit_split": {
            "rows": int(len(train_df)),
            "episodes": int(train_df["episode_index"].nunique()),
        },
        "eval_split": {
            "rows": int(len(val_df)),
            "episodes": int(val_df["episode_index"].nunique()),
        },
        "metrics": metrics,
    }
    with open(output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)

    logger.info("Saved strict report to %s", output_dir / "report.json")


if __name__ == "__main__":
    main()
