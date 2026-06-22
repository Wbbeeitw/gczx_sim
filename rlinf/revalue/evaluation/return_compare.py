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

"""Compare raw base and fused return prediction quality."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rlinf.revalue.io import save_json


@dataclass(frozen=True)
class ReturnComparisonConfig:
    """Options for base-vs-fused return prediction comparison."""

    advantages_path: str
    predictions_path: str
    output_path: str
    return_min: float = -700.0
    return_max: float = 0.0
    splits: tuple[str, ...] = ("all", "train", "val", "test")


def _prediction_metrics(part: pd.DataFrame, pred_col: str) -> dict[str, float]:
    err = part[pred_col] - part["target_return"]
    mse = float(np.mean(err**2))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
    }


def _split_frame_report(part: pd.DataFrame) -> dict[str, Any]:
    base = _prediction_metrics(part, "base_pred_return")
    fused = _prediction_metrics(part, "fused_pred_return")
    mse_improvement = 0.0
    if base["mse"] > 0.0:
        mse_improvement = 100.0 * (1.0 - fused["mse"] / base["mse"])
    mae_improvement = 0.0
    if base["mae"] > 0.0:
        mae_improvement = 100.0 * (1.0 - fused["mae"] / base["mae"])
    rmse_improvement = 0.0
    if base["rmse"] > 0.0:
        rmse_improvement = 100.0 * (1.0 - fused["rmse"] / base["rmse"])
    bias_abs_improvement = 0.0
    base_bias_abs = abs(base["bias"])
    fused_bias_abs = abs(fused["bias"])
    if base_bias_abs > 0.0:
        bias_abs_improvement = 100.0 * (1.0 - fused_bias_abs / base_bias_abs)
    return {
        "rows": int(len(part)),
        "episodes": int(part["episode_index"].nunique()),
        "base": base,
        "shared_mlp_fusion": fused,
        "mse_improvement_pct": float(mse_improvement),
        "rmse_improvement_pct": float(rmse_improvement),
        "mae_improvement_pct": float(mae_improvement),
        "bias_abs_improvement_pct": float(bias_abs_improvement),
    }


def _episode_report(part: pd.DataFrame) -> dict[str, Any]:
    ep = (
        part.groupby("episode_index")
        .agg(
            base_mse=("base_sqerr", "mean"),
            fused_mse=("fused_sqerr", "mean"),
            base_mae=("base_abserr", "mean"),
            fused_mae=("fused_abserr", "mean"),
            frames=("frame_index", "count"),
        )
        .reset_index()
    )
    ep["mse_gain"] = ep["base_mse"] - ep["fused_mse"]
    ep["mae_gain"] = ep["base_mae"] - ep["fused_mae"]
    return {
        "episodes": int(len(ep)),
        "episodes_improved_mse": int((ep["mse_gain"] > 0.0).sum()),
        "mean_base_mse": float(ep["base_mse"].mean()),
        "mean_fused_mse": float(ep["fused_mse"].mean()),
        "mean_mse_gain": float(ep["mse_gain"].mean()),
        "median_mse_gain": float(ep["mse_gain"].median()),
        "mean_base_mae": float(ep["base_mae"].mean()),
        "mean_fused_mae": float(ep["fused_mae"].mean()),
        "mean_mae_gain": float(ep["mae_gain"].mean()),
        "worst_episodes_by_mse_gain": [
            {
                "episode_index": int(row.episode_index),
                "frames": int(row.frames),
                "base_mse": float(row.base_mse),
                "fused_mse": float(row.fused_mse),
                "mse_gain": float(row.mse_gain),
            }
            for row in ep.sort_values("mse_gain").head(5).itertuples(index=False)
        ],
    }


def _load_comparison_frame(cfg: ReturnComparisonConfig) -> pd.DataFrame:
    adv = pd.read_parquet(cfg.advantages_path)
    pred = pd.read_parquet(cfg.predictions_path)
    required_adv = {"episode_index", "frame_index", "return", "value_current"}
    required_pred = {"episode_index", "frame_index", "value_fused"}
    missing_adv = required_adv - set(adv.columns)
    missing_pred = required_pred - set(pred.columns)
    if missing_adv:
        raise ValueError(f"Advantages missing columns: {sorted(missing_adv)}")
    if missing_pred:
        raise ValueError(f"Predictions missing columns: {sorted(missing_pred)}")

    keep = ["episode_index", "frame_index", "value_fused"]
    if "split" in pred.columns:
        keep.append("split")
    merged = pred[keep].merge(
        adv[["episode_index", "frame_index", "return", "value_current"]],
        on=["episode_index", "frame_index"],
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlap between predictions and advantages.")
    if "split" not in merged.columns:
        merged["split"] = "all"

    ret_range = float(cfg.return_max) - float(cfg.return_min)
    if ret_range <= 0.0:
        raise ValueError("return_max must be greater than return_min")
    merged["base_pred_return"] = (
        (merged["value_current"] + 1.0) * ret_range + float(cfg.return_min)
    )
    merged["fused_pred_return"] = (
        (merged["value_fused"] + 1.0) * ret_range + float(cfg.return_min)
    )
    merged["target_return"] = merged["return"]
    merged["base_sqerr"] = (merged["base_pred_return"] - merged["target_return"]) ** 2
    merged["fused_sqerr"] = (
        merged["fused_pred_return"] - merged["target_return"]
    ) ** 2
    merged["base_abserr"] = (
        merged["base_pred_return"] - merged["target_return"]
    ).abs()
    merged["fused_abserr"] = (
        merged["fused_pred_return"] - merged["target_return"]
    ).abs()
    return merged


def compare_return_predictions(cfg: ReturnComparisonConfig) -> dict[str, Any]:
    """Compare base and fused return predictions and save a JSON report."""
    df = _load_comparison_frame(cfg)
    report: dict[str, Any] = {
        "advantages_path": str(Path(cfg.advantages_path)),
        "predictions_path": str(Path(cfg.predictions_path)),
        "return_min": float(cfg.return_min),
        "return_max": float(cfg.return_max),
        "frame_level": {},
        "episode_level": {},
    }

    for split in cfg.splits:
        part = df if split == "all" else df[df["split"] == split]
        if part.empty:
            continue
        report["frame_level"][split] = _split_frame_report(part)
        report["episode_level"][split] = _episode_report(part)

    save_json(report, cfg.output_path)
    return report


def summarize_return_report(report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Flatten a return comparison report into frame and episode summary rows."""
    frame_rows: list[dict[str, Any]] = []
    for split, part in report.get("frame_level", {}).items():
        base = part.get("base", {})
        fused = part.get("shared_mlp_fusion", {})
        base_mse = base.get("mse")
        fused_mse = fused.get("mse")
        base_rmse = base.get("rmse")
        fused_rmse = fused.get("rmse")
        base_mae = base.get("mae")
        fused_mae = fused.get("mae")
        base_bias = base.get("bias")
        fused_bias = fused.get("bias")

        mse_improvement = part.get("mse_improvement_pct")
        if mse_improvement is None and base_mse not in (None, 0) and fused_mse is not None:
            mse_improvement = 100.0 * (1.0 - float(fused_mse) / float(base_mse))

        rmse_improvement = part.get("rmse_improvement_pct")
        if (
            rmse_improvement is None
            and base_rmse not in (None, 0)
            and fused_rmse is not None
        ):
            rmse_improvement = 100.0 * (
                1.0 - float(fused_rmse) / float(base_rmse)
            )

        mae_improvement = part.get("mae_improvement_pct")
        if mae_improvement is None and base_mae not in (None, 0) and fused_mae is not None:
            mae_improvement = 100.0 * (1.0 - float(fused_mae) / float(base_mae))

        bias_abs_improvement = part.get("bias_abs_improvement_pct")
        if (
            bias_abs_improvement is None
            and base_bias is not None
            and fused_bias is not None
            and abs(float(base_bias)) > 0.0
        ):
            bias_abs_improvement = 100.0 * (
                1.0 - abs(float(fused_bias)) / abs(float(base_bias))
            )
        frame_rows.append(
            {
                "split": split,
                "rows": part.get("rows"),
                "episodes": part.get("episodes"),
                "base_mse": base_mse,
                "fused_mse": fused_mse,
                "mse_improvement_pct": mse_improvement,
                "base_rmse": base_rmse,
                "fused_rmse": fused_rmse,
                "rmse_improvement_pct": rmse_improvement,
                "base_mae": base_mae,
                "fused_mae": fused_mae,
                "mae_improvement_pct": mae_improvement,
                "base_bias": base_bias,
                "fused_bias": fused_bias,
                "bias_abs_improvement_pct": bias_abs_improvement,
            }
        )

    episode_rows: list[dict[str, Any]] = []
    for split, part in report.get("episode_level", {}).items():
        episodes = int(part.get("episodes", 0) or 0)
        improved = int(part.get("episodes_improved_mse", 0) or 0)
        improved_pct = 100.0 * improved / episodes if episodes > 0 else 0.0
        episode_rows.append(
            {
                "split": split,
                "episodes": episodes,
                "episodes_improved_mse": improved,
                "episodes_improved_mse_pct": improved_pct,
                "mean_base_mse": part.get("mean_base_mse"),
                "mean_fused_mse": part.get("mean_fused_mse"),
                "mean_mse_gain": part.get("mean_mse_gain"),
                "median_mse_gain": part.get("median_mse_gain"),
                "mean_base_mae": part.get("mean_base_mae"),
                "mean_fused_mae": part.get("mean_fused_mae"),
                "mean_mae_gain": part.get("mean_mae_gain"),
            }
        )

    return {
        "frame_rows": frame_rows,
        "episode_rows": episode_rows,
    }
