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

"""Export fused Revalue predictions as a standard ReCap advantage tag."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rlinf.revalue.data.advantage_table import (
    read_advantages,
    resolve_advantage_path,
)


@dataclass
class ExportConfig:
    """Options for exporting fused predictions to ReCap advantage format."""

    dataset_path: str
    source_advantages_path: str
    predictions_path: str
    output_tag: str
    lookahead_step: int
    gamma: float = 1.0
    positive_quantile: float = 0.3
    discount_next_value: bool = True
    split: str | None = "train"
    report_path: str | None = None


def _load_predictions(path: str | Path, split: str | None) -> pd.DataFrame:
    pred_df = pd.read_parquet(path)
    required = {"episode_index", "frame_index", "value_fused"}
    missing = required - set(pred_df.columns)
    if missing:
        raise ValueError(f"Predictions missing columns: {sorted(missing)}")
    if split is not None and "split" in pred_df.columns:
        pred_df = pred_df[pred_df["split"] == split].copy()
    if pred_df.empty:
        raise ValueError(f"No predictions remain for split={split!r}")
    return pred_df


def compute_fused_advantages(
    *,
    source_advantages: pd.DataFrame,
    predictions: pd.DataFrame,
    lookahead_step: int,
    gamma: float,
    discount_next_value: bool,
) -> pd.DataFrame:
    """Recompute advantages using fused values and source reward sums."""
    required_source = {"episode_index", "frame_index", "reward_sum", "return"}
    missing = required_source - set(source_advantages.columns)
    if missing:
        raise ValueError(f"Source advantages missing columns: {sorted(missing)}")

    pred_keep = [
        "episode_index",
        "frame_index",
        "value_fused",
        "phase_pred",
        "phase_progress_pred",
        "global_progress_pred",
        "split",
    ]
    pred_keep = [col for col in pred_keep if col in predictions.columns]
    merged = source_advantages.merge(
        predictions[pred_keep],
        on=["episode_index", "frame_index"],
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlap between source advantages and predictions.")
    merged = merged.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    if "num_valid_rewards" in merged.columns:
        num_valid = merged["num_valid_rewards"].to_numpy(dtype=np.int64)
    else:
        num_valid = np.full(len(merged), lookahead_step, dtype=np.int64)

    values = merged["value_fused"].to_numpy(dtype=np.float64)
    next_values = np.zeros(len(merged), dtype=np.float64)
    grouped = merged.groupby("episode_index", sort=False).indices
    for _, indices in grouped.items():
        idx = np.asarray(indices, dtype=np.int64)
        frame_to_pos = {
            int(merged.loc[pos, "frame_index"]): int(pos)
            for pos in idx
        }
        for pos in idx:
            frame = int(merged.loc[pos, "frame_index"])
            next_frame = frame + int(num_valid[pos])
            next_pos = frame_to_pos.get(next_frame)
            next_values[pos] = 0.0 if next_pos is None else values[next_pos]

    gamma_k = (
        np.power(float(gamma), np.clip(num_valid, a_min=0, a_max=None))
        if discount_next_value
        else np.ones(len(merged), dtype=np.float64)
    )
    reward_sum = merged["reward_sum"].to_numpy(dtype=np.float64)
    advantage = reward_sum + gamma_k * next_values - values

    out = merged.copy()
    out["value_current"] = values
    out["value_next"] = next_values
    out["advantage_continuous"] = advantage
    return out


def build_save_advantages_df(
    fused_df: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    """Build a standard ReCap advantage dataframe."""
    save_cols = [
        "episode_index",
        "frame_index",
        "advantage_continuous",
        "return",
        "value_current",
        "value_next",
        "reward_sum",
        "reward_sum_raw",
        "num_valid_rewards",
        "dataset_name",
        "phase_pred",
        "phase_progress_pred",
        "global_progress_pred",
    ]
    save_cols = [col for col in save_cols if col in fused_df.columns]
    save_df = fused_df[save_cols].copy()
    save_df["advantage"] = save_df["advantage_continuous"] >= float(threshold)
    return save_df


def update_mixture_config(
    dataset_path: str | Path,
    *,
    tag: str,
    threshold: float,
    positive_quantile: float,
) -> None:
    """Record the exported tag in ``mixture_config.yaml`` when PyYAML exists."""
    try:
        import yaml
    except ImportError:
        return

    dataset_path = Path(dataset_path)
    path = dataset_path / "mixture_config.yaml"
    if path.exists():
        with open(path, "r", encoding="utf-8") as file:
            mixture = yaml.safe_load(file) or {}
    else:
        mixture = {}
    mixture.setdefault("tags", {})
    mixture["tags"][tag] = {
        "unified_threshold": float(threshold),
        "positive_quantile": float(positive_quantile),
    }
    mixture["advantage_tag"] = tag
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(mixture, file, default_flow_style=False)


def export_fused_advantages(cfg: ExportConfig) -> Path:
    """Write ``meta/advantages_<output_tag>.parquet`` for downstream CFG/ReCap."""
    source_df = read_advantages(cfg.source_advantages_path)
    pred_df = _load_predictions(cfg.predictions_path, cfg.split)
    fused_df = compute_fused_advantages(
        source_advantages=source_df,
        predictions=pred_df,
        lookahead_step=cfg.lookahead_step,
        gamma=cfg.gamma,
        discount_next_value=cfg.discount_next_value,
    )
    threshold = float(
        np.percentile(
            fused_df["advantage_continuous"].to_numpy(dtype=np.float64),
            (1.0 - cfg.positive_quantile) * 100.0,
        )
    )
    save_df = build_save_advantages_df(fused_df, threshold=threshold)
    out_path = resolve_advantage_path(cfg.dataset_path, cfg.output_tag)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_df.to_parquet(out_path, index=False)
    update_mixture_config(
        cfg.dataset_path,
        tag=cfg.output_tag,
        threshold=threshold,
        positive_quantile=cfg.positive_quantile,
    )

    report = {
        "dataset_path": str(Path(cfg.dataset_path).resolve()),
        "source_advantages_path": str(Path(cfg.source_advantages_path).resolve()),
        "predictions_path": str(Path(cfg.predictions_path).resolve()),
        "output_tag": cfg.output_tag,
        "advantage_path": str(out_path),
        "lookahead_step": int(cfg.lookahead_step),
        "gamma": float(cfg.gamma),
        "discount_next_value": bool(cfg.discount_next_value),
        "positive_quantile": float(cfg.positive_quantile),
        "threshold": threshold,
        "rows_exported": int(len(save_df)),
        "episodes_exported": int(save_df["episode_index"].nunique()),
        "positive_ratio": float(save_df["advantage"].mean()),
    }
    report_path = (
        Path(cfg.report_path)
        if cfg.report_path
        else Path(cfg.dataset_path) / "meta" / f"{cfg.output_tag}_revalue_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    return out_path
