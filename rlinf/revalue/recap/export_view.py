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

"""Export a re-indexed child-dataset view of raw or fused advantages."""

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
from rlinf.revalue.recap.export import (
    build_save_advantages_df,
    compute_fused_advantages,
    compute_gated_positive_mask,
    infer_episode_success,
    load_full_positive_episodes,
    update_mixture_config,
)


@dataclass
class ExportDatasetViewConfig:
    """Options for exporting advantages onto a child dataset view.

    A round's freshly collected episodes live inside the merged parent
    dataset at ``source_episode_start <= episode_index < source_episode_end``.
    This view slices those rows out of the parent base table and, in fused
    mode, the prediction table. It re-indexes them by
    ``child_episode_offset``, recomputes the positive threshold on the child
    rows only, and writes a standard ReCap advantage sidecar into the child
    dataset. No new model inference is required.
    """

    source_advantages_path: str
    predictions_path: str | None
    child_dataset_path: str
    output_tag: str
    source_episode_start: int
    mode: str = "fused"
    source_episode_end: int | None = None
    child_episode_offset: int = 0
    lookahead_step: int = 10
    gamma: float = 1.0
    positive_quantile: float = 0.3
    discount_next_value: bool = True
    expected_episodes: int | None = None
    report_path: str | None = None
    success_gate: bool = False
    failure_positive_cap: float = 0.2
    failure_reward: float = -300.0
    demo_backstop: bool = False


def _child_total_frames(child_dataset_path: Path) -> int | None:
    info_path = child_dataset_path / "meta" / "info.json"
    if not info_path.exists():
        return None
    with open(info_path, "r", encoding="utf-8") as file:
        info = json.load(file)
    total = info.get("total_frames")
    return int(total) if total is not None else None


def export_dataset_view(cfg: ExportDatasetViewConfig) -> Path:
    """Write ``meta/advantages_<output_tag>.parquet`` for the child dataset."""
    child_path = Path(cfg.child_dataset_path)
    if not child_path.exists():
        raise FileNotFoundError(f"child dataset does not exist: {child_path}")

    source_df = read_advantages(cfg.source_advantages_path)
    if cfg.mode not in {"raw", "fused"}:
        raise ValueError("mode must be either 'raw' or 'fused'")
    if "episode_index" not in source_df.columns:
        raise ValueError("source advantages table is missing episode_index")
    pred_df = None
    if cfg.mode == "fused":
        if not cfg.predictions_path:
            raise ValueError("predictions_path is required in fused mode")
        pred_df = pd.read_parquet(cfg.predictions_path)
        if "episode_index" not in pred_df.columns:
            raise ValueError("predictions table is missing episode_index")

    start = int(cfg.source_episode_start)
    end = None if cfg.source_episode_end is None else int(cfg.source_episode_end)

    def _slice_and_reindex(df: pd.DataFrame) -> pd.DataFrame:
        mask = df["episode_index"] >= start
        if end is not None:
            mask &= df["episode_index"] < end
        out = df[mask].copy()
        out["episode_index"] = out["episode_index"] + int(cfg.child_episode_offset)
        return out

    child_source = _slice_and_reindex(source_df)
    child_pred = _slice_and_reindex(pred_df) if pred_df is not None else None
    if child_source.empty or (child_pred is not None and child_pred.empty):
        raise ValueError(
            f"no rows found for episodes [{start}, {end}); check "
            "source_episode_start/end against the parent dataset"
        )
    tables = [("source advantages", child_source)]
    if child_pred is not None:
        tables.append(("predictions", child_pred))
    for name, df in tables:
        if int(df["episode_index"].min()) < 0:
            raise ValueError(
                f"child_episode_offset={cfg.child_episode_offset} produced "
                f"negative episode_index in {name}"
            )

    if cfg.mode == "fused":
        exported_df = compute_fused_advantages(
            source_advantages=child_source,
            predictions=child_pred,
            lookahead_step=cfg.lookahead_step,
            gamma=cfg.gamma,
            discount_next_value=cfg.discount_next_value,
        )
    else:
        if "advantage_continuous" not in child_source.columns:
            raise ValueError(
                "raw source advantages are missing advantage_continuous"
            )
        exported_df = child_source
    threshold = float(
        np.percentile(
            exported_df["advantage_continuous"].to_numpy(dtype=np.float64),
            (1.0 - cfg.positive_quantile) * 100.0,
        )
    )
    positive_mask = None
    gate_stats: dict[str, float | int | bool] = {}
    if cfg.success_gate or cfg.demo_backstop:
        forced = (
            load_full_positive_episodes(child_path) if cfg.demo_backstop else None
        )
        positive_mask = compute_gated_positive_mask(
            exported_df,
            positive_quantile=cfg.positive_quantile,
            failure_positive_cap=cfg.failure_positive_cap,
            failure_reward=cfg.failure_reward,
            full_positive_episodes=forced,
        )
        success_by_episode = infer_episode_success(
            exported_df, failure_reward=cfg.failure_reward
        )
        positive_episodes = exported_df["episode_index"][positive_mask]
        num_positive = int(positive_mask.sum())
        num_success_positive = int(
            positive_episodes.map(success_by_episode).fillna(False).sum()
        )
        gate_stats = {
            "success_gate": bool(cfg.success_gate),
            "demo_backstop": bool(cfg.demo_backstop),
            "failure_positive_cap": float(cfg.failure_positive_cap),
            "num_full_positive_episodes": len(forced) if forced else 0,
            "num_success_positive_frames": num_success_positive,
            "num_failure_positive_frames": num_positive - num_success_positive,
            "positive_success_purity": (
                num_success_positive / num_positive if num_positive else 0.0
            ),
        }
    save_df = build_save_advantages_df(
        exported_df, threshold=threshold, positive_mask=positive_mask
    )

    episodes = int(save_df["episode_index"].nunique())
    if cfg.expected_episodes is not None and episodes != int(cfg.expected_episodes):
        raise ValueError(
            f"expected {cfg.expected_episodes} child episodes, got {episodes}"
        )
    total_frames = _child_total_frames(child_path)
    if total_frames is not None and len(save_df) != total_frames:
        raise ValueError(
            f"child dataset meta/info.json reports {total_frames} frames, "
            f"but the exported view has {len(save_df)} rows"
        )

    out_path = resolve_advantage_path(child_path, cfg.output_tag)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_df.to_parquet(out_path, index=False)
    update_mixture_config(
        child_path,
        tag=cfg.output_tag,
        threshold=threshold,
        positive_quantile=cfg.positive_quantile,
    )

    report = {
        "child_dataset_path": str(child_path.resolve()),
        "mode": cfg.mode,
        "source_advantages_path": str(Path(cfg.source_advantages_path).resolve()),
        "predictions_path": (
            str(Path(cfg.predictions_path).resolve())
            if cfg.predictions_path
            else None
        ),
        "output_tag": cfg.output_tag,
        "advantage_path": str(out_path),
        "source_episode_start": start,
        "source_episode_end": end,
        "child_episode_offset": int(cfg.child_episode_offset),
        "lookahead_step": int(cfg.lookahead_step),
        "gamma": float(cfg.gamma),
        "discount_next_value": bool(cfg.discount_next_value),
        "positive_quantile": float(cfg.positive_quantile),
        "threshold": threshold,
        "rows_exported": int(len(save_df)),
        "episodes_exported": episodes,
        "child_total_frames": total_frames,
        "positive_ratio": float(save_df["advantage"].mean()),
        **gate_stats,
    }
    report_path = (
        Path(cfg.report_path)
        if cfg.report_path
        else child_path / "meta" / f"{cfg.output_tag}_revalue_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    return out_path
