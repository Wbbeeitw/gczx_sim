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

"""Export a re-indexed child-dataset view of fused Revalue advantages."""

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
    update_mixture_config,
)


@dataclass
class ExportDatasetViewConfig:
    """Options for exporting fused advantages onto a child dataset view.

    A round's freshly collected episodes live inside the merged parent
    dataset at ``source_episode_start <= episode_index < source_episode_end``.
    This view slices those rows out of the parent base/prediction tables,
    re-indexes them by ``child_episode_offset``, recomputes the positive
    threshold on the child rows only, and writes a standard ReCap advantage
    sidecar into the child dataset. No new model inference is required: the
    child rows reuse the parent critic/Revalue predictions.
    """

    source_advantages_path: str
    predictions_path: str
    child_dataset_path: str
    output_tag: str
    source_episode_start: int
    source_episode_end: int | None = None
    child_episode_offset: int = 0
    lookahead_step: int = 10
    gamma: float = 1.0
    positive_quantile: float = 0.3
    discount_next_value: bool = True
    expected_episodes: int | None = None
    report_path: str | None = None


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
    pred_df = pd.read_parquet(cfg.predictions_path)
    for name, df in (("source advantages", source_df), ("predictions", pred_df)):
        if "episode_index" not in df.columns:
            raise ValueError(f"{name} table is missing episode_index")

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
    child_pred = _slice_and_reindex(pred_df)
    if child_source.empty or child_pred.empty:
        raise ValueError(
            f"no rows found for episodes [{start}, {end}); check "
            "source_episode_start/end against the parent dataset"
        )
    for name, df in (("source advantages", child_source), ("predictions", child_pred)):
        if int(df["episode_index"].min()) < 0:
            raise ValueError(
                f"child_episode_offset={cfg.child_episode_offset} produced "
                f"negative episode_index in {name}"
            )

    fused_df = compute_fused_advantages(
        source_advantages=child_source,
        predictions=child_pred,
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
        "source_advantages_path": str(Path(cfg.source_advantages_path).resolve()),
        "predictions_path": str(Path(cfg.predictions_path).resolve()),
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
