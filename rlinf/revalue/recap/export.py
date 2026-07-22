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
    success_gate: bool = False
    failure_positive_cap: float = 0.2
    failure_reward: float = -300.0
    demo_backstop: bool = False


def infer_episode_success(
    df: pd.DataFrame,
    *,
    failure_reward: float,
) -> pd.Series:
    """Infer per-episode success from the first-frame return.

    ``compute_returns`` assigns -1 per step and adds ``failure_reward`` (e.g.
    -300) to the final frame of failed episodes, so a failed episode of
    length n has first-frame return around ``-(n + |failure_reward|)`` while
    a successful one has first-frame return around ``-n``. The midpoint
    ``-(n + |failure_reward| / 2)`` separates the two cases.
    """

    ordered = df.sort_values(["episode_index", "frame_index"])
    grouped = ordered.groupby("episode_index")
    episode_len = grouped["frame_index"].size()
    first_return = grouped["return"].first()
    return first_return > -(episode_len + abs(float(failure_reward)) / 2.0)


def compute_gated_positive_mask(
    df: pd.DataFrame,
    *,
    positive_quantile: float,
    failure_positive_cap: float,
    failure_reward: float,
    full_positive_episodes: set[int] | None = None,
) -> np.ndarray:
    """Build the positive mask with success gating and a demo backstop.

    - Episodes in ``full_positive_episodes`` are forced positive on every
      frame (expert demos) and do not count against the quantile budget.
    - All remaining rollout frames are ranked together by continuous
      advantage. Frames are accepted in descending order until the
      ``positive_quantile`` budget is filled, while failed-episode frames are
      skipped once ``failure_positive_cap`` of the budget has been accepted.
    """

    n_rows = len(df)
    if n_rows == 0:
        return np.zeros(0, dtype=bool)
    if not 0.0 <= float(positive_quantile) <= 1.0:
        raise ValueError("positive_quantile must be between 0 and 1")
    if not 0.0 <= float(failure_positive_cap) <= 1.0:
        raise ValueError("failure_positive_cap must be between 0 and 1")

    success_by_episode = infer_episode_success(df, failure_reward=failure_reward)
    episode_index = df["episode_index"].to_numpy()
    is_success_frame = np.array(
        [bool(success_by_episode.get(ep, False)) for ep in episode_index]
    )
    forced = np.zeros(n_rows, dtype=bool)
    if full_positive_episodes:
        forced_set = {int(ep) for ep in full_positive_episodes}
        forced = np.array([ep in forced_set for ep in episode_index])

    eligible = ~forced
    budget = int(round(float(positive_quantile) * float(eligible.sum())))
    advantages = df["advantage_continuous"].to_numpy(dtype=np.float64)

    chosen = np.zeros(n_rows, dtype=bool)
    failure_limit = int(float(failure_positive_cap) * budget)
    eligible_idx = np.flatnonzero(eligible)
    order = eligible_idx[np.argsort(-advantages[eligible_idx], kind="stable")]
    selected = 0
    selected_failures = 0
    for index in order:
        if selected >= budget:
            break
        if not is_success_frame[index]:
            if selected_failures >= failure_limit:
                continue
            selected_failures += 1
        chosen[index] = True
        selected += 1

    return chosen | forced


def summarize_gated_positive_mask(
    df: pd.DataFrame,
    positive_mask: np.ndarray,
    *,
    positive_quantile: float,
    failure_positive_cap: float,
    failure_reward: float,
    full_positive_episodes: set[int] | None = None,
) -> dict[str, float | int | bool]:
    """Summarize expert backstop and constrained rollout selection."""

    episode_index = df["episode_index"].to_numpy(dtype=np.int64)
    forced_set = {int(ep) for ep in full_positive_episodes or set()}
    forced = np.array([ep in forced_set for ep in episode_index], dtype=bool)
    rollout = ~forced
    positive = np.asarray(positive_mask, dtype=bool)
    rollout_positive = positive & rollout
    success_by_episode = infer_episode_success(
        df,
        failure_reward=failure_reward,
    )
    success_frame = np.array(
        [bool(success_by_episode.get(ep, False)) for ep in episode_index],
        dtype=bool,
    )

    rollout_frames = int(rollout.sum())
    rollout_budget = int(round(float(positive_quantile) * rollout_frames))
    rollout_positive_frames = int(rollout_positive.sum())
    rollout_failure_positive = int((rollout_positive & ~success_frame).sum())
    rollout_success_positive = rollout_positive_frames - rollout_failure_positive
    rollout_failure_ratio = (
        rollout_failure_positive / rollout_positive_frames
        if rollout_positive_frames
        else 0.0
    )
    total_positive = int(positive.sum())
    total_success_positive = int((positive & success_frame).sum())
    return {
        "failure_positive_cap": float(failure_positive_cap),
        "num_full_positive_episodes": len(forced_set),
        "num_full_positive_frames": int(forced.sum()),
        "num_rollout_frames": rollout_frames,
        "rollout_positive_budget": rollout_budget,
        "num_rollout_positive_frames": rollout_positive_frames,
        "num_rollout_success_positive_frames": rollout_success_positive,
        "num_rollout_failure_positive_frames": rollout_failure_positive,
        "rollout_positive_ratio": (
            rollout_positive_frames / rollout_frames if rollout_frames else 0.0
        ),
        "rollout_failure_positive_ratio": rollout_failure_ratio,
        "rollout_budget_filled": rollout_positive_frames == rollout_budget,
        "failure_cap_satisfied": (
            rollout_failure_ratio <= float(failure_positive_cap) + 1e-12
        ),
        "num_success_positive_frames": total_success_positive,
        "num_failure_positive_frames": total_positive - total_success_positive,
        "positive_success_purity": (
            total_success_positive / total_positive if total_positive else 0.0
        ),
    }


def load_full_positive_episodes(dataset_path: str | Path) -> set[int]:
    """Load demo episode indices from ``meta/full_positive_episodes.json``."""

    path = Path(dataset_path) / "meta" / "full_positive_episodes.json"
    if not path.exists():
        raise FileNotFoundError(
            f"demo_backstop requires {path}; write it when merging expert demos"
        )
    with open(path, "r", encoding="utf-8") as file:
        episodes = json.load(file)
    if not isinstance(episodes, list):
        raise ValueError(f"{path} must contain a JSON list of episode indices")
    return {int(ep) for ep in episodes}


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
    positive_mask: np.ndarray | None = None,
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
    if positive_mask is not None:
        save_df["advantage"] = np.asarray(positive_mask, dtype=bool)
    else:
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
    positive_mask = None
    gate_stats: dict[str, float | int | bool] = {}
    if cfg.success_gate or cfg.demo_backstop:
        forced = (
            load_full_positive_episodes(cfg.dataset_path)
            if cfg.demo_backstop
            else None
        )
        positive_mask = compute_gated_positive_mask(
            fused_df,
            positive_quantile=cfg.positive_quantile,
            failure_positive_cap=cfg.failure_positive_cap,
            failure_reward=cfg.failure_reward,
            full_positive_episodes=forced,
        )
        gate_stats = {
            "success_gate": bool(cfg.success_gate),
            "demo_backstop": bool(cfg.demo_backstop),
            **summarize_gated_positive_mask(
                fused_df,
                positive_mask,
                positive_quantile=cfg.positive_quantile,
                failure_positive_cap=cfg.failure_positive_cap,
                failure_reward=cfg.failure_reward,
                full_positive_episodes=forced,
            ),
        }
    save_df = build_save_advantages_df(
        fused_df, threshold=threshold, positive_mask=positive_mask
    )
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
        **gate_stats,
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
