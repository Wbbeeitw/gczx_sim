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

"""Export joint-critic fused predictions into a new ReCap advantage tag.

This script bridges the phase-progress joint critic experiments with downstream
CFG/ReCap training. It takes frame-level fused value predictions produced by
``analyze_joint_logit_fusion.py`` (or an equivalent parquet with
``episode_index``, ``frame_index``, and ``value_fused``), recomputes

    A_t = normalize(r_{t:t+N}) + gamma^k * V_fused(o_{t+N}) - V_fused(o_t),

and saves a standard ``meta/advantages_<tag>.parquet`` file that existing CFG
training can consume via ``data.advantage_tag=<tag>``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from episode_subset_utils import (
    compute_frame_indices_for_episodes,
    load_episode_subset_file,
    resolve_episode_split_for_dataset,
    resolve_episode_subset_for_dataset,
)
from recompute_advantages_from_value_reward import (
    build_save_advantages_df,
    load_existing_advantages,
    update_mixture_config,
)
from rlinf.data.datasets.recap.utils import load_return_stats_from_dataset

logger = logging.getLogger(__name__)


def _resolve_selected_keys(
    dataset_path: Path,
    episode_subset_path: str | None,
    episode_split_path: str | None,
    split_name: str,
) -> set[tuple[int, int]] | None:
    """Resolve optional episode filtering into a key set."""
    if episode_split_path:
        raw = load_episode_subset_file(episode_split_path)
        split_spec = resolve_episode_split_for_dataset(raw, dataset_path)
        if split_spec is None:
            raise ValueError(
                f"No episode split entry found for dataset '{dataset_path.name}' "
                f"in {episode_split_path}"
            )
        split_name = split_name.lower()
        if split_name == "train":
            episode_ids = split_spec.train_episodes
        elif split_name == "val":
            episode_ids = split_spec.val_episodes
        elif split_name == "selected":
            episode_ids = split_spec.selected_episodes
        else:
            raise ValueError(
                f"split_name must be one of ['train', 'val', 'selected'], got {split_name!r}"
            )
    elif episode_subset_path:
        raw = load_episode_subset_file(episode_subset_path)
        subset_spec = resolve_episode_subset_for_dataset(raw, dataset_path)
        if subset_spec is None:
            raise ValueError(
                f"No episode subset entry found for dataset '{dataset_path.name}' "
                f"in {episode_subset_path}"
            )
        episode_ids = subset_spec.episodes
    else:
        return None

    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(str(dataset_path))
    frame_indices = compute_frame_indices_for_episodes(
        meta.episode_data_index["to"], episode_ids
    )

    # Build (episode_index, frame_index) pairs from episode boundaries.
    keys: set[tuple[int, int]] = set()
    ep_to = [int(x) for x in meta.episode_data_index["to"]]
    for global_idx in frame_indices:
        ep_idx = int(np.searchsorted(ep_to, global_idx, side="right"))
        ep_start = 0 if ep_idx == 0 else ep_to[ep_idx - 1]
        frame_idx = global_idx - ep_start
        keys.add((ep_idx, frame_idx))
    return keys


def _load_predictions(
    predictions_path: Path,
    selected_keys: set[tuple[int, int]] | None,
) -> pd.DataFrame:
    """Load fused predictions and optionally restrict to a subset/split."""
    pred_df = pd.read_parquet(predictions_path)
    required = {"episode_index", "frame_index", "value_fused"}
    missing = required - set(pred_df.columns)
    if missing:
        raise ValueError(
            f"Predictions parquet missing required columns: {sorted(missing)}"
        )

    keep_cols = [
        "episode_index",
        "frame_index",
        "value_fused",
        "phase_pred",
        "phase_progress_pred",
        "global_progress_pred",
        "split",
    ]
    keep_cols = [col for col in keep_cols if col in pred_df.columns]
    pred_df = pred_df[keep_cols].copy()

    pred_df["episode_index"] = pred_df["episode_index"].astype(np.int64)
    pred_df["frame_index"] = pred_df["frame_index"].astype(np.int64)
    pred_df["value_fused"] = pred_df["value_fused"].astype(np.float64)

    if selected_keys is not None:
        mask = [
            (int(ep), int(fr)) in selected_keys
            for ep, fr in zip(pred_df["episode_index"], pred_df["frame_index"])
        ]
        pred_df = pred_df.loc[mask].reset_index(drop=True)

    if pred_df.empty:
        raise ValueError("No prediction rows remain after applying selection filters.")

    return pred_df


def _compute_fused_advantages(
    predictions_df: pd.DataFrame,
    source_adv_df: pd.DataFrame,
    lookahead_step: int,
    gamma: float,
    discount_next_value: bool,
) -> pd.DataFrame:
    """Recompute advantages using fused values while keeping original rewards/returns."""
    merged = source_adv_df.merge(
        predictions_df,
        on=["episode_index", "frame_index"],
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlapping rows between source advantages and predictions.")

    merged = merged.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    required_cols = {"reward_sum", "return"}
    missing = required_cols - set(merged.columns)
    if missing:
        raise ValueError(
            f"Source advantages parquet missing required columns: {sorted(missing)}"
        )

    if "num_valid_rewards" in merged.columns:
        num_valid_arr = merged["num_valid_rewards"].to_numpy(dtype=np.int64)
    else:
        num_valid_arr = np.full(len(merged), lookahead_step, dtype=np.int64)

    fused_next = np.zeros(len(merged), dtype=np.float64)
    fused_values = merged["value_fused"].to_numpy(dtype=np.float64)

    grouped = merged.groupby("episode_index", sort=False).indices
    for _, indices in grouped.items():
        idx = np.asarray(indices, dtype=np.int64)
        local_len = len(idx)
        cutoff = local_len - lookahead_step
        if cutoff > 0:
            fused_next[idx[:cutoff]] = fused_values[idx[lookahead_step:]]

    if discount_next_value:
        gamma_k = np.power(gamma, np.clip(num_valid_arr, a_min=0, a_max=None))
    else:
        gamma_k = np.ones(len(merged), dtype=np.float64)

    reward_sum = merged["reward_sum"].to_numpy(dtype=np.float64)
    advantage_continuous = reward_sum + gamma_k * fused_next - fused_values

    out_df = merged.copy()
    out_df["value_current"] = fused_values
    out_df["value_next"] = fused_next
    out_df["advantage_continuous"] = advantage_continuous

    if "value_logits_current" in out_df.columns:
        logger.warning(
            "Keeping source value logits/probs unchanged in output parquet. "
            "Downstream CFG training only reads the boolean advantage label."
        )

    return out_df


def _write_tag(
    dataset_path: Path,
    output_tag: str,
    save_df: pd.DataFrame,
    positive_quantile: float,
    threshold: float,
) -> Path:
    """Save the recomputed advantage tag into dataset meta/."""
    meta_dir = dataset_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    adv_path = meta_dir / f"advantages_{output_tag}.parquet"
    save_df.to_parquet(adv_path, index=False)

    ret_min, ret_max = load_return_stats_from_dataset(dataset_path)
    update_mixture_config(
        mixture_root=dataset_path,
        dataset_paths=[dataset_path],
        positive_quantile=positive_quantile,
        unified_threshold=threshold,
        advantage_tag=output_tag,
        global_return_min=ret_min,
        global_return_max=ret_max,
    )
    return adv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--predictions_path", required=True)
    parser.add_argument("--source_tag", required=True)
    parser.add_argument("--output_tag", required=True)
    parser.add_argument("--lookahead_step", type=int, required=True)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--positive_quantile", type=float, default=0.3)
    parser.add_argument(
        "--discount_next_value",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use gamma^k * V(o_{t+k}) in the exported advantage formula.",
    )
    parser.add_argument("--episode_subset_path", default=None)
    parser.add_argument("--episode_split_path", default=None)
    parser.add_argument(
        "--split_name",
        choices=["train", "val", "selected"],
        default="train",
        help="Which split from episode_split_path to export.",
    )
    parser.add_argument(
        "--report_path",
        default=None,
        help="Optional JSON report path. Defaults to meta/<output_tag>_export_report.json.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    dataset_path = Path(args.dataset_path).resolve()
    predictions_path = Path(args.predictions_path).resolve()

    selected_keys = _resolve_selected_keys(
        dataset_path=dataset_path,
        episode_subset_path=args.episode_subset_path,
        episode_split_path=args.episode_split_path,
        split_name=args.split_name,
    )
    pred_df = _load_predictions(predictions_path, selected_keys)
    source_adv_df = load_existing_advantages(dataset_path, source_tag=args.source_tag)

    fused_df = _compute_fused_advantages(
        predictions_df=pred_df,
        source_adv_df=source_adv_df,
        lookahead_step=args.lookahead_step,
        gamma=args.gamma,
        discount_next_value=args.discount_next_value,
    )

    threshold = float(
        np.percentile(
            fused_df["advantage_continuous"].to_numpy(dtype=np.float64),
            (1.0 - args.positive_quantile) * 100.0,
        )
    )
    save_df = build_save_advantages_df(fused_df, threshold)
    adv_path = _write_tag(
        dataset_path=dataset_path,
        output_tag=args.output_tag,
        save_df=save_df,
        positive_quantile=args.positive_quantile,
        threshold=threshold,
    )

    report = {
        "dataset_path": str(dataset_path),
        "predictions_path": str(predictions_path),
        "source_tag": args.source_tag,
        "output_tag": args.output_tag,
        "lookahead_step": int(args.lookahead_step),
        "gamma": float(args.gamma),
        "discount_next_value": bool(args.discount_next_value),
        "positive_quantile": float(args.positive_quantile),
        "threshold": threshold,
        "rows_exported": int(len(save_df)),
        "episodes_exported": int(save_df["episode_index"].nunique()),
        "positive_count": int(save_df["advantage"].sum()),
        "positive_ratio": float(save_df["advantage"].mean()),
        "value_current_mse_vs_return": float(
            np.mean(
                (
                    save_df["value_current"].to_numpy(dtype=np.float64)
                    - save_df["return"].to_numpy(dtype=np.float64)
                )
                ** 2
            )
        ),
        "advantage_path": str(adv_path),
    }

    report_path = (
        Path(args.report_path).resolve()
        if args.report_path
        else dataset_path / "meta" / f"{args.output_tag}_export_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved fused advantages tag to %s", adv_path)
    logger.info("Saved export report to %s", report_path)
    logger.info(
        "rows=%d episodes=%d positive_ratio=%.4f threshold=%.6f",
        report["rows_exported"],
        report["episodes_exported"],
        report["positive_ratio"],
        report["threshold"],
    )


if __name__ == "__main__":
    main()
