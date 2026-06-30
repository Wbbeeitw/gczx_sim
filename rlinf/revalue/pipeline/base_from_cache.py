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

"""Build raw ReCap advantages directly from extracted Revalue feature caches."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rlinf.revalue.data.advantage_table import resolve_advantage_path
from rlinf.revalue.io import save_json
from rlinf.revalue.value_scale import (
    map_returns_to_value_scale,
    validate_atoms_match_value_scale,
)

logger = logging.getLogger(__name__)

_FEATURE_CACHE_REQUIRED_KEYS = {"episode_index", "frame_index", "raw_logits"}


@dataclass(frozen=True)
class BaseFromCacheConfig:
    """Options for rebuilding a base advantage parquet from cached features."""

    dataset_path: str
    features_dir: str
    tag: str
    returns_tag: str | None = None
    lookahead_step: int = 10
    gamma: float = 1.0
    positive_quantile: float = 0.3
    discount_next_value: bool = True
    return_min: float = -700.0
    return_max: float = 0.0
    value_min: float = -1.0
    value_max: float = 0.0
    dataset_type: str = "rollout"
    report_path: str | None = None


def _returns_sidecar_path(dataset_path: str | Path, returns_tag: str | None) -> Path:
    dataset_path = Path(dataset_path)
    filename = f"returns_{returns_tag}.parquet" if returns_tag else "returns.parquet"
    return dataset_path / "meta" / filename


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    denom = np.sum(exp, axis=1, keepdims=True)
    if np.any(denom <= 0.0):
        raise ValueError("Encountered invalid logits with zero softmax denominator.")
    return exp / denom


def _load_single_cache(
    cache_path: Path,
    *,
    split: str,
    row_offset: int,
    value_min: float,
    value_max: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    if not cache_path.exists():
        raise FileNotFoundError(f"Feature cache not found: {cache_path}")

    data = torch.load(cache_path, map_location="cpu", weights_only=False)
    missing = _FEATURE_CACHE_REQUIRED_KEYS - set(data)
    if missing:
        raise ValueError(
            f"Feature cache {cache_path} missing required keys: {sorted(missing)}"
        )
    if data.get("raw_value") is None:
        raise ValueError(
            f"Feature cache {cache_path} is missing raw_value. "
            "Re-run extract_features with a value model that exports raw_value."
        )
    validate_atoms_match_value_scale(
        data.get("atoms"),
        value_min=value_min,
        value_max=value_max,
        source=f"Feature cache {cache_path}",
    )

    episode_index = np.asarray(data["episode_index"], dtype=np.int64).reshape(-1)
    frame_index = np.asarray(data["frame_index"], dtype=np.int64).reshape(-1)
    raw_value = np.asarray(data["raw_value"], dtype=np.float32).reshape(-1)
    raw_logits = np.asarray(data["raw_logits"], dtype=np.float32)
    if raw_logits.ndim != 2:
        raise ValueError(
            f"Feature cache {cache_path} raw_logits must be rank-2, got {raw_logits.shape}"
        )
    row_count = len(episode_index)
    if len(frame_index) != row_count or len(raw_value) != row_count:
        raise ValueError(
            f"Feature cache {cache_path} has inconsistent row counts: "
            f"episode_index={len(episode_index)}, frame_index={len(frame_index)}, "
            f"raw_value={len(raw_value)}"
        )
    if raw_logits.shape[0] != row_count:
        raise ValueError(
            f"Feature cache {cache_path} raw_logits rows={raw_logits.shape[0]}, "
            f"expected {row_count}"
        )

    rows = pd.DataFrame(
        {
            "episode_index": episode_index,
            "frame_index": frame_index,
            "value_current": raw_value.astype(np.float64),
            "cache_row_id": np.arange(row_offset, row_offset + row_count, dtype=np.int64),
            "split": split,
        }
    )
    return rows, raw_logits


def _load_feature_rows(
    features_dir: str | Path,
    *,
    value_min: float,
    value_max: float,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, dict[str, int]]]:
    features_dir = Path(features_dir)
    cache_specs = [
        ("train", features_dir / "train.pt"),
        ("val", features_dir / "val.pt"),
    ]

    frames: list[pd.DataFrame] = []
    logits_parts: list[np.ndarray] = []
    row_offset = 0
    split_stats: dict[str, dict[str, int]] = {}

    for split, cache_path in cache_specs:
        rows, logits = _load_single_cache(
            cache_path,
            split=split,
            row_offset=row_offset,
            value_min=value_min,
            value_max=value_max,
        )
        frames.append(rows)
        logits_parts.append(logits)
        split_stats[split] = {
            "rows": int(len(rows)),
            "episodes": int(rows["episode_index"].nunique()),
        }
        row_offset += len(rows)

    frame_df = pd.concat(frames, ignore_index=True)
    dup = frame_df.duplicated(subset=["episode_index", "frame_index"], keep=False)
    if dup.any():
        preview = frame_df.loc[dup, ["episode_index", "frame_index"]].head(5)
        raise ValueError(
            "Feature caches contain duplicate (episode_index, frame_index) rows. "
            f"First duplicates:\n{preview.to_string(index=False)}"
        )

    frame_df = frame_df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    logits = np.concatenate(logits_parts, axis=0)
    return frame_df, logits, split_stats


def _load_returns(dataset_path: str | Path, returns_tag: str | None) -> tuple[Path, pd.DataFrame]:
    returns_path = _returns_sidecar_path(dataset_path, returns_tag)
    if not returns_path.exists():
        raise FileNotFoundError(
            f"Returns sidecar not found: {returns_path}. "
            "Run build_base or compute_returns first, or enable compute_returns for this stage."
        )

    df = pd.read_parquet(returns_path, columns=["episode_index", "frame_index", "return", "reward"])
    required = {"episode_index", "frame_index", "return", "reward"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Returns sidecar {returns_path} missing required columns: {sorted(missing)}"
        )

    df["episode_index"] = df["episode_index"].astype(np.int64)
    df["frame_index"] = df["frame_index"].astype(np.int64)
    df["return"] = df["return"].astype(np.float64)
    df["reward"] = df["reward"].astype(np.float64)
    return returns_path, df


def _merge_feature_and_returns(
    feature_df: pd.DataFrame,
    returns_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = feature_df.merge(
        returns_df,
        on=["episode_index", "frame_index"],
        how="left",
        validate="one_to_one",
    )
    missing = merged["return"].isna() | merged["reward"].isna()
    if missing.any():
        preview = merged.loc[missing, ["episode_index", "frame_index"]].head(5)
        raise ValueError(
            f"{int(missing.sum())} feature rows are missing return/reward data. "
            f"First missing keys:\n{preview.to_string(index=False)}"
        )
    return merged.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)


def _compute_base_arrays(
    merged: pd.DataFrame,
    *,
    lookahead_step: int,
    gamma: float,
    discount_next_value: bool,
    return_min: float,
    return_max: float,
    value_min: float,
    value_max: float,
) -> dict[str, np.ndarray]:
    if lookahead_step <= 0:
        raise ValueError(f"lookahead_step must be positive, got {lookahead_step}")
    row_count = len(merged)
    value_next = np.zeros(row_count, dtype=np.float64)
    reward_sum_raw = np.zeros(row_count, dtype=np.float64)
    reward_sum = np.zeros(row_count, dtype=np.float64)
    num_valid_rewards = np.zeros(row_count, dtype=np.int64)
    next_row_id = np.full(row_count, -1, dtype=np.int64)

    gamma_powers = np.array([gamma**i for i in range(lookahead_step)], dtype=np.float64)

    for _, indices in merged.groupby("episode_index", sort=False).indices.items():
        idx = np.asarray(indices, dtype=np.int64)
        returns = merged.loc[idx, "return"].to_numpy(dtype=np.float64)
        rewards = merged.loc[idx, "reward"].to_numpy(dtype=np.float64)
        values = merged.loc[idx, "value_current"].to_numpy(dtype=np.float64)
        row_ids = merged.loc[idx, "cache_row_id"].to_numpy(dtype=np.int64)

        ep_len = len(idx)
        for local_pos, global_pos in enumerate(idx):
            num_valid = min(lookahead_step, ep_len - local_pos)
            num_valid_rewards[global_pos] = num_valid

            next_local = local_pos + lookahead_step
            has_bootstrap = next_local < ep_len
            if has_bootstrap:
                value_next[global_pos] = values[next_local]
                next_row_id[global_pos] = row_ids[next_local]
            else:
                value_next[global_pos] = 0.0

            if abs(gamma - 1.0) < 1.0e-8:
                reward_sum_raw[global_pos] = (
                    returns[local_pos] - returns[next_local]
                    if has_bootstrap
                    else returns[local_pos]
                )
            else:
                reward_sum_raw[global_pos] = float(
                    np.sum(gamma_powers[:num_valid] * rewards[local_pos : local_pos + num_valid])
                )

            reward_sum[global_pos] = (
                map_returns_to_value_scale(
                    reward_sum_raw[global_pos],
                    return_min=return_min,
                    return_max=return_max,
                    value_min=value_min,
                    value_max=value_max,
                )
            )

    gamma_k = (
        np.power(float(gamma), np.clip(num_valid_rewards, a_min=0, a_max=None))
        if discount_next_value
        else np.ones(row_count, dtype=np.float64)
    )
    advantage_cont = reward_sum + gamma_k * value_next - merged["value_current"].to_numpy(
        dtype=np.float64
    )
    return {
        "advantage_continuous": advantage_cont,
        "value_next": value_next,
        "reward_sum_raw": reward_sum_raw,
        "reward_sum": reward_sum,
        "num_valid_rewards": num_valid_rewards,
        "next_row_id": next_row_id,
    }


def _build_save_df(
    merged: pd.DataFrame,
    *,
    raw_logits: np.ndarray,
    computed: dict[str, np.ndarray],
    threshold: float,
    dataset_type: str,
) -> pd.DataFrame:
    row_ids = merged["cache_row_id"].to_numpy(dtype=np.int64)
    current_logits = raw_logits[row_ids]
    current_probs = _stable_softmax(current_logits)

    next_row_id = computed["next_row_id"]
    next_logits = np.zeros_like(current_logits)
    next_probs = np.zeros_like(current_probs)
    valid_next = next_row_id >= 0
    if valid_next.any():
        next_logits[valid_next] = raw_logits[next_row_id[valid_next]]
        next_probs[valid_next] = _stable_softmax(next_logits[valid_next])

    save_df = pd.DataFrame(
        {
            "episode_index": merged["episode_index"].to_numpy(dtype=np.int64),
            "frame_index": merged["frame_index"].to_numpy(dtype=np.int64),
            "advantage_continuous": computed["advantage_continuous"].astype(np.float64),
            "return": merged["return"].to_numpy(dtype=np.float64),
            "value_current": merged["value_current"].to_numpy(dtype=np.float64),
            "value_next": computed["value_next"].astype(np.float64),
            "reward_sum": computed["reward_sum"].astype(np.float64),
            "reward_sum_raw": computed["reward_sum_raw"].astype(np.float64),
            "num_valid_rewards": computed["num_valid_rewards"].astype(np.int64),
            "value_logits_current": current_logits.tolist(),
            "value_probs_current": current_probs.tolist(),
            "value_logits_next": next_logits.tolist(),
            "value_probs_next": next_probs.tolist(),
        }
    )
    if dataset_type.lower() == "sft":
        save_df["advantage"] = True
    else:
        save_df["advantage"] = (
            save_df["advantage_continuous"] >= float(threshold)
        ).astype(bool)
    return save_df


def _update_mixture_config(
    dataset_path: str | Path,
    *,
    tag: str,
    threshold: float,
    positive_quantile: float,
    return_min: float,
    return_max: float,
) -> None:
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed; skipping mixture_config.yaml update")
        return

    dataset_path = Path(dataset_path)
    mixture_path = dataset_path / "mixture_config.yaml"
    if mixture_path.exists():
        with open(mixture_path, "r", encoding="utf-8") as file:
            mixture = yaml.safe_load(file) or {}
    else:
        mixture = {}

    mixture["datasets"] = [{"name": dataset_path.name, "weight": 1.0}]
    mixture["global_return_min"] = float(return_min)
    mixture["global_return_max"] = float(return_max)
    mixture["unified_threshold"] = float(threshold)
    mixture["positive_quantile"] = float(positive_quantile)
    mixture.setdefault("tags", {})
    mixture["tags"][tag] = {
        "unified_threshold": float(threshold),
        "positive_quantile": float(positive_quantile),
    }
    mixture["advantage_tag"] = tag

    with open(mixture_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(mixture, file, default_flow_style=False)


def build_revalue_base_from_cache(cfg: BaseFromCacheConfig) -> Path:
    """Write ``meta/advantages_<tag>.parquet`` using cached raw value outputs."""
    if not 0.0 < float(cfg.positive_quantile) <= 1.0:
        raise ValueError(
            f"positive_quantile must be in (0, 1], got {cfg.positive_quantile}"
        )

    feature_df, raw_logits, split_stats = _load_feature_rows(
        cfg.features_dir,
        value_min=float(cfg.value_min),
        value_max=float(cfg.value_max),
    )
    returns_path, returns_df = _load_returns(cfg.dataset_path, cfg.returns_tag)
    merged = _merge_feature_and_returns(feature_df, returns_df)
    computed = _compute_base_arrays(
        merged,
        lookahead_step=cfg.lookahead_step,
        gamma=float(cfg.gamma),
        discount_next_value=cfg.discount_next_value,
        return_min=float(cfg.return_min),
        return_max=float(cfg.return_max),
        value_min=float(cfg.value_min),
        value_max=float(cfg.value_max),
    )

    threshold = float(
        np.percentile(
            computed["advantage_continuous"],
            (1.0 - float(cfg.positive_quantile)) * 100.0,
        )
    )
    save_df = _build_save_df(
        merged,
        raw_logits=raw_logits,
        computed=computed,
        threshold=threshold,
        dataset_type=cfg.dataset_type,
    )

    out_path = resolve_advantage_path(cfg.dataset_path, cfg.tag)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_df.to_parquet(out_path, index=False)
    _update_mixture_config(
        cfg.dataset_path,
        tag=cfg.tag,
        threshold=threshold,
        positive_quantile=float(cfg.positive_quantile),
        return_min=float(cfg.return_min),
        return_max=float(cfg.return_max),
    )

    if cfg.report_path:
        report = {
            "dataset_path": str(Path(cfg.dataset_path).resolve()),
            "features_dir": str(Path(cfg.features_dir).resolve()),
            "returns_path": str(returns_path.resolve()),
            "advantage_path": str(out_path.resolve()),
            "tag": cfg.tag,
            "dataset_type": cfg.dataset_type,
            "lookahead_step": int(cfg.lookahead_step),
            "gamma": float(cfg.gamma),
            "discount_next_value": bool(cfg.discount_next_value),
            "positive_quantile": float(cfg.positive_quantile),
            "unified_threshold": float(threshold),
            "num_rows": int(len(save_df)),
            "num_episodes": int(save_df["episode_index"].nunique()),
            "num_positive": int(save_df["advantage"].sum()),
            "positive_rate": float(save_df["advantage"].mean()),
            "return_min": float(cfg.return_min),
            "return_max": float(cfg.return_max),
            "value_min": float(cfg.value_min),
            "value_max": float(cfg.value_max),
            "logit_bins": int(raw_logits.shape[1]),
            "feature_splits": split_stats,
        }
        save_json(report, cfg.report_path)
        logger.info("saved cache-base report to %s", cfg.report_path)

    logger.info(
        "built Revalue base advantages from cache: %s rows=%d threshold=%.6f",
        out_path,
        len(save_df),
        threshold,
    )
    return out_path
