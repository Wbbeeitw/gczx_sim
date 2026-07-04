"""Helpers for opt-in CFG training strategies driven by advantage tables."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
import pandas as pd

QUALITY_LABEL_POSITIVE = 0
QUALITY_LABEL_NEUTRAL = 1
QUALITY_LABEL_NEGATIVE = 2

CFG_STRATEGY_BINARY = "binary"
CFG_STRATEGY_CSA_SOFT = "csa_soft"
SUPPORTED_CFG_STRATEGIES = (CFG_STRATEGY_BINARY, CFG_STRATEGY_CSA_SOFT)


@dataclass(frozen=True)
class CFGStrategyConfig:
    """Config for deriving CFG sample metadata from exported advantages."""

    strategy: str = CFG_STRATEGY_BINARY
    positive_quantile: float = 0.30
    bottom_quantile: float = 0.15
    bottom_negative_prob: float = 0.50
    weight_lambda: float = 0.20
    seed: int = 42

    def __post_init__(self) -> None:
        if self.strategy not in SUPPORTED_CFG_STRATEGIES:
            raise ValueError(
                f"Unsupported CFG strategy {self.strategy!r}; "
                f"expected one of {SUPPORTED_CFG_STRATEGIES}."
            )
        if not 0.0 < self.positive_quantile < 1.0:
            raise ValueError("positive_quantile must be in (0, 1).")
        if not 0.0 < self.bottom_quantile < 1.0:
            raise ValueError("bottom_quantile must be in (0, 1).")
        if self.positive_quantile + self.bottom_quantile >= 1.0:
            raise ValueError(
                "positive_quantile + bottom_quantile must stay below 1.0."
            )
        if not 0.0 <= self.bottom_negative_prob <= 1.0:
            raise ValueError("bottom_negative_prob must be in [0, 1].")


def _stable_uniform_01(*, dataset_id: str, episode_index: int, frame_index: int, seed: int) -> float:
    payload = f"{dataset_id}|{episode_index}|{frame_index}|{seed}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    int_value = int.from_bytes(digest, byteorder="big", signed=False)
    return int_value / float(2**64 - 1)


def _normalized_percentile_rank(values: pd.Series) -> np.ndarray:
    count = int(values.shape[0])
    if count == 0:
        return np.empty((0,), dtype=np.float32)
    if count == 1:
        return np.array([0.5], dtype=np.float32)
    ranks = values.rank(method="average", ascending=True).to_numpy(dtype=np.float64)
    return ((ranks - 1.0) / float(count - 1)).astype(np.float32)


def build_cfg_sample_metadata(
    advantages_df: pd.DataFrame,
    *,
    strategy_cfg: CFGStrategyConfig,
    dataset_id: str,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Build per-sample metadata consumed by downstream CFG training.

    The returned mapping always includes the legacy ``advantage`` boolean to keep the
    existing training path intact. CSA-CFG adds:
    - ``cfg_quality_label``: 0=positive, 1=neutral, 2=negative
    - ``cfg_percentile_rank``: rank of continuous advantage in [0, 1]
    - ``cfg_loss_weight``: smooth weight derived from percentile rank
    """

    required_keys = {"episode_index", "frame_index", "advantage"}
    missing = required_keys - set(advantages_df.columns)
    if missing:
        raise ValueError(
            f"Advantage table missing required columns for CFG metadata: {sorted(missing)}"
        )

    if strategy_cfg.strategy == CFG_STRATEGY_BINARY:
        return {
            (int(row["episode_index"]), int(row["frame_index"])): {
                "advantage": bool(row["advantage"]),
            }
            for row in advantages_df.to_dict("records")
        }

    if "advantage_continuous" not in advantages_df.columns:
        raise ValueError(
            "CSA-CFG requires 'advantage_continuous' in the advantage parquet. "
            "Export fused advantages before launching CSA-CFG."
        )

    values = advantages_df["advantage_continuous"].astype(float)
    positive_threshold = float(
        np.percentile(values.to_numpy(dtype=np.float64), (1.0 - strategy_cfg.positive_quantile) * 100.0)
    )
    bottom_threshold = float(
        np.percentile(values.to_numpy(dtype=np.float64), strategy_cfg.bottom_quantile * 100.0)
    )
    percentile_rank = _normalized_percentile_rank(values)

    metadata: dict[tuple[int, int], dict[str, Any]] = {}
    for idx, row in enumerate(advantages_df.to_dict("records")):
        episode_index = int(row["episode_index"])
        frame_index = int(row["frame_index"])
        value = float(row["advantage_continuous"])
        if value >= positive_threshold:
            quality_label = QUALITY_LABEL_POSITIVE
        elif value <= bottom_threshold:
            sample_uniform = _stable_uniform_01(
                dataset_id=dataset_id,
                episode_index=episode_index,
                frame_index=frame_index,
                seed=strategy_cfg.seed,
            )
            if sample_uniform < strategy_cfg.bottom_negative_prob:
                quality_label = QUALITY_LABEL_NEGATIVE
            else:
                quality_label = QUALITY_LABEL_NEUTRAL
        else:
            quality_label = QUALITY_LABEL_NEUTRAL

        rank = float(percentile_rank[idx])
        metadata[(episode_index, frame_index)] = {
            "advantage": quality_label == QUALITY_LABEL_POSITIVE,
            "cfg_quality_label": int(quality_label),
            "cfg_percentile_rank": rank,
            "cfg_loss_weight": float(1.0 + strategy_cfg.weight_lambda * (2.0 * rank - 1.0)),
        }
    return metadata
