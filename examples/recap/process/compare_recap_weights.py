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

"""
Compare Recap/CFG positive-label (weight) distributions between raw and
phase-corrected advantages.

Usage:
    python compare_recap_weights.py \
        --raw /path/to/advantages_raw.parquet \
        --corrected /path/to/advantages_corrected.parquet \
        --output /path/to/comparison.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def describe(series: pd.Series) -> dict:
    return {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "median": float(series.median()),
        "max": float(series.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="Raw advantages parquet")
    parser.add_argument("--corrected", required=True, help="Corrected advantages parquet")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    print(f"Loading raw: {args.raw}")
    raw = pd.read_parquet(args.raw)
    print(f"Loading corrected: {args.corrected}")
    corr = pd.read_parquet(args.corrected)

    # Align
    raw_cols = ["episode_index", "frame_index", "advantage_continuous", "advantage", "return"]
    corr_cols = ["episode_index", "frame_index", "advantage_continuous", "advantage", "phase"]
    if "progress" in corr.columns:
        corr_cols.append("progress")
    merged = raw[raw_cols].merge(
        corr[corr_cols],
        on=["episode_index", "frame_index"],
        suffixes=("_raw", "_corr"),
    )

    results = {}

    # 1. Overall positive ratio
    pos_raw = merged["advantage_raw"].mean()
    pos_corr = merged["advantage_corr"].mean()
    results["overall_positive_ratio"] = {"raw": float(pos_raw), "corrected": float(pos_corr)}
    print("\n=== Overall positive ratio ===")
    print(f"  raw:      {pos_raw:.3f}")
    print(f"  corrected:{pos_corr:.3f}")

    # 2. Advantage continuous distribution
    results["advantage_continuous"] = {
        "raw": describe(merged["advantage_continuous_raw"]),
        "corrected": describe(merged["advantage_continuous_corr"]),
    }
    print("\n=== Advantage continuous distribution ===")
    for tag in ("raw", "corrected"):
        key = f"advantage_continuous_{'raw' if tag == 'raw' else 'corr'}"
        s = merged[key]
        print(
            f"  {tag:10s}: mean={s.mean():.4f}, std={s.std():.4f}, "
            f"min={s.min():.4f}, median={s.median():.4f}, max={s.max():.4f}"
        )

    # 3. Per-phase positive ratio and mean advantage
    print("\n=== Per-phase statistics ===")
    phase_stats = []
    for ph in sorted(merged["phase"].unique().astype(int)):
        sub = merged[merged["phase"] == ph]
        stats = {
            "phase": int(ph),
            "progress_mid": float((sub["progress"].min() + sub["progress"].max()) / 2) if "progress" in sub.columns else float(ph),
            "count": len(sub),
            "positive_ratio_raw": float(sub["advantage_raw"].mean()),
            "positive_ratio_corr": float(sub["advantage_corr"].mean()),
            "mean_advantage_raw": float(sub["advantage_continuous_raw"].mean()),
            "mean_advantage_corr": float(sub["advantage_continuous_corr"].mean()),
            "mean_return": float(sub["return"].mean()),
        }
        phase_stats.append(stats)
        print(
            f"  phase={ph} (progress ~{stats['progress_mid']:.2f}): "
            f"pos_raw={stats['positive_ratio_raw']:.3f}, "
            f"pos_corr={stats['positive_ratio_corr']:.3f}, "
            f"adv_raw={stats['mean_advantage_raw']:.3f}, "
            f"adv_corr={stats['mean_advantage_corr']:.3f}, "
            f"return={stats['mean_return']:.1f}"
        )
    results["per_phase"] = phase_stats

    # 4. Label flip matrix
    raw_labels = merged["advantage_raw"].astype(bool)
    corr_labels = merged["advantage_corr"].astype(bool)
    flip = pd.crosstab(
        raw_labels,
        corr_labels,
        rownames=["raw"],
        colnames=["corrected"],
    )
    print("\n=== Label flip matrix (raw vs corrected) ===")
    print(flip)
    results["flip_counts"] = {
        "false_to_false": int(flip.loc[False, False]) if False in flip.index and False in flip.columns else 0,
        "false_to_true": int(flip.loc[False, True]) if False in flip.index and True in flip.columns else 0,
        "true_to_false": int(flip.loc[True, False]) if True in flip.index and False in flip.columns else 0,
        "true_to_true": int(flip.loc[True, True]) if True in flip.index and True in flip.columns else 0,
    }
    total = len(merged)
    print("\n=== Label flip percentages ===")
    for k, v in results["flip_counts"].items():
        print(f"  {k}: {v} ({v/total*100:.2f}%)")

    # 5. Per-episode positive rate statistics
    ep_raw = merged.groupby("episode_index")["advantage_raw"].mean()
    ep_corr = merged.groupby("episode_index")["advantage_corr"].mean()
    results["per_episode_positive_rate"] = {
        "raw": describe(ep_raw),
        "corrected": describe(ep_corr),
    }
    print("\n=== Per-episode positive rate ===")
    print(f"  raw:       mean={ep_raw.mean():.3f}, std={ep_raw.std():.3f}")
    print(f"  corrected: mean={ep_corr.mean():.3f}, std={ep_corr.std():.3f}")

    # 6. Correlation with return
    corr_raw = merged["advantage_continuous_raw"].corr(merged["return"])
    corr_corr = merged["advantage_continuous_corr"].corr(merged["return"])
    results["advantage_return_correlation"] = {
        "raw": float(corr_raw),
        "corrected": float(corr_corr),
    }
    print("\n=== Advantage vs return correlation ===")
    print(f"  raw:      {corr_raw:.4f}")
    print(f"  corrected:{corr_corr:.4f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved comparison to {out_path}")


if __name__ == "__main__":
    main()
