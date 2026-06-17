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
Apply a coarse phase/progress correction to pre-computed advantages.

This is a minimal feasibility check: it uses z_t (phase) and p_t (progress)
to compute a per-phase mean value bias, subtracts it, and rewrites the
advantage parquet. No model training, no Recap code changes.

Usage:
    cd examples/recap/process
    python apply_phase_correction.py \
        --input /path/to/advantages_raw.parquet \
        --phase /path/to/phase_progress_gt.parquet \
        --output /path/to/advantages_corrected.parquet
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Raw advantages parquet")
    parser.add_argument("--phase", required=True, help="Phase/progress parquet")
    parser.add_argument("--output", required=True, help="Output corrected parquet")
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor")
    parser.add_argument(
        "--positive-quantile",
        type=float,
        default=0.3,
        help="Top X%% of corrected advantages become advantage=True",
    )
    parser.add_argument(
        "--return-min",
        type=float,
        default=-700.0,
        help="Min return used to normalize return to the same scale as value output",
    )
    parser.add_argument(
        "--return-max",
        type=float,
        default=0.0,
        help="Max return used to normalize return to the same scale as value output",
    )
    args = parser.parse_args()

    print(f"Loading raw advantages: {args.input}")
    adv_df = pd.read_parquet(args.input)
    print(f"  rows={len(adv_df)}, episodes={adv_df['episode_index'].nunique()}")

    print(f"Loading phase/progress: {args.phase}")
    zp_df = pd.read_parquet(args.phase)
    print(f"  rows={len(zp_df)}")

    # Merge phase/progress into advantages
    df = adv_df.merge(zp_df, on=["episode_index", "frame_index"], how="left")

    # Normalize returns to the same [-1, 0] scale used by ValueCriticModel
    # (matches compute_advantages.py normalization)
    ret_range = args.return_max - args.return_min

    def normalize_return(x):
        if ret_range <= 0:
            return -0.5
        return (x - args.return_min) / ret_range - 1.0

    df["return_norm"] = df["return"].apply(normalize_return)

    # Episode lengths, needed for next-frame clamping
    ep_lengths = df.groupby("episode_index")["frame_index"].max() + 1
    df["episode_length"] = df["episode_index"].map(ep_lengths)

    # Next frame index used by value_next (use num_valid_rewards if available, else 1)
    step_key = "num_valid_rewards" if "num_valid_rewards" in df.columns else None
    if step_key:
        df["next_frame_index"] = (
            df["frame_index"] + df["num_valid_rewards"]
        ).clip(upper=df["episode_length"] - 1)
    else:
        df["next_frame_index"] = (
            df["frame_index"] + 1
        ).clip(upper=df["episode_length"] - 1)

    # Look up phase of the next frame
    phase_next = zp_df.rename(
        columns={"phase": "phase_next", "progress": "progress_next"}
    )[["episode_index", "frame_index", "phase_next", "progress_next"]]
    df = df.merge(
        phase_next,
        left_on=["episode_index", "next_frame_index"],
        right_on=["episode_index", "frame_index"],
        how="left",
        suffixes=("", "_next_lookup"),
    )
    # Drop duplicated frame_index from the merge
    df.drop(columns=["frame_index_next_lookup"], inplace=True, errors="ignore")

    # Fallback: if next phase is missing, use current phase
    df["phase_next"] = df["phase_next"].fillna(df["phase"]).astype(int)

    # Compute per-phase value bias: E[value_current - normalized_return]
    df["value_error"] = df["value_current"] - df["return_norm"]
    bias = df.groupby("phase")["value_error"].mean().to_dict()
    print("\nPhase bias (value_current - return):")
    for ph in sorted(bias):
        print(f"  phase={ph}: bias={bias[ph]:.4f}")

    # Apply correction
    df["value_current_corr"] = df["value_current"] - df["phase"].map(bias)
    df["value_next_corr"] = df["value_next"] - df["phase_next"].map(bias)

    # Recompute advantage: A = reward_sum + gamma^N * V_next - V_curr
    # For variable lookahead, use gamma raised to the actual step count.
    if step_key:
        n_steps = df["num_valid_rewards"].astype(float).values
    else:
        n_steps = np.ones(len(df))
    gamma_factor = args.gamma ** n_steps
    df["advantage_continuous_raw"] = df["advantage_continuous"]
    df["advantage_continuous"] = (
        df["reward_sum"] + gamma_factor * df["value_next_corr"] - df["value_current_corr"]
    )

    # Recompute boolean advantage threshold from corrected distribution
    threshold = np.percentile(
        df["advantage_continuous"].values, 100.0 * (1.0 - args.positive_quantile)
    )
    df["advantage"] = df["advantage_continuous"] >= threshold
    print(f"\nCorrected advantage threshold (top {args.positive_quantile*100:.0f}%): {threshold:.4f}")
    print(f"  positive ratio: {df['advantage'].mean():.3f}")

    # Metrics (in normalized return space)
    mse_raw = (df["value_current"] - df["return_norm"]).pow(2).mean()
    mse_corr = (df["value_current_corr"] - df["return_norm"]).pow(2).mean()
    print("\n=== Value MSE ===")
    print(f"  raw:      {mse_raw:.6f}")
    print(f"  corrected:{mse_corr:.6f}")
    print(f"  relative improvement: {(1 - mse_corr / mse_raw) * 100:.2f}%")

    # Per-phase summary
    print("\n=== Per-phase value error ===")
    summary = []
    for ph in sorted(df["phase"].unique().astype(int)):
        sub = df[df["phase"] == ph]
        summary.append(
            {
                "phase": int(ph),
                "count": len(sub),
                "mse_raw": float(((sub["value_current"] - sub["return_norm"]) ** 2).mean()),
                "mse_corr": float(
                    ((sub["value_current_corr"] - sub["return_norm"]) ** 2).mean()
                ),
            }
        )
        print(
            f"  phase={ph}: n={summary[-1]['count']}, "
            f"mse_raw={summary[-1]['mse_raw']:.6f}, "
            f"mse_corr={summary[-1]['mse_corr']:.6f}, "
            f"improvement={(1 - summary[-1]['mse_corr']/summary[-1]['mse_raw'])*100:.1f}%"
        )

    # Save corrected parquet
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the standard columns expected by downstream code
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
        "advantage",
        # Extra diagnostic columns
        "value_current_corr",
        "value_next_corr",
        "phase",
        "progress",
        "phase_next",
        "value_error",
    ]
    save_cols = [c for c in save_cols if c in df.columns]
    df[save_cols].to_parquet(out_path, index=False)
    print(f"\nSaved corrected advantages to {out_path}")

    # Save bias table for reference
    bias_path = out_path.parent / f"{out_path.stem}_bias.json"
    with open(bias_path, "w") as f:
        json.dump({int(k): float(v) for k, v in bias.items()}, f, indent=2)
    print(f"Saved bias table to {bias_path}")


if __name__ == "__main__":
    main()
