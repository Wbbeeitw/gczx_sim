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
Apply phase+progress correction to pre-computed advantages.

For each semantic phase, fit a linear bias model:
    bias(z, p) = a_z * p + b_z
where p is the progress within phase z.  This uses both z_t and p_t.

Usage:
    python apply_phase_progress_correction.py \
        --input /path/to/advantages_raw.parquet \
        --phase /path/to/phase_progress_semantic.parquet \
        --output /path/to/advantages_zp_corrected.parquet
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--positive-quantile", type=float, default=0.3)
    parser.add_argument("--return-min", type=float, default=-700.0)
    parser.add_argument("--return-max", type=float, default=0.0)
    args = parser.parse_args()

    print(f"Loading raw advantages: {args.input}")
    adv_df = pd.read_parquet(args.input)
    print(f"  rows={len(adv_df)}, episodes={adv_df['episode_index'].nunique()}")

    print(f"Loading phase/progress: {args.phase}")
    zp_df = pd.read_parquet(args.phase)
    print(f"  rows={len(zp_df)}")

    # Merge
    df = adv_df.merge(zp_df, on=["episode_index", "frame_index"], how="left")

    # Normalize return
    ret_range = args.return_max - args.return_min

    def normalize_return(x):
        return -0.5 if ret_range <= 0 else (x - args.return_min) / ret_range - 1.0

    df["return_norm"] = df["return"].apply(normalize_return)

    # Episode lengths for next-frame lookup
    ep_lengths = df.groupby("episode_index")["frame_index"].max() + 1
    df["episode_length"] = df["episode_index"].map(ep_lengths)

    step_key = "num_valid_rewards" if "num_valid_rewards" in df.columns else None
    if step_key:
        df["next_frame_index"] = (
            df["frame_index"] + df["num_valid_rewards"]
        ).clip(upper=df["episode_length"] - 1)
    else:
        df["next_frame_index"] = (df["frame_index"] + 1).clip(
            upper=df["episode_length"] - 1
        )

    # Look up next-frame phase & phase_progress
    zp_next = zp_df.rename(
        columns={
            "phase": "phase_next",
            "phase_progress": "phase_progress_next",
            "progress": "progress_next",
        }
    )[["episode_index", "frame_index", "phase_next", "phase_progress_next"]]
    df = df.merge(
        zp_next,
        left_on=["episode_index", "next_frame_index"],
        right_on=["episode_index", "frame_index"],
        how="left",
        suffixes=("", "_next_lookup"),
    )
    df.drop(columns=["frame_index_next_lookup"], inplace=True, errors="ignore")
    df["phase_next"] = df["phase_next"].fillna(df["phase"]).astype(int)
    df["phase_progress_next"] = df["phase_progress_next"].fillna(df["phase_progress"])

    # Compute value error in normalized space
    df["value_error"] = df["value_current"] - df["return_norm"]

    # Fit per-phase linear bias: error = a * phase_progress + b
    coeffs = {}
    print("\nPer-phase linear bias fit (bias = a * phase_progress + b):")
    for ph in sorted(df["phase"].dropna().unique().astype(int)):
        sub = df[df["phase"] == ph]
        x = sub["phase_progress"].values.astype(np.float64)
        y = sub["value_error"].values.astype(np.float64)
        if len(x) >= 2 and np.std(x) > 1e-6:
            a, b = np.polyfit(x, y, 1)
        else:
            a, b = 0.0, float(y.mean()) if len(y) > 0 else 0.0
        coeffs[int(ph)] = (float(a), float(b))
        print(f"  phase={ph}: a={a:.4f}, b={b:.4f}, n={len(x)}")

    def predict_bias(row):
        a, b = coeffs[int(row["phase"])]
        return a * row["phase_progress"] + b

    def predict_bias_next(row):
        a, b = coeffs[int(row["phase_next"])]
        return a * row["phase_progress_next"] + b

    df["bias_current"] = df.apply(predict_bias, axis=1)
    df["bias_next"] = df.apply(predict_bias_next, axis=1)

    df["value_current_corr"] = df["value_current"] - df["bias_current"]
    df["value_next_corr"] = df["value_next"] - df["bias_next"]

    # Recompute advantage
    n_steps = df["num_valid_rewards"].astype(float).values if step_key else np.ones(len(df))
    gamma_factor = args.gamma ** n_steps
    df["advantage_continuous_raw"] = df["advantage_continuous"]
    df["advantage_continuous"] = (
        df["reward_sum"] + gamma_factor * df["value_next_corr"] - df["value_current_corr"]
    )

    threshold = np.percentile(
        df["advantage_continuous"].values, 100.0 * (1.0 - args.positive_quantile)
    )
    df["advantage"] = df["advantage_continuous"] >= threshold
    print(f"\nCorrected advantage threshold (top {args.positive_quantile*100:.0f}%): {threshold:.4f}")
    print(f"  positive ratio: {df['advantage'].mean():.3f}")

    # Metrics
    mse_raw = (df["value_current"] - df["return_norm"]).pow(2).mean()
    mse_corr = (df["value_current_corr"] - df["return_norm"]).pow(2).mean()
    print("\n=== Value MSE ===")
    print(f"  raw:      {mse_raw:.6f}")
    print(f"  corrected:{mse_corr:.6f}")
    print(f"  relative improvement: {(1 - mse_corr / mse_raw) * 100:.2f}%")

    print("\n=== Per-phase value error ===")
    for ph in sorted(df["phase"].unique().astype(int)):
        sub = df[df["phase"] == ph]
        mse_raw_ph = ((sub["value_current"] - sub["return_norm"]) ** 2).mean()
        mse_corr_ph = ((sub["value_current_corr"] - sub["return_norm"]) ** 2).mean()
        print(
            f"  phase={ph}: n={len(sub)}, mse_raw={mse_raw_ph:.6f}, "
            f"mse_corr={mse_corr_ph:.6f}, improvement={(1-mse_corr_ph/mse_raw_ph)*100:.1f}%"
        )

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
        "value_current_corr",
        "value_next_corr",
        "phase",
        "phase_progress",
        "global_progress",
        "value_error",
    ]
    save_cols = [c for c in save_cols if c in df.columns]
    df[save_cols].to_parquet(out_path, index=False)
    print(f"\nSaved corrected advantages to {out_path}")

    bias_path = out_path.parent / f"{out_path.stem}_bias.json"
    with open(bias_path, "w") as f:
        json.dump({int(k): {"a": v[0], "b": v[1]} for k, v in coeffs.items()}, f, indent=2)
    print(f"Saved per-phase linear bias coeffs to {bias_path}")


if __name__ == "__main__":
    main()
