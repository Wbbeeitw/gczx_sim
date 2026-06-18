import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr


def main():
    dataset_path = Path("/workspace/datasets/sequential_pickcube_lerobot")
    adv_path = dataset_path / "meta" / "advantages_maniskill_eval_5k.parquet"
    output_dir = dataset_path / "meta"

    adv_df = pd.read_parquet(adv_path)

    # Use eval episodes only (episode 20-99)
    eval_df = adv_df[adv_df["episode_index"] >= 20].copy()

    # Compute progress and phase per episode
    ep_lengths = eval_df.groupby("episode_index")["frame_index"].max() + 1
    eval_df["episode_length"] = eval_df["episode_index"].map(ep_lengths)
    eval_df["progress"] = eval_df["frame_index"] / (eval_df["episode_length"] - 1).clip(lower=1)
    num_phases = 5
    eval_df["phase"] = (eval_df["progress"] * num_phases).clip(upper=num_phases - 1).astype(int)

    # Phase bias table
    bias_table = {}
    for phase in sorted(eval_df["phase"].unique().astype(int)):
        phase_df = eval_df[eval_df["phase"] == phase]
        mean_adv = phase_df["advantage_continuous"].mean()
        std_adv = phase_df["advantage_continuous"].std()
        bias = -float(mean_adv)
        bias_table[int(phase)] = {
            "count": int(len(phase_df)),
            "mean_advantage": float(mean_adv),
            "std_advantage": float(std_adv),
            "bias": bias,
        }
        print(f"phase={phase}: count={len(phase_df)}, mean_adv={mean_adv:.6f}, std={std_adv:.6f}, bias={bias:.6f}")

    bias_path = output_dir / "phase_bias_table_maniskill_eval_5k.json"
    with open(bias_path, "w") as f:
        json.dump(bias_table, f, indent=2)
    print(f"\nSaved phase bias table to {bias_path}")

    bias_values = [v["bias"] for v in bias_table.values()]
    print(f"Bias range (max - min): {max(bias_values) - min(bias_values):.6f}")

    # Progress correlation / decile bias
    corr, pval = spearmanr(eval_df["progress"], eval_df["advantage_continuous"])
    print(f"\nSpearman correlation(progress, advantage): {corr:.6f} (p={pval:.2e})")

    deciles = pd.qcut(eval_df["progress"], 10, labels=False, duplicates="drop")
    eval_df["progress_decile"] = deciles
    decile_table = {}
    for d in sorted(eval_df["progress_decile"].unique().astype(int)):
        ddf = eval_df[eval_df["progress_decile"] == d]
        decile_table[int(d)] = {
            "progress_range": [float(ddf["progress"].min()), float(ddf["progress"].max())],
            "mean_advantage": float(ddf["advantage_continuous"].mean()),
            "std_advantage": float(ddf["advantage_continuous"].std()),
            "count": int(len(ddf)),
        }
    print("\nProgress decile mean advantages:")
    for d, info in decile_table.items():
        print(f"  decile={d}: progress={info['progress_range'][0]:.3f}-{info['progress_range'][1]:.3f}, "
              f"mean_adv={info['mean_advantage']:.6f}, std={info['std_advantage']:.6f}")

    progress_path = output_dir / "progress_bias_table_maniskill_eval_5k.json"
    with open(progress_path, "w") as f:
        json.dump({"spearman_corr": float(corr), "p_value": float(pval), "deciles": decile_table}, f, indent=2)
    print(f"\nSaved progress bias table to {progress_path}")


if __name__ == "__main__":
    main()
