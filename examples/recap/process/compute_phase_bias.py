import json
import pandas as pd
from pathlib import Path


def main():
    dataset_path = Path("/workspace/datasets/recap_libero10_task0/libero10_task0_train")
    adv_path = dataset_path / "meta" / "advantages_base_train_5k_N10_q30_full.parquet"
    zp_path = dataset_path / "meta" / "phase_progress_gt.parquet"

    adv_df = pd.read_parquet(adv_path)
    zp_df = pd.read_parquet(zp_path)

    df = adv_df.merge(zp_df, on=["episode_index", "frame_index"], how="left")

    print("Phase bias table (bias = -mean_advantage):")
    bias_table = {}
    for phase in sorted(df["phase"].dropna().unique().astype(int)):
        phase_df = df[df["phase"] == phase]
        mean_adv = phase_df["advantage_continuous"].mean()
        bias = -mean_adv
        bias_table[int(phase)] = float(bias)
        print(f"  phase={phase}: count={len(phase_df)}, mean_advantage={mean_adv:.4f}, bias={bias:.4f}")

    bias_path = dataset_path / "meta" / "phase_bias_table.json"
    with open(bias_path, "w") as f:
        json.dump(bias_table, f, indent=2)
    print(f"\nSaved bias table to {bias_path}")

    bias_values = list(bias_table.values())
    bias_range = max(bias_values) - min(bias_values)
    print(f"Bias range: {bias_range:.4f}")


if __name__ == "__main__":
    main()
