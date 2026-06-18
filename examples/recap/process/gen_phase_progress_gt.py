import pandas as pd
from pathlib import Path


def main():
    dataset_path = Path("/workspace/datasets/recap_libero10_task0/libero10_task0_train")
    adv_path = dataset_path / "meta" / "advantages_base_train_5k_N10_q30_full.parquet"
    output_path = dataset_path / "meta" / "phase_progress_gt.parquet"

    df = pd.read_parquet(adv_path)

    ep_lengths = df.groupby("episode_index")["frame_index"].max() + 1
    df["episode_length"] = df["episode_index"].map(ep_lengths)
    df["progress"] = df["frame_index"] / (df["episode_length"] - 1).clip(lower=1)
    num_phases = 5
    df["phase"] = (df["progress"] * num_phases).clip(upper=num_phases - 1).astype(int)

    out_df = df[["episode_index", "frame_index", "phase", "progress"]]
    out_df.to_parquet(output_path, index=False)

    print(f"Saved {len(out_df)} entries to {output_path}")
    print("Phase distribution:")
    print(out_df["phase"].value_counts().sort_index())
    print(f"Progress range: {out_df['progress'].min():.4f} - {out_df['progress'].max():.4f}")


if __name__ == "__main__":
    main()
