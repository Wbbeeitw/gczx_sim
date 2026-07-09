"""Convert boundary tables into phase/progress parquet labels.

Supports:
    * CSV emitted by ``task1_boundary_vlm.py``
    * JSONL cache emitted by ``task1_boundary_vlm.py``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from task1_boundary_vlm import build_phase_df_from_boundaries
except ModuleNotFoundError:
    from phase_split_script.task1_boundary_vlm import build_phase_df_from_boundaries


def _load_boundary_table(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        rows: list[dict[str, Any]] = []
        for row in df.to_dict(orient="records"):
            rows.append({
                "episode_index": int(row["episode_index"]),
                "episode_length": int(row["episode_length"]),
                "is_success": row.get("is_success"),
                "final_boundaries": {
                    "b1": None if pd.isna(row.get("b1_final")) else int(row["b1_final"]),
                    "b2": None if pd.isna(row.get("b2_final")) else int(row["b2_final"]),
                    "b3": None if pd.isna(row.get("b3_final")) else int(row["b3_final"]),
                },
            })
        return rows

    if path.suffix.lower() == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    raise ValueError(f"Unsupported boundary table format: {path}")


def convert_boundaries_to_parquet(
    dataset_path: str | Path,
    boundaries_path: str | Path,
    output_name: str,
) -> Path:
    dataset_path = Path(dataset_path)
    boundaries_path = Path(boundaries_path)
    entries = _load_boundary_table(boundaries_path)
    if not entries:
        raise RuntimeError(f"No boundary entries found in {boundaries_path}")

    records = []
    for entry in entries:
        records.append(
            build_phase_df_from_boundaries(
                episode_index=int(entry["episode_index"]),
                episode_length=int(entry["episode_length"]),
                boundaries=entry["final_boundaries"],
                is_success=entry.get("is_success"),
            )
        )

    out_df = pd.concat(records, ignore_index=True)
    out_df = out_df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    meta_dir = dataset_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_path = meta_dir / f"{output_name}.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"Saved {len(out_df)} frame labels to {out_path}")
    print("Phase distribution:")
    print(out_df["phase"].value_counts().sort_index())
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert task1 boundary tables to phase/progress parquet labels."
    )
    parser.add_argument("--dataset_path", required=True, help="Path to dataset root.")
    parser.add_argument(
        "--boundaries_path",
        required=True,
        help="CSV or JSONL boundary table emitted by task1_boundary_vlm.py.",
    )
    parser.add_argument(
        "--output_name",
        default="phase_progress_semantic_boundary_vlm",
        help="Output parquet name under dataset/meta/.",
    )
    args = parser.parse_args()

    out_path = convert_boundaries_to_parquet(
        dataset_path=args.dataset_path,
        boundaries_path=args.boundaries_path,
        output_name=args.output_name,
    )
    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
