"""Write a comparison descriptor for raw Value predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def write_raw_value_comparison(
    *,
    advantages_path: Path,
    output_path: Path,
    return_min: float,
    return_max: float,
    value_min: float = -1.0,
    value_max: float = 0.0,
) -> dict:
    """Validate raw advantages and write their normalization metadata."""
    advantages = pd.read_parquet(advantages_path)
    required = {"episode_index", "frame_index", "return", "value_current"}
    missing = required - set(advantages.columns)
    if missing:
        raise ValueError(f"Raw advantages missing columns: {sorted(missing)}")
    if advantages.empty:
        raise ValueError("Raw advantages are empty")
    if return_max <= return_min:
        raise ValueError("return_max must be greater than return_min")
    if value_max <= value_min:
        raise ValueError("value_max must be greater than value_min")

    result = {
        "schema_version": "1.0",
        "mode": "raw",
        "advantages_path": str(advantages_path.resolve()),
        "predictions_path": None,
        "return_min": float(return_min),
        "return_max": float(return_max),
        "value_min": float(value_min),
        "value_max": float(value_max),
        "rows": int(len(advantages)),
        "episodes": int(advantages["episode_index"].nunique()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--advantages-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--return-min", type=float, required=True)
    parser.add_argument("--return-max", type=float, required=True)
    parser.add_argument("--value-min", type=float, default=-1.0)
    parser.add_argument("--value-max", type=float, default=0.0)
    args = parser.parse_args()
    result = write_raw_value_comparison(
        advantages_path=args.advantages_path,
        output_path=args.output_path,
        return_min=args.return_min,
        return_max=args.return_max,
        value_min=args.value_min,
        value_max=args.value_max,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
