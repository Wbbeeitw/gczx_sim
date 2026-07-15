"""Create a paper-ready result record for one ReCap experiment round.

The script combines rollout summary, return comparison, episode outcomes,
z/p metrics, and fusion metrics into one JSON record and one flat CSV row.
It treats the ``all`` split of an independent evaluation dataset as the
primary Critic result and keeps policy metrics separate from Critic metrics.

Example:

    python examples/recap/process/record_round_results.py \
        --task task1 \
        --round 0 \
        --policy-label SFT \
        --critic-label Value0-fused \
        --train-dataset /data/libero_long/task1_d0_40 \
        --eval-dataset /data/libero_long/eval/task1_r0_eval10 \
        --comparison /data/libero_long/feature/D1_eval10/return_compare_paper_nocw.json \
        --zp-metrics /data/libero_long/feature/D1_feature/zp_head_paper_nocw/metrics.json \
        --fusion-metrics /data/libero_long/feature/D1_feature/fusion_paper_nocw/metrics.json \
        --output-dir /data/libero_long/experiment_results/task1 \
        --output-name round0_sft_value0
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _dataset_episode_stats(dataset_path: Path) -> dict[int, dict[str, Any]]:
    episode_path = dataset_path / "meta" / "episodes.jsonl"
    stats_path = dataset_path / "meta" / "episodes_stats.jsonl"
    episodes = {}
    if episode_path.exists():
        for row in _read_jsonl(episode_path):
            episode = int(row["episode_index"])
            episodes[episode] = {"length": int(row["length"])}
    if stats_path.exists():
        for row in _read_jsonl(stats_path):
            episode = int(row["episode_index"])
            success_stats = row.get("stats", {}).get("is_success", {})
            if "mean" in success_stats:
                episodes.setdefault(episode, {})["is_success"] = bool(
                    float(success_stats["mean"]) >= 0.5
                )
    return episodes


def _dataset_summary(dataset_path: Path) -> dict[str, Any]:
    episodes = _dataset_episode_stats(dataset_path)
    lengths = [int(row["length"]) for row in episodes.values() if "length" in row]
    outcomes = [row["is_success"] for row in episodes.values() if "is_success" in row]
    return {
        "episodes": len(episodes),
        "frames": int(sum(lengths)),
        "successes": int(sum(outcomes)) if outcomes else None,
        "failures": int(len(outcomes) - sum(outcomes)) if outcomes else None,
        "mean_episode_length": float(np.mean(lengths)) if lengths else None,
        "success_only_act": (
            float(
                np.mean(
                    [
                        row["length"]
                        for row in episodes.values()
                        if row.get("is_success") and "length" in row
                    ]
                )
            )
            if any(row.get("is_success") for row in episodes.values())
            else None
        ),
        "success_only_act_std": (
            float(
                np.std(
                    [
                        row["length"]
                        for row in episodes.values()
                        if row.get("is_success") and "length" in row
                    ]
                )
            )
            if any(row.get("is_success") for row in episodes.values())
            else None
        ),
    }


def _load_comparison_frame(comparison: dict[str, Any]) -> pd.DataFrame:
    advantages = pd.read_parquet(comparison["advantages_path"])
    predictions = pd.read_parquet(comparison["predictions_path"])
    merged = predictions[
        ["episode_index", "frame_index", "value_fused"]
    ].merge(
        advantages[["episode_index", "frame_index", "return", "value_current"]],
        on=["episode_index", "frame_index"],
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlapping frames in comparison inputs.")

    return_min = float(comparison["return_min"])
    return_max = float(comparison["return_max"])
    value_min = float(comparison["value_min"])
    value_max = float(comparison["value_max"])
    scale = (return_max - return_min) / (value_max - value_min)
    merged["base_pred_return"] = (
        (merged["value_current"] - value_min) * scale + return_min
    )
    merged["fused_pred_return"] = (
        (merged["value_fused"] - value_min) * scale + return_min
    )
    merged["base_abs_error"] = (
        merged["base_pred_return"] - merged["return"]
    ).abs()
    merged["fused_abs_error"] = (
        merged["fused_pred_return"] - merged["return"]
    ).abs()
    merged["base_squared_error"] = (
        merged["base_pred_return"] - merged["return"]
    ) ** 2
    merged["fused_squared_error"] = (
        merged["fused_pred_return"] - merged["return"]
    ) ** 2
    return merged


def _error_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"frames": 0, "episodes": 0}
    return {
        "frames": int(len(frame)),
        "episodes": int(frame["episode_index"].nunique()),
        "base_mae": float(frame["base_abs_error"].mean()),
        "fused_mae": float(frame["fused_abs_error"].mean()),
        "base_rmse": float(np.sqrt(frame["base_squared_error"].mean())),
        "fused_rmse": float(np.sqrt(frame["fused_squared_error"].mean())),
        "base_bias": float(
            (frame["base_pred_return"] - frame["return"]).mean()
        ),
        "fused_bias": float(
            (frame["fused_pred_return"] - frame["return"]).mean()
        ),
        "mae_improvement_pct": float(
            100.0
            * (1.0 - frame["fused_abs_error"].mean() / frame["base_abs_error"].mean())
        ),
    }


def _episode_error_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"episodes": 0}
    grouped = frame.groupby("episode_index")
    base_mae = grouped["base_abs_error"].mean()
    fused_mae = grouped["fused_abs_error"].mean()
    return {
        "episodes": int(len(grouped)),
        "mean_base_mae": float(base_mae.mean()),
        "mean_fused_mae": float(fused_mae.mean()),
        "mean_mae_gain": float((base_mae - fused_mae).mean()),
        "episodes_improved_mae": int((base_mae > fused_mae).sum()),
    }


def _load_zp_metrics(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    data = _read_json(path)
    return {
        "best_epoch": data.get("best_epoch"),
        "best_val_metrics": data.get("best_val_metrics", {}),
    }


def _load_fusion_metrics(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    data = _read_json(path)
    best_epoch = data.get("best_epoch")
    history = data.get("history", [])
    best = history[int(best_epoch) - 1] if best_epoch and history else {}
    return {
        "best_epoch": best_epoch,
        "best_val_metrics": {
            "value_mse": best.get("val_value_mse"),
            "value_mae": best.get("val_value_mae"),
            "raw_value_mse": best.get("val_raw_value_mse"),
            "improvement_pct": best.get("val_improvement_pct"),
        },
    }


def _build_row(result: dict[str, Any]) -> dict[str, Any]:
    eval_data = result["evaluation"]
    critic = result["critic"]
    all_frame = critic["frame_level_all"]
    all_episode = critic["episode_level_all"]
    status = critic["status_frame_metrics"]
    zp = result.get("zp") or {}
    zp_metrics = zp.get("best_val_metrics", {})
    fusion = result.get("fusion") or {}
    fusion_metrics = fusion.get("best_val_metrics", {})
    return {
        "task": result["task"],
        "round": result["round"],
        "policy": result["policy"],
        "critic": result["critic_label"],
        "train_episodes": result["training"].get("episodes"),
        "train_frames": result["training"].get("frames"),
        "eval_episodes": eval_data.get("episodes"),
        "eval_frames": eval_data.get("frames"),
        "eval_successes": eval_data.get("successes"),
        "eval_failures": eval_data.get("failures"),
        "eval_success_rate": eval_data.get("success_rate"),
        "eval_act_all_mean": eval_data.get("mean_episode_length"),
        "eval_act_success_mean": eval_data.get("success_only_act"),
        "eval_act_success_std": eval_data.get("success_only_act_std"),
        "raw_frame_mae": all_frame.get("base_mae"),
        "fused_frame_mae": all_frame.get("fused_mae"),
        "fused_frame_mae_improvement_pct": all_frame.get("mae_improvement_pct"),
        "raw_episode_mean_mae": all_episode.get("mean_base_mae"),
        "fused_episode_mean_mae": all_episode.get("mean_fused_mae"),
        "success_raw_frame_mae": status["success"].get("base_mae"),
        "success_fused_frame_mae": status["success"].get("fused_mae"),
        "failure_raw_frame_mae": status["failure"].get("base_mae"),
        "failure_fused_frame_mae": status["failure"].get("fused_mae"),
        "zp_best_epoch": zp.get("best_epoch"),
        "zp_phase_acc": zp_metrics.get("phase_acc"),
        "zp_macro_phase_acc": zp_metrics.get("macro_phase_acc"),
        "zp_progress_mae": zp_metrics.get("progress_mae"),
        "zp_global_progress_mae": zp_metrics.get("global_progress_mae"),
        "fusion_best_epoch": fusion.get("best_epoch"),
        "fusion_val_mae": fusion_metrics.get("value_mae"),
        "fusion_val_raw_mse": fusion_metrics.get("raw_value_mse"),
        "fusion_val_improvement_pct": fusion_metrics.get("improvement_pct"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record one paper-ready ReCap experiment round."
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--policy-label", required=True)
    parser.add_argument("--critic-label", required=True)
    parser.add_argument("--train-dataset", type=Path, default=None)
    parser.add_argument("--eval-dataset", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--zp-metrics", type=Path, default=None)
    parser.add_argument("--fusion-metrics", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = _read_json(args.comparison)
    collection_path = args.eval_dataset / "collection_summary.json"
    collection = _read_json(collection_path) if collection_path.exists() else {}
    eval_data = _dataset_summary(args.eval_dataset)
    eval_data["successes"] = collection.get("successes", eval_data["successes"])
    eval_data["success_rate"] = collection.get(
        "success_rate",
        (
            eval_data["successes"] / eval_data["episodes"]
            if eval_data["successes"] is not None and eval_data["episodes"]
            else None
        ),
    )
    eval_data["mean_episode_length"] = collection.get(
        "mean_episode_length", eval_data["mean_episode_length"]
    )

    frame = _load_comparison_frame(comparison)
    outcomes = _dataset_episode_stats(args.eval_dataset)
    frame["outcome"] = frame["episode_index"].map(
        lambda episode: "success"
        if outcomes.get(int(episode), {}).get("is_success") is True
        else "failure"
        if outcomes.get(int(episode), {}).get("is_success") is False
        else "unknown"
    )
    status_metrics = {
        status: _error_metrics(frame[frame["outcome"] == status])
        for status in ("success", "failure", "unknown")
    }

    training = _dataset_summary(args.train_dataset) if args.train_dataset else {}
    result = {
        "schema_version": "1.0",
        "task": args.task,
        "round": args.round,
        "policy": args.policy_label,
        "critic_label": args.critic_label,
        "training": training,
        "evaluation": eval_data,
        "critic": {
            "frame_level_all": _error_metrics(frame),
            "episode_level_all": _episode_error_metrics(frame),
            "status_frame_metrics": status_metrics,
            "reported_comparison": comparison.get("frame_level", {}).get("all", {}),
        },
        "zp": _load_zp_metrics(args.zp_metrics),
        "fusion": _load_fusion_metrics(args.fusion_metrics),
        "policy_metrics": {
            "success_rate": eval_data.get("success_rate"),
            "success_only_act": eval_data.get("success_only_act"),
            "note": "These are policy rollout metrics; they are not Critic MAE metrics.",
        },
        "missing_metrics": [
            "phase_macro_f1",
            "value_nll",
            "value_calibration_ece",
            "policy_confidence_interval",
        ],
        "source_paths": {
            "eval_dataset": str(args.eval_dataset),
            "train_dataset": str(args.train_dataset)
            if args.train_dataset
            else None,
            "comparison": str(args.comparison),
            "collection_summary": str(collection_path)
            if collection_path.exists()
            else None,
            "zp_metrics": str(args.zp_metrics) if args.zp_metrics else None,
            "fusion_metrics": (
                str(args.fusion_metrics) if args.fusion_metrics else None
            ),
        },
    }
    row = _build_row(result)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.output_name}.json"
    csv_path = output_dir / f"{args.output_name}.csv"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(f"saved JSON result: {json_path}")
    print(f"saved CSV result: {csv_path}")
    print(json.dumps(row, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
