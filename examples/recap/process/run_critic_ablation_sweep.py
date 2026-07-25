#!/usr/bin/env python
"""Train and evaluate GLC-Critic architecture ablations on one fixed pool."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from rlinf.revalue.data.feature_cache import (
    HEAD_TYPE_TEMPORAL_LOCAL_STAGE_GATED,
)
from rlinf.revalue.pipeline.predict import PredictionConfig, predict_fused_values
from rlinf.revalue.pipeline.train import (
    FusionTrainingConfig,
    ZPTrainingConfig,
    train_fusion,
    train_zp_head,
)
from rlinf.revalue.value_scale import map_values_to_return_scale


KEY_COLUMNS = ["episode_index", "frame_index"]
VARIANT_ORDER = [
    "raw",
    "without_local_temporal",
    "without_motion_difference",
    "without_phase_z",
    "without_progress_p",
    "full",
]
VARIANT_LABELS = {
    "raw": "Raw Critic",
    "without_local_temporal": "w/o local temporal aggregation",
    "without_motion_difference": "w/o motion-difference module",
    "without_phase_z": "w/o phase correction head z",
    "without_progress_p": "w/o intra-phase progress head p",
    "full": "Full GLC-Critic",
}
VARIANT_FLAGS = {
    "without_local_temporal": {
        "use_local_temporal": False,
        "use_motion_difference": True,
        "use_phase_head": True,
        "use_progress_head": True,
    },
    "without_motion_difference": {
        "use_local_temporal": True,
        "use_motion_difference": False,
        "use_phase_head": True,
        "use_progress_head": True,
    },
    "without_phase_z": {
        "use_local_temporal": True,
        "use_motion_difference": True,
        "use_phase_head": False,
        "use_progress_head": True,
    },
    "without_progress_p": {
        "use_local_temporal": True,
        "use_motion_difference": True,
        "use_phase_head": True,
        "use_progress_head": False,
    },
    "full": {
        "use_local_temporal": True,
        "use_motion_difference": True,
        "use_phase_head": True,
        "use_progress_head": True,
    },
}


@dataclass(frozen=True)
class SweepConfig:
    """Serializable training and evaluation settings for one sweep."""

    num_tasks: int
    num_phases: int
    return_min: float
    return_max: float
    value_min: float
    value_max: float
    num_bins: int
    boundary_window: int
    seed: int
    batch_size: int
    prediction_batch_size: int
    max_epochs: int
    early_stop_patience: int
    hidden_dim: int
    dropout: float
    trunk_depth: int
    window_size: int
    num_layers: int
    num_heads: int
    ffn_dim: int
    stage_embedding_dim: int
    progress_hidden_dim: int
    progress_depth: int
    use_class_weights: bool
    fusion_hidden_dim: int
    fusion_depth: int
    fusion_dropout: float
    fusion_alpha: float
    device: str


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _prediction_is_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        frame = pd.read_parquet(
            path,
            columns=["split", "episode_index", "frame_index", "value_fused"],
        )
    except (OSError, ValueError, KeyError):
        return False
    return bool(len(frame)) and not frame.duplicated(KEY_COLUMNS).any()


def _cache_keys(path: Path, task: str) -> pd.DataFrame:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    required = {"episode_index", "frame_index", "phase"}
    missing = required - set(cache)
    if missing:
        raise ValueError(f"{path} missing cache keys: {sorted(missing)}")
    rows = len(cache["episode_index"])
    if any(len(cache[key]) != rows for key in required):
        raise ValueError(f"{path} has inconsistent cache tensor lengths")
    return pd.DataFrame(
        {
            "task": task,
            "episode_index": cache["episode_index"].cpu().numpy(),
            "frame_index": cache["frame_index"].cpu().numpy(),
            "phase_true": cache["phase"].cpu().numpy(),
        }
    )


def audit_inputs(
    *,
    revalue_root: Path,
    advantages_path: Path,
    num_tasks: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate cached task splits and return the held-out frame index."""
    if not advantages_path.is_file():
        raise FileNotFoundError(f"Advantages not found: {advantages_path}")
    advantages = pd.read_parquet(advantages_path)
    required_adv = {
        "episode_index",
        "frame_index",
        "return",
        "value_current",
        "value_logits_current",
    }
    missing_adv = required_adv - set(advantages.columns)
    if missing_adv:
        raise ValueError(
            f"Advantages missing columns required by fusion: {sorted(missing_adv)}"
        )
    if advantages.duplicated(KEY_COLUMNS).any():
        raise ValueError("Advantages contain duplicate episode/frame keys")

    val_frames: list[pd.DataFrame] = []
    task_report: dict[str, Any] = {}
    for task_index in range(num_tasks):
        task = f"task{task_index}"
        features_dir = revalue_root / "features" / task
        train_path = features_dir / "train.pt"
        val_path = features_dir / "val.pt"
        for path in (train_path, val_path):
            if not path.is_file():
                raise FileNotFoundError(f"Feature cache not found: {path}")
        train = _cache_keys(train_path, task)
        val = _cache_keys(val_path, task)
        episode_overlap = set(train["episode_index"]) & set(val["episode_index"])
        if episode_overlap:
            raise ValueError(
                f"{task} train/val caches share episodes: "
                f"{sorted(episode_overlap)[:10]}"
            )
        overlap = train.merge(val, on=KEY_COLUMNS, how="inner")
        if not overlap.empty:
            raise ValueError(f"{task} train/val caches overlap by {len(overlap)} rows")
        val_frames.append(val)
        task_report[task] = {
            "train_frames": int(len(train)),
            "train_episodes": int(train["episode_index"].nunique()),
            "val_frames": int(len(val)),
            "val_episodes": int(val["episode_index"].nunique()),
        }

    val_index = pd.concat(val_frames, ignore_index=True)
    if val_index.duplicated(KEY_COLUMNS).any():
        duplicates = val_index[val_index.duplicated(KEY_COLUMNS, keep=False)].head()
        raise ValueError(
            "Task validation caches share episode/frame keys:\n"
            + duplicates.to_string(index=False)
        )
    adv_keys = advantages[KEY_COLUMNS]
    covered = val_index.merge(adv_keys, on=KEY_COLUMNS, how="left", indicator=True)
    missing_rows = covered["_merge"] != "both"
    if missing_rows.any():
        raise ValueError(
            f"{int(missing_rows.sum())} held-out cache rows lack advantages"
        )
    report = {
        "revalue_root": str(revalue_root),
        "advantages_path": str(advantages_path),
        "advantages_frames": int(len(advantages)),
        "held_out_frames": int(len(val_index)),
        "held_out_episodes": int(val_index["episode_index"].nunique()),
        "tasks": task_report,
    }
    return val_index, report


def _variant_manifest(
    variant: str,
    flags: dict[str, bool],
    cfg: SweepConfig,
    advantages_path: Path,
) -> dict[str, Any]:
    return {
        "variant": variant,
        "label": VARIANT_LABELS[variant],
        "flags": flags,
        "sweep": asdict(cfg),
        "advantages_path": str(advantages_path.resolve()),
        "neutral_ablation_inputs": {
            "without_phase_z": {
                "phase_probs": "uniform",
                "global_progress": 0.5,
                "progress_conditioning": "phase-independent",
            },
            "without_progress_p": {
                "phase_progress": 0.5,
                "global_progress": "predicted phase midpoint",
            },
        },
    }


def _prepare_variant_dir(
    path: Path,
    manifest: dict[str, Any],
    *,
    force: bool,
) -> None:
    manifest_path = path / "variant_config.json"
    if manifest_path.is_file() and not force:
        previous = _read_json(manifest_path)
        if previous != manifest:
            raise ValueError(
                f"Existing variant config differs at {manifest_path}; "
                "use a new --output-root or pass --force"
            )
    _write_json(manifest, manifest_path)


def train_variant_task(
    *,
    variant: str,
    task: str,
    flags: dict[str, bool],
    revalue_root: Path,
    advantages_path: Path,
    output_root: Path,
    cfg: SweepConfig,
    force: bool,
) -> Path:
    """Train one task head/fusion and predict both cached splits."""
    task_root = output_root / variant / task
    zp_path = task_root / "zp_head" / "zp_head.pt"
    fusion_path = task_root / "fusion" / "fusion.pt"
    predictions_path = task_root / "predictions.parquet"
    if _prediction_is_complete(predictions_path) and not force:
        print(f"[RESUME] {variant}/{task}: predictions already complete")
        return predictions_path

    features_dir = revalue_root / "features" / task
    if not zp_path.is_file() or force:
        print(f"[TRAIN ZP] {variant}/{task}")
        train_zp_head(
            ZPTrainingConfig(
                features_dir=str(features_dir),
                output_dir=str(zp_path.parent),
                num_phases=cfg.num_phases,
                head_type=HEAD_TYPE_TEMPORAL_LOCAL_STAGE_GATED,
                hidden_dim=cfg.hidden_dim,
                dropout=cfg.dropout,
                trunk_depth=cfg.trunk_depth,
                window_size=cfg.window_size,
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                ffn_dim=cfg.ffn_dim,
                stage_embedding_dim=cfg.stage_embedding_dim,
                progress_hidden_dim=cfg.progress_hidden_dim,
                progress_depth=cfg.progress_depth,
                batch_size=cfg.batch_size,
                max_epochs=cfg.max_epochs,
                early_stop_patience=cfg.early_stop_patience,
                use_class_weights=cfg.use_class_weights,
                use_local_temporal=flags["use_local_temporal"],
                use_motion_difference=flags["use_motion_difference"],
                use_phase_head=flags["use_phase_head"],
                use_progress_head=flags["use_progress_head"],
                device=cfg.device,
                seed=cfg.seed,
            )
        )
    else:
        print(f"[RESUME] {variant}/{task}: z/p checkpoint found")

    if not fusion_path.is_file() or force:
        print(f"[TRAIN FUSION] {variant}/{task}")
        train_fusion(
            FusionTrainingConfig(
                features_dir=str(features_dir),
                advantages_path=str(advantages_path),
                zp_head_path=str(zp_path),
                output_dir=str(fusion_path.parent),
                return_min=cfg.return_min,
                return_max=cfg.return_max,
                value_min=cfg.value_min,
                value_max=cfg.value_max,
                num_bins=cfg.num_bins,
                num_phases=cfg.num_phases,
                fusion_hidden_dim=cfg.fusion_hidden_dim,
                fusion_depth=cfg.fusion_depth,
                fusion_dropout=cfg.fusion_dropout,
                alpha=cfg.fusion_alpha,
                batch_size=cfg.batch_size,
                max_epochs=cfg.max_epochs,
                early_stop_patience=cfg.early_stop_patience,
                device=cfg.device,
                seed=cfg.seed,
            )
        )
    else:
        print(f"[RESUME] {variant}/{task}: fusion checkpoint found")

    print(f"[PREDICT] {variant}/{task}")
    return predict_fused_values(
        PredictionConfig(
            features_dir=str(features_dir),
            advantages_path=str(advantages_path),
            zp_head_path=str(zp_path),
            fusion_path=str(fusion_path),
            output_path=str(predictions_path),
            batch_size=cfg.prediction_batch_size,
            device=cfg.device,
        )
    )


def _combine_predictions(paths: list[Path], output_path: Path) -> Path:
    frames = [pd.read_parquet(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(KEY_COLUMNS).any():
        raise ValueError("Combined predictions contain duplicate keys")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    return output_path


def _mark_boundaries(frame: pd.DataFrame, window: int) -> pd.Series:
    ordered = frame.sort_values(["task", *KEY_COLUMNS])
    boundary = pd.Series(False, index=ordered.index)
    for _, part in ordered.groupby(["task", "episode_index"], sort=False):
        phases = part["phase_true"].to_numpy(dtype=np.int64)
        frames = part["frame_index"].to_numpy(dtype=np.int64)
        positions = np.flatnonzero(phases[1:] != phases[:-1]) + 1
        if not len(positions):
            continue
        distances = np.min(
            np.abs(frames[:, None] - frames[positions][None, :]), axis=1
        )
        boundary.loc[part.index] = distances <= window
    return boundary.reindex(frame.index)


def evaluate_predictions(
    *,
    variant: str,
    predictions_path: Path,
    advantages: pd.DataFrame,
    val_index: pd.DataFrame,
    cfg: SweepConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one prediction artifact using exactly the cached val rows."""
    predictions = pd.read_parquet(predictions_path)
    required = {*KEY_COLUMNS, "value_fused"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"{predictions_path} missing columns: {sorted(missing)}")
    predictions = predictions[[*KEY_COLUMNS, "value_fused"]]
    frame = val_index.merge(
        predictions,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    ).merge(
        advantages[[*KEY_COLUMNS, "return", "value_current"]],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    if frame[["value_fused", "return", "value_current"]].isna().any().any():
        raise ValueError(f"{variant} does not cover every held-out cache row")
    value_column = "value_current" if variant == "raw" else "value_fused"
    frame["predicted_return"] = map_values_to_return_scale(
        frame[value_column].to_numpy(dtype=np.float64),
        return_min=cfg.return_min,
        return_max=cfg.return_max,
        value_min=cfg.value_min,
        value_max=cfg.value_max,
    )
    frame["absolute_error"] = np.abs(
        frame["predicted_return"].to_numpy(dtype=np.float64)
        - frame["return"].to_numpy(dtype=np.float64)
    )
    frame["is_boundary"] = _mark_boundaries(frame, cfg.boundary_window)

    task_rows: list[dict[str, Any]] = []
    for task, part in frame.groupby("task", sort=True):
        boundary = part[part["is_boundary"]]
        if boundary.empty:
            raise ValueError(
                f"{task} has no phase boundary in the held-out split"
            )
        task_rows.append(
            {
                "variant": variant,
                "critic_variant": VARIANT_LABELS[variant],
                "task": task,
                "held_out_frames": int(len(part)),
                "held_out_episodes": int(part["episode_index"].nunique()),
                "held_out_mae": float(part["absolute_error"].mean()),
                "boundary_frames": int(len(boundary)),
                "boundary_mae": float(boundary["absolute_error"].mean()),
            }
        )
    by_task = pd.DataFrame(task_rows)
    boundary = frame[frame["is_boundary"]]
    summary = {
        "variant": variant,
        "critic_variant": VARIANT_LABELS[variant],
        "held_out_frames": int(len(frame)),
        "held_out_episodes": int(frame["episode_index"].nunique()),
        "held_out_mae": float(frame["absolute_error"].mean()),
        "task_macro_held_out_mae": float(by_task["held_out_mae"].mean()),
        "boundary_frames": int(len(boundary)),
        "boundary_mae": float(boundary["absolute_error"].mean()),
        "task_macro_boundary_mae": float(by_task["boundary_mae"].mean()),
        "predictions_path": str(predictions_path),
    }
    return summary, task_rows


def _write_reports(
    *,
    output_root: Path,
    audit: dict[str, Any],
    cfg: SweepConfig,
    summaries: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
) -> None:
    order = {variant: index for index, variant in enumerate(VARIANT_ORDER)}
    summaries.sort(key=lambda row: order[row["variant"]])
    task_rows.sort(key=lambda row: (order[row["variant"]], row["task"]))
    table = pd.DataFrame(summaries)
    by_task = pd.DataFrame(task_rows)
    table.to_csv(output_root / "critic_ablation_table.csv", index=False)
    by_task.to_csv(output_root / "critic_ablation_by_task.csv", index=False)
    report = {
        "evaluation_scale": (
            "Predictions are inverse-mapped from normalized value space to "
            "the original return/cost scale before MAE computation."
        ),
        "held_out_definition": "Exact val.pt rows from every task feature cache.",
        "boundary_definition": (
            f"Frames within +/-{cfg.boundary_window} of a phase_true transition."
        ),
        "audit": audit,
        "config": asdict(cfg),
        "variants": summaries,
    }
    _write_json(report, output_root / "critic_ablation_summary.json")

    display = table[
        [
            "critic_variant",
            "held_out_mae",
            "boundary_mae",
            "held_out_frames",
            "boundary_frames",
        ]
    ].copy()
    text = [
        "GLC-CRITIC ARCHITECTURE ABLATION",
        "",
        report["evaluation_scale"],
        report["held_out_definition"],
        report["boundary_definition"],
        "",
        display.to_string(index=False, float_format=lambda value: f"{value:.3f}"),
        "",
        f"CSV: {output_root / 'critic_ablation_table.csv'}",
        f"BY TASK: {output_root / 'critic_ablation_by_task.csv'}",
    ]
    summary_path = output_root / "critic_ablation_summary.txt"
    summary_path.write_text("\n".join(text) + "\n", encoding="utf-8")
    print(summary_path.read_text(encoding="utf-8"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revalue-root", type=Path, required=True)
    parser.add_argument("--advantages", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANT_FLAGS),
        default=list(VARIANT_FLAGS),
    )
    parser.add_argument("--full-predictions", type=Path)
    parser.add_argument("--retrain-full", action="store_true")
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument("--num-phases", type=int, default=4)
    parser.add_argument("--return-min", type=float, default=-900.0)
    parser.add_argument("--return-max", type=float, default=0.0)
    parser.add_argument("--value-min", type=float, default=-1.0)
    parser.add_argument("--value-max", type=float, default=0.0)
    parser.add_argument("--num-bins", type=int, default=201)
    parser.add_argument("--boundary-window", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--prediction-batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--trunk-depth", type=int, default=1)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--stage-embedding-dim", type=int, default=32)
    parser.add_argument("--progress-hidden-dim", type=int, default=256)
    parser.add_argument("--progress-depth", type=int, default=2)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--fusion-depth", type=int, default=2)
    parser.add_argument("--fusion-dropout", type=float, default=0.1)
    parser.add_argument("--fusion-alpha", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    revalue_root = args.revalue_root.expanduser().resolve()
    advantages_path = args.advantages.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cfg = SweepConfig(
        num_tasks=args.num_tasks,
        num_phases=args.num_phases,
        return_min=args.return_min,
        return_max=args.return_max,
        value_min=args.value_min,
        value_max=args.value_max,
        num_bins=args.num_bins,
        boundary_window=args.boundary_window,
        seed=args.seed,
        batch_size=args.batch_size,
        prediction_batch_size=args.prediction_batch_size,
        max_epochs=args.max_epochs,
        early_stop_patience=args.early_stop_patience,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        trunk_depth=args.trunk_depth,
        window_size=args.window_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        stage_embedding_dim=args.stage_embedding_dim,
        progress_hidden_dim=args.progress_hidden_dim,
        progress_depth=args.progress_depth,
        use_class_weights=args.use_class_weights,
        fusion_hidden_dim=args.fusion_hidden_dim,
        fusion_depth=args.fusion_depth,
        fusion_dropout=args.fusion_dropout,
        fusion_alpha=args.fusion_alpha,
        device=args.device,
    )
    val_index, audit = audit_inputs(
        revalue_root=revalue_root,
        advantages_path=advantages_path,
        num_tasks=cfg.num_tasks,
    )
    _write_json(audit, output_root / "input_audit.json")
    print(json.dumps(audit, indent=2))
    if args.dry_run:
        print("[DRY RUN] Input audit passed; no training was started.")
        return

    variant_predictions: dict[str, Path] = {}
    full_predictions = args.full_predictions
    if full_predictions is None:
        candidate = revalue_root / "predictions.parquet"
        if candidate.is_file():
            full_predictions = candidate

    for variant in args.variants:
        flags = VARIANT_FLAGS[variant]
        manifest = _variant_manifest(variant, flags, cfg, advantages_path)
        _prepare_variant_dir(
            output_root / variant,
            manifest,
            force=args.force,
        )
        if (
            variant == "full"
            and not args.retrain_full
            and full_predictions is not None
        ):
            full_predictions = full_predictions.expanduser().resolve()
            if not _prediction_is_complete(full_predictions):
                raise ValueError(
                    f"Full predictions are missing or incomplete: {full_predictions}"
                )
            print(f"[REUSE FULL] {full_predictions}")
            variant_predictions[variant] = full_predictions
            continue

        task_predictions = []
        for task_index in range(cfg.num_tasks):
            task = f"task{task_index}"
            task_predictions.append(
                train_variant_task(
                    variant=variant,
                    task=task,
                    flags=flags,
                    revalue_root=revalue_root,
                    advantages_path=advantages_path,
                    output_root=output_root,
                    cfg=cfg,
                    force=args.force,
                )
            )
        variant_predictions[variant] = _combine_predictions(
            task_predictions,
            output_root / variant / "predictions.parquet",
        )

    if not variant_predictions:
        raise ValueError("No variants were selected")
    reference_predictions = next(iter(variant_predictions.values()))
    advantages = pd.read_parquet(advantages_path)
    summaries: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    raw_summary, raw_tasks = evaluate_predictions(
        variant="raw",
        predictions_path=reference_predictions,
        advantages=advantages,
        val_index=val_index,
        cfg=cfg,
    )
    summaries.append(raw_summary)
    task_rows.extend(raw_tasks)
    for variant, predictions_path in variant_predictions.items():
        summary, rows = evaluate_predictions(
            variant=variant,
            predictions_path=predictions_path,
            advantages=advantages,
            val_index=val_index,
            cfg=cfg,
        )
        summaries.append(summary)
        task_rows.extend(rows)
    _write_reports(
        output_root=output_root,
        audit=audit,
        cfg=cfg,
        summaries=summaries,
        task_rows=task_rows,
    )


if __name__ == "__main__":
    main()
