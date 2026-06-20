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

"""Run a strict 4-method baseline suite on cached 80/20 train-val features.

Methods:
1. Raw critic
2. Single-frame shared MLP trunk + two heads
3. Temporal phase-prior head
4. Temporal z->MLP->p head

For fairness, the three learned baselines use roughly matched parameter counts:
- shared_mlp_two_heads: hidden_dim=640, trunk_depth=6      (~2.47M)
- temporal_phase_prior: default 256/2/2/4/512/32          (~2.43M)
- temporal_z_mlp_p: hidden_dim=336, 2 layers, 6 heads     (~2.40M)

Evaluation protocol:
- train split predictions fit per-phase correction coefficients
- val split applies frozen coefficients
- report only raw vs predicted on val (no oracle)
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MethodSpec:
    """Train/analyze specification for one learned baseline."""

    key: str
    display_name: str
    train_script: str
    analyze_script: str
    train_extra_args: tuple[str, ...]


METHOD_SPECS: tuple[MethodSpec, ...] = (
    MethodSpec(
        key="shared_mlp_two_heads",
        display_name="Shared MLP Trunk + Two Heads",
        train_script="train_head.py",
        analyze_script="predict_and_analyze.py",
        train_extra_args=(
            "--hidden_dim",
            "640",
            "--trunk_depth",
            "6",
            "--dropout",
            "0.1",
            "--label_smoothing",
            "0.05",
            "--global_progress_loss_weight",
            "0.25",
        ),
    ),
    MethodSpec(
        key="temporal_phase_prior",
        display_name="Temporal Phase Prior",
        train_script="train_head_temporal_phase_prior.py",
        analyze_script="predict_and_analyze_temporal_phase_prior.py",
        train_extra_args=(
            "--window_size",
            "5",
            "--hidden_dim",
            "256",
            "--stage_layers",
            "2",
            "--progress_layers",
            "2",
            "--num_heads",
            "4",
            "--ffn_dim",
            "512",
            "--stage_embedding_dim",
            "32",
            "--attention_dropout",
            "0.1",
        ),
    ),
    MethodSpec(
        key="temporal_z_mlp_p",
        display_name="Temporal z->MLP->p",
        train_script="train_head_temporal_z_mlp_p.py",
        analyze_script="predict_and_analyze_temporal_z_mlp_p.py",
        train_extra_args=(
            "--window_size",
            "5",
            "--hidden_dim",
            "336",
            "--num_layers",
            "2",
            "--num_heads",
            "6",
            "--ffn_dim",
            "672",
            "--stage_embedding_dim",
            "64",
            "--progress_hidden_dim",
            "336",
            "--progress_depth",
            "3",
        ),
    ),
)


def _run_command(
    cmd: list[str],
    cwd: Path,
    method_label: str,
    stage_label: str,
    log_path: Path,
    skip_if_exists: Path | None = None,
    force: bool = False,
) -> None:
    """Run a subprocess command and append prefixed output to the suite log."""
    prefix = f"[{method_label}][{stage_label}]"
    if skip_if_exists is not None and skip_if_exists.exists() and not force:
        logger.info("%s Skip existing artifact: %s", prefix, skip_if_exists)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{prefix} Skip existing artifact: {skip_if_exists}\n")
        return

    logger.info("%s Running: %s", prefix, " ".join(cmd))
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{prefix} Running: {' '.join(cmd)}\n")
        f.flush()

        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip("\n")
            logger.info("%s %s", prefix, line)
            f.write(f"{prefix} {line}\n")
        ret = process.wait()
        f.write(f"{prefix} Exit code: {ret}\n")
        f.flush()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, cmd)


def _count_checkpoint_params(head_path: Path) -> int:
    """Count parameters from a saved checkpoint state dict."""
    ckpt = torch.load(head_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    return int(sum(t.numel() for t in state_dict.values()))


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_markdown_table(rows: list[dict[str, Any]]) -> str:
    """Render a compact markdown summary table."""
    headers = [
        "Method",
        "Params",
        "Val MSE",
        "Rel. Improve",
        "Phase Acc",
        "Progress MAE",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(row["params"]),
                    f'{row["val_mse"]:.6f}',
                    f'{row["improvement_pct"]:.1f}%',
                    row["phase_acc"],
                    row["progress_mae"],
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", required=True)
    parser.add_argument("--advantages_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--return_min", type=float, default=-700.0)
    parser.add_argument("--return_max", type=float, default=0.0)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--analyze_batch_size", type=int, default=1024)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun all steps even if output artifacts already exist.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    repo_root = Path(__file__).resolve().parents[3]
    script_dir = Path(__file__).resolve().parent
    python_bin = sys.executable

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "suite_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=float)
    suite_log_path = output_root / "suite.log"
    with open(suite_log_path, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write(f"Baseline suite start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"features_dir={args.features_dir}\n")
        f.write(f"advantages_path={args.advantages_path}\n")
        f.write(f"output_root={args.output_root}\n")
        f.write("=" * 80 + "\n")

    learned_rows: list[dict[str, Any]] = []
    raw_mse: float | None = None

    for spec in METHOD_SPECS:
        method_root = output_root / spec.key
        train_dir = method_root / "train"
        analysis_dir = method_root / "analysis"
        strict_dir = method_root / "strict_eval"
        train_dir.mkdir(parents=True, exist_ok=True)
        analysis_dir.mkdir(parents=True, exist_ok=True)
        strict_dir.mkdir(parents=True, exist_ok=True)

        train_script_path = script_dir / spec.train_script
        analyze_script_path = script_dir / spec.analyze_script
        strict_script_path = script_dir / "evaluate_raw_vs_predicted_strict.py"

        train_cmd = [
            python_bin,
            str(train_script_path),
            "--features_dir",
            args.features_dir,
            "--output_dir",
            str(train_dir),
            "--batch_size",
            str(args.train_batch_size),
            "--max_epochs",
            str(args.max_epochs),
            "--seed",
            str(args.seed),
            *spec.train_extra_args,
        ]
        _run_command(
            train_cmd,
            cwd=repo_root,
            method_label=spec.key,
            stage_label="train",
            log_path=suite_log_path,
            skip_if_exists=train_dir / "head.pt",
            force=args.force,
        )

        analyze_cmd = [
            python_bin,
            str(analyze_script_path),
            "--features_dir",
            args.features_dir,
            "--head_checkpoint",
            str(train_dir / "head.pt"),
            "--output_dir",
            str(analysis_dir),
            "--batch_size",
            str(args.analyze_batch_size),
            "--splits",
            "train",
            "val",
        ]
        _run_command(
            analyze_cmd,
            cwd=repo_root,
            method_label=spec.key,
            stage_label="predict",
            log_path=suite_log_path,
            skip_if_exists=analysis_dir / "phase_predictions.parquet",
            force=args.force,
        )

        strict_cmd = [
            python_bin,
            str(strict_script_path),
            "--advantages_path",
            args.advantages_path,
            "--predictions_path",
            str(analysis_dir / "phase_predictions.parquet"),
            "--output_dir",
            str(strict_dir),
            "--return_min",
            str(args.return_min),
            "--return_max",
            str(args.return_max),
        ]
        _run_command(
            strict_cmd,
            cwd=repo_root,
            method_label=spec.key,
            stage_label="strict_eval",
            log_path=suite_log_path,
            skip_if_exists=strict_dir / "report.json",
            force=args.force,
        )

        params = _count_checkpoint_params(train_dir / "head.pt")
        analysis_report = _load_json(analysis_dir / "report.json")
        strict_report = _load_json(strict_dir / "report.json")

        pred_metrics = analysis_report["prediction_metrics"]
        strict_metrics = strict_report["metrics"]
        if raw_mse is None:
            raw_mse = float(strict_metrics["mse_raw"])
        else:
            current_raw = float(strict_metrics["mse_raw"])
            if abs(current_raw - raw_mse) > 1e-9:
                raise ValueError(
                    f"Raw MSE mismatch across methods: {current_raw} vs {raw_mse}"
                )

        learned_rows.append(
            {
                "key": spec.key,
                "method": spec.display_name,
                "params": params,
                "val_mse": float(strict_metrics["mse_predicted"]),
                "improvement_pct": float(strict_metrics["predicted_improvement_pct"]),
                "phase_acc_value": float(pred_metrics["phase_acc"]),
                "progress_mae_value": float(pred_metrics["phase_progress_mae"]),
                "phase_acc": f'{pred_metrics["phase_acc"]:.4f}',
                "progress_mae": f'{pred_metrics["phase_progress_mae"]:.4f}',
                "artifacts": {
                    "train_dir": str(train_dir),
                    "analysis_dir": str(analysis_dir),
                    "strict_dir": str(strict_dir),
                },
            }
        )

    if raw_mse is None:
        raise RuntimeError("No learned method completed, raw MSE unavailable.")

    summary_rows = [
        {
            "key": "raw_critic",
            "method": "Raw Critic",
            "params": 0,
            "val_mse": raw_mse,
            "improvement_pct": 0.0,
            "phase_acc_value": None,
            "progress_mae_value": None,
            "phase_acc": "-",
            "progress_mae": "-",
            "artifacts": {},
        },
        *learned_rows,
    ]

    summary = {
        "raw_mse": raw_mse,
        "methods": summary_rows,
        "matched_param_note": {
            "shared_mlp_two_heads": "hidden_dim=640, trunk_depth=6",
            "temporal_phase_prior": "hidden_dim=256, stage_layers=2, progress_layers=2, num_heads=4, ffn_dim=512, stage_embedding_dim=32",
            "temporal_z_mlp_p": "hidden_dim=336, num_layers=2, num_heads=6, ffn_dim=672, stage_embedding_dim=64, progress_hidden_dim=336, progress_depth=3",
        },
    }

    with open(output_root / "baseline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)

    markdown_table = _format_markdown_table(summary_rows)
    with open(output_root / "baseline_summary.md", "w", encoding="utf-8") as f:
        f.write(markdown_table + "\n")

    logger.info("=== Strict Baseline Summary ===")
    with open(suite_log_path, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("FINAL SUMMARY\n")
        f.write("=" * 80 + "\n")
    for row in summary_rows:
        logger.info(
            "%s | params=%s | val_mse=%.6f | improvement=%.1f%% | phase_acc=%s | progress_mae=%s",
            row["method"],
            row["params"],
            row["val_mse"],
            row["improvement_pct"],
            row["phase_acc"],
            row["progress_mae"],
        )
        with open(suite_log_path, "a", encoding="utf-8") as f:
            f.write(
                f'{row["method"]} | params={row["params"]} | '
                f'val_mse={row["val_mse"]:.6f} | '
                f'improvement={row["improvement_pct"]:.1f}% | '
                f'phase_acc={row["phase_acc"]} | '
                f'progress_mae={row["progress_mae"]}\n'
            )
    logger.info("Saved summary JSON to %s", output_root / "baseline_summary.json")
    logger.info("Saved summary Markdown to %s", output_root / "baseline_summary.md")
    logger.info("Saved suite log to %s", suite_log_path)
    with open(suite_log_path, "a", encoding="utf-8") as f:
        f.write(f"summary_json={output_root / 'baseline_summary.json'}\n")
        f.write(f"summary_md={output_root / 'baseline_summary.md'}\n")
        f.write(f"suite_log={suite_log_path}\n")


if __name__ == "__main__":
    main()
