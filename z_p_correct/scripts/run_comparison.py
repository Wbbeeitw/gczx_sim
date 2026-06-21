#!/usr/bin/env python
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

"""Launch multiple z/p head methods and compare their fusion improvements."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from z_p_correct.models import heads  # noqa: F401  # registers heads
from z_p_correct.registry import list_heads

logger = logging.getLogger(__name__)

DEFAULT_METHODS = [
    "base_shared_mlp",
    "temporal_phase_prior",
    "temporal_z_mlp_p",
    "our_phase_progress",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple z/p methods and compare.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config with methods and common args.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        choices=list_heads(),
        help="Methods to run. Overrides config.",
    )
    parser.add_argument("--features_dir", default=None)
    parser.add_argument("--advantages_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--train_script",
        default=str(Path(__file__).parent / "train_z_p_correct.py"),
        help="Path to train_z_p_correct.py.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def _load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_command(
    train_script: str,
    method: str,
    common: dict[str, Any],
    method_overrides: dict[str, Any] | None,
) -> list[str]:
    cmd = [sys.executable, train_script, "--head_type", method]

    merged = dict(common)
    if method_overrides:
        merged.update(method_overrides)

    for key, value in merged.items():
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    return cmd


def _collect_report(method: str, output_dir: Path) -> dict[str, Any] | None:
    report_path = output_dir / method / "eval_report.json"
    if not report_path.exists():
        return None
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.config:
        cfg = _load_config(args.config)
        methods = args.methods or cfg.get("methods", DEFAULT_METHODS)
        common = cfg.get("common", {})
        method_overrides = cfg.get("method_overrides", {})
        if args.features_dir:
            common["features_dir"] = args.features_dir
        if args.advantages_path:
            common["advantages_path"] = args.advantages_path
        if args.output_dir:
            common["output_dir"] = args.output_dir
    else:
        methods = args.methods or DEFAULT_METHODS
        common = {
            "features_dir": args.features_dir,
            "advantages_path": args.advantages_path,
            "output_dir": args.output_dir,
        }
        common = {k: v for k, v in common.items() if v is not None}
        method_overrides = {}

    missing = [k for k in ("features_dir", "advantages_path", "output_dir") if not common.get(k)]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    logger.info("Running comparison for methods: %s", methods)

    results: dict[str, dict[str, Any] | None] = {}
    for method in methods:
        logger.info("=" * 60)
        logger.info("Running method: %s", method)
        cmd = _build_command(
            train_script=args.train_script,
            method=method,
            common=common,
            method_overrides=method_overrides.get(method),
        )
        logger.info("Command: %s", " ".join(cmd))

        if args.dry_run:
            continue

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error("Method %s failed: %s", method, e)
            results[method] = None
            continue

        report = _collect_report(method, Path(common["output_dir"]))
        results[method] = report

    if args.dry_run:
        return

    # Summarize.
    logger.info("=" * 60)
    logger.info("Comparison summary")
    rows = []
    for method, report in results.items():
        if report is None:
            rows.append({
                "method": method,
                "raw_value_mse": None,
                "value_mse": None,
                "improvement_pct": None,
            })
        else:
            rows.append({
                "method": method,
                "raw_value_mse": report.get("raw_value_mse"),
                "value_mse": report.get("value_mse"),
                "improvement_pct": report.get("improvement_pct"),
            })

    for row in rows:
        logger.info(
            "%-30s raw_mse=%s fused_mse=%s improvement=%s%%",
            row["method"],
            "N/A" if row["raw_value_mse"] is None else f"{row['raw_value_mse']:.6f}",
            "N/A" if row["value_mse"] is None else f"{row['value_mse']:.6f}",
            "N/A" if row["improvement_pct"] is None else f"{row['improvement_pct']:.2f}",
        )

    summary_path = Path(common["output_dir"]) / "comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"results": rows}, f, indent=2)
    logger.info("Saved comparison summary to %s", summary_path)


if __name__ == "__main__":
    main()
