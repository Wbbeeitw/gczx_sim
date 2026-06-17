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

"""One-click runner for the phase/progress probe pipeline.

Reads a YAML config, runs feature extraction -> head training -> prediction
analysis sequentially, and prints a summary.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _flag(name: str, value: Any) -> list[str]:
    """Return a CLI flag only if value is not None."""
    if value is None:
        return []
    return [f"--{name}", str(value)]


def _run_step(cmd: list[str]) -> None:
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="examples/recap/phase_progress_probe/configs/libero_task0.yaml",
    )
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--value_checkpoint", default=None)
    parser.add_argument("--siglip_path", default=None)
    parser.add_argument("--gemma3_path", default=None)
    parser.add_argument("--advantages_path", default=None)
    parser.add_argument("--output_root", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = _load_config(args.config)

    dataset_path = args.dataset_path or config["dataset_path"]
    value_checkpoint = args.value_checkpoint or config["value_checkpoint"]
    siglip_path = args.siglip_path or config["siglip_path"]
    gemma3_path = args.gemma3_path or config["gemma3_path"]
    advantages_path = args.advantages_path or config.get("advantages_path")
    output_root = args.output_root or config["output_root"]

    output_root = Path(output_root)
    features_dir = output_root / "features"
    head_dir = output_root / "head"
    analysis_dir = output_root / "analysis"

    features_dir.mkdir(parents=True, exist_ok=True)
    head_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    extract_cfg = config.get("extract", {})
    train_cfg = config.get("train", {})
    predict_cfg = config.get("predict", {})

    # Step 1: extract features.
    extract_cmd = [
        sys.executable,
        "examples/recap/phase_progress_probe/extract_features.py",
        "--dataset_path", dataset_path,
        "--value_checkpoint", value_checkpoint,
        "--siglip_path", siglip_path,
        "--gemma3_path", gemma3_path,
        "--output_dir", str(features_dir),
    ]
    for key in ("batch_size", "num_workers", "val_episode_ratio", "seed"):
        if key in extract_cfg:
            extract_cmd.extend(_flag(key, extract_cfg[key]))
    _run_step(extract_cmd)

    # Step 2: train head.
    train_cmd = [
        sys.executable,
        "examples/recap/phase_progress_probe/train_head.py",
        "--features_dir", str(features_dir),
        "--output_dir", str(head_dir),
    ]
    for key, value in train_cfg.items():
        train_cmd.extend(_flag(key, value))
    _run_step(train_cmd)

    # Step 3: predict and analyze.
    predict_cmd = [
        sys.executable,
        "examples/recap/phase_progress_probe/predict_and_analyze.py",
        "--features_dir", str(features_dir),
        "--head_checkpoint", str(head_dir / "head.pt"),
        "--output_dir", str(analysis_dir),
    ]
    for key, value in predict_cfg.items():
        predict_cmd.extend(_flag(key, value))
    predict_cmd.extend(_flag("advantages_path", advantages_path))
    _run_step(predict_cmd)

    logger.info("=" * 60)
    logger.info("Probe pipeline complete.")
    logger.info("  Features:  %s", features_dir)
    logger.info("  Head:      %s", head_dir / "head.pt")
    logger.info("  Analysis:  %s", analysis_dir)


if __name__ == "__main__":
    main()
