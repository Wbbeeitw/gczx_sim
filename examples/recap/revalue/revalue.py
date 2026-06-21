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

"""CLI for Revalue two-stage value re-estimation."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlinf.revalue.config import (
    RevalueConfig,
    RevalueDataConfig,
    RevalueFusionConfig,
    RevalueOutputConfig,
    RevalueRecapConfig,
    RevalueReturnsConfig,
    RevalueTrainConfig,
    RevalueValueConfig,
    RevalueZPConfig,
)
from rlinf.revalue.pipeline.run import run_revalue

logger = logging.getLogger(__name__)


def _to_dataclass(cfg: DictConfig) -> RevalueConfig:
    obj = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected mapping config, got {type(obj).__name__}")
    return RevalueConfig(
        method=obj.get("method", "shared_mlp_fusion"),
        stage=obj.get("stage", "all"),
        data=RevalueDataConfig(**(obj.get("data") or {})),
        value=RevalueValueConfig(**(obj.get("value") or {})),
        returns=RevalueReturnsConfig(**(obj.get("returns") or {})),
        output=RevalueOutputConfig(**(obj.get("output") or {})),
        zp=RevalueZPConfig(**(obj.get("zp") or {})),
        fusion=RevalueFusionConfig(**(obj.get("fusion") or {})),
        train=RevalueTrainConfig(**(obj.get("train") or {})),
        recap=RevalueRecapConfig(**(obj.get("recap") or {})),
    )


@hydra.main(version_base=None, config_path="config", config_name="revalue")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info("Revalue config:\n%s", OmegaConf.to_yaml(cfg))
    run_revalue(_to_dataclass(cfg))


if __name__ == "__main__":
    main()
