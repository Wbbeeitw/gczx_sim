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

"""Model-weights-only initialization checkpoint loading for CFG models.

Iterated policy training (e.g. ReCap rounds) needs to start from the
previous round policy's weights while keeping a fresh optimizer, a fresh
lr schedule and ``global_step=0``. ``runner.resume_dir`` cannot express
this because it also restores training state, and pointing
``actor.model.model_path`` at a checkpoint directory fails norm-stats
loading. These helpers overlay a ``full_weights.pt`` checkpoint on top of
an already-constructed model and log an audit trail.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)

_INIT_WEIGHTS_RELPATHS = (
    os.path.join("actor", "model_state_dict", "full_weights.pt"),
    os.path.join("model_state_dict", "full_weights.pt"),
)


def resolve_init_checkpoint_file(init_checkpoint_path: str) -> str:
    """Resolve an init checkpoint file or ``global_step_*`` dir to a .pt file."""
    path = os.path.expanduser(str(init_checkpoint_path))
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        for relpath in _INIT_WEIGHTS_RELPATHS:
            candidate = os.path.join(path, relpath)
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            f"init checkpoint directory {path} contains no full_weights.pt "
            f"(looked for: {', '.join(_INIT_WEIGHTS_RELPATHS)})"
        )
    raise FileNotFoundError(f"init checkpoint path does not exist: {path}")


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    """Return the hex sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_init_checkpoint(model, init_checkpoint_path: str) -> dict[str, Any]:
    """Overlay model weights from an init checkpoint and log an audit trail.

    The overlay uses ``strict=False``, matching the base-weight loading
    convention in ``get_model``. A checkpoint that matches zero model keys
    is treated as an error rather than a benign no-op, since that means the
    file is unrelated to this model.
    """
    weights_file = resolve_init_checkpoint_file(init_checkpoint_path)
    file_sha256 = sha256_file(weights_file)
    state_dict = torch.load(weights_file, map_location="cpu")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(
            f"init checkpoint {weights_file} did not contain a non-empty "
            f"state dict (got {type(state_dict).__name__})"
        )
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)
    matched_keys = len(state_dict) - len(unexpected_keys)
    if matched_keys == 0:
        raise ValueError(
            f"init checkpoint {weights_file} matched 0 model keys; "
            "refusing to initialize from an unrelated checkpoint"
        )
    logger.info(
        "loaded init checkpoint %s sha256=%s matched=%d missing=%d unexpected=%d",
        weights_file,
        file_sha256[:16],
        matched_keys,
        len(missing_keys),
        len(unexpected_keys),
    )
    if missing_keys:
        logger.info("init checkpoint missing key examples: %s", missing_keys[:5])
    if unexpected_keys:
        logger.info(
            "init checkpoint unexpected key examples: %s", unexpected_keys[:5]
        )
    return {
        "init_checkpoint_path": str(init_checkpoint_path),
        "init_checkpoint_file": weights_file,
        "init_checkpoint_sha256": file_sha256,
        "matched_keys": matched_keys,
        "missing_keys": len(missing_keys),
        "unexpected_keys": len(unexpected_keys),
    }
