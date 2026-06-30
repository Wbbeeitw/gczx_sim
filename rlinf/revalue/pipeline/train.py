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

"""Training stages for Revalue."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from rlinf.revalue.data import build_feature_loaders, build_fusion_loaders
from rlinf.revalue.io import save_json
from rlinf.revalue.models import LogitFusionMLP, SharedMLPPhaseProgressHead
from rlinf.revalue.training import (
    FusionTrainer,
    FusionTrainerConfig,
    ZPHeadTrainer,
    ZPHeadTrainerConfig,
)
from rlinf.revalue.value_scale import validate_atoms_match_value_scale

logger = logging.getLogger(__name__)


@dataclass
class ZPTrainingConfig:
    """Stage-1 z/p training options."""

    features_dir: str
    output_dir: str
    num_phases: int = 5
    hidden_dim: int = 256
    dropout: float = 0.1
    trunk_depth: int = 1
    batch_size: int = 256
    num_workers: int = 0
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    max_epochs: int = 100
    early_stop_patience: int = 10
    device: str = "cuda"
    seed: int = 42


@dataclass
class FusionTrainingConfig:
    """Stage-2 fusion training options."""

    features_dir: str
    advantages_path: str
    zp_head_path: str
    output_dir: str
    return_min: float
    return_max: float
    value_min: float = -1.0
    value_max: float = 0.0
    num_bins: int = 201
    num_phases: int = 5
    fusion_hidden_dim: int = 256
    fusion_depth: int = 2
    fusion_dropout: float = 0.1
    alpha: float = 1.0
    batch_size: int = 256
    num_workers: int = 0
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    max_epochs: int = 100
    early_stop_patience: int = 10
    device: str = "cuda"
    seed: int = 42


def train_zp_head(cfg: ZPTrainingConfig) -> Path:
    """Train stage-1 z/p head and save ``zp_head.pt``."""
    torch.manual_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_feature_loaders(
        cfg.features_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    feature_dim = train_loader.dataset.feature_dim
    head = SharedMLPPhaseProgressHead(
        feature_dim=feature_dim,
        num_phases=cfg.num_phases,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        trunk_depth=cfg.trunk_depth,
    )
    trainer_cfg = ZPHeadTrainerConfig(
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        max_epochs=cfg.max_epochs,
        early_stop_patience=cfg.early_stop_patience,
        device=cfg.device,
    )
    trainer = ZPHeadTrainer(head, trainer_cfg)
    metrics = trainer.fit(train_loader, val_loader)
    checkpoint = {
        "state_dict": head.state_dict(),
        "feature_dim": feature_dim,
        "num_phases": cfg.num_phases,
        "hidden_dim": cfg.hidden_dim,
        "dropout": cfg.dropout,
        "trunk_depth": cfg.trunk_depth,
    }
    output_path = output_dir / "zp_head.pt"
    torch.save(checkpoint, output_path)
    save_json(metrics, output_dir / "metrics.json")
    logger.info("saved z/p head to %s", output_path)
    return output_path


def load_zp_head(path: str | Path, *, device: str = "cpu") -> SharedMLPPhaseProgressHead:
    """Load a z/p head checkpoint saved by :func:`train_zp_head`."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    head = SharedMLPPhaseProgressHead(
        feature_dim=int(checkpoint["feature_dim"]),
        num_phases=int(checkpoint["num_phases"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
        trunk_depth=int(checkpoint["trunk_depth"]),
    )
    head.load_state_dict(checkpoint["state_dict"])
    return head.to(device)


def train_fusion(cfg: FusionTrainingConfig) -> Path:
    """Train stage-2 fusion MLP with the z/p head frozen."""
    torch.manual_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_fusion_loaders(
        cfg.features_dir,
        cfg.advantages_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    atoms = train_loader.dataset.atoms
    if atoms is None:
        atoms = torch.linspace(cfg.value_min, cfg.value_max, cfg.num_bins)
    validate_atoms_match_value_scale(
        atoms,
        value_min=cfg.value_min,
        value_max=cfg.value_max,
        source=f"Feature cache {cfg.features_dir}",
    )
    if len(atoms) != cfg.num_bins:
        raise ValueError(
            f"atoms length {len(atoms)} does not match num_bins={cfg.num_bins}"
        )

    zp_head = load_zp_head(cfg.zp_head_path, device=cfg.device)
    fusion = LogitFusionMLP(
        num_bins=cfg.num_bins,
        num_phases=cfg.num_phases,
        hidden_dim=cfg.fusion_hidden_dim,
        dropout=cfg.fusion_dropout,
        depth=cfg.fusion_depth,
    )
    trainer_cfg = FusionTrainerConfig(
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        max_epochs=cfg.max_epochs,
        early_stop_patience=cfg.early_stop_patience,
        alpha=cfg.alpha,
        device=cfg.device,
    )
    trainer = FusionTrainer(zp_head, fusion, atoms, trainer_cfg)
    metrics = trainer.fit(
        train_loader,
        val_loader,
        return_min=cfg.return_min,
        return_max=cfg.return_max,
        value_min=cfg.value_min,
        value_max=cfg.value_max,
    )
    checkpoint = {
        "state_dict": fusion.state_dict(),
        "num_bins": cfg.num_bins,
        "num_phases": cfg.num_phases,
        "hidden_dim": cfg.fusion_hidden_dim,
        "dropout": cfg.fusion_dropout,
        "depth": cfg.fusion_depth,
        "alpha": cfg.alpha,
        "atoms": atoms.cpu(),
    }
    output_path = output_dir / "fusion.pt"
    torch.save(checkpoint, output_path)
    save_json(metrics, output_dir / "metrics.json")
    logger.info("saved fusion MLP to %s", output_path)
    return output_path


def load_fusion(
    path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[LogitFusionMLP, torch.Tensor, float]:
    """Load a fusion checkpoint saved by :func:`train_fusion`."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    fusion = LogitFusionMLP(
        num_bins=int(checkpoint["num_bins"]),
        num_phases=int(checkpoint["num_phases"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
        depth=int(checkpoint["depth"]),
    )
    fusion.load_state_dict(checkpoint["state_dict"])
    atoms = checkpoint["atoms"].float().to(device)
    alpha = float(checkpoint["alpha"])
    return fusion.to(device), atoms, alpha
