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

"""Stage-2 trainer for the fusion MLP with a frozen z/p head."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from rlinf.revalue.models.fusion import fuse_logits, value_from_logits
from rlinf.revalue.metrics import value_regression_metrics
from rlinf.revalue.value_scale import map_returns_to_value_scale

logger = logging.getLogger(__name__)


@dataclass
class FusionTrainerConfig:
    """Configuration for stage-2 fusion MLP training."""

    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    max_epochs: int = 100
    early_stop_patience: int = 10
    early_stop_delta: float = 1.0e-5
    lr_patience: int = 5
    alpha: float = 1.0
    max_grad_norm: float | None = None
    device: str = "cuda"


class FusionTrainer:
    """Train only the fusion MLP while the z/p head is frozen."""

    def __init__(
        self,
        zp_head: nn.Module,
        fusion: nn.Module,
        atoms: torch.Tensor,
        cfg: FusionTrainerConfig,
    ) -> None:
        self.zp_head = zp_head.to(cfg.device)
        self.fusion = fusion.to(cfg.device)
        self.atoms = atoms.to(cfg.device)
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.zp_head.eval()
        for param in self.zp_head.parameters():
            param.requires_grad = False

        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_epoch = -1
        self.best_val_loss = float("inf")

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        kwargs: dict[str, Any] = {
            "mode": "min",
            "factor": 0.5,
            "patience": self.cfg.lr_patience,
        }
        if "verbose" in inspect.signature(
            torch.optim.lr_scheduler.ReduceLROnPlateau.__init__
        ).parameters:
            kwargs["verbose"] = True
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)

    @staticmethod
    def _normalize_return(
        returns: torch.Tensor,
        *,
        return_min: float,
        return_max: float,
        value_min: float,
        value_max: float,
    ) -> torch.Tensor:
        return map_returns_to_value_scale(
            returns,
            return_min=return_min,
            return_max=return_max,
            value_min=value_min,
            value_max=value_max,
        )

    def _forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_min: float,
        return_max: float,
        value_min: float,
        value_max: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if "raw_logits" not in batch:
            raise ValueError(
                "Fusion training requires raw_logits in the feature cache. "
                "Regenerate features from a base advantage tag created with "
                "advantage.save_value_distribution=true."
            )
        if "return" not in batch:
            raise ValueError("Fusion training batch missing raw return targets.")

        raw_logits = batch["raw_logits"].to(self.device)
        returns = batch["return"].to(self.device)
        raw_value = batch.get("raw_value")
        raw_value = raw_value.to(self.device) if raw_value is not None else None
        target_value = self._normalize_return(
            returns,
            return_min=return_min,
            return_max=return_max,
            value_min=value_min,
            value_max=value_max,
        )

        with torch.no_grad():
            if "feature_window" in batch:
                head_out = self.zp_head(
                    batch["feature_window"].to(self.device),
                    stage_prior=None,
                )
            else:
                head_out = self.zp_head(batch["features"].to(self.device))

        delta_logits = self.fusion(
            raw_logits,
            head_out["phase_probs"],
            head_out["phase_progress"],
            head_out["global_progress"],
            phase_progress_all=head_out.get("phase_progress_all"),
        )
        fused_logits = fuse_logits(raw_logits, delta_logits, self.cfg.alpha)
        fused_value = value_from_logits(fused_logits, self.atoms)
        loss = F.mse_loss(fused_value, target_value)
        if raw_value is None:
            raw_value = value_from_logits(raw_logits, self.atoms)
        return loss, fused_value, target_value, raw_value

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        *,
        return_min: float,
        return_max: float,
        value_min: float,
        value_max: float,
    ) -> float:
        """Run one fusion training epoch."""
        self.zp_head.eval()
        self.fusion.train()
        total_loss = 0.0
        total_count = 0
        for batch in tqdm(train_loader, desc="train_fusion", leave=False):
            optimizer.zero_grad(set_to_none=True)
            loss, _, _, _ = self._forward(
                batch,
                return_min=return_min,
                return_max=return_max,
                value_min=value_min,
                value_max=value_max,
            )
            loss.backward()
            if self.cfg.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(
                    self.fusion.parameters(), self.cfg.max_grad_norm
                )
            optimizer.step()
            batch_size = int(batch["raw_logits"].shape[0])
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
        return total_loss / max(total_count, 1)

    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader,
        *,
        return_min: float,
        return_max: float,
        value_min: float,
        value_max: float,
    ) -> dict[str, float]:
        """Evaluate fusion value prediction."""
        self.zp_head.eval()
        self.fusion.eval()
        total_loss = 0.0
        total_count = 0
        all_pred: list[torch.Tensor] = []
        all_target: list[torch.Tensor] = []
        all_raw: list[torch.Tensor] = []

        for batch in data_loader:
            loss, fused_value, target_value, raw_value = self._forward(
                batch,
                return_min=return_min,
                return_max=return_max,
                value_min=value_min,
                value_max=value_max,
            )
            batch_size = int(batch["features"].shape[0])
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
            all_pred.append(fused_value.detach().cpu())
            all_target.append(target_value.detach().cpu())
            all_raw.append(raw_value.detach().cpu())

        metrics = value_regression_metrics(
            pred_value=torch.cat(all_pred, dim=0),
            target_value=torch.cat(all_target, dim=0),
            raw_value=torch.cat(all_raw, dim=0),
        )
        metrics["loss"] = total_loss / max(total_count, 1)
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        return_min: float,
        return_max: float,
        value_min: float,
        value_max: float,
    ) -> dict[str, Any]:
        """Train fusion with the z/p head frozen."""
        frozen_params = [param for param in self.zp_head.parameters()]
        if any(param.requires_grad for param in frozen_params):
            raise RuntimeError("z/p head must be frozen before fusion training.")

        optimizer = torch.optim.AdamW(
            self.fusion.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = self._build_scheduler(optimizer)
        patience = 0
        history: list[dict[str, Any]] = []

        for epoch in range(1, self.cfg.max_epochs + 1):
            train_loss = self.train_epoch(
                train_loader,
                optimizer,
                return_min=return_min,
                return_max=return_max,
                value_min=value_min,
                value_max=value_max,
            )
            val_metrics = self.evaluate(
                val_loader,
                return_min=return_min,
                return_max=return_max,
                value_min=value_min,
                value_max=value_max,
            )
            scheduler.step(val_metrics["loss"])
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    **{f"val_{key}": value for key, value in val_metrics.items()},
                }
            )
            logger.info(
                "fusion epoch=%d train_loss=%.6f val_loss=%.6f "
                "val_mse=%.6f raw_mse=%.6f improvement=%.2f%%",
                epoch,
                train_loss,
                val_metrics["loss"],
                val_metrics["value_mse"],
                val_metrics.get("raw_value_mse", 0.0),
                val_metrics.get("improvement_pct", 0.0),
            )

            if val_metrics["loss"] < self.best_val_loss - self.cfg.early_stop_delta:
                self.best_val_loss = val_metrics["loss"]
                self.best_epoch = epoch
                self.best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.fusion.state_dict().items()
                }
                patience = 0
            else:
                patience += 1

            if patience >= self.cfg.early_stop_patience:
                logger.info(
                    "early stopping fusion at epoch=%d best_epoch=%d",
                    epoch,
                    self.best_epoch,
                )
                break

        if self.best_state is not None:
            self.fusion.load_state_dict(self.best_state)

        return {
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "history": history,
        }
