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

"""Generic trainer for logit-space fusion MLP."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.fusion.logit_fusion import fuse_logits
from ..utils.metrics import compute_fusion_metrics, value_from_logits

logger = logging.getLogger(__name__)


@dataclass
class FusionTrainerConfig:
    """Configuration for fusion MLP training."""

    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 200
    early_stop_patience: int = 20
    early_stop_delta: float = 1e-5
    lr_patience: int = 10
    alpha: float = 1.0
    max_grad_norm: Optional[float] = None
    device: str = "cuda"
    history: list[dict[str, Any]] = field(default_factory=list)


class FusionTrainer:
    """Train a fusion MLP with a frozen phase/progress head.

    The head is kept in eval mode and its parameters are frozen. Only the fusion
    MLP parameters are optimized.
    """

    def __init__(
        self,
        head: nn.Module,
        fusion: nn.Module,
        atoms: torch.Tensor,
        cfg: FusionTrainerConfig,
    ):
        self.head = head.to(cfg.device)
        self.fusion = fusion.to(cfg.device)
        self.atoms = atoms.to(cfg.device)
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # Freeze head: this is the key two-stage training guarantee.
        self.head.eval()
        for param in self.head.parameters():
            param.requires_grad = False

        self.best_state: Optional[dict[str, Any]] = None
        self.best_val_loss = float("inf")
        self.best_epoch = -1

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

    def _unpack_batch(
        self, batch: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Unpack a batch from either dict or tuple format."""
        if isinstance(batch, dict):
            raw_logits = batch["raw_logits"]
            phase_probs = batch.get("phase_probs")
            phase_progress = batch.get("phase_progress")
            global_progress = batch.get("global_progress")
            raw_value = batch.get("raw_value")
            return_norm = batch["return_norm"]
            head_input = batch
        else:
            (
                raw_logits,
                phase_probs,
                phase_progress,
                global_progress,
                raw_value,
                return_norm,
            ) = batch
            head_input = None
        return (
            raw_logits,
            phase_probs,
            phase_progress,
            global_progress,
            raw_value,
            return_norm,
            head_input,
        )

    def _forward_head(
        self,
        head_input: Optional[dict[str, torch.Tensor]],
        phase_probs: Optional[torch.Tensor] = None,
        phase_progress: Optional[torch.Tensor] = None,
        global_progress: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Run the frozen head on the batch or use provided z/p signals.

        Supports both single-frame features and temporal feature windows.
        """
        if phase_probs is not None and phase_progress is not None and global_progress is not None:
            return {
                "phase_probs": phase_probs,
                "phase_progress": phase_progress,
                "global_progress": global_progress,
            }
        with torch.no_grad():
            assert head_input is not None
            if "feature_window" in head_input:
                return self.head(head_input["feature_window"])
            return self.head(head_input["features"])

    def _compute_loss(
        self,
        raw_logits: torch.Tensor,
        head_out: dict[str, torch.Tensor],
        return_norm: torch.Tensor,
    ) -> torch.Tensor:
        delta = self.fusion(
            raw_logits,
            head_out["phase_probs"],
            head_out["phase_progress"],
            head_out["global_progress"],
        )
        fused_logits = fuse_logits(raw_logits, delta, self.cfg.alpha)
        fused_value = value_from_logits(fused_logits, self.atoms)
        return F.mse_loss(fused_value, return_norm)

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Run one training epoch for the fusion MLP."""
        self.fusion.train()
        epoch_loss = 0.0
        epoch_samples = 0

        pbar = tqdm(train_loader, desc="Fusion train epoch")
        for batch in pbar:
            (
                raw_logits,
                phase_probs,
                phase_progress,
                global_progress,
                _raw_value,
                return_norm,
                head_input,
            ) = self._unpack_batch(batch)

            raw_logits = raw_logits.to(self.device)
            phase_probs = phase_probs.to(self.device)
            phase_progress = phase_progress.to(self.device)
            global_progress = global_progress.to(self.device)
            return_norm = return_norm.to(self.device)

            self._optimizer.zero_grad()
            head_out = self._forward_head(
                head_input, phase_probs, phase_progress, global_progress
            )
            loss = self._compute_loss(raw_logits, head_out, return_norm)

            loss.backward()
            if self.cfg.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.fusion.parameters(), self.cfg.max_grad_norm)
            self._optimizer.step()

            batch_size = raw_logits.shape[0]
            epoch_loss += loss.item() * batch_size
            epoch_samples += batch_size
            pbar.set_postfix({"loss": epoch_loss / epoch_samples})

        return epoch_loss / epoch_samples

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> dict[str, float]:
        """Evaluate the fusion MLP on a validation set."""
        self.fusion.eval()
        total_loss = 0.0
        total_samples = 0
        all_pred: list[torch.Tensor] = []
        all_true: list[torch.Tensor] = []
        all_raw: list[torch.Tensor] = []

        for batch in val_loader:
            (
                raw_logits,
                phase_probs,
                phase_progress,
                global_progress,
                raw_value,
                return_norm,
                head_input,
            ) = self._unpack_batch(batch)

            raw_logits = raw_logits.to(self.device)
            phase_probs = phase_probs.to(self.device)
            phase_progress = phase_progress.to(self.device)
            global_progress = global_progress.to(self.device)
            return_norm = return_norm.to(self.device)

            head_out = self._forward_head(
                head_input, phase_probs, phase_progress, global_progress
            )
            loss = self._compute_loss(raw_logits, head_out, return_norm)

            delta = self.fusion(
                raw_logits,
                head_out["phase_probs"],
                head_out["phase_progress"],
                head_out["global_progress"],
            )
            fused_logits = fuse_logits(raw_logits, delta, self.cfg.alpha)
            fused_value = value_from_logits(fused_logits, self.atoms)

            batch_size = raw_logits.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_pred.append(fused_value.cpu())
            all_true.append(return_norm.cpu())
            if raw_value is not None:
                all_raw.append(raw_value.cpu())

        pred_all = torch.cat(all_pred)
        true_all = torch.cat(all_true)
        raw_all = torch.cat(all_raw) if all_raw else true_all.clone()

        metrics = compute_fusion_metrics(pred_all, true_all, raw_all)
        metrics["loss"] = total_loss / total_samples
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict[str, Any]:
        """Train the fusion MLP with early stopping.

        Returns:
            Dict with best_epoch, best_val_loss, history.
        """
        self._optimizer = torch.optim.AdamW(
            self.fusion.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = self._build_scheduler(self._optimizer)

        patience_counter = 0
        history: list[dict[str, Any]] = []

        for epoch in range(1, self.cfg.max_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            scheduler.step(val_metrics["loss"])

            logger.info(
                "Fusion epoch %d | train_loss=%.4f | val_loss=%.4f | val_value_mse=%.4f | "
                "val_improvement=%.2f%%",
                epoch,
                train_loss,
                val_metrics["loss"],
                val_metrics["value_mse"],
                val_metrics["improvement_pct"],
            )

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
            )

            if val_metrics["loss"] < self.best_val_loss - self.cfg.early_stop_delta:
                self.best_val_loss = val_metrics["loss"]
                self.best_epoch = epoch
                self.best_state = {k: v.cpu().clone() for k, v in self.fusion.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.cfg.early_stop_patience:
                logger.info(
                    "Fusion early stopping at epoch %d (best epoch %d)",
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
