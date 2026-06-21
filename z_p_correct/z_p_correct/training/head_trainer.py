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

"""Generic trainer for phase/progress heads."""

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

from ..utils.metrics import compute_head_metrics

logger = logging.getLogger(__name__)


@dataclass
class HeadTrainerConfig:
    """Configuration for head training."""

    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 200
    early_stop_patience: int = 20
    early_stop_delta: float = 1e-5
    lr_patience: int = 10
    phase_loss_weight: float = 1.0
    progress_loss_weight: float = 1.0
    global_progress_loss_weight: float = 0.0
    label_smoothing: float = 0.0
    use_class_weights: bool = False
    max_grad_norm: Optional[float] = None
    device: str = "cuda"
    log_interval: int = 10
    history: list[dict[str, Any]] = field(default_factory=list)


class HeadTrainer:
    """Train a phase/progress head with cross-entropy + MSE losses."""

    def __init__(self, head: nn.Module, cfg: HeadTrainerConfig):
        self.head = head.to(cfg.device)
        self.cfg = cfg
        self.device = torch.device(cfg.device)
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

    def _compute_loss(
        self,
        out: dict[str, torch.Tensor],
        phase: torch.Tensor,
        phase_progress: torch.Tensor,
        global_progress: torch.Tensor,
        phase_criterion: nn.Module,
        progress_criterion: nn.Module,
        global_progress_criterion: nn.Module,
    ) -> torch.Tensor:
        loss = (
            self.cfg.phase_loss_weight * phase_criterion(out["phase_logits"], phase)
            + self.cfg.progress_loss_weight
            * progress_criterion(out["phase_progress"], phase_progress)
        )
        if self.cfg.global_progress_loss_weight > 0.0:
            loss = loss + self.cfg.global_progress_loss_weight * global_progress_criterion(
                out["global_progress"], global_progress
            )
        return loss

    def _get_head_input(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return either ``features`` or ``feature_window`` depending on cache config."""
        if "feature_window" in batch:
            return batch["feature_window"].to(self.device)
        return batch["features"].to(self.device)

    def _build_criteria(self, train_loader: DataLoader) -> tuple[nn.Module, nn.Module, nn.Module]:
        if self.cfg.use_class_weights:
            phase_counts = torch.zeros(self.head.num_phases, dtype=torch.float32)
            for batch in train_loader:
                phase = batch["phase"]
                for ph in range(self.head.num_phases):
                    phase_counts[ph] += (phase == ph).sum().item()
            weights = 1.0 / (phase_counts + 1.0)
            weights = weights / weights.sum() * self.head.num_phases
            logger.info("Class weights: %s", weights.tolist())
            phase_criterion = nn.CrossEntropyLoss(
                weight=weights.to(self.device), label_smoothing=self.cfg.label_smoothing
            )
        else:
            phase_criterion = nn.CrossEntropyLoss(label_smoothing=self.cfg.label_smoothing)

        return phase_criterion, nn.MSELoss(), nn.MSELoss()

    def train_epoch(self, train_loader: DataLoader, criteria: tuple[nn.Module, nn.Module, nn.Module]) -> float:
        """Run one training epoch."""
        self.head.train()
        phase_criterion, progress_criterion, global_criterion = criteria
        epoch_loss = 0.0
        epoch_samples = 0

        pbar = tqdm(train_loader, desc=f"Head train epoch")
        for batch in pbar:
            head_input = self._get_head_input(batch)
            phase = batch["phase"].to(self.device)
            phase_progress = batch["phase_progress"].to(self.device)
            global_progress = batch["global_progress"].to(self.device)

            self._optimizer.zero_grad()
            out = self.head(head_input)
            loss = self._compute_loss(
                out, phase, phase_progress, global_progress,
                phase_criterion, progress_criterion, global_criterion
            )

            loss.backward()
            if self.cfg.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.head.parameters(), self.cfg.max_grad_norm)
            self._optimizer.step()

            batch_size = head_input.shape[0]
            epoch_loss += loss.item() * batch_size
            epoch_samples += batch_size
            pbar.set_postfix({"loss": epoch_loss / epoch_samples})

        return epoch_loss / epoch_samples

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, criteria: tuple[nn.Module, nn.Module, nn.Module]) -> dict[str, Any]:
        """Evaluate the head on a validation set."""
        self.head.eval()
        phase_criterion, progress_criterion, global_criterion = criteria
        total_loss = 0.0
        total_samples = 0
        all_out: list[dict[str, torch.Tensor]] = []
        all_phase: list[torch.Tensor] = []
        all_progress: list[torch.Tensor] = []
        all_global: list[torch.Tensor] = []

        for batch in val_loader:
            head_input = self._get_head_input(batch)
            phase = batch["phase"].to(self.device)
            phase_progress = batch["phase_progress"].to(self.device)
            global_progress = batch["global_progress"].to(self.device)

            out = self.head(head_input)
            loss = self._compute_loss(
                out, phase, phase_progress, global_progress,
                phase_criterion, progress_criterion, global_criterion
            )

            batch_size = head_input.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_out.append({k: v.cpu() for k, v in out.items()})
            all_phase.append(phase.cpu())
            all_progress.append(phase_progress.cpu())
            all_global.append(global_progress.cpu())

        phase_logits = torch.cat([o["phase_logits"] for o in all_out])
        phase_progress_pred = torch.cat([o["phase_progress"] for o in all_out])
        global_progress_pred = torch.cat([o["global_progress"] for o in all_out])
        phase_true = torch.cat(all_phase)
        phase_progress_true = torch.cat(all_progress)
        global_progress_true = torch.cat(all_global)

        metrics = compute_head_metrics(
            phase_logits=phase_logits,
            phase_progress_pred=phase_progress_pred,
            global_progress_pred=global_progress_pred,
            phase_true=phase_true,
            phase_progress_true=phase_progress_true,
            global_progress_true=global_progress_true,
            num_phases=self.head.num_phases,
        )
        metrics["loss"] = total_loss / total_samples
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict[str, Any]:
        """Train the head with early stopping.

        Returns:
            Dict with best_epoch, best_val_loss, history.
        """
        self._optimizer = torch.optim.AdamW(
            self.head.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = self._build_scheduler(self._optimizer)
        criteria = self._build_criteria(train_loader)

        patience_counter = 0
        history: list[dict[str, Any]] = []

        for epoch in range(1, self.cfg.max_epochs + 1):
            train_loss = self.train_epoch(train_loader, criteria)
            val_metrics = self.evaluate(val_loader, criteria)
            scheduler.step(val_metrics["loss"])

            logger.info(
                "Head epoch %d | train_loss=%.4f | val_loss=%.4f | val_phase_acc=%.4f | "
                "val_progress_mae=%.4f | val_global_mae=%.4f",
                epoch,
                train_loss,
                val_metrics["loss"],
                val_metrics["phase_acc"],
                val_metrics["progress_mae"],
                val_metrics["global_progress_mae"],
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
                self.best_state = {k: v.cpu().clone() for k, v in self.head.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.cfg.early_stop_patience:
                logger.info(
                    "Head early stopping at epoch %d (best epoch %d)",
                    epoch,
                    self.best_epoch,
                )
                break

        if self.best_state is not None:
            self.head.load_state_dict(self.best_state)

        return {
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "history": history,
        }
