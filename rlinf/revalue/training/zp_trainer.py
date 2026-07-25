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

"""Stage-1 trainer for the shared MLP z/p head."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from rlinf.revalue.metrics import phase_progress_metrics

logger = logging.getLogger(__name__)


@dataclass
class ZPHeadTrainerConfig:
    """Configuration for stage-1 z/p head training."""

    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    max_epochs: int = 100
    early_stop_patience: int = 10
    early_stop_delta: float = 1.0e-5
    lr_patience: int = 5
    phase_loss_weight: float = 1.0
    progress_loss_weight: float = 1.0
    global_progress_loss_weight: float = 0.0
    label_smoothing: float = 0.0
    max_grad_norm: float | None = None
    device: str = "cuda"


@dataclass
class TemporalZPHeadTrainerConfig:
    """Configuration for stage-1 temporal z->p head training."""

    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    max_epochs: int = 100
    early_stop_patience: int = 10
    early_stop_delta: float = 1.0e-5
    lr_patience: int = 5
    phase_loss_weight: float = 1.0
    progress_loss_weight: float = 1.0
    global_progress_loss_weight: float = 0.25
    label_smoothing: float = 0.05
    use_class_weights: bool = True
    progress_beta: float = 0.05
    max_grad_norm: float | None = None
    stage_only_epochs: int = 4
    gt_stage_prior_epochs: int = 6
    stage_prior_ramp_epochs: int = 10
    max_pred_stage_prior_weight: float = 0.7
    num_phases: int = 5
    device: str = "cuda"


class ZPHeadTrainer:
    """Train only the z/p head on frozen VLM features."""

    def __init__(self, head: nn.Module, cfg: ZPHeadTrainerConfig) -> None:
        self.head = head.to(cfg.device)
        self.cfg = cfg
        self.device = torch.device(cfg.device)
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

    def _loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        features = batch["features"].to(self.device)
        phase = batch["phase"].to(self.device)
        phase_progress = batch["phase_progress"].to(self.device)
        global_progress = batch["global_progress"].to(self.device)

        out = self.head(features)
        loss = self.cfg.phase_loss_weight * F.cross_entropy(
            out["phase_logits"],
            phase,
            label_smoothing=self.cfg.label_smoothing,
        )
        loss = loss + self.cfg.progress_loss_weight * F.mse_loss(
            out["phase_progress"], phase_progress
        )
        if self.cfg.global_progress_loss_weight > 0.0:
            loss = loss + self.cfg.global_progress_loss_weight * F.mse_loss(
                out["global_progress"], global_progress
            )
        return loss, out

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Run one z/p head training epoch."""
        self.head.train()
        total_loss = 0.0
        total_count = 0
        for batch in tqdm(train_loader, desc="train_zp", leave=False):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = self._loss(batch)
            loss.backward()
            if self.cfg.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.head.parameters(), self.cfg.max_grad_norm)
            optimizer.step()
            batch_size = int(batch["features"].shape[0])
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
        return total_loss / max(total_count, 1)

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> dict[str, float]:
        """Evaluate the z/p head."""
        self.head.eval()
        total_loss = 0.0
        total_count = 0
        all_logits: list[torch.Tensor] = []
        all_phase: list[torch.Tensor] = []
        all_progress_pred: list[torch.Tensor] = []
        all_progress: list[torch.Tensor] = []
        all_global_pred: list[torch.Tensor] = []
        all_global: list[torch.Tensor] = []

        for batch in data_loader:
            loss, out = self._loss(batch)
            batch_size = int(batch["features"].shape[0])
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
            all_logits.append(out["phase_logits"].detach().cpu())
            all_phase.append(batch["phase"].detach().cpu())
            all_progress_pred.append(out["phase_progress"].detach().cpu())
            all_progress.append(batch["phase_progress"].detach().cpu())
            all_global_pred.append(out["global_progress"].detach().cpu())
            all_global.append(batch["global_progress"].detach().cpu())

        metrics = phase_progress_metrics(
            phase_logits=torch.cat(all_logits, dim=0),
            phase_progress_pred=torch.cat(all_progress_pred, dim=0),
            global_progress_pred=torch.cat(all_global_pred, dim=0),
            phase_true=torch.cat(all_phase, dim=0),
            phase_progress_true=torch.cat(all_progress, dim=0),
            global_progress_true=torch.cat(all_global, dim=0),
        )
        metrics["loss"] = total_loss / max(total_count, 1)
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict[str, Any]:
        """Train with early stopping and restore the best state."""
        optimizer = torch.optim.AdamW(
            self.head.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = self._build_scheduler(optimizer)
        patience = 0
        history: list[dict[str, Any]] = []

        for epoch in range(1, self.cfg.max_epochs + 1):
            train_loss = self.train_epoch(train_loader, optimizer)
            val_metrics = self.evaluate(val_loader)
            scheduler.step(val_metrics["loss"])
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    **{f"val_{key}": value for key, value in val_metrics.items()},
                }
            )
            logger.info(
                "zp epoch=%d train_loss=%.6f val_loss=%.6f "
                "val_phase_acc=%.4f val_progress_mae=%.4f",
                epoch,
                train_loss,
                val_metrics["loss"],
                val_metrics["phase_acc"],
                val_metrics["progress_mae"],
            )

            if val_metrics["loss"] < self.best_val_loss - self.cfg.early_stop_delta:
                self.best_val_loss = val_metrics["loss"]
                self.best_epoch = epoch
                self.best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.head.state_dict().items()
                }
                patience = 0
            else:
                patience += 1

            if patience >= self.cfg.early_stop_patience:
                logger.info(
                    "early stopping z/p head at epoch=%d best_epoch=%d",
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


def _build_stage_prior(
    phase_center: torch.Tensor,
    phase_logits: torch.Tensor | None,
    *,
    epoch: int,
    cfg: TemporalZPHeadTrainerConfig,
    dtype: torch.dtype,
) -> torch.Tensor:
    gt_onehot = F.one_hot(
        phase_center.long(),
        num_classes=cfg.num_phases,
    ).to(dtype)
    if (
        epoch <= cfg.stage_only_epochs + cfg.gt_stage_prior_epochs
        or phase_logits is None
    ):
        return gt_onehot

    pred_probs = F.softmax(phase_logits.detach(), dim=-1).to(dtype)
    mix_progress = min(
        1.0,
        max(
            0.0,
            (
                epoch - cfg.stage_only_epochs - cfg.gt_stage_prior_epochs
            )
            / max(1, cfg.stage_prior_ramp_epochs),
        ),
    )
    pred_weight = cfg.max_pred_stage_prior_weight * mix_progress
    gt_weight = 1.0 - pred_weight
    return gt_weight * gt_onehot + pred_weight * pred_probs


def _compute_temporal_metrics(
    *,
    phase_logits: torch.Tensor,
    phase_progress_pred: torch.Tensor,
    global_progress_pred: torch.Tensor,
    phase_true: torch.Tensor,
    phase_progress_true: torch.Tensor,
    global_progress_true: torch.Tensor,
    num_phases: int,
) -> dict[str, float]:
    phase_pred = phase_logits.argmax(dim=-1)
    phase_acc = (phase_pred == phase_true).float().mean().item()

    per_phase_acc: dict[str, float] = {}
    per_phase_progress_mae: dict[str, float] = {}
    for phase_id in range(num_phases):
        mask = phase_true == phase_id
        if mask.any():
            per_phase_acc[f"phase_{phase_id}_acc"] = (
                (phase_pred[mask] == phase_true[mask]).float().mean().item()
            )
            per_phase_progress_mae[f"phase_{phase_id}_progress_mae"] = (
                phase_progress_pred[mask] - phase_progress_true[mask]
            ).abs().mean().item()
        else:
            per_phase_acc[f"phase_{phase_id}_acc"] = float("nan")
            per_phase_progress_mae[f"phase_{phase_id}_progress_mae"] = float("nan")

    macro_phase_acc = float(
        np.nanmean([per_phase_acc[f"phase_{phase_id}_acc"] for phase_id in range(num_phases)])
    )
    late_phase_acc = float(
        np.nanmean(
            [
                per_phase_acc.get("phase_3_acc", float("nan")),
                per_phase_acc.get("phase_4_acc", float("nan")),
            ]
        )
    )
    progress_mae = (phase_progress_pred - phase_progress_true).abs().mean().item()
    progress_mse = ((phase_progress_pred - phase_progress_true) ** 2).mean().item()
    global_mae = (global_progress_pred - global_progress_true).abs().mean().item()
    global_mse = ((global_progress_pred - global_progress_true) ** 2).mean().item()

    return {
        "phase_acc": phase_acc,
        "macro_phase_acc": macro_phase_acc,
        "late_phase_acc": late_phase_acc,
        "progress_mae": progress_mae,
        "progress_mse": progress_mse,
        "global_progress_mae": global_mae,
        "global_progress_mse": global_mse,
        **per_phase_acc,
        **per_phase_progress_mae,
    }


def _is_better_temporal_checkpoint(
    metrics: dict[str, float],
    best_metrics: dict[str, float] | None,
    *,
    min_delta: float,
    use_phase_head: bool = True,
    use_progress_head: bool = True,
) -> bool:
    if best_metrics is None:
        return True

    comparisons: list[tuple[str, bool]] = []
    if use_phase_head and use_progress_head:
        comparisons.append(("global_progress_mae", False))
    if use_progress_head:
        comparisons.append(("progress_mae", False))
    if use_phase_head:
        comparisons.extend(
            [("late_phase_acc", True), ("macro_phase_acc", True)]
        )
    comparisons.append(("loss", False))
    for key, higher_is_better in comparisons:
        current = float(metrics[key])
        best = float(best_metrics[key])
        if higher_is_better:
            if current > best + min_delta:
                return True
            if current < best - min_delta:
                return False
        else:
            if current < best - min_delta:
                return True
            if current > best + min_delta:
                return False
    return False


def _should_track_temporal_checkpoint(
    *,
    epoch: int,
    cfg: TemporalZPHeadTrainerConfig,
    use_phase_head: bool = True,
    use_progress_head: bool = True,
) -> bool:
    """Avoid selecting a best checkpoint before progress supervision starts."""
    if use_phase_head and use_progress_head:
        return epoch > cfg.stage_only_epochs
    return True


class TemporalZPHeadTrainer:
    """Train a temporal z->p head on windowed frozen VLM features."""

    def __init__(self, head: nn.Module, cfg: TemporalZPHeadTrainerConfig) -> None:
        self.head = head.to(cfg.device)
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_epoch = -1
        self.best_val_loss = float("inf")
        self.best_val_metrics: dict[str, float] | None = None

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

    def _phase_criterion(self, train_loader: DataLoader) -> nn.Module:
        if not self.cfg.use_class_weights:
            return nn.CrossEntropyLoss(label_smoothing=self.cfg.label_smoothing)

        phase_counts = torch.zeros(self.cfg.num_phases, dtype=torch.float32)
        for batch in train_loader:
            phase_center = batch["phase_center"]
            for phase_id in range(self.cfg.num_phases):
                phase_counts[phase_id] += (phase_center == phase_id).sum().item()
        weights = 1.0 / (phase_counts + 1.0)
        weights = weights / weights.sum() * self.cfg.num_phases
        logger.info("temporal z/p class weights: %s", weights.tolist())
        return nn.CrossEntropyLoss(
            weight=weights.to(self.device),
            label_smoothing=self.cfg.label_smoothing,
        )

    @property
    def _use_phase_head(self) -> bool:
        return bool(getattr(self.head, "use_phase_head", True))

    @property
    def _use_progress_head(self) -> bool:
        return bool(getattr(self.head, "use_progress_head", True))

    def _head_forward(
        self,
        feature_window: torch.Tensor,
        valid_mask: torch.Tensor | None,
        *,
        phase_center: torch.Tensor,
        epoch: int | None,
    ) -> dict[str, torch.Tensor]:
        head_kwargs: dict[str, torch.Tensor | None] = {"stage_prior": None}
        if getattr(self.head, "USES_VALID_MASK", False):
            head_kwargs["valid_mask"] = valid_mask
        if (
            epoch is None
            or not self._use_phase_head
            or not self._use_progress_head
            or epoch <= self.cfg.stage_only_epochs
        ):
            return self.head(feature_window, **head_kwargs)

        stage_out = self.head(feature_window, **head_kwargs)
        head_kwargs["stage_prior"] = _build_stage_prior(
            phase_center,
            stage_out["phase_logits"],
            epoch=epoch,
            cfg=self.cfg,
            dtype=feature_window.dtype,
        )
        return self.head(feature_window, **head_kwargs)

    def _active_loss(
        self,
        out: dict[str, torch.Tensor],
        *,
        phase_center: torch.Tensor,
        phase_progress_center: torch.Tensor,
        global_progress_center: torch.Tensor,
        phase_criterion: nn.Module,
        include_progress: bool,
    ) -> torch.Tensor:
        terms: list[torch.Tensor] = []
        if self._use_phase_head:
            terms.append(
                self.cfg.phase_loss_weight
                * phase_criterion(out["phase_logits"], phase_center)
            )
        if self._use_progress_head and include_progress:
            terms.append(
                self.cfg.progress_loss_weight
                * F.smooth_l1_loss(
                    out["phase_progress"],
                    phase_progress_center,
                    beta=self.cfg.progress_beta,
                )
            )
        if (
            self._use_phase_head
            and self._use_progress_head
            and include_progress
            and self.cfg.global_progress_loss_weight > 0.0
        ):
            terms.append(
                self.cfg.global_progress_loss_weight
                * F.smooth_l1_loss(
                    out["global_progress"],
                    global_progress_center,
                    beta=self.cfg.progress_beta,
                )
            )
        if not terms:
            raise RuntimeError("Temporal head has no active training objective")
        return torch.stack(terms).sum()

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        phase_criterion: nn.Module,
        *,
        epoch: int,
    ) -> float:
        self.head.train()
        total_loss = 0.0
        total_count = 0

        for batch in tqdm(train_loader, desc="train_zp", leave=False):
            feature_window = batch["feature_window"].to(self.device)
            valid_mask = batch.get("valid_mask")
            valid_mask = valid_mask.to(self.device) if valid_mask is not None else None
            phase_center = batch["phase_center"].to(self.device)
            phase_progress_center = batch["phase_progress_center"].to(self.device)
            global_progress_center = batch["global_progress_center"].to(self.device)

            optimizer.zero_grad(set_to_none=True)
            out = self._head_forward(
                feature_window,
                valid_mask,
                phase_center=phase_center,
                epoch=epoch,
            )
            loss = self._active_loss(
                out,
                phase_center=phase_center,
                phase_progress_center=phase_progress_center,
                global_progress_center=global_progress_center,
                phase_criterion=phase_criterion,
                include_progress=(
                    not self._use_phase_head
                    or epoch > self.cfg.stage_only_epochs
                ),
            )

            loss.backward()
            if self.cfg.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.head.parameters(), self.cfg.max_grad_norm)
            optimizer.step()

            batch_size = int(feature_window.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

        return total_loss / max(total_count, 1)

    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader,
        phase_criterion: nn.Module,
    ) -> dict[str, float]:
        self.head.eval()
        total_loss = 0.0
        total_count = 0
        all_logits: list[torch.Tensor] = []
        all_phase_true: list[torch.Tensor] = []
        all_progress_pred: list[torch.Tensor] = []
        all_progress_true: list[torch.Tensor] = []
        all_global_pred: list[torch.Tensor] = []
        all_global_true: list[torch.Tensor] = []

        for batch in data_loader:
            feature_window = batch["feature_window"].to(self.device)
            valid_mask = batch.get("valid_mask")
            valid_mask = valid_mask.to(self.device) if valid_mask is not None else None
            phase_center = batch["phase_center"].to(self.device)
            phase_progress_center = batch["phase_progress_center"].to(self.device)
            global_progress_center = batch["global_progress_center"].to(self.device)

            out = self._head_forward(
                feature_window,
                valid_mask,
                phase_center=phase_center,
                epoch=None,
            )
            loss = self._active_loss(
                out,
                phase_center=phase_center,
                phase_progress_center=phase_progress_center,
                global_progress_center=global_progress_center,
                phase_criterion=phase_criterion,
                include_progress=True,
            )

            batch_size = int(feature_window.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
            all_logits.append(out["phase_logits"].detach().cpu())
            all_phase_true.append(phase_center.detach().cpu())
            all_progress_pred.append(out["phase_progress"].detach().cpu())
            all_progress_true.append(phase_progress_center.detach().cpu())
            all_global_pred.append(out["global_progress"].detach().cpu())
            all_global_true.append(global_progress_center.detach().cpu())

        metrics = _compute_temporal_metrics(
            phase_logits=torch.cat(all_logits, dim=0),
            phase_progress_pred=torch.cat(all_progress_pred, dim=0),
            global_progress_pred=torch.cat(all_global_pred, dim=0),
            phase_true=torch.cat(all_phase_true, dim=0),
            phase_progress_true=torch.cat(all_progress_true, dim=0),
            global_progress_true=torch.cat(all_global_true, dim=0),
            num_phases=self.cfg.num_phases,
        )
        metrics["loss"] = total_loss / max(total_count, 1)
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict[str, Any]:
        phase_criterion = self._phase_criterion(train_loader)
        optimizer = torch.optim.AdamW(
            self.head.parameters(),
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
                phase_criterion,
                epoch=epoch,
            )
            val_metrics = self.evaluate(val_loader, phase_criterion)
            scheduler.step(val_metrics["loss"])
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    **{f"val_{key}": value for key, value in val_metrics.items()},
                }
            )
            logger.info(
                "zp epoch=%d train_loss=%.6f val_loss=%.6f "
                "val_phase_acc=%.4f val_progress_mae=%.4f val_global_mae=%.4f",
                epoch,
                train_loss,
                val_metrics["loss"],
                val_metrics["phase_acc"],
                val_metrics["progress_mae"],
                val_metrics["global_progress_mae"],
            )

            if not _should_track_temporal_checkpoint(
                epoch=epoch,
                cfg=self.cfg,
                use_phase_head=self._use_phase_head,
                use_progress_head=self._use_progress_head,
            ):
                continue

            if _is_better_temporal_checkpoint(
                val_metrics,
                self.best_val_metrics,
                min_delta=self.cfg.early_stop_delta,
                use_phase_head=self._use_phase_head,
                use_progress_head=self._use_progress_head,
            ):
                self.best_val_loss = float(val_metrics["loss"])
                self.best_epoch = epoch
                self.best_val_metrics = {
                    key: float(value) for key, value in val_metrics.items()
                }
                self.best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.head.state_dict().items()
                }
                patience = 0
            else:
                patience += 1

            if patience >= self.cfg.early_stop_patience:
                logger.info(
                    "early stopping temporal z/p head at epoch=%d best_epoch=%d",
                    epoch,
                    self.best_epoch,
                )
                break

        if self.best_state is not None:
            self.head.load_state_dict(self.best_state)

        return {
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "best_val_metrics": self.best_val_metrics,
            "history": history,
        }
