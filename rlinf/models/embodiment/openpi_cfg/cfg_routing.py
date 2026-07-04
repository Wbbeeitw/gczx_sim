"""Pure routing helpers for CFG and CSA-CFG training/inference."""

from __future__ import annotations

from typing import Any

import torch

QUALITY_LABEL_POSITIVE = 0
QUALITY_LABEL_NEUTRAL = 1
QUALITY_LABEL_NEGATIVE = 2


def compute_binary_cfg_routing_masks(
    advantage: torch.Tensor,
    *,
    positive_only_conditional: bool,
    unconditional_prob: float,
    random_values: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Backward-compatible routing for legacy boolean advantage CFG."""

    advantage = advantage.to(dtype=torch.bool)
    batch_size = advantage.shape[0]
    device = advantage.device

    if random_values is None:
        random_values = torch.rand(batch_size, device=device)
    else:
        random_values = random_values.to(device=device)

    positive_mask = advantage
    negative_mask = ~positive_mask

    if positive_only_conditional:
        positive_conditional_mask = positive_mask & (random_values > unconditional_prob)
        negative_conditional_mask = torch.zeros_like(positive_mask)
    else:
        guidance_mask = random_values > unconditional_prob
        positive_conditional_mask = positive_mask & guidance_mask
        negative_conditional_mask = negative_mask & guidance_mask

    conditional_mask = positive_conditional_mask | negative_conditional_mask
    positive_unconditional_mask = positive_mask & ~positive_conditional_mask
    negative_unconditional_mask = negative_mask & ~negative_conditional_mask

    return {
        "positive_mask": positive_mask,
        "negative_mask": negative_mask,
        "conditional_mask": conditional_mask,
        "positive_conditional_mask": positive_conditional_mask,
        "positive_unconditional_mask": positive_unconditional_mask,
        "negative_conditional_mask": negative_conditional_mask,
        "negative_unconditional_mask": negative_unconditional_mask,
    }


def compute_csa_cfg_routing_masks(
    quality_label: torch.Tensor,
    *,
    unconditional_prob: float,
    positive_prompt_prob: float,
    unconditional_random: torch.Tensor | None = None,
    positive_prompt_random: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Routing for CSA-CFG with explicit positive/neutral/negative conditions."""

    quality_label = quality_label.to(dtype=torch.long)
    batch_size = quality_label.shape[0]
    device = quality_label.device

    if unconditional_random is None:
        unconditional_random = torch.rand(batch_size, device=device)
    else:
        unconditional_random = unconditional_random.to(device=device)
    if positive_prompt_random is None:
        positive_prompt_random = torch.rand(batch_size, device=device)
    else:
        positive_prompt_random = positive_prompt_random.to(device=device)

    positive_label_mask = quality_label == QUALITY_LABEL_POSITIVE
    neutral_label_mask = quality_label == QUALITY_LABEL_NEUTRAL
    negative_label_mask = quality_label == QUALITY_LABEL_NEGATIVE

    unconditional_mask = unconditional_random < unconditional_prob
    conditional_mask = ~unconditional_mask

    positive_conditional_mask = (
        conditional_mask
        & positive_label_mask
        & (positive_prompt_random < positive_prompt_prob)
    )
    positive_to_neutral_mask = (
        conditional_mask
        & positive_label_mask
        & ~positive_conditional_mask
    )
    neutral_conditional_mask = (
        (conditional_mask & neutral_label_mask)
        | positive_to_neutral_mask
    )
    negative_conditional_mask = conditional_mask & negative_label_mask

    positive_unconditional_mask = unconditional_mask & positive_label_mask
    neutral_unconditional_mask = unconditional_mask & neutral_label_mask
    negative_unconditional_mask = unconditional_mask & negative_label_mask

    return {
        "positive_label_mask": positive_label_mask,
        "neutral_label_mask": neutral_label_mask,
        "negative_label_mask": negative_label_mask,
        "conditional_mask": conditional_mask,
        "unconditional_mask": unconditional_mask,
        "positive_conditional_mask": positive_conditional_mask,
        "positive_to_neutral_mask": positive_to_neutral_mask,
        "neutral_conditional_mask": neutral_conditional_mask,
        "negative_conditional_mask": negative_conditional_mask,
        "positive_unconditional_mask": positive_unconditional_mask,
        "neutral_unconditional_mask": neutral_unconditional_mask,
        "negative_unconditional_mask": negative_unconditional_mask,
    }


def masked_loss_sum(per_sample_loss: torch.Tensor, mask: torch.Tensor) -> float:
    """Return the detached summed loss over a boolean mask."""

    if mask.numel() == 0 or not torch.any(mask):
        return 0.0
    return (per_sample_loss.detach() * mask.float()).sum().item()


def build_weighted_loss(
    per_sample_loss: torch.Tensor,
    sample_weight: torch.Tensor | None,
) -> torch.Tensor:
    """Return a normalized weighted mean when weights are provided."""

    if sample_weight is None:
        return per_sample_loss.mean()
    weights = sample_weight.to(device=per_sample_loss.device, dtype=per_sample_loss.dtype)
    weight_sum = torch.clamp(weights.sum(), min=torch.finfo(per_sample_loss.dtype).eps)
    return (per_sample_loss * weights).sum() / weight_sum


def summarize_weight_metrics(sample_weight: torch.Tensor | None) -> dict[str, Any]:
    """Small helper to keep optional weight logging consistent."""

    if sample_weight is None:
        return {}
    return {
        "cfg_loss_weight_mean": float(sample_weight.mean().item()),
        "cfg_loss_weight_min": float(sample_weight.min().item()),
        "cfg_loss_weight_max": float(sample_weight.max().item()),
    }
