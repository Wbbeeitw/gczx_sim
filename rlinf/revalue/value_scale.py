"""Helpers for mapping between raw return space and critic value space."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _validate_ranges(
    *,
    return_min: float,
    return_max: float,
    value_min: float,
    value_max: float,
) -> tuple[float, float]:
    ret_range = float(return_max) - float(return_min)
    if ret_range <= 0.0:
        raise ValueError(
            f"Invalid return range [{return_min}, {return_max}]. "
            "return_max must be greater than return_min."
        )
    value_range = float(value_max) - float(value_min)
    if value_range <= 0.0:
        raise ValueError(
            f"Invalid value range [{value_min}, {value_max}]. "
            "value_max must be greater than value_min."
        )
    return ret_range, value_range


def map_returns_to_value_scale(
    returns: Any,
    *,
    return_min: float,
    return_max: float,
    value_min: float,
    value_max: float,
) -> Any:
    """Map raw returns into the configured critic value scale."""

    ret_range, value_range = _validate_ranges(
        return_min=return_min,
        return_max=return_max,
        value_min=value_min,
        value_max=value_max,
    )
    return (
        (returns - float(return_min)) / ret_range * value_range + float(value_min)
    )


def map_values_to_return_scale(
    values: Any,
    *,
    return_min: float,
    return_max: float,
    value_min: float,
    value_max: float,
) -> Any:
    """Map critic values back into raw return space."""

    ret_range, value_range = _validate_ranges(
        return_min=return_min,
        return_max=return_max,
        value_min=value_min,
        value_max=value_max,
    )
    return (
        (values - float(value_min)) / value_range * ret_range + float(return_min)
    )


def validate_atoms_match_value_scale(
    atoms: torch.Tensor | np.ndarray | None,
    *,
    value_min: float,
    value_max: float,
    source: str,
) -> None:
    """Raise when cached categorical supports disagree with the configured value scale."""

    if atoms is None:
        return

    if isinstance(atoms, torch.Tensor):
        atom_min = float(atoms.min().item())
        atom_max = float(atoms.max().item())
    else:
        atom_arr = np.asarray(atoms, dtype=np.float64).reshape(-1)
        if atom_arr.size == 0:
            raise ValueError(f"{source} atoms must not be empty.")
        atom_min = float(np.min(atom_arr))
        atom_max = float(np.max(atom_arr))

    scale = max(abs(float(value_min)), abs(float(value_max)), 1.0)
    atol = 1.0e-4 * scale
    if not np.isclose(atom_min, float(value_min), atol=atol, rtol=0.0) or not np.isclose(
        atom_max, float(value_max), atol=atol, rtol=0.0
    ):
        raise ValueError(
            f"{source} atoms range [{atom_min}, {atom_max}] does not match the "
            f"configured value range [{value_min}, {value_max}]."
        )
