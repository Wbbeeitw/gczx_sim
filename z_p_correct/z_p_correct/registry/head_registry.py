"""Registry for phase/progress head implementations."""

from typing import Dict, Type
import torch.nn as nn

_HEAD_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_head(name: str):
    """Decorator to register a phase/progress head class under ``name``.

    Args:
        name: Lower-case identifier used in configs and CLI.

    Returns:
        Decorator that registers the class.
    """

    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in _HEAD_REGISTRY:
            raise ValueError(f"Head '{name}' already registered.")
        _HEAD_REGISTRY[name] = cls
        return cls

    return decorator


def get_head_class(name: str) -> Type[nn.Module]:
    """Look up a registered head class by name."""
    name = name.lower()
    if name not in _HEAD_REGISTRY:
        raise KeyError(
            f"Unknown head '{name}'. Available: {sorted(_HEAD_REGISTRY.keys())}"
        )
    return _HEAD_REGISTRY[name]


def list_heads() -> list[str]:
    """Return all registered head names."""
    return sorted(_HEAD_REGISTRY.keys())
