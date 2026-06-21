"""Registry for phase/progress heads."""

from .head_registry import register_head, get_head_class, list_heads

__all__ = ["register_head", "get_head_class", "list_heads"]
