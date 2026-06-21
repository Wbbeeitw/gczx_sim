"""Integration with RLinf value critic and ReCap advantages."""

from .online_fusion_critic import OnlineFusionCritic, VLMFeatureExtractor
from .export_fused_advantages import export_fused_advantages

__all__ = [
    "OnlineFusionCritic",
    "VLMFeatureExtractor",
    "export_fused_advantages",
]
