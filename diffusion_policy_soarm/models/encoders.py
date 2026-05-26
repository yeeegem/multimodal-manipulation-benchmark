"""Observation encoder: per-camera image encoder + proprioception → conditioning vector.

Phase 3 implementation target:
- ResNet18 per camera (ImageNet init optional), global average pool, linear head
  projecting to ``encoder.feature_dim``.
- Linear embedding of the proprioceptive state to ``encoder.state_embed_dim``.
- Concatenate all obs_horizon * (n_cameras * feature_dim + state_embed_dim)
  features into a single conditioning vector for the denoiser.

Design note: we share weights across the obs_horizon time dimension (the same
ResNet processes each frame); temporal structure is handled by the denoiser,
not the encoder. This matches the paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


class ImageEncoder(nn.Module):
    """Single-camera encoder: ResNet18 → feature vector.

    Args:
        cfg: Encoder config sub-tree.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        raise NotImplementedError("Phase 3")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images.

        Args:
            images: ``(B, C, H, W)`` float tensor, pixel values in [0, 1].

        Returns:
            Feature vectors of shape ``(B, feature_dim)``.
        """
        raise NotImplementedError("Phase 3")


class ObservationEncoder(nn.Module):
    """Encode full observation history into a flat conditioning vector.

    Args:
        cfg: Full resolved config (``encoder`` and ``model`` sub-trees used).
        camera_keys: Ordered list of camera names (determines concatenation order).
        state_dim: Dimensionality of the proprioceptive state vector.
    """

    def __init__(self, cfg: DictConfig, camera_keys: list[str], state_dim: int) -> None:
        super().__init__()
        raise NotImplementedError("Phase 3")

    @property
    def output_dim(self) -> int:
        """Dimensionality of the output conditioning vector."""
        raise NotImplementedError("Phase 3")

    def forward(
        self,
        images: dict[str, torch.Tensor],
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Encode observations into a conditioning vector.

        Args:
            images: Dict mapping camera key → ``(B, obs_horizon, C, H, W)`` tensor.
            state: ``(B, obs_horizon, state_dim)`` tensor.

        Returns:
            Conditioning vector of shape ``(B, output_dim)``.
        """
        raise NotImplementedError("Phase 3")
