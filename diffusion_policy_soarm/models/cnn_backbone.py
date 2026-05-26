"""1-D temporal convolutional U-Net denoiser with FiLM conditioning.

Phase 3 implementation target:

Architecture (faithful to Chi et al. 2023, §4.2):
- Input: noisy action chunk x_t of shape (B, pred_horizon, action_dim).
  Transposed to (B, action_dim, pred_horizon) for 1-D convolution.
- Timestep embedding: sinusoidal → 2-layer MLP → scalar pair (γ, β) per block.
- Observation embedding: linear → scalar pair (γ, β) per block (FiLM conditioning).
- U-Net: `len(channels)` downsample stages, each a residual block + stride-2 conv.
  Skip connections concatenated on the up-path. Final 1-D conv → action_dim output.
- FiLM: each residual block applies feature-wise linear modulation
  x ← γ ⊙ x + β with (γ, β) derived from the concatenated [timestep_emb; obs_emb].

Design notes:
- FiLM parameters are produced by a shared linear layer inside each block, NOT a
  global broadcast, so the modulation is block-depth-aware.
- GroupNorm (not BatchNorm) is used inside residual blocks for training-stability
  with small batch sizes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal embedding for the diffusion timestep scalar."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        raise NotImplementedError("Phase 3")

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed integer timestep indices.

        Args:
            t: ``(B,)`` integer tensor of timestep indices.

        Returns:
            ``(B, dim)`` float embedding.
        """
        raise NotImplementedError("Phase 3")


class FiLMResBlock1d(nn.Module):
    """1-D residual block with FiLM conditioning.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        cond_dim: Dimension of the combined conditioning vector (timestep + obs).
        kernel_size: Convolution kernel size (must be odd).
        n_groups: Group count for GroupNorm.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 5,
        n_groups: int = 8,
    ) -> None:
        super().__init__()
        raise NotImplementedError("Phase 3")

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward pass with FiLM conditioning.

        Args:
            x: ``(B, in_channels, T)`` feature map.
            cond: ``(B, cond_dim)`` conditioning vector.

        Returns:
            ``(B, out_channels, T)`` feature map.
        """
        raise NotImplementedError("Phase 3")


class ConditionalUNet1d(nn.Module):
    """1-D temporal U-Net denoiser for action diffusion.

    Args:
        cfg: Full resolved config (``denoiser.cnn``, ``diffusion``, ``model`` sub-trees).
        action_dim: Dimensionality of one action step (6 for SO-101).
        cond_dim: Dimensionality of the observation conditioning vector.
    """

    def __init__(self, cfg: DictConfig, action_dim: int, cond_dim: int) -> None:
        super().__init__()
        raise NotImplementedError("Phase 3")

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Predict the noise added to the action chunk.

        Args:
            x: Noisy action chunk ``(B, pred_horizon, action_dim)``.
            t: Integer diffusion timestep indices ``(B,)``.
            cond: Observation conditioning vector ``(B, cond_dim)``.

        Returns:
            Predicted noise ``(B, pred_horizon, action_dim)``.
        """
        raise NotImplementedError("Phase 3")
