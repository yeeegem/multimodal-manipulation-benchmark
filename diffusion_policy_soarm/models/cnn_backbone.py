"""1-D temporal convolutional U-Net denoiser with FiLM conditioning.

Architecture overview (default channels=[256, 512, 1024], T_p=16):

  Input x_t: (B, T_p=16, action_dim=6)
  ↓ transpose to (B, action_dim, T_p)
  ↓ input_proj: Conv1d(action_dim → 256)

  Encoder:
    level 0 — 2×FiLMResBlock(256, cond) → skip₀ at (B, 256, 16)
              → Downsample Conv1d stride-2 → (B, 512, 8)
    level 1 — 2×FiLMResBlock(512, cond) → skip₁ at (B, 512, 8)
              → Downsample Conv1d stride-2 → (B, 1024, 4)

  Bottleneck:
    2×FiLMResBlock(1024, cond) at (B, 1024, 4)

  Decoder:
    level 1 — ConvTranspose1d stride-2 → (B, 512, 8)
              cat(skip₁) → (B, 1024, 8)
              2×FiLMResBlock(1024→512, cond)
    level 0 — ConvTranspose1d stride-2 → (B, 256, 16)
              cat(skip₀) → (B, 512, 16)
              2×FiLMResBlock(512→256, cond)

  ↓ output_proj: Conv1d(256 → action_dim)
  ↓ transpose back to (B, T_p, action_dim)

Conditioning (FiLM):
  - Diffusion timestep t → sinusoidal embedding → 2-layer MLP → t_emb (256-d)
  - Observation vector c → Linear + SiLU → obs_emb (256-d)
  - film_cond = cat([t_emb, obs_emb])  (512-d)
  - Each FiLMResBlock: Linear(512 → 2×out_channels) → (γ, β) for feature-wise modulation

The observation projection down to 256-d is intentional: it creates a learned
bottleneck between the high-dimensional observation encoding (≈2176-d) and the
per-block FiLM parameters, reducing the parameter count in each block's film_proj
from O(obs_dim × channels) to O(256 × channels).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------

class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal positional encoding for the diffusion timestep.

    Maps integer t ∈ [0, T) to a ``dim``-dimensional embedding via:
      emb[2i]   = sin(t / 10000^(2i/dim))
      emb[2i+1] = cos(t / 10000^(2i/dim))
    then passes through a 2-layer MLP (SiLU activation).

    Args:
        dim: Output embedding dimension (must be even).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0, "dim must be even for sinusoidal embedding"
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed integer timestep indices.

        Args:
            t: ``(B,)`` integer timestep tensor.

        Returns:
            ``(B, dim)`` float embedding.
        """
        half = self.dim // 2
        # Frequencies: 1 / 10000^(2i/dim) for i in [0, half)
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t.float()[:, None] * freqs[None, :]   # (B, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# FiLM residual block
# ---------------------------------------------------------------------------

class FiLMResBlock1d(nn.Module):
    """1-D residual block with Feature-wise Linear Modulation (FiLM).

    Block structure:
      h = Conv1d(x)   → GroupNorm → SiLU
        + FiLM(cond)  : h = (1 + γ) * h + β  where (γ,β) = Linear(cond)
      h = Conv1d(h)   → GroupNorm → SiLU
      out = h + residual_conv(x)

    The ``1 + γ`` formulation (instead of bare γ) initialises the scale near 1,
    which is a mild inductive bias that preserves the residual magnitude early
    in training.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        cond_dim: Conditioning vector dimension (timestep + obs embeddings).
        kernel_size: Conv1d kernel size (must be odd for 'same' padding).
        n_groups: Number of groups for GroupNorm.
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
        pad = kernel_size // 2  # 'same' padding for odd kernel

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad)
        self.norm1 = nn.GroupNorm(n_groups, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad)
        self.norm2 = nn.GroupNorm(n_groups, out_channels)

        # FiLM projection: cond → (γ, β), each of dim out_channels
        self.film_proj = nn.Linear(cond_dim, 2 * out_channels)

        # Residual path: 1×1 conv when channel counts differ
        self.res_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward pass with FiLM conditioning.

        Args:
            x: ``(B, in_channels, T)`` feature map.
            cond: ``(B, cond_dim)`` conditioning vector.

        Returns:
            ``(B, out_channels, T)`` feature map.
        """
        h = F.silu(self.norm1(self.conv1(x)))       # (B, out_ch, T)

        # FiLM modulation after first conv
        film = self.film_proj(cond)                 # (B, 2 * out_ch)
        gamma, beta = film.chunk(2, dim=-1)         # each (B, out_ch)
        # Unsqueeze for broadcasting over the time dimension
        h = h * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)

        h = F.silu(self.norm2(self.conv2(h)))       # (B, out_ch, T)

        return h + self.res_conv(x)


# ---------------------------------------------------------------------------
# 1-D Temporal U-Net
# ---------------------------------------------------------------------------

class ConditionalUNet1d(nn.Module):
    """1-D temporal U-Net denoiser for action diffusion.

    Args:
        cfg: Full resolved config.  Reads ``denoiser.cnn``, ``model``.
        action_dim: Dimensionality of one action step (6 for SO-101).
        obs_cond_dim: Dimensionality of the observation conditioning vector
            (output of ObservationEncoder).
    """

    def __init__(self, cfg: DictConfig, action_dim: int, obs_cond_dim: int) -> None:
        super().__init__()

        channels: list[int] = list(cfg.denoiser.cnn.channels)   # e.g. [256, 512, 1024]
        kernel_size: int = cfg.denoiser.cnn.kernel_size
        n_groups: int = cfg.denoiser.cnn.n_groups

        t_emb_dim = channels[0]          # timestep embed dim = first channel width

        # --- Conditioning components ---
        self.t_embed = SinusoidalTimestepEmbedding(t_emb_dim)
        # Project high-dim obs vector to the same size as t_emb for concatenation
        self.obs_proj = nn.Sequential(
            nn.Linear(obs_cond_dim, t_emb_dim),
            nn.SiLU(),
        )
        film_dim = t_emb_dim * 2         # combined FiLM conditioning dim

        # --- Encoder ---
        self.input_proj = nn.Conv1d(action_dim, channels[0], kernel_size=1)

        self.down_blocks: nn.ModuleList = nn.ModuleList()
        self.down_samples: nn.ModuleList = nn.ModuleList()
        in_ch = channels[0]
        for out_ch in channels[1:]:
            self.down_blocks.append(nn.ModuleList([
                FiLMResBlock1d(in_ch, in_ch, film_dim, kernel_size, n_groups),
                FiLMResBlock1d(in_ch, in_ch, film_dim, kernel_size, n_groups),
            ]))
            # Stride-2 conv doubles channels and halves the time dimension
            self.down_samples.append(
                nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
            )
            in_ch = out_ch

        # --- Bottleneck (at channels[-1] spatial resolution) ---
        self.mid_blocks: nn.ModuleList = nn.ModuleList([
            FiLMResBlock1d(in_ch, in_ch, film_dim, kernel_size, n_groups),
            FiLMResBlock1d(in_ch, in_ch, film_dim, kernel_size, n_groups),
        ])

        # --- Decoder ---
        self.up_samples: nn.ModuleList = nn.ModuleList()
        self.up_blocks: nn.ModuleList = nn.ModuleList()
        for out_ch in reversed(channels[:-1]):
            # ConvTranspose1d doubles the time dimension
            self.up_samples.append(
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
            )
            # First block merges the skip connection (cat doubles channels)
            self.up_blocks.append(nn.ModuleList([
                FiLMResBlock1d(out_ch * 2, out_ch, film_dim, kernel_size, n_groups),
                FiLMResBlock1d(out_ch, out_ch, film_dim, kernel_size, n_groups),
            ]))
            in_ch = out_ch

        self.output_proj = nn.Conv1d(channels[0], action_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, obs_cond: torch.Tensor) -> torch.Tensor:
        """Predict the noise added to the action chunk.

        Args:
            x: Noisy action chunk ``(B, T_p, action_dim)``.
            t: Integer diffusion timestep indices ``(B,)``.
            obs_cond: Observation conditioning vector ``(B, obs_cond_dim)``.

        Returns:
            Predicted noise ``(B, T_p, action_dim)``.
        """
        # Build combined FiLM conditioning vector
        cond = torch.cat([self.t_embed(t), self.obs_proj(obs_cond)], dim=-1)  # (B, film_dim)

        # Transpose for 1-D convolution: (B, action_dim, T_p)
        h = self.input_proj(x.transpose(1, 2))   # (B, channels[0], T_p)

        # Encoder
        skips: list[torch.Tensor] = []
        for blocks, downsample in zip(self.down_blocks, self.down_samples):
            for block in blocks:
                h = block(h, cond)
            skips.append(h)
            h = downsample(h)

        # Bottleneck
        for block in self.mid_blocks:
            h = block(h, cond)

        # Decoder
        for upsample, blocks, skip in zip(self.up_samples, self.up_blocks, reversed(skips)):
            h = upsample(h)
            h = torch.cat([h, skip], dim=1)    # concat skip connection along channels
            for block in blocks:
                h = block(h, cond)

        # Transpose back: (B, action_dim, T_p) → (B, T_p, action_dim)
        return self.output_proj(h).transpose(1, 2)
