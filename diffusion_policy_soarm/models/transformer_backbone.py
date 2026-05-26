"""Transformer denoiser with cross-attention conditioning (ablation variant).

Phase 3 (ablation) implementation target:

Architecture:
- Noisy action tokens: linear projection of each action step → d_model, then
  add sinusoidal position encoding over the pred_horizon axis.
- Timestep token: sinusoidal timestep embedding projected to d_model; prepended
  to the action sequence.
- Conditioning via cross-attention: the observation embedding is projected to
  (n_kv_tokens, d_model) key-value pairs; each transformer layer has a
  cross-attention sublayer attending to these.
- Output: linear head d_model → action_dim per token; drop the timestep token.

Design note: cross-attention conditioning is preferred over concatenating the
observation token into the self-attention sequence because it keeps the
self-attention quadratic cost proportional only to pred_horizon.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


class TransformerDenoiser(nn.Module):
    """Transformer-based noise predictor (ablation against ConditionalUNet1d).

    Args:
        cfg: Full resolved config (``denoiser.transformer``, ``model`` sub-trees).
        action_dim: Dimensionality of one action step.
        cond_dim: Dimensionality of the observation conditioning vector.
    """

    def __init__(self, cfg: DictConfig, action_dim: int, cond_dim: int) -> None:
        super().__init__()
        raise NotImplementedError("Phase 3 ablation")

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Predict noise for the action chunk.

        Args:
            x: ``(B, pred_horizon, action_dim)`` noisy action chunk.
            t: ``(B,)`` integer timestep indices.
            cond: ``(B, cond_dim)`` observation conditioning vector.

        Returns:
            ``(B, pred_horizon, action_dim)`` predicted noise.
        """
        raise NotImplementedError("Phase 3 ablation")
