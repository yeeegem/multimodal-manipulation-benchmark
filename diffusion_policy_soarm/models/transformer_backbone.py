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

from diffusion_policy_soarm.models.cnn_backbone import SinusoidalTimestepEmbedding


class TransformerDenoiser(nn.Module):
    """Transformer-based noise predictor (ablation against ConditionalUNet1d).

    Args:
        cfg: Full resolved config (``denoiser.transformer``, ``model`` sub-trees).
        action_dim: Dimensionality of one action step.
        cond_dim: Dimensionality of the observation conditioning vector.
    """

    def __init__(self, cfg: DictConfig, action_dim: int, cond_dim: int) -> None:
        super().__init__()

        d_model: int = cfg.denoiser.transformer.d_model
        n_heads: int = cfg.denoiser.transformer.n_heads
        n_layers: int = cfg.denoiser.transformer.n_layers
        dropout: float = cfg.denoiser.transformer.dropout
        pred_horizon: int = cfg.model.pred_horizon

        self.action_proj = nn.Linear(action_dim, d_model)
        self.t_embed = SinusoidalTimestepEmbedding(d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, pred_horizon + 1, d_model))
        self.cond_proj = nn.Sequential(nn.Linear(cond_dim, d_model), nn.SiLU())

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.output_head = nn.Linear(d_model, action_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Predict noise for the action chunk.

        Args:
            x: ``(B, pred_horizon, action_dim)`` noisy action chunk.
            t: ``(B,)`` integer timestep indices.
            cond: ``(B, cond_dim)`` observation conditioning vector.

        Returns:
            ``(B, pred_horizon, action_dim)`` predicted noise.
        """
        tokens = self.action_proj(x)              # (B, T_p, d_model)
        t_tok = self.t_embed(t).unsqueeze(1)       # (B, 1, d_model)
        seq = torch.cat([t_tok, tokens], dim=1)    # (B, T_p + 1, d_model)
        seq = seq + self.pos_emb

        memory = self.cond_proj(cond).unsqueeze(1)  # (B, 1, d_model)

        out = self.decoder(seq, memory)
        out = out[:, 1:]                            # drop timestep token

        return self.output_head(out)
