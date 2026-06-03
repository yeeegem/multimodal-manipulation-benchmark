"""Model factory: assemble the policy selected by ``cfg.model.type``.

Centralises model construction so the training loop (``train.py``) and the
inference loop (``infer.py``) build the encoder and dispatch on model type in
exactly one place. Supported types:

  - ``"diffusion"`` (default): ObservationEncoder + denoiser + DiffusionModule.
  - ``"bc"``:                   ObservationEncoder + MLPActionHead + BCModule.
"""

from __future__ import annotations

import torch.nn as nn
from omegaconf import DictConfig

from diffusion_policy_soarm.models.bc import BCModule, MLPActionHead
from diffusion_policy_soarm.models.diffusion import DiffusionModule, build_denoiser
from diffusion_policy_soarm.models.encoders import ObservationEncoder


def build_policy(
    cfg: DictConfig,
    action_dim: int,
    state_dim: int,
    action_normalizer: nn.Module,
    state_normalizer: nn.Module,
) -> nn.Module:
    """Build the policy module selected by ``cfg.model.type``.

    Args:
        cfg: Full resolved config.
        action_dim: Dimensionality of one action step.
        state_dim: Dimensionality of the proprioceptive state vector.
        action_normalizer: Normalizer for the action chunk.
        state_normalizer: Normalizer for the proprioceptive state.

    Returns:
        A ``DiffusionModule`` or ``BCModule`` exposing ``compute_loss`` and
        ``predict_actions``.
    """
    camera_keys = list(cfg.dataset.camera_keys)
    encoder = ObservationEncoder(cfg, camera_keys=camera_keys, state_dim=state_dim)

    model_type = cfg.model.get("type", "diffusion")
    if model_type == "diffusion":
        denoiser = build_denoiser(cfg, action_dim=action_dim, obs_cond_dim=encoder.output_dim)
        return DiffusionModule(cfg, encoder, denoiser, action_normalizer, state_normalizer)
    elif model_type == "bc":
        head = MLPActionHead(cfg, action_dim=action_dim, obs_cond_dim=encoder.output_dim)
        return BCModule(cfg, encoder, head, action_normalizer, state_normalizer)
    else:
        raise ValueError(f"Unknown model type: {model_type!r} (expected 'diffusion' or 'bc')")
