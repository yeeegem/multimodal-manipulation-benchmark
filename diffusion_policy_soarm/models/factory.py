"""Model factory: assemble networks and the policy selected by ``cfg.model.type``.

Single home for "build a network/policy" so the training loop (``train.py``) and the
inference loop (``infer.py``) construct things in exactly one place. Supported policy
types:

  - ``"diffusion"`` (default): ObservationEncoder + denoiser + DiffusionModule.
  - ``"bc"``:                   ObservationEncoder + MLPActionHead + BCModule.
"""

from __future__ import annotations

import torch.nn as nn
from omegaconf import DictConfig

from diffusion_policy_soarm.models.bc import BCModule, MLPActionHead
from diffusion_policy_soarm.models.cnn_backbone import ConditionalUNet1d
from diffusion_policy_soarm.models.diffusion import DiffusionModule
from diffusion_policy_soarm.models.encoders import ObservationEncoder
from diffusion_policy_soarm.models.transformer_backbone import TransformerDenoiser


def build_denoiser(cfg: DictConfig, action_dim: int, obs_cond_dim: int) -> nn.Module:
    """Factory: instantiate the denoiser selected by ``cfg.denoiser.backbone``.

    Args:
        cfg: Full resolved config.
        action_dim: Dimensionality of one action step.
        obs_cond_dim: ObservationEncoder output dimension.

    Returns:
        ConditionalUNet1d or TransformerDenoiser.
    """
    backbone = cfg.denoiser.backbone
    if backbone == "cnn":
        return ConditionalUNet1d(cfg, action_dim=action_dim, obs_cond_dim=obs_cond_dim)
    elif backbone == "transformer":
        return TransformerDenoiser(cfg, action_dim=action_dim, cond_dim=obs_cond_dim)
    else:
        raise ValueError(f"Unknown denoiser backbone: {backbone!r}")


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
