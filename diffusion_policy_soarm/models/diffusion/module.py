"""Training objective and the top-level DiffusionModule.

``diffusion_loss`` is the epsilon-prediction MSE used during training.
``DiffusionModule`` ties together the observation encoder, the denoiser, the
normalizers, the noise schedule, and the inference sampler.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from diffusion_policy_soarm.models.diffusion.samplers import DDIMSampler, DDPMSampler
from diffusion_policy_soarm.models.diffusion.schedule import (
    NoiseSchedule,
    make_noise_schedule,
    q_sample,
)


def diffusion_loss(
    denoiser: nn.Module,
    x_0: torch.Tensor,
    cond: torch.Tensor,
    schedule: NoiseSchedule,
) -> torch.Tensor:
    """Epsilon-prediction MSE loss for one batch.

    Samples t ~ Uniform[0, T−1] and ε ~ N(0,I), computes x_t via
    q_sample, runs the denoiser to get ε̂, returns MSE(ε̂, ε).

    Args:
        denoiser: Noise-prediction network callable as ``denoiser(x_t, t, cond)``.
        x_0: Clean action chunk ``(B, T_p, action_dim)`` on device.
        cond: Observation conditioning vector ``(B, cond_dim)`` on device.
        schedule: Pre-computed noise schedule (tensors moved to device internally).

    Returns:
        Scalar MSE loss.
    """
    B = x_0.shape[0]
    T = schedule.betas.shape[0]
    t = torch.randint(0, T, (B,), device=x_0.device)
    x_t, noise = q_sample(x_0, t, schedule)
    noise_pred = denoiser(x_t, t, cond)
    return F.mse_loss(noise_pred, noise)


# ---------------------------------------------------------------------------
# DiffusionModule — top-level model
# ---------------------------------------------------------------------------

class DiffusionModule(nn.Module):
    """Top-level model: encoder + denoiser + normalizers + schedule + sampler.

    Owns all trainable parameters and the inference sampler.  The EMA shadow
    copy of the denoiser weights is managed externally in the training loop
    (see train.py) so that this module stays serialisable without extra state.

    Args:
        cfg: Full resolved config.
        encoder: ObservationEncoder.
        denoiser: ConditionalUNet1d or TransformerDenoiser.
        action_normalizer: Normalizer for the action chunk.
        state_normalizer: Normalizer for the proprioceptive state.
    """

    def __init__(
        self,
        cfg: DictConfig,
        encoder: nn.Module,
        denoiser: nn.Module,
        action_normalizer: nn.Module,
        state_normalizer: nn.Module,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.denoiser = denoiser
        self.action_normalizer = action_normalizer
        self.state_normalizer = state_normalizer

        self.schedule = make_noise_schedule(cfg.diffusion)
        self._build_sampler()

    def _build_sampler(self) -> None:
        # Sampler config lives in the `infer` block (inference-only); the schedule
        # itself comes from `diffusion`.
        clip = self.cfg.infer.clip_sample
        if self.cfg.infer.sampler == "ddpm":
            self._sampler = DDPMSampler(self.schedule, clip_sample=clip)
        elif self.cfg.infer.sampler == "ddim":
            self._sampler = DDIMSampler(
                self.schedule,
                num_inference_steps=self.cfg.infer.num_inference_steps,
                clip_sample=clip,
            )
        else:
            raise ValueError(f"Unknown sampler: {self.cfg.infer.sampler!r}")

    def compute_loss(self, batch: dict) -> torch.Tensor:
        """Training forward pass.

        Normalises observations and actions, encodes observations, and computes
        the epsilon-prediction MSE loss.

        Args:
            batch: Dict with keys ``"images"``, ``"state"``, ``"actions"``.

        Returns:
            Scalar loss tensor (on the same device as the batch).
        """
        state_norm = self.state_normalizer(batch["state"])
        action_norm = self.action_normalizer(batch["actions"])
        obs_cond = self.encoder(batch["images"], state_norm)
        return diffusion_loss(self.denoiser, action_norm, obs_cond, self.schedule)

    @torch.no_grad()
    def predict_actions(
        self,
        batch: dict,
        warm_start_actions_norm: torch.Tensor | None = None,
        return_normalized: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Inference forward pass.

        Encodes observations and runs the configured sampler to produce a
        clean (normalised) action chunk, then denormalises before returning.

        Args:
            batch: Dict with keys ``"images"`` and ``"state"``.

        Returns:
            Denormalised action chunk ``(B, pred_horizon, action_dim)``.
        """
        device = next(self.parameters()).device
        B = batch["state"].shape[0]
        state_norm = self.state_normalizer(batch["state"])
        obs_cond = self.encoder(batch["images"], state_norm)

        action_dim = self.cfg.model.get("action_dim", 6)
        pred_horizon = self.cfg.model.pred_horizon
        shape = (B, pred_horizon, action_dim)

        actions_norm = self._sampler.sample(
            self.denoiser,
            shape,
            obs_cond,
            device,
            init_clean=warm_start_actions_norm,
        )
        actions = self.action_normalizer.inverse(actions_norm)
        if return_normalized:
            return actions, actions_norm
        return actions
