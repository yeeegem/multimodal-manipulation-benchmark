"""Diffusion machinery: noise schedule, forward process, loss, and samplers.

All components are decoupled from the denoiser network so they can be unit-
tested in isolation and swapped independently.

Math notation follows Ho et al. (DDPM, 2020) and Song et al. (DDIM, 2021):
  - T          : number of training diffusion steps
  - β_t        : noise variance at step t
  - α_t = 1−β_t
  - ᾱ_t = ∏_{s=1}^{t} α_s  (cumulative product)
  - q(x_t|x_0) = N(x_t; √ᾱ_t x_0, (1−ᾱ_t)I)   forward process
  - ε_θ(x_t,t,c): denoiser predicts the added noise ε (epsilon-prediction)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

@dataclass
class NoiseSchedule:
    """Pre-computed, device-agnostic noise schedule tensors.

    All tensors have shape ``(T,)`` and live on CPU at construction time.
    Call ``.to(device)`` on individual tensors inside training/sampling loops.

    Attributes:
        betas: β_t
        alphas: α_t = 1 − β_t
        alphas_cumprod: ᾱ_t
        alphas_cumprod_prev: ᾱ_{t−1}  (with ᾱ_{−1} ≡ 1 at index 0)
        sqrt_alphas_cumprod: √ᾱ_t
        sqrt_one_minus_alphas_cumprod: √(1−ᾱ_t)
        posterior_variance: β̃_t = β_t(1−ᾱ_{t−1})/(1−ᾱ_t)   (DDPM reverse)
        posterior_mean_coef1: √ᾱ_{t−1} β_t / (1−ᾱ_t)
        posterior_mean_coef2: √α_t (1−ᾱ_{t−1}) / (1−ᾱ_t)
    """

    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    posterior_variance: torch.Tensor
    posterior_mean_coef1: torch.Tensor
    posterior_mean_coef2: torch.Tensor


def make_noise_schedule(cfg: DictConfig) -> NoiseSchedule:
    """Build a noise schedule from config.

    Args:
        cfg: ``diffusion`` sub-tree of the resolved config.
            Must contain ``num_train_timesteps``, ``noise_schedule``,
            and (if linear) ``beta_start`` / ``beta_end``.

    Returns:
        :class:`NoiseSchedule` with all derived tensors on CPU.
    """
    T: int = cfg.num_train_timesteps

    if cfg.noise_schedule == "cosine":
        betas = _cosine_betas(T)
    elif cfg.noise_schedule == "linear":
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, T, dtype=torch.float64)
    else:
        raise ValueError(f"Unknown noise_schedule: {cfg.noise_schedule!r}")

    betas = betas.float()
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    # ᾱ_{t-1}: shift right by one; ᾱ_{-1} ≡ 1 (used for DDPM posterior at t=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

    sqrt_alphas_cumprod = alphas_cumprod.sqrt()
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()

    # DDPM reverse-process posterior q(x_{t-1}|x_t, x_0)
    # β̃_t = β_t * (1−ᾱ_{t-1}) / (1−ᾱ_t)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    # Coefficients of the posterior mean (Eq. 7 in Ho et al.)
    posterior_mean_coef1 = betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod)
    posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod)

    return NoiseSchedule(
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        alphas_cumprod_prev=alphas_cumprod_prev,
        sqrt_alphas_cumprod=sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
        posterior_variance=posterior_variance,
        posterior_mean_coef1=posterior_mean_coef1,
        posterior_mean_coef2=posterior_mean_coef2,
    )


def _cosine_betas(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule (Nichol & Dhariwal, 2021).

    f(t) = cos²(((t/T) + s) / (1+s) · π/2)
    ᾱ_t  = f(t) / f(0)
    β_t  = clip(1 − ᾱ_t/ᾱ_{t-1}, 0, 0.999)

    The offset ``s=0.008`` prevents β from being too small near t=0,
    which would cause numerical instability.
    """
    steps = torch.linspace(0, T, T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    f = f / f[0]  # normalise so ᾱ_0 = 1
    # betas[i] corresponds to step i+1 in paper notation (1-indexed)
    betas = (1.0 - f[1:] / f[:-1]).clamp(0.0, 0.999)
    return betas.float()


# ---------------------------------------------------------------------------
# Forward process
# ---------------------------------------------------------------------------

def q_sample(
    x_0: torch.Tensor,
    t: torch.Tensor,
    schedule: NoiseSchedule,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a noisy action from the forward process q(x_t | x_0).

    x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε,    ε ~ N(0, I)

    Args:
        x_0: Clean action chunk ``(B, T_p, action_dim)``.
        t: Integer timestep indices ``(B,)`` in ``[0, T−1]``.
        schedule: Pre-computed noise schedule.
        noise: Optional pre-sampled noise (same shape as x_0).
            Drawn from N(0,I) if None.

    Returns:
        Tuple ``(x_t, noise)`` — both shaped like ``x_0``.
    """
    if noise is None:
        noise = torch.randn_like(x_0)
    sqrt_a = _extract(schedule.sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_1ma = _extract(schedule.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    return sqrt_a * x_0 + sqrt_1ma * noise, noise


# ---------------------------------------------------------------------------
# Training loss
# ---------------------------------------------------------------------------

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
# DDPM sampler
# ---------------------------------------------------------------------------

class DDPMSampler:
    """Full ancestral DDPM sampler (Algorithm 2, Ho et al. 2020).

    Args:
        schedule: Pre-computed noise schedule.
        clip_sample: If True, clip the x_0 estimate to [−1, 1] each step.
            Required when actions are normalised to that range.
    """

    def __init__(self, schedule: NoiseSchedule, clip_sample: bool = True) -> None:
        self.schedule = schedule
        self.clip_sample = clip_sample
        self.num_steps = schedule.betas.shape[0]

    @torch.no_grad()
    def sample(
        self,
        denoiser: nn.Module,
        shape: tuple[int, ...],
        cond: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Run the full reverse diffusion chain.

        Args:
            denoiser: Trained noise-prediction network (in eval mode).
            shape: Output shape ``(B, T_p, action_dim)``.
            cond: Observation conditioning ``(B, cond_dim)`` on ``device``.
            device: Target device.

        Returns:
            Denoised action chunk ``(B, T_p, action_dim)``.
        """
        x = torch.randn(shape, device=device)

        for t_idx in reversed(range(self.num_steps)):
            t_batch = torch.full((shape[0],), t_idx, dtype=torch.long, device=device)

            # Predict noise
            noise_pred = denoiser(x, t_batch, cond)

            # Estimate clean action (x̂_0)
            sqrt_a = _extract(self.schedule.sqrt_alphas_cumprod, t_batch, x.shape)
            sqrt_1ma = _extract(self.schedule.sqrt_one_minus_alphas_cumprod, t_batch, x.shape)
            x_0_pred = (x - sqrt_1ma * noise_pred) / sqrt_a

            if self.clip_sample:
                x_0_pred = x_0_pred.clamp(-1.0, 1.0)

            # Compute posterior mean μ̃_t(x_t, x̂_0)
            coef1 = _extract(self.schedule.posterior_mean_coef1, t_batch, x.shape)
            coef2 = _extract(self.schedule.posterior_mean_coef2, t_batch, x.shape)
            posterior_mean = coef1 * x_0_pred + coef2 * x

            if t_idx > 0:
                # Add posterior noise (zero at final step)
                posterior_var = _extract(self.schedule.posterior_variance, t_batch, x.shape)
                noise = torch.randn_like(x)
                x = posterior_mean + posterior_var.sqrt() * noise
            else:
                x = posterior_mean

        return x


# ---------------------------------------------------------------------------
# DDIM sampler
# ---------------------------------------------------------------------------

class DDIMSampler:
    """Deterministic DDIM sampler (η=0, Song et al. 2021).

    Sub-samples the training schedule to ``num_inference_steps`` steps,
    enabling ≥10× speed-up with minimal quality loss.

    Args:
        schedule: Pre-computed training noise schedule (T steps).
        num_inference_steps: Number of denoising steps at inference.
            Must be ≤ T.  10–20 steps typically match DDPM quality.
        clip_sample: Clip the x_0 estimate to [−1, 1] each step.
    """

    def __init__(
        self,
        schedule: NoiseSchedule,
        num_inference_steps: int,
        clip_sample: bool = True,
    ) -> None:
        T = schedule.betas.shape[0]
        if num_inference_steps > T:
            raise ValueError(f"num_inference_steps ({num_inference_steps}) > T ({T})")
        self.schedule = schedule
        self.clip_sample = clip_sample
        self.num_steps = num_inference_steps

        # Sub-sample T training steps evenly into num_inference_steps steps.
        # We pick indices that are roughly evenly spaced in [0, T-1], ending at T-1.
        step_ratio = T // num_inference_steps
        timesteps = (torch.arange(num_inference_steps) * step_ratio).long()
        timesteps = timesteps + (T - 1 - timesteps[-1])  # shift so last = T-1
        self._timesteps: list[int] = timesteps.tolist()[::-1]  # T-1 → 0

    @torch.no_grad()
    def sample(
        self,
        denoiser: nn.Module,
        shape: tuple[int, ...],
        cond: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Run the DDIM reverse chain.

        Args:
            denoiser: Trained noise-prediction network (in eval mode).
            shape: Output shape ``(B, T_p, action_dim)``.
            cond: Observation conditioning ``(B, cond_dim)`` on ``device``.
            device: Target device.

        Returns:
            Denoised action chunk ``(B, T_p, action_dim)``.
        """
        x = torch.randn(shape, device=device)

        for i, t_idx in enumerate(self._timesteps):
            t_batch = torch.full((shape[0],), t_idx, dtype=torch.long, device=device)

            # Predict noise
            noise_pred = denoiser(x, t_batch, cond)

            # Estimate x̂_0
            sqrt_a = _extract(self.schedule.sqrt_alphas_cumprod, t_batch, x.shape)
            sqrt_1ma = _extract(self.schedule.sqrt_one_minus_alphas_cumprod, t_batch, x.shape)
            x_0_pred = (x - sqrt_1ma * noise_pred) / sqrt_a

            if self.clip_sample:
                x_0_pred = x_0_pred.clamp(-1.0, 1.0)

            # Previous timestep (or virtual step −1 where ᾱ = 1)
            if i + 1 < len(self._timesteps):
                t_prev = self._timesteps[i + 1]
                sqrt_a_prev = self.schedule.sqrt_alphas_cumprod[t_prev].to(device)
                sqrt_1ma_prev = self.schedule.sqrt_one_minus_alphas_cumprod[t_prev].to(device)
            else:
                # Last step: ᾱ_{t_prev} = 1 → x = x̂_0
                sqrt_a_prev = torch.ones(1, device=device)
                sqrt_1ma_prev = torch.zeros(1, device=device)

            # Deterministic DDIM update (η = 0)
            x = sqrt_a_prev * x_0_pred + sqrt_1ma_prev * noise_pred

        return x


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _extract(a: torch.Tensor, t: torch.Tensor, broadcast_shape: tuple[int, ...]) -> torch.Tensor:
    """Gather scalar schedule values at timestep indices and broadcast.

    Args:
        a: 1-D schedule tensor ``(T,)``, on any device.
        t: Integer timestep indices ``(B,)``.
        broadcast_shape: Shape of the target tensor, e.g. ``(B, T_p, action_dim)``.

    Returns:
        Tensor of shape ``(B, 1, 1, ...)`` ready to broadcast over trailing dims.
    """
    out = a.to(t.device).gather(0, t).float()
    return out.reshape(t.shape[0], *([1] * (len(broadcast_shape) - 1)))
