"""Noise schedule and the forward (q) process.

Math notation follows Ho et al. (DDPM, 2020):
  - T          : number of training diffusion steps
  - beta_t     : noise variance at step t
  - alpha_t    = 1 - beta_t
  - alphabar_t = prod_{s=1}^{t} alpha_s  (cumulative product)
  - q(x_t|x_0) = N(x_t; sqrt(alphabar_t) x_0, (1-alphabar_t) I)   forward process

These components are decoupled from the denoiser and the samplers so they can be
unit-tested in isolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from omegaconf import DictConfig


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
