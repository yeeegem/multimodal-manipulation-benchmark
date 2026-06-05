"""Reverse-process samplers: DDPM (ancestral) and DDIM (deterministic).

Both consume a trained epsilon-prediction denoiser and a :class:`NoiseSchedule`,
and produce a clean action chunk. Notation follows Ho et al. (DDPM, 2020) and
Song et al. (DDIM, 2021).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from diffusion_policy_soarm.models.diffusion.schedule import NoiseSchedule, _extract, q_sample


def shift_action_chunk_for_warm_start(
    action_chunk: torch.Tensor,
    exec_horizon: int,
) -> torch.Tensor:
    """Shift a clean action chunk forward by ``exec_horizon`` steps.

    The unexecuted suffix is preserved and the tail is padded by repeating the
    final action so the next chunk starts from the previous plan's continuation.
    """
    if exec_horizon <= 0:
        return action_chunk

    pred_horizon = action_chunk.shape[1]
    if exec_horizon >= pred_horizon:
        return action_chunk[:, -1:, :].expand(-1, pred_horizon, -1).clone()

    suffix = action_chunk[:, exec_horizon:, :]
    pad = action_chunk[:, -1:, :].expand(-1, exec_horizon, -1)
    return torch.cat([suffix, pad], dim=1)


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
        init_clean: torch.Tensor | None = None,
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
        if init_clean is None:
            x = torch.randn(shape, device=device)
        else:
            if init_clean.shape != shape:
                raise ValueError(f"init_clean shape {init_clean.shape} does not match {shape}")
            t_batch = torch.full((shape[0],), self.num_steps - 1, dtype=torch.long, device=device)
            x, _ = q_sample(init_clean.to(device), t_batch, self.schedule)

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
        # Pick indices that are roughly evenly spaced in [0, T-1], ending at T-1.
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
        init_clean: torch.Tensor | None = None,
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
        if init_clean is None:
            x = torch.randn(shape, device=device)
        else:
            if init_clean.shape != shape:
                raise ValueError(f"init_clean shape {init_clean.shape} does not match {shape}")
            t_start = self._timesteps[0]
            t_batch = torch.full((shape[0],), t_start, dtype=torch.long, device=device)
            x, _ = q_sample(init_clean.to(device), t_batch, self.schedule)

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
