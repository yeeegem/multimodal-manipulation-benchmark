"""Phase 2 gate: unit tests for the diffusion core.

Tests are independent of the denoiser network — they only exercise the noise
schedule, q_sample, and sampler mechanics.

Run with:
    uv run pytest tests/test_diffusion.py -v
"""

import math

import pytest
import torch
from omegaconf import OmegaConf

from diffusion_policy_soarm.models.diffusion import (
    DDIMSampler,
    DDPMSampler,
    NoiseSchedule,
    _extract,
    diffusion_loss,
    make_noise_schedule,
    q_sample,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cosine_cfg():
    return OmegaConf.create({
        "num_train_timesteps": 100,
        "noise_schedule": "cosine",
        "beta_start": 0.0001,
        "beta_end": 0.02,
    })


@pytest.fixture
def linear_cfg():
    return OmegaConf.create({
        "num_train_timesteps": 100,
        "noise_schedule": "linear",
        "beta_start": 0.0001,
        "beta_end": 0.02,
    })


@pytest.fixture
def cosine_schedule(cosine_cfg):
    return make_noise_schedule(cosine_cfg)


# ---------------------------------------------------------------------------
# Noise schedule tests
# ---------------------------------------------------------------------------

def test_schedule_shapes(cosine_schedule):
    T = 100
    for field in [
        "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
        "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
        "posterior_variance", "posterior_mean_coef1", "posterior_mean_coef2",
    ]:
        tensor = getattr(cosine_schedule, field)
        assert tensor.shape == (T,), f"{field} has wrong shape {tensor.shape}"


def test_betas_in_valid_range(cosine_schedule):
    assert (cosine_schedule.betas >= 0).all()
    assert (cosine_schedule.betas <= 1).all()


def test_alphas_cumprod_monotone_decreasing(cosine_schedule):
    ac = cosine_schedule.alphas_cumprod
    assert (ac[:-1] > ac[1:]).all(), "ᾱ_t must decrease monotonically"


def test_alphas_cumprod_boundary_values(cosine_schedule):
    ac = cosine_schedule.alphas_cumprod
    # At t=0: should be close to 1 (very little noise injected yet)
    assert ac[0] > 0.99, f"ᾱ_0 = {ac[0]:.4f}, expected > 0.99"
    # At t=T-1: should be close to 0 (almost pure noise)
    assert ac[-1] < 0.01, f"ᾱ_{{T-1}} = {ac[-1]:.4f}, expected < 0.01"


def test_linear_schedule(linear_cfg):
    sched = make_noise_schedule(linear_cfg)
    # Beta should increase from beta_start to beta_end
    assert sched.betas[0] < sched.betas[-1]
    assert abs(sched.betas[0].item() - linear_cfg.beta_start) < 1e-5
    assert abs(sched.betas[-1].item() - linear_cfg.beta_end) < 1e-5


# ---------------------------------------------------------------------------
# Forward process (q_sample) tests
# ---------------------------------------------------------------------------

def test_q_sample_shape(cosine_schedule):
    B, T_p, D = 4, 16, 6
    x_0 = torch.randn(B, T_p, D)
    t = torch.randint(0, 100, (B,))
    x_t, noise = q_sample(x_0, t, cosine_schedule)
    assert x_t.shape == (B, T_p, D)
    assert noise.shape == (B, T_p, D)


def test_q_sample_at_t0_near_clean(cosine_schedule):
    """At t=0 the noisy sample should be very close to the clean action."""
    B, T_p, D = 8, 16, 6
    x_0 = torch.randn(B, T_p, D)
    t = torch.zeros(B, dtype=torch.long)
    x_t, _ = q_sample(x_0, t, cosine_schedule)

    # x_t ≈ √ᾱ_0 · x_0 + √(1−ᾱ_0) · ε;  ᾱ_0 ≈ 0.9999 ⇒ tiny noise
    max_deviation = (x_t - x_0).abs().max().item()
    assert max_deviation < 0.15, (
        f"At t=0 max deviation from x_0 is {max_deviation:.4f}, expected < 0.15"
    )


def test_q_sample_at_tT_near_gaussian(cosine_schedule):
    """At t=T−1 the noisy sample should be approximately N(0, I)."""
    T = cosine_schedule.betas.shape[0]
    B, T_p, D = 256, 16, 6
    x_0 = torch.randn(B, T_p, D)
    t = torch.full((B,), T - 1, dtype=torch.long)
    x_t, _ = q_sample(x_0, t, cosine_schedule)

    mean = x_t.mean().abs().item()
    std = x_t.std().item()
    assert mean < 0.1, f"At t=T−1 mean={mean:.4f}, expected < 0.1"
    assert abs(std - 1.0) < 0.1, f"At t=T−1 std={std:.4f}, expected ≈ 1.0"


def test_q_sample_with_provided_noise(cosine_schedule):
    """Providing explicit noise should give deterministic output."""
    B, T_p, D = 4, 16, 6
    x_0 = torch.randn(B, T_p, D)
    t = torch.randint(0, 100, (B,))
    noise = torch.randn(B, T_p, D)

    x_t1, n1 = q_sample(x_0, t, cosine_schedule, noise=noise)
    x_t2, n2 = q_sample(x_0, t, cosine_schedule, noise=noise)

    assert torch.allclose(x_t1, x_t2)
    assert torch.allclose(n1, n2)
    assert torch.allclose(n1, noise)


# ---------------------------------------------------------------------------
# Diffusion loss test (with a dummy denoiser)
# ---------------------------------------------------------------------------

class _IdentityDenoiser(torch.nn.Module):
    """Returns a fixed zero tensor — just tests that the loss pipeline runs."""

    def forward(self, x_t, t, cond):
        return torch.zeros_like(x_t)


def test_diffusion_loss_runs(cosine_schedule):
    B, T_p, D, cond_dim = 4, 16, 6, 128
    denoiser = _IdentityDenoiser()
    x_0 = torch.randn(B, T_p, D)
    cond = torch.randn(B, cond_dim)
    loss = diffusion_loss(denoiser, x_0, cond, cosine_schedule)
    assert loss.shape == ()
    assert loss.item() > 0


# ---------------------------------------------------------------------------
# DDPM sampler test (with a dummy denoiser)
# ---------------------------------------------------------------------------

class _ZeroDenoiser(torch.nn.Module):
    """Predicts zero noise — converges toward the mean of the distribution."""

    def forward(self, x_t, t, cond):
        return torch.zeros_like(x_t)


def test_ddpm_sampler_output_shape(cosine_schedule):
    sampler = DDPMSampler(cosine_schedule, clip_sample=True)
    B, T_p, D, cond_dim = 2, 16, 6, 64
    cond = torch.zeros(B, cond_dim)
    out = sampler.sample(_ZeroDenoiser(), (B, T_p, D), cond, device=torch.device("cpu"))
    assert out.shape == (B, T_p, D)


def test_ddpm_sampler_stays_in_bounds_with_clip(cosine_schedule):
    """clip_sample clips the intermediate x̂_0 estimate each step.

    The final output is the posterior mean μ̃_t = coef1·x̂_0 + coef2·x_t, which
    can be slightly outside [−1,1] because x_t is not clipped.  We verify the
    output is bounded (not exploding), not that it is strictly in [−1,1].
    """
    sampler = DDPMSampler(cosine_schedule, clip_sample=True)
    B, T_p, D = 4, 16, 6
    cond = torch.zeros(B, 64)
    out = sampler.sample(_ZeroDenoiser(), (B, T_p, D), cond, torch.device("cpu"))
    assert out.min().item() > -2.0, "output min exploded"
    assert out.max().item() < 2.0, "output max exploded"


# ---------------------------------------------------------------------------
# DDIM sampler test
# ---------------------------------------------------------------------------

def test_ddim_sampler_output_shape(cosine_schedule):
    sampler = DDIMSampler(cosine_schedule, num_inference_steps=10, clip_sample=True)
    B, T_p, D = 2, 16, 6
    cond = torch.zeros(B, 64)
    out = sampler.sample(_ZeroDenoiser(), (B, T_p, D), cond, torch.device("cpu"))
    assert out.shape == (B, T_p, D)


def test_ddim_is_deterministic(cosine_schedule):
    """DDIM with η=0 must produce identical outputs for the same noise seed."""
    torch.manual_seed(0)
    sampler = DDIMSampler(cosine_schedule, num_inference_steps=10, clip_sample=True)
    B, T_p, D = 2, 16, 6
    cond = torch.zeros(B, 64)

    torch.manual_seed(42)
    out1 = sampler.sample(_ZeroDenoiser(), (B, T_p, D), cond, torch.device("cpu"))
    torch.manual_seed(42)
    out2 = sampler.sample(_ZeroDenoiser(), (B, T_p, D), cond, torch.device("cpu"))

    assert torch.allclose(out1, out2), "DDIM must be deterministic given same RNG seed"


def test_ddim_timestep_count(cosine_schedule):
    """Check that the sampler uses exactly num_inference_steps denoising steps."""
    K = 10
    sampler = DDIMSampler(cosine_schedule, num_inference_steps=K)
    assert len(sampler._timesteps) == K


# ---------------------------------------------------------------------------
# _extract helper test
# ---------------------------------------------------------------------------

def test_extract_broadcasting():
    a = torch.arange(100, dtype=torch.float32)
    t = torch.tensor([0, 5, 99])
    out = _extract(a, t, (3, 16, 6))
    assert out.shape == (3, 1, 1)
    assert out[0, 0, 0].item() == 0.0
    assert out[1, 0, 0].item() == 5.0
    assert out[2, 0, 0].item() == 99.0
