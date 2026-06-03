"""Diffusion subsystem: noise schedule, forward process, samplers, training module.

Submodules:
- ``schedule``: NoiseSchedule, make_noise_schedule, q_sample (forward process).
- ``samplers``: DDPMSampler, DDIMSampler (reverse process).
- ``module``:   diffusion_loss, DiffusionModule (training objective + top-level model).

Public names are re-exported here so callers can keep using
``from diffusion_policy_soarm.models.diffusion import <name>``.
"""

from diffusion_policy_soarm.models.diffusion.module import DiffusionModule, diffusion_loss
from diffusion_policy_soarm.models.diffusion.samplers import (
    DDIMSampler,
    DDPMSampler,
    shift_action_chunk_for_warm_start,
)
from diffusion_policy_soarm.models.diffusion.schedule import (
    NoiseSchedule,
    _extract,
    make_noise_schedule,
    q_sample,
)

__all__ = [
    "NoiseSchedule",
    "make_noise_schedule",
    "q_sample",
    "_extract",
    "DDPMSampler",
    "DDIMSampler",
    "shift_action_chunk_for_warm_start",
    "diffusion_loss",
    "DiffusionModule",
]
