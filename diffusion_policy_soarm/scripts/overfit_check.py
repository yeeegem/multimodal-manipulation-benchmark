"""Phase 4 gate: overfit 5 demonstration episodes to verify the pipeline.

Trains on only 5 episodes for many epochs and checks:
1. Training loss drops to near zero.
2. DDPM sampling from the trained model reproduces the ground-truth trajectories
   (mean per-joint absolute error < 0.01 rad after normalisation).

Usage:
    uv run python -m diffusion_policy_soarm.scripts.overfit_check \\
        --config diffusion_policy_soarm/configs/base.yaml \\
        [key=value overrides]
"""

from __future__ import annotations


def run_overfit_check(cfg_path: str, overrides: list[str] | None = None) -> None:
    """Run the overfit sanity check and print pass/fail to stdout."""
    raise NotImplementedError("Phase 4")


if __name__ == "__main__":
    raise NotImplementedError("Phase 4")
