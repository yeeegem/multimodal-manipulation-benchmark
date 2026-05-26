"""Real-time receding-horizon inference loop for the SO-ARM101 arm.

Usage:
    uv run python -m diffusion_policy_soarm.infer \\
        --checkpoint runs/<run>/checkpoints/ema_latest.pt \\
        --config runs/<run>/config.yaml \\
        [key=value overrides]

Phase 6 implementation target:
- Load EMA checkpoint, build model in eval mode.
- Connect to the arm via lerobot's robot interface.
- Observation loop: capture images + state → encode → DDIM sample → execute T_a steps.
- Report per-inference latency (mean and p95) over a 100-step window.
- Optional torch.compile path controlled by infer.compile flag in config.
"""

from __future__ import annotations


def main() -> None:
    """Real-time inference loop entry point."""
    raise NotImplementedError("Phase 6")


if __name__ == "__main__":
    main()
