"""Training entry point for Diffusion Policy on SO-ARM101.

Usage:
    uv run python -m diffusion_policy_soarm.train \\
        --config diffusion_policy_soarm/configs/base.yaml \\
        [key=value overrides, e.g. training.batch_size=32]

Phase 4 implementation target:
- Parse config, seed, build dataset/loader, build model, run training loop.
- EMA weight shadow, TensorBoard logging, periodic checkpointing.
- Phase 4 gate: overfit 5 demos to near-zero loss.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    """Main training entry point."""
    raise NotImplementedError("Phase 4")


if __name__ == "__main__":
    main()
