"""Batch visualisation script — Phase 1 sanity gate.

Loads one batch from DiffusionDataset, prints tensor shapes, and saves a figure
with the front-camera image sequence and the corresponding action chunk as a
joint-position line plot.

Usage:
    python -m diffusion_policy_soarm.data.visualize_batch \\
        --config diffusion_policy_soarm/configs/base.yaml \\
        [key=value overrides]
"""

from __future__ import annotations


def visualize_batch(cfg_path: str, overrides: list[str] | None = None) -> None:
    """Print shapes and save a visualisation of one training batch."""
    raise NotImplementedError("Phase 1")


if __name__ == "__main__":
    raise NotImplementedError("Phase 1")
