"""Observation and action normalisation statistics.

Phase 1 implementation target:
- Compute per-dimension min/max from the training split.
- Save stats to ``<run_dir>/norm_stats.json`` so eval uses identical scaling.
- Provide forward (normalise to [-1, 1]) and inverse transforms as nn.Modules
  so they can be composed cleanly in the inference graph.

Design note: we use min-max scaling to [-1, 1] (not z-score) because the
diffusion noise schedule and the ``clip_sample`` flag assume actions live in
a bounded interval.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig


class Normalizer(nn.Module):
    """Per-dimension min-max normaliser to the interval [-1, 1].

    Args:
        mins: Tensor of per-dimension minima, shape ``(D,)``.
        maxs: Tensor of per-dimension maxima, shape ``(D,)``.
    """

    def __init__(self, mins: torch.Tensor, maxs: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mins", mins)
        self.register_buffer("maxs", maxs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise ``x`` to [-1, 1]."""
        raise NotImplementedError("Phase 1")

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Map normalised values back to original scale."""
        raise NotImplementedError("Phase 1")

    def save(self, path: Path) -> None:
        """Persist stats as a JSON file for reproducible eval loading."""
        raise NotImplementedError("Phase 1")

    @classmethod
    def from_file(cls, path: Path) -> "Normalizer":
        """Load a Normalizer from a previously saved JSON file."""
        raise NotImplementedError("Phase 1")

    @classmethod
    def from_dataset(cls, cfg: DictConfig, key: str) -> "Normalizer":
        """Compute min/max statistics from the training split for ``key``."""
        raise NotImplementedError("Phase 1")
