"""Per-dimension min-max normalisation to the interval [-1, 1].

Design decisions:
- Min-max (not z-score) because the DDPM noise schedule and ``clip_sample`` flag
  assume the clean signal lives in a bounded interval.  Z-score does not guarantee
  this and can produce outliers outside the clipping range.
- Stats are read from LeRobotDataset's pre-computed ``meta.stats`` (derived from
  the full training set at recording time), so we never need to decode video frames
  just to compute normalisation constants.
- The Normalizer is an nn.Module so that ``register_buffer`` keeps tensors on the
  same device as the model without manual ``.to(device)`` calls at every use site.
- Stats are saved to disk alongside each checkpoint as a human-readable JSON file,
  ensuring that eval and inference use identical scaling.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from omegaconf import DictConfig


class Normalizer(nn.Module):
    """Per-dimension min-max normaliser mapping to [-1, 1].

    Args:
        mins: Per-dimension minima, shape ``(D,)``.
        maxs: Per-dimension maxima, shape ``(D,)``.
    """

    def __init__(self, mins: torch.Tensor, maxs: torch.Tensor) -> None:
        super().__init__()
        if mins.shape != maxs.shape or mins.ndim != 1:
            raise ValueError("mins and maxs must be 1-D tensors of equal shape.")
        # A very small epsilon prevents division by zero for constant features.
        eps = torch.tensor(1e-8)
        self.register_buffer("mins", mins.float())
        self.register_buffer("maxs", maxs.float())
        self.register_buffer("_range", torch.maximum(maxs.float() - mins.float(), eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise ``x`` to [-1, 1].

        Args:
            x: Float tensor of any shape ``(..., D)`` where D matches the
               number of dimensions the stats were computed over.

        Returns:
            Normalised tensor with the same shape as ``x``.
        """
        return (x - self.mins) / self._range * 2.0 - 1.0

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Map normalised values in [-1, 1] back to the original scale.

        Args:
            x: Normalised float tensor of shape ``(..., D)``.

        Returns:
            Denormalised tensor with the same shape as ``x``.
        """
        return (x + 1.0) / 2.0 * self._range + self.mins

    def save(self, path: Path) -> None:
        """Write stats to a JSON file for human inspection and reproducibility."""
        payload = {
            "mins": self.mins.tolist(),
            "maxs": self.maxs.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def from_file(cls, path: Path) -> "Normalizer":
        """Load a Normalizer from a previously saved JSON file.

        Args:
            path: Path to the JSON file written by :meth:`save`.

        Returns:
            Normalizer instance with stats loaded from disk.
        """
        payload = json.loads(path.read_text())
        mins = torch.tensor(payload["mins"], dtype=torch.float32)
        maxs = torch.tensor(payload["maxs"], dtype=torch.float32)
        return cls(mins, maxs)

    @classmethod
    def from_lerobot_stats(cls, lerobot_ds: LeRobotDataset, key: str) -> "Normalizer":
        """Build a Normalizer from the pre-computed stats in the LeRobotDataset.

        Uses ``meta.stats[key]['min']`` and ``meta.stats[key]['max']``, which are
        computed over the full training split at recording time.

        Args:
            lerobot_ds: The underlying LeRobotDataset (access via
                ``DiffusionDataset.lerobot_dataset``).
            key: Feature key to read stats for, e.g. ``"action"`` or
                ``"observation.state"``.

        Returns:
            Normalizer for the given feature key.
        """
        stats = lerobot_ds.meta.stats[key]
        mins = torch.tensor(np.asarray(stats["min"]), dtype=torch.float32)
        maxs = torch.tensor(np.asarray(stats["max"]), dtype=torch.float32)
        return cls(mins, maxs)


def build_normalizers(
    lerobot_ds: LeRobotDataset, cfg: DictConfig
) -> dict[str, Normalizer]:
    """Build Normalizer instances for actions and proprioceptive state.

    Args:
        lerobot_ds: Underlying LeRobotDataset instance.
        cfg: Resolved config (reads ``dataset.state_key`` and ``dataset.action_key``).

    Returns:
        Dict with keys ``"action"`` and ``"state"``.
    """
    return {
        "action": Normalizer.from_lerobot_stats(lerobot_ds, cfg.dataset.action_key),
        "state": Normalizer.from_lerobot_stats(lerobot_ds, cfg.dataset.state_key),
    }
