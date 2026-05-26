"""LeRobot dataset wrapper producing (observation history, action chunk) samples.

Phase 1 implementation target:
- Wrap a LeRobotDataset to yield fixed-length windows aligned to the receding-horizon
  convention: obs_horizon frames of observations and pred_horizon frames of actions.
- Handle episode boundaries (pad with edge values rather than crossing episodes).
- Decode video frames on-the-fly via the LeRobot video backend (pyav).
- Apply image resize and tensor conversion; return raw (unnormalised) tensors.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset


class Batch(TypedDict):
    """One training sample returned by DiffusionDataset.__getitem__."""

    # Shape: (obs_horizon, C, H, W) per camera key
    images: dict[str, torch.Tensor]
    # Shape: (obs_horizon, state_dim)
    state: torch.Tensor
    # Shape: (pred_horizon, action_dim)
    actions: torch.Tensor


class DiffusionDataset(Dataset):
    """Maps a LeRobotDataset to (obs_window, action_chunk) training pairs.

    Args:
        cfg: Resolved OmegaConf config (``dataset`` and ``model`` sub-trees used).
        split: Dataset split string passed to LeRobotDataset (e.g. ``"train"``).
    """

    def __init__(self, cfg: DictConfig, split: str = "train") -> None:
        raise NotImplementedError("Phase 1")

    def __len__(self) -> int:
        raise NotImplementedError("Phase 1")

    def __getitem__(self, idx: int) -> Batch:
        raise NotImplementedError("Phase 1")

    @classmethod
    def from_path(cls, dataset_path: str | Path, cfg: DictConfig) -> "DiffusionDataset":
        """Convenience constructor accepting a raw path override."""
        raise NotImplementedError("Phase 1")
