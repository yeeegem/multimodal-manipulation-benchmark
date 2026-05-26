"""LeRobot dataset wrapper producing (observation history, action chunk) samples.

Design decisions:
- We use LeRobotDataset's ``delta_timestamps`` to fetch obs_horizon past frames and
  pred_horizon future frames in a single __getitem__ call, avoiding any manual
  frame-lookup logic.
- Valid indices exclude the last (pred_horizon - 1) frames of each episode so that
  every training sample has a complete, unpadded action chunk. Episode-start padding
  for observations is kept (the policy must handle it at deployment time too).
- Images are resized in __getitem__ from native 480×640 to the config-specified size.
  We use torch.nn.functional.interpolate treating the time dimension as batch.
- The dataset returns raw (unnormalised) tensors; normalisation is applied in the
  training loop via Normalizer so that the stats can be inspected and saved separately.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import torch
import torch.nn.functional as F
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from omegaconf import DictConfig
from torch.utils.data import Dataset


class Batch(TypedDict):
    """One training sample returned by DiffusionDataset.__getitem__.

    All tensors are raw (unnormalised).  Images are float32 in [0, 1].
    """

    # (obs_horizon, C, H, W) per camera key, after resize
    images: dict[str, torch.Tensor]
    # (obs_horizon, state_dim) — raw joint angles in degrees
    state: torch.Tensor
    # (pred_horizon, action_dim) — raw target joint angles in degrees
    actions: torch.Tensor


class DiffusionDataset(Dataset):
    """Maps a LeRobotDataset to (obs_window, action_chunk) training pairs.

    Args:
        cfg: Resolved OmegaConf config.  Uses ``dataset`` and ``model`` sub-trees.
        split: Dataset split string passed to LeRobotDataset (only ``"train"``
            is supported for this dataset).

    Attributes:
        camera_keys: Ordered list of camera feature names.
        obs_horizon: Number of consecutive observation frames per sample.
        pred_horizon: Length of the action chunk per sample.
        image_size: ``(H, W)`` that images are resized to.
    """

    def __init__(self, cfg: DictConfig, split: str = "train") -> None:
        self.camera_keys: list[str] = list(cfg.dataset.camera_keys)
        self.state_key: str = cfg.dataset.state_key
        self.action_key: str = cfg.dataset.action_key
        self.obs_horizon: int = cfg.model.obs_horizon
        self.pred_horizon: int = cfg.model.pred_horizon
        self.image_size: tuple[int, int] = tuple(cfg.dataset.image_size)  # type: ignore[assignment]

        fps: float = cfg.dataset.fps
        # Relative time offsets (seconds) for observation and action windows.
        # Observation: [-(T_o-1)/fps, ..., 0]  (most-recent frame last)
        # Action:      [0, 1/fps, ..., (T_p-1)/fps]
        obs_delta = [-(self.obs_horizon - 1 - i) / fps for i in range(self.obs_horizon)]
        act_delta = [i / fps for i in range(self.pred_horizon)]
        delta_timestamps = {
            **{key: obs_delta for key in self.camera_keys},
            self.state_key: obs_delta,
            self.action_key: act_delta,
        }

        dataset_path = Path(cfg.dataset.path)
        self._lerobot_ds = LeRobotDataset(
            repo_id=dataset_path.name,
            root=dataset_path,
            download_videos=False,
            video_backend="pyav",
            delta_timestamps=delta_timestamps,
            tolerance_s=0.5 / fps,
        )

        # Build valid index list: exclude the last (pred_horizon - 1) frames of each
        # episode so every sample has a complete, unpadded action chunk.
        self._valid_indices = self._compute_valid_indices(self._lerobot_ds)

    def _compute_valid_indices(self, ds: LeRobotDataset) -> list[int]:
        """Return global frame indices with complete pred_horizon-step futures."""
        valid: list[int] = []
        for i in range(ds.meta.total_episodes):
            ep = ds.meta.episodes[i]
            from_idx: int = ep["dataset_from_index"]
            to_idx: int = ep["dataset_to_index"]  # exclusive upper bound
            # Last valid start: needs pred_horizon frames including itself
            last_valid = to_idx - self.pred_horizon
            if last_valid >= from_idx:
                valid.extend(range(from_idx, last_valid + 1))
        return valid

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> Batch:
        global_idx = self._valid_indices[idx]
        raw = self._lerobot_ds[global_idx]

        images: dict[str, torch.Tensor] = {}
        for key in self.camera_keys:
            # LeRobot returns (obs_horizon, C, H, W) for windowed camera keys
            frames: torch.Tensor = raw[key]
            images[key] = self._resize(frames)

        return Batch(
            images=images,
            state=raw[self.state_key],           # (obs_horizon, state_dim)
            actions=raw[self.action_key],         # (pred_horizon, action_dim)
        )

    def _resize(self, frames: torch.Tensor) -> torch.Tensor:
        """Resize (T, C, H, W) frames to (T, C, *image_size).

        Uses bilinear interpolation; treats the time dimension as batch.
        """
        T, C, H, W = frames.shape
        h, w = self.image_size
        if (H, W) == (h, w):
            return frames
        return F.interpolate(frames, size=(h, w), mode="bilinear", align_corners=False)

    @property
    def action_dim(self) -> int:
        """Dimensionality of one action step."""
        return self._lerobot_ds.features[self.action_key]["shape"][0]

    @property
    def state_dim(self) -> int:
        """Dimensionality of the proprioceptive state vector."""
        return self._lerobot_ds.features[self.state_key]["shape"][0]

    @property
    def lerobot_dataset(self) -> LeRobotDataset:
        """Expose the underlying LeRobotDataset for stat computation."""
        return self._lerobot_ds
