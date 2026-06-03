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

import sys
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub.errors import RepositoryNotFoundError
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from omegaconf import DictConfig, OmegaConf
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


def frame_cache_dir(dataset_path: Path, image_size: tuple[int, int]) -> Path:
    """Resolution-qualified frame-cache directory: ``<path>/frame_cache/<H>x<W>``.

    Keying the cache by resolution lets multiple image sizes coexist (e.g. the
    96x96 main run and the 480x640 ablation) and prevents silently loading frames
    that were extracted at a different resolution than the current config wants.
    """
    h, w = image_size
    return dataset_path / "frame_cache" / f"{h}x{w}"


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

    def __init__(
        self,
        cfg: DictConfig,
        split: str = "train",
    ) -> None:
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
        hf_repo_id: str = OmegaConf.select(cfg, "dataset.hf_repo_id", default=dataset_path.name)
        try:
            self._lerobot_ds = LeRobotDataset(
                repo_id=hf_repo_id,
                root=dataset_path,
                download_videos=True,
                video_backend="pyav",
                delta_timestamps=delta_timestamps,
                tolerance_s=0.5 / fps,
            )
        except (RepositoryNotFoundError, FileNotFoundError) as exc:
            print(
                f"\nERROR: Dataset not found locally at '{dataset_path}' "
                f"and HF Hub repo '{hf_repo_id}' was not accessible.\n"
                f"Check dataset.hf_repo_id in your config. Details: {exc}\n"
            )
            sys.exit(1)

        # Build valid index list: exclude the last (pred_horizon - 1) frames of each
        # episode so every sample has a complete, unpadded action chunk.
        self._valid_indices = self._compute_valid_indices(self._lerobot_ds)

        # Frame cache: if pre-extracted numpy arrays exist, use them instead of
        # PyAV video decoding to keep the GPU fed at ~100%.  The cache is keyed by
        # resolution so caches for different image_size values never collide.
        cache_dir = frame_cache_dir(dataset_path, self.image_size)
        self._use_cache = cache_dir.exists()
        self._preload_cache: bool = bool(getattr(cfg.dataset, "preload_cache", False))
        if self._use_cache:
            self._init_frame_cache(cache_dir, dataset_path, fps)

    def _init_frame_cache(self, cache_dir: Path, dataset_path: Path, fps: float) -> None:
        """Load per-episode mmap arrays and build a state/action-only LeRobot instance."""
        total_eps = self._lerobot_ds.meta.total_episodes

        # Load frame arrays: fully into RAM (preload_cache=true) or memory-mapped.
        load_mode = None if self._preload_cache else "r"
        self._frame_mmaps: dict[str, list[np.ndarray]] = {}
        for cam_key in self.camera_keys:
            safe_key = cam_key.replace("/", ".")
            cam_dir = cache_dir / safe_key
            self._frame_mmaps[cam_key] = [
                np.load(str(cam_dir / f"episode_{i:06d}.npy"), mmap_mode=load_mode)
                for i in range(total_eps)
            ]

        # Global frame index → (episode_idx, local_offset within episode)
        total_frames = self._lerobot_ds.meta.episodes[-1]["dataset_to_index"]
        self._global_ep_idx = np.empty(total_frames, dtype=np.int32)
        self._global_local_off = np.empty(total_frames, dtype=np.int32)
        for ep_idx, ep in enumerate(self._lerobot_ds.meta.episodes):
            f, t = ep["dataset_from_index"], ep["dataset_to_index"]
            self._global_ep_idx[f:t] = ep_idx
            self._global_local_off[f:t] = np.arange(t - f, dtype=np.int32)

        # Pre-cache state and action from parquet — eliminates all LeRobot __getitem__
        # overhead in the training hot-path. State+action are ~1.5 MB total.
        data_files = sorted((dataset_path / "data").rglob("*.parquet"), key=str)
        df = pd.concat([pd.read_parquet(f) for f in data_files]).sort_values("index")
        self._state_cache: np.ndarray = np.stack(df[self.state_key].values).astype(np.float32)
        self._action_cache: np.ndarray = np.stack(df[self.action_key].values).astype(np.float32)

    def _compute_valid_indices(self, ds: LeRobotDataset) -> list[int]:
        """Return global frame indices with complete pred_horizon-step futures."""
        valid: list[int] = []
        for i in range(ds.meta.total_episodes):
            ep = ds.meta.episodes[i]
            from_idx: int = ep["dataset_from_index"]
            to_idx: int = ep["dataset_to_index"]  # exclusive upper bound
            last_valid = to_idx - self.pred_horizon
            if last_valid >= from_idx:
                valid.extend(range(from_idx, last_valid + 1))
        return valid

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> Batch:
        global_idx = self._valid_indices[idx]
        if self._use_cache:
            return self._getitem_cached(global_idx)
        return self._getitem_lerobot(global_idx)

    def _getitem_lerobot(self, global_idx: int) -> Batch:
        raw = self._lerobot_ds[global_idx]
        images: dict[str, torch.Tensor] = {}
        for key in self.camera_keys:
            images[key] = self._resize(raw[key])
        return Batch(
            images=images,
            state=raw[self.state_key],
            actions=raw[self.action_key],
        )

    def _getitem_cached(self, global_idx: int) -> Batch:
        ep_idx = int(self._global_ep_idx[global_idx])
        local_off = int(self._global_local_off[global_idx])

        # Observation window: clamp to episode start (repeats first frame like LeRobot).
        obs_locals = [
            max(0, local_off - (self.obs_horizon - 1 - i))
            for i in range(self.obs_horizon)
        ]

        images: dict[str, torch.Tensor] = {}
        for cam_key in self.camera_keys:
            mmap = self._frame_mmaps[cam_key][ep_idx]
            # (T, H, W, 3) uint8 → (T, C, H, W) float32 in [0, 1]
            frames_np = np.stack([mmap[j] for j in obs_locals])
            frames_t = torch.from_numpy(frames_np.copy()).permute(0, 3, 1, 2).float().div_(255.0)
            images[cam_key] = self._resize(frames_t)

        ep_from = self._lerobot_ds.meta.episodes[ep_idx]["dataset_from_index"]
        obs_globals = [ep_from + j for j in obs_locals]
        state = torch.from_numpy(self._state_cache[obs_globals].copy())
        actions = torch.from_numpy(
            self._action_cache[global_idx : global_idx + self.pred_horizon].copy()
        )
        return Batch(images=images, state=state, actions=actions)

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
