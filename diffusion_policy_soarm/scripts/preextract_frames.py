"""Pre-extract all video frames to disk as episode numpy arrays.

Eliminates PyAV video decoding from the training hot-path: __getitem__ becomes
a numpy mmap read instead of a seek + decode, pushing GPU utilisation to ~100%.

Video files contain extra reset frames between episodes.  We use LeRobot's own
decoder so the timestamp mapping is always correct regardless of how the video
was encoded.

Cache layout:
    {dataset_path}/frame_cache/
        {sanitized_cam_key}/episode_000000.npy   # uint8 (N, H, W, 3)
        ...

Usage:
    uv run python -m diffusion_policy_soarm.scripts.preextract_frames \\
        --config diffusion_policy_soarm/configs/base.yaml \\
        [--num-workers N]   # default: all CPU cores

Safe to interrupt and re-run: episodes with existing .npy files are skipped.
"""

from __future__ import annotations

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
from tqdm import tqdm

from diffusion_policy_soarm.utils.config import load_config

# ---------------------------------------------------------------------------
# Per-worker state (initialised once per process via Pool initializer)
# ---------------------------------------------------------------------------

_ds: LeRobotDataset | None = None
_camera_keys: list[str] = []
_target_hw: tuple[int, int] = (480, 640)
_cache_dir: Path = Path()


def _worker_init(cfg_path: str, overrides: list[str] | None) -> None:
    global _ds, _camera_keys, _target_hw, _cache_dir

    cfg = load_config(cfg_path, overrides)
    dataset_path = Path(cfg.dataset.path)
    _camera_keys = list(cfg.dataset.camera_keys)
    _target_hw = tuple(cfg.dataset.image_size)  # type: ignore[assignment]
    _cache_dir = dataset_path / "frame_cache"
    fps: float = cfg.dataset.fps

    _ds = LeRobotDataset(
        repo_id=dataset_path.name,
        root=dataset_path,
        download_videos=False,
        video_backend="pyav",
        delta_timestamps={cam_key: [0.0] for cam_key in _camera_keys},
        tolerance_s=0.5 / fps,
    )


def _extract_episode(ep_idx: int) -> int:
    """Extract one episode and save per-camera numpy arrays. Returns ep_idx."""
    assert _ds is not None

    out_paths = {
        cam_key: _cache_dir / cam_key.replace("/", ".") / f"episode_{ep_idx:06d}.npy"
        for cam_key in _camera_keys
    }
    if all(p.exists() for p in out_paths.values()):
        return ep_idx

    ep = _ds.meta.episodes[ep_idx]
    from_idx: int = ep["dataset_from_index"]
    to_idx: int = ep["dataset_to_index"]

    ep_frames: dict[str, list[np.ndarray]] = {k: [] for k in _camera_keys}
    h, w = _target_hw

    for global_idx in range(from_idx, to_idx):
        sample = _ds[global_idx]
        for cam_key in _camera_keys:
            frame_chw = sample[cam_key]
            if frame_chw.ndim == 4:
                frame_chw = frame_chw[0]  # (1, C, H, W) → (C, H, W)
            frame_u8 = (frame_chw.permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            if frame_u8.shape[:2] != (h, w):
                frame_u8 = np.asarray(
                    Image.fromarray(frame_u8).resize((w, h), Image.BILINEAR),
                    dtype=np.uint8,
                )
            ep_frames[cam_key].append(frame_u8)

    for cam_key, frames in ep_frames.items():
        np.save(str(out_paths[cam_key]), np.stack(frames))  # (N, H, W, 3) uint8

    return ep_idx


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def preextract(cfg_path: str, overrides: list[str] | None, num_workers: int) -> None:
    cfg = load_config(cfg_path, overrides)
    dataset_path = Path(cfg.dataset.path)
    camera_keys: list[str] = list(cfg.dataset.camera_keys)

    cache_dir = dataset_path / "frame_cache"
    for cam_key in camera_keys:
        (cache_dir / cam_key.replace("/", ".")).mkdir(parents=True, exist_ok=True)

    # Peek at episode count without full init
    from lerobot.datasets.lerobot_dataset import LeRobotDataset as _LR
    _meta_ds = _LR(
        repo_id=dataset_path.name,
        root=dataset_path,
        download_videos=False,
        video_backend="pyav",
    )
    total_episodes: int = _meta_ds.meta.total_episodes
    del _meta_ds

    print(f"Extracting {total_episodes} episodes × {len(camera_keys)} cameras "
          f"with {num_workers} workers → {cache_dir}")

    with Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(cfg_path, overrides),
    ) as pool:
        for _ in tqdm(
            pool.imap_unordered(_extract_episode, range(total_episodes)),
            total=total_episodes,
            unit="ep",
        ):
            pass

    print(f"\nDone. Cache written to: {cache_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-extract video frames to disk.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-workers", type=int, default=os.cpu_count())
    args, overrides = parser.parse_known_args()
    preextract(args.config, overrides or None, args.num_workers)


if __name__ == "__main__":
    main()
