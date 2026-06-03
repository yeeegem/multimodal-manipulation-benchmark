"""Real-time receding-horizon inference loop for the SO-ARM101 arm.

Usage:
    uv run python -m diffusion_policy_soarm.infer \\
        --checkpoint runs/<experiment>/<timestamp>/checkpoints/best.pt \\
        --config runs/<experiment>/<timestamp>/config.yaml \\
        [--dry-run] \\
        [key=value overrides, e.g. infer.num_ddim_steps=5]

Latency budget: observation capture + DDIM sampling must finish within one
step period (1/fps ≈ 33 ms at 30 Hz).  Use --dry-run to test without the arm.
"""

from __future__ import annotations

import argparse
import collections
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower
from omegaconf import OmegaConf

from diffusion_policy_soarm.data.normalization import Normalizer
from diffusion_policy_soarm.models.diffusion import shift_action_chunk_for_warm_start
from diffusion_policy_soarm.models.factory import build_policy
from diffusion_policy_soarm.utils.config import load_config


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def _build_inference_model(cfg, run_dir: Path) -> nn.Module:
    """Build the policy from saved normalizer files (no dataset needed).

    Derives action_dim and state_dim from the normalizer JSON files saved
    alongside the checkpoint, then assembles the policy selected by
    ``cfg.model.type`` (diffusion or BC) via ``build_policy``.
    """
    action_norm = Normalizer.from_file(run_dir / "action_normalizer.json")
    state_norm = Normalizer.from_file(run_dir / "state_normalizer.json")
    action_dim = int(action_norm.mins.shape[0])
    state_dim = int(state_norm.mins.shape[0])

    return build_policy(
        cfg,
        action_dim=action_dim,
        state_dim=state_dim,
        action_normalizer=action_norm,
        state_normalizer=state_norm,
    )


# ---------------------------------------------------------------------------
# Observation preprocessing
# ---------------------------------------------------------------------------

def _preprocess_obs_buffer(
    obs_buffer: collections.deque,
    camera_key_map: dict[str, str],
    motor_names: list[str],
    image_size: tuple[int, int],
    device: torch.device,
) -> dict:
    """Convert a deque of raw robot obs dicts into a model-ready batch.

    Each element of *obs_buffer* is a single flat dict returned by
    ``robot.get_observation()`` containing both camera frames and motor
    positions, e.g.::

        {
          "front":              np.ndarray (H_raw, W_raw, 3) uint8,
          "wrist":              np.ndarray (H_raw, W_raw, 3) uint8,
          "shoulder_pan.pos":   12.3,
          "shoulder_lift.pos": -45.7,
          ...
        }

    Args:
        obs_buffer: Deque of length obs_horizon, each element a dict from
            ``robot.get_observation()``.
        camera_key_map: Maps the robot's short camera name (e.g. "front") →
            the dataset feature key (e.g. "observation.images.front") that
            the encoder's per-camera ResNet was registered under during
            training.
        motor_names: Ordered motor names matching the training state vector.
        image_size: Target (H, W) to resize camera frames to.
        device: Target device for output tensors.

    Returns:
        Dict with:
          "images": {dataset_key: (1, T_o, C, H, W) float32}
          "state":  (1, T_o, state_dim) float32
    """
    target_h, target_w = image_size

    # One (1, T_o, C, H, W) tensor per camera, keyed by the dataset feature
    # name the encoder expects.
    images: dict[str, torch.Tensor] = {}
    for cam_name, dataset_key in camera_key_map.items():
        # Walk the obs_horizon-long rolling buffer to stack this camera's
        # frames in temporal order: index 0 = oldest, -1 = newest.
        frames: list[torch.Tensor] = []
        for obs in obs_buffer:
            # uint8 HWC numpy → float32 CHW tensor in [0, 1].
            # The encoder applies ImageNet normalisation internally.
            raw_frame = obs[cam_name]
            frame = (
                torch.from_numpy(np.ascontiguousarray(raw_frame))
                .permute(2, 0, 1)
                .float()
                .div_(255.0)
            )
            frames.append(frame)

        # (T_o, C, H_raw, W_raw)
        stacked = torch.stack(frames, dim=0)

        # Cameras capture at 480x640; the encoder was trained on 96x96.
        # Skip the interpolate call when shapes already match (dry-run uses
        # synthetic frames already at target size).
        if stacked.shape[-2:] != (target_h, target_w):
            stacked = F.interpolate(
                stacked, size=(target_h, target_w), mode="bilinear", align_corners=False
            )

        # Prepend batch dim → (1, T_o, C, H, W) and move to GPU.
        images[dataset_key] = stacked.unsqueeze(0).to(device)

    # State vector: one row per buffered observation, columns in motor_names
    # order so dimension i always means the same joint as during training.
    state_frames: list[torch.Tensor] = []
    for obs in obs_buffer:
        positions = [float(obs[f"{m}.pos"]) for m in motor_names]
        state_frames.append(torch.tensor(positions, dtype=torch.float32))
    state = torch.stack(state_frames, dim=0).unsqueeze(0).to(device)  # (1, T_o, state_dim)

    return {"images": images, "state": state}


# ---------------------------------------------------------------------------
# Control loop
# ---------------------------------------------------------------------------

def _run_loop(
    cfg,
    model: nn.Module,
    robot: SOFollower | None,
    motor_names: list[str],
    dry_run: bool,
    device: torch.device,
) -> None:
    """Receding-horizon control loop.

    Each iteration:
      1. Capture one fresh observation (or use synthetic obs in dry-run).
      2. Build model batch from obs_horizon-length rolling buffer.
      3. Run DDIM inference to get a pred_horizon-step action chunk.
      4. Execute exec_horizon steps at 1/fps Hz, then replan.

    Latency (obs capture + encode + DDIM sample) is tracked over a 100-step
    rolling window and printed each iteration as mean and p95.
    """
    obs_horizon: int = cfg.model.obs_horizon
    exec_horizon: int = cfg.infer.exec_horizon
    fps: float = cfg.dataset.fps
    image_size: tuple[int, int] = tuple(cfg.dataset.image_size)
    camera_key_map: dict[str, str] = dict(cfg.infer.camera_key_map)
    step_duration = 1.0 / fps

    latencies: collections.deque = collections.deque(maxlen=100)
    obs_buffer: collections.deque = collections.deque(maxlen=obs_horizon)

    # Pre-fill observation buffer with obs_horizon frames before the loop.
    if dry_run:
        cam_names = list(camera_key_map.keys())
        cam_h = int(cfg.infer.cameras[cam_names[0]].height)
        cam_w = int(cfg.infer.cameras[cam_names[0]].width)
        for _ in range(obs_horizon):
            fake: dict = {name: np.zeros((cam_h, cam_w, 3), dtype=np.uint8) for name in cam_names}
            for m in motor_names:
                fake[f"{m}.pos"] = 0.0
            obs_buffer.append(fake)
    else:
        for _ in range(obs_horizon):
            obs_buffer.append(robot.get_observation())

    warm_start_actions_norm: torch.Tensor | None = None

    print("Inference loop started. Press Ctrl-C to stop.")

    while True:
        t0 = time.perf_counter()

        if not dry_run:
            obs_buffer.append(robot.get_observation())

        batch = _preprocess_obs_buffer(obs_buffer, camera_key_map, motor_names, image_size, device)

        with torch.no_grad():
            actions, actions_norm = model.predict_actions(
                batch,
                warm_start_actions_norm=warm_start_actions_norm,
                return_normalized=True,
            )

        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        arr = list(latencies)
        mean_lat = float(np.mean(arr))
        p95_lat = float(np.percentile(arr, 95))
        print(f"inference {latency_ms:.1f} ms | mean {mean_lat:.1f} | p95 {p95_lat:.1f}")

        chunk = actions[0]  # (pred_horizon, action_dim)
        warm_start_actions_norm = shift_action_chunk_for_warm_start(
            actions_norm, exec_horizon
        ).detach()

        # print("actions:", actions[0, :4].cpu().numpy())  # first 4 steps
        # print("state:  ", batch["state"][0, -1].cpu().numpy())  # current position

        for i in range(exec_horizon):
            step_t0 = time.perf_counter()
            if not dry_run:
                action_vec = chunk[i].cpu().numpy().astype(np.float32)
                action_dict = {f"{m}.pos": float(v) for m, v in zip(motor_names, action_vec)}
                robot.send_action(action_dict)
            elapsed = time.perf_counter() - step_t0
            remaining = step_duration - elapsed
            if remaining > 0:
                time.sleep(remaining)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    # Surface LeRobot's own logging (camera/motor warnings) at a known level.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Real-time Diffusion Policy inference on SO-ARM101.")
    parser.add_argument("--checkpoint", required=True, help="Path to EMA checkpoint .pt file.")
    parser.add_argument("--config", required=True, help="Path to run config.yaml.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip robot connect/send; run inference on synthetic observations.",
    )
    args, overrides = parser.parse_known_args(argv)

    cfg = load_config(args.config, overrides or None)

    # Set inference steps from infer.num_ddim_steps; sampler comes from config/CLI.
    cfg = OmegaConf.merge(cfg, OmegaConf.create({
        "diffusion": {
            "num_inference_steps": int(cfg.infer.num_ddim_steps),
        }
    }))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # run_dir is the parent of the checkpoints/ directory.
    run_dir = Path(args.checkpoint).parent.parent

    model = _build_inference_model(cfg, run_dir)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["ema"])
    model = model.to(device)
    model.eval()

    if cfg.infer.compile:
        print("Compiling model with torch.compile (~30 s one-time warmup)...")
        model = torch.compile(model)

    if not args.dry_run:
        cameras = {
            name: OpenCVCameraConfig(
                index_or_path=cam_cfg.path,
                width=cam_cfg.width,
                height=cam_cfg.height,
                fps=cam_cfg.fps,
                fourcc=cam_cfg.fourcc,
                backend=cam_cfg.backend,
            )
            for name, cam_cfg in cfg.infer.cameras.items()
        }
        robot_cfg = SOFollowerRobotConfig(
            port=cfg.infer.robot_port,
            id=cfg.infer.robot_id,
            cameras=cameras,
        )
        robot = SOFollower(robot_cfg)
        robot.connect()
        motor_names = list(robot.bus.motors.keys())
        print(f"Robot connected. Motors: {motor_names}")

        # Lower the Feetech firmware-side acceleration ramp so commanded deltas
        # execute more gently. Persists until power cycle.
        accel = OmegaConf.select(cfg, "infer.motor_acceleration", default=None)
        if accel is not None:
            for motor in motor_names:
                robot.bus.write("Acceleration", motor, int(accel))
            print(f"Wrote Acceleration={int(accel)} to all {len(motor_names)} motors.")
    else:
        robot = None
        state_norm = Normalizer.from_file(run_dir / "state_normalizer.json")
        state_dim = int(state_norm.mins.shape[0])
        motor_names = [f"motor_{i}" for i in range(state_dim)]
        print(f"Dry-run mode. Simulating {state_dim} motors, {list(cfg.infer.cameras)!r} cameras.")

    try:
        _run_loop(cfg, model, robot, motor_names, args.dry_run, device)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if robot is not None:
            robot.disconnect()
            print("Robot disconnected.")


if __name__ == "__main__":
    main()
