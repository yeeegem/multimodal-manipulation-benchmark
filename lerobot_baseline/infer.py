"""Real-time inference on SO-ARM101 using a LeRobot DiffusionPolicy checkpoint.

Usage:
    uv run python lerobot_baseline/infer.py \
        --checkpoint lerobot_baseline/runs/main/checkpoints/step_100000

LeRobot's policy manages observation queuing (n_obs_steps=2) and action
chunking (n_action_steps=8) internally.  The loop passes one fresh observation
per step and receives one action -- replanning happens automatically every
n_action_steps calls.
"""

from __future__ import annotations

import argparse
import collections
import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

ROBOT_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B42076683-if00"
ROBOT_ID = "valera"
FRONT_CAM_PATH = "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_111DD85F-video-index0"
WRIST_CAM_PATH = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0"
IMAGE_H, IMAGE_W = 96, 96
FPS = 30
MOTOR_ACCELERATION = 64


def build_obs(
    front: np.ndarray,
    wrist: np.ndarray,
    state: list[float],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    def frame_tensor(frame: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float().div_(255.0)
        if t.shape[-2:] != (IMAGE_H, IMAGE_W):
            t = F.interpolate(t.unsqueeze(0), size=(IMAGE_H, IMAGE_W), mode="bilinear", align_corners=False).squeeze(0)
        return t.unsqueeze(0).to(device)  # (1, C, H, W)

    return {
        "observation.images.front": frame_tensor(front),
        "observation.images.wrist": frame_tensor(wrist),
        "observation.state": torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device),
    }


def run_loop(policy: DiffusionPolicy, robot: SOFollower, motor_names: list[str]) -> None:
    device = next(policy.parameters()).device
    step_duration = 1.0 / FPS
    latencies: collections.deque = collections.deque(maxlen=100)

    policy.reset()
    print("Inference loop started. Press Ctrl-C to stop.")

    while True:
        t0 = time.perf_counter()

        raw = robot.get_observation()
        state = [float(raw[f"{m}.pos"]) for m in motor_names]
        obs = build_obs(raw["front"], raw["wrist"], state, device)

        with torch.no_grad():
            action = policy.select_action(obs)  # (action_dim,)

        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)
        arr = list(latencies)
        print(f"inference {latency_ms:.1f} ms | mean {np.mean(arr):.1f} | p95 {np.percentile(arr, 95):.1f}")

        action_np = action.cpu().numpy().astype(np.float32)
        robot.send_action({f"{m}.pos": float(v) for m, v in zip(motor_names, action_np)})

        elapsed = time.perf_counter() - t0
        if (remaining := step_duration - elapsed) > 0:
            time.sleep(remaining)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="LeRobot DiffusionPolicy inference on SO-ARM101.")
    parser.add_argument("--checkpoint", required=True, help="Path to LeRobot checkpoint directory.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    policy = DiffusionPolicy.from_pretrained(args.checkpoint)
    policy = policy.to(device).eval()

    cameras = {
        "front": OpenCVCameraConfig(
            index_or_path=FRONT_CAM_PATH, width=640, height=480, fps=FPS, fourcc="MJPG", backend=200
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=WRIST_CAM_PATH, width=640, height=480, fps=FPS, fourcc="MJPG", backend=200
        ),
    }
    robot = SOFollower(SOFollowerRobotConfig(port=ROBOT_PORT, id=ROBOT_ID, cameras=cameras))
    robot.connect()
    motor_names = list(robot.bus.motors.keys())
    print(f"Robot connected. Motors: {motor_names}")
    for motor in motor_names:
        robot.bus.write("Acceleration", motor, MOTOR_ACCELERATION)

    try:
        run_loop(policy, robot, motor_names)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
