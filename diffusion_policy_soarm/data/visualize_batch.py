"""Phase 1 gate: print tensor shapes and visualise one training sample.

Shows the front-camera image sequence (obs_horizon frames) and the corresponding
action chunk as per-joint position traces.

Usage:
    uv run python -m diffusion_policy_soarm.data.visualize_batch \\
        --config diffusion_policy_soarm/configs/base.yaml \\
        [key=value overrides, e.g. dataset.path=recordings/redcubes_bluecup]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig

from diffusion_policy_soarm.data.dataset import DiffusionDataset
from diffusion_policy_soarm.data.normalization import build_normalizers
from diffusion_policy_soarm.utils.config import load_config
from diffusion_policy_soarm.utils.seed import seed_everything

# Joint names for the action / state plots
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def print_shapes(sample: dict, prefix: str = "") -> None:
    """Recursively print tensor shapes for a sample dict."""
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            print(f"  {prefix}{key}: {tuple(val.shape)} {val.dtype}")
        elif isinstance(val, dict):
            print(f"  {prefix}{key}:")
            print_shapes(val, prefix=prefix + "  ")


def visualize_batch(cfg: DictConfig, save_path: Path = Path("batch_sample.png")) -> None:
    """Load one sample, print shapes, and save a visualisation figure.

    Args:
        cfg: Resolved OmegaConf config.
        save_path: Where to write the output PNG.
    """
    seed_everything(cfg.training.seed)

    print("Building dataset …")
    ds = DiffusionDataset(cfg)
    print(f"Dataset length (valid training samples): {len(ds)}")

    sample = ds[0]

    # --- print tensor shapes -------------------------------------------------
    print("\n--- Tensor shapes (one sample) ---")
    print(f"  images:")
    for cam_key, img in sample["images"].items():
        print(f"    {cam_key}: {tuple(img.shape)} {img.dtype}  (T, C, H, W)")
    print(f"  state:   {tuple(sample['state'].shape)}  (T_o, state_dim)")
    print(f"  actions: {tuple(sample['actions'].shape)}  (T_p, action_dim)")
    print()

    # --- verify normaliser ---------------------------------------------------
    normalizers = build_normalizers(ds.lerobot_dataset, cfg)
    norm_a = normalizers["action"].forward(sample["actions"])
    print(f"Normalised action range: [{norm_a.min():.3f}, {norm_a.max():.3f}]  (expect ≈ [-1, 1])")

    # --- plot ----------------------------------------------------------------
    n_cameras = len(sample["images"])
    obs_horizon = cfg.model.obs_horizon
    n_cols = obs_horizon + 1  # obs frames + action plot

    fig, axes = plt.subplots(
        n_cameras, n_cols, figsize=(4 * n_cols, 4 * n_cameras)
    )
    # Ensure 2-D axes array for uniform indexing
    if n_cameras == 1:
        axes = axes[None, :]

    for row, (cam_key, frames) in enumerate(sample["images"].items()):
        # frames: (T_o, C, H, W) float32 in [0, 1]
        for t in range(obs_horizon):
            ax = axes[row, t]
            img = frames[t].permute(1, 2, 0).numpy()  # (H, W, C)
            ax.imshow(img)
            ax.set_title(f"{cam_key.split('.')[-1]}  t-{obs_horizon - 1 - t}" if t < obs_horizon - 1 else f"{cam_key.split('.')[-1]}  t")
            ax.axis("off")

    # Action chunk traces (last column, spanning all camera rows)
    ax_act = axes[0, -1]
    actions = sample["actions"].numpy()  # (T_p, action_dim)
    t_axis = range(actions.shape[0])
    for j, name in enumerate(JOINT_NAMES):
        ax_act.plot(t_axis, actions[:, j], label=name)
    ax_act.set_title(f"Action chunk (T_p={cfg.model.pred_horizon})")
    ax_act.set_xlabel("Step")
    ax_act.set_ylabel("Joint angle (°)")
    ax_act.legend(fontsize=7)

    # Hide remaining camera rows in the action column
    for row in range(1, n_cameras):
        axes[row, -1].axis("off")

    fig.suptitle("DiffusionDataset — sample 0", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    print(f"Figure saved to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise one DiffusionDataset sample.")
    parser.add_argument("--config", required=True, help="Path to base config YAML.")
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.config, overrides or None)
    visualize_batch(cfg)


if __name__ == "__main__":
    main()
