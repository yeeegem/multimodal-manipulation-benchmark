"""Open-loop diagnostic: replay recorded demos through a trained checkpoint.

No robot is involved. For every frame of the recorded demos the policy is fed the
observation that was actually recorded, and its predicted action is compared to what
the human actually did next. Because the policy never drives the arm, errors do not
compound, so this isolates one question: given the correct observation, can the policy
produce the correct action?

The task is deliberately multimodal: two identical cubes (left and right) are present,
so picking either side is valid. That has two consequences for how error is measured:

  - The horizontal choice (shoulder_pan) is *free* at the start. Scoring it against a
    single demo is meaningless, so direction is measured by mode coverage instead:
    sample the policy many times on the ambiguous home observation and see whether it
    ever picks both sides, or always collapses to one.
  - The descent and grasp (shoulder_lift, elbow_flex, wrist_flex, gripper) are
    *determined* once the arm is over a cube. In open loop the observation (including the
    arm's own joint angles) has already fixed which cube by then, so per-joint error is
    meaningful. It is reported over the grasp window (the frames where the demo is
    descending/closing). This is the main metric: it directly measures the "reaches but
    never goes down to grasp" failure.

Usage:
    uv run python -m scripts.diagnose_openloop \\
        --checkpoint runs/main_96x96/20260602_084823/checkpoints/best.pt \\
        --config     runs/main_96x96/20260602_084823/config.yaml \\
        [--episodes 10] [--stride 1] [--samples 8] [--output <dir>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy_soarm.data.dataset import DiffusionDataset
from diffusion_policy_soarm.data.normalization import Normalizer
from diffusion_policy_soarm.models.factory import build_policy
from diffusion_policy_soarm.utils.config import load_config

# Action/state column order (from the dataset's meta/info.json).
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
PAN = 0       # base rotation: the free left/right choice (multimodal)
GRIPPER = 5   # open/close
# The joints that actually carry the "descend and grasp" motion - the main metric.
DESCENT = [1, 2, 3]  # shoulder_lift, elbow_flex, wrist_flex


def ensure_sampler_keys(cfg) -> None:
    """Backfill the inference sampler keys when loading an older run config.

    The sampler knobs (sampler/num_inference_steps/clip_sample) used to live in the
    `diffusion:` block and were later moved to `infer:`. Older saved configs lack the
    `infer:` versions, which `DiffusionModule` reads. Filling in DDIM defaults lets the
    historical run load without editing its saved file. No-op for current configs.
    """
    defaults = {"sampler": "ddim", "num_inference_steps": 10, "clip_sample": True}
    for key, value in defaults.items():
        if OmegaConf.select(cfg, f"infer.{key}") is None:
            OmegaConf.update(cfg, f"infer.{key}", value)


def build_model(cfg, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    """Build the policy from the saved normalizers and load EMA weights."""
    run_dir = checkpoint.parent.parent
    action_norm = Normalizer.from_file(run_dir / "action_normalizer.json")
    state_norm = Normalizer.from_file(run_dir / "state_normalizer.json")
    model = build_policy(
        cfg,
        action_dim=int(action_norm.mins.shape[0]),
        state_dim=int(state_norm.mins.shape[0]),
        action_normalizer=action_norm,
        state_normalizer=state_norm,
    )
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["ema"])
    return model.to(device).eval()


def group_positions_by_episode(dataset: DiffusionDataset) -> list[tuple[int, list[int]]]:
    """Return [(episode_idx, [dataset positions])] in episode order.

    A "position" is an index into the dataset (what __getitem__ takes), not a global
    frame index. Positions within an episode are returned in time order.
    """
    episodes = dataset.lerobot_dataset.meta.episodes
    valid = dataset._valid_indices
    groups: list[tuple[int, list[int]]] = []
    for ep_idx, ep in enumerate(episodes):
        start, end = ep["dataset_from_index"], ep["dataset_to_index"]
        positions = [pos for pos, g in enumerate(valid) if start <= g < end]
        if positions:
            groups.append((ep_idx, positions))
    return groups


def make_batch(sample: dict, device: torch.device, repeat: int = 1) -> dict:
    """Turn one dataset sample into a model batch, optionally repeated `repeat` times.

    Repeating the same observation lets the policy draw several independent action
    samples for it in a single forward pass (used for mode coverage).
    """
    images = {
        k: v.unsqueeze(0).repeat(repeat, *([1] * v.ndim)).to(device)
        for k, v in sample["images"].items()
    }
    state = sample["state"].unsqueeze(0).repeat(repeat, *([1] * sample["state"].ndim)).to(device)
    return {"images": images, "state": state}


@torch.no_grad()
def predict_episode(model, dataset, positions, device) -> tuple[np.ndarray, np.ndarray]:
    """Run the policy on each frame of one episode (open loop).

    Returns (pred, gt), each of shape (n_frames, action_dim): the action the policy
    would execute next vs the action the human actually did next.
    """
    preds, gts = [], []
    for pos in positions:
        sample = dataset[pos]
        pred_chunk = model.predict_actions(make_batch(sample, device))  # (1, T_pred, A), degrees
        preds.append(pred_chunk[0, 0].cpu().numpy())  # first step = the next action
        gts.append(sample["actions"][0].numpy())
    return np.stack(preds), np.stack(gts)


def reached_side(pan_trajectory: np.ndarray) -> int:
    """Which side the base rotates toward over a chunk: +1 if pan rises, else -1."""
    return 1 if pan_trajectory[-1] >= pan_trajectory[0] else -1


@torch.no_grad()
def sample_start_sides(model, dataset, first_pos, device, n_samples: int) -> list[int]:
    """Sample `n_samples` action chunks for an episode's first (ambiguous) frame.

    Returns the chosen side (+1 / -1) for each sample, from the net pan travel of the
    predicted chunk. A healthy multimodal policy mixes both signs; a collapsed one
    (or any deterministic BC) returns all the same.
    """
    sample = dataset[first_pos]
    chunks = model.predict_actions(make_batch(sample, device, repeat=n_samples))  # (N, T, A)
    pan = chunks[:, :, PAN].cpu().numpy()  # (N, T_pred)
    return [reached_side(pan[i]) for i in range(n_samples)]


def grasp_window_mask(gt_gripper: np.ndarray) -> np.ndarray:
    """Boolean mask of frames where the demo has actuated the gripper from its start.

    Sign-agnostic: marks frames where the gripper has moved more than half its range
    away from its initial (resting/open) value, i.e. the descend-and-hold phase.
    """
    rng = gt_gripper.max() - gt_gripper.min()
    if rng < 1e-6:
        return np.zeros(len(gt_gripper), dtype=bool)
    return np.abs(gt_gripper - gt_gripper[0]) > 0.5 * rng


def plot_episode(
    ep_idx: int, pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, out_path: Path
) -> None:
    """Save a 6-joint predicted-vs-demo overlay, with the grasp window shaded."""
    frames = np.arange(len(gt))
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for joint, ax in enumerate(axes.ravel()):
        ax.plot(frames, gt[:, joint], label="demo (truth)", color="tab:green")
        ax.plot(frames, pred[:, joint], label="policy", color="tab:red", linestyle="--")
        if mask.any():
            ax.axvspan(frames[mask][0], frames[mask][-1], color="tab:blue", alpha=0.08)
        ax.set_ylabel(f"{JOINT_NAMES[joint]} (deg)")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="best")
    fig.suptitle(f"Episode {ep_idx}: open-loop prediction vs demo (grasp window shaded)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Open-loop diagnostic for a trained policy.")
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint .pt file.")
    parser.add_argument("--config", required=True, help="Path to the run config.yaml.")
    parser.add_argument("--episodes", default="10", help="How many episodes to use, or 'all'.")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth frame (speed).")
    parser.add_argument("--samples", type=int, default=8, help="Mode-coverage samples per start.")
    parser.add_argument("--output", default=None, help="Output dir (default <run>/diagnostics).")
    args = parser.parse_args(argv)

    plt.switch_backend("Agg")  # no display needed; write PNGs to disk

    checkpoint = Path(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = load_config(args.config)
    ensure_sampler_keys(cfg)

    out_dir = Path(args.output) if args.output else checkpoint.parent.parent / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = DiffusionDataset(cfg)
    model = build_model(cfg, checkpoint, device)

    groups = group_positions_by_episode(dataset)
    if args.episodes != "all":
        groups = groups[: int(args.episodes)]

    all_errors: list[np.ndarray] = []     # per-frame |pred - gt|, every frame
    grasp_errors: list[np.ndarray] = []   # per-frame |pred - gt|, grasp-window frames only
    start_sides: list[int] = []           # +1/-1 mode-coverage samples across episode starts

    for ep_idx, positions in groups:
        positions = positions[:: args.stride]
        pred, gt = predict_episode(model, dataset, positions, device)
        mask = grasp_window_mask(gt[:, GRIPPER])

        all_errors.append(np.abs(pred - gt))
        if mask.any():
            grasp_errors.append(np.abs(pred[mask] - gt[mask]))
        start_sides.extend(sample_start_sides(model, dataset, positions[0], device, args.samples))

        plot_episode(ep_idx, pred, gt, mask, out_dir / f"episode_{ep_idx:03d}.png")
        print(f"episode {ep_idx}: {len(positions)} frames, grasp-window {int(mask.sum())} frames")

    mae_all = np.concatenate(all_errors, axis=0).mean(axis=0)
    mae_grasp = (
        np.concatenate(grasp_errors, axis=0).mean(axis=0) if grasp_errors else np.full(6, np.nan)
    )
    went_right = sum(1 for s in start_sides if s > 0)

    summary = {
        "n_episodes": len(groups),
        "per_joint_mae_deg_all_frames": {n: float(mae_all[i]) for i, n in enumerate(JOINT_NAMES)},
        "per_joint_mae_deg_grasp_window": {
            n: float(mae_grasp[i]) for i, n in enumerate(JOINT_NAMES)
        },
        "mode_coverage": {
            "samples": len(start_sides),
            "went_right": went_right,
            "went_left": len(start_sides) - went_right,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== open-loop summary (degrees) ===")
    print(f"{'joint':14s} {'all frames':>11s} {'grasp window':>13s}")
    for i, name in enumerate(JOINT_NAMES):
        marker = "  <- descent/grasp" if (i in DESCENT or i == GRIPPER) else ""
        print(f"{name:14s} {mae_all[i]:11.2f} {mae_grasp[i]:13.2f}{marker}")
    print(
        f"\nmode coverage (home obs, {args.samples} samples x {len(groups)} episodes): "
        f"left {len(start_sides) - went_right}, right {went_right}  "
        f"(a healthy multimodal policy mixes both; all-one-side = mode collapse)"
    )
    print(f"wrote {out_dir}/summary.json and per-episode plots")


if __name__ == "__main__":
    main()
