"""Phase 4 gate: overfit 5 demonstration episodes to verify the pipeline end-to-end.

A correctly wired pipeline should be able to memorise 5 short episodes:
- Loss must reach < 0.01 within max_steps gradient steps.
- Mean action reconstruction error (after DDPM sampling) must be < 0.05 in
  normalised space (i.e. < 2.5% of the [-1, 1] action range per joint).

If either check fails, there is a bug in the model, loss, or data pipeline —
not a generalisation problem, since we are deliberately overfitting.

Usage:
    uv run python -m diffusion_policy_soarm.scripts.overfit_check \\
        --config diffusion_policy_soarm/configs/base.yaml \\
        [key=value overrides]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from diffusion_policy_soarm.data.dataset import DiffusionDataset
from diffusion_policy_soarm.train import build_model, move_batch
from diffusion_policy_soarm.utils.config import load_config, resolve_run_dir, save_config
from diffusion_policy_soarm.utils.seed import seed_everything

OVERFIT_EPISODES = list(range(5))   # episodes 0–4
MAX_STEPS = 3000                     # hard step cap
LOSS_THRESHOLD = 0.01                # must reach this loss
RECON_THRESHOLD = 0.05               # normalised MAE on action reconstruction


def run_overfit_check(cfg_path: str, overrides: list[str] | None = None) -> None:
    from diffusion_policy_soarm.train import EMAModel, make_lr_scheduler, run_training

    cfg = load_config(cfg_path, overrides)

    # Smaller batch and more aggressive LR for fast memorisation
    overfit_overrides = [
        "training.batch_size=8",
        "training.num_epochs=200",
        "training.lr=3e-4",
        "training.warmup_steps=100",
        "training.log_every=50",
        "training.save_every=999",   # suppress epoch saves; we only need latest/best
        "training.num_workers=2",
    ]
    cfg = load_config(cfg_path, (overrides or []) + overfit_overrides)

    seed_everything(cfg.training.seed)
    run_dir = Path("runs") / "overfit_check"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir)

    print("=" * 60)
    print("Phase 4 overfit sanity check")
    print(f"  Episodes: {OVERFIT_EPISODES}")
    print(f"  Max steps: {MAX_STEPS}")
    print(f"  Loss threshold: < {LOSS_THRESHOLD}")
    print(f"  Recon threshold (normalised MAE): < {RECON_THRESHOLD}")
    print("=" * 60)

    # Train on 5 episodes
    ema_model = run_training(cfg, run_dir, episode_ids=OVERFIT_EPISODES, max_steps=MAX_STEPS)

    # --- Evaluate final loss on the same 5 episodes -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = DiffusionDataset(cfg, episode_ids=OVERFIT_EPISODES)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    ema_model = ema_model.to(device).eval()

    losses, recon_errors = [], []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)

            # Loss on EMA model
            loss = ema_model.compute_loss(batch)
            losses.append(loss.item())

            # Reconstruction: sample action chunk and compare to normalised GT
            act_norm = ema_model.action_normalizer(batch["actions"])
            pred_norm = ema_model.predict_actions(
                {"images": batch["images"], "state": batch["state"]}
            )
            pred_norm_rescaled = ema_model.action_normalizer(pred_norm)
            mae = (pred_norm_rescaled - act_norm).abs().mean().item()
            recon_errors.append(mae)

    final_loss = sum(losses) / len(losses)
    final_recon = sum(recon_errors) / len(recon_errors)

    print()
    print("Results:")
    print(f"  Final loss (EMA model):          {final_loss:.5f}  (threshold < {LOSS_THRESHOLD})")
    print(f"  Reconstruction MAE (normalised): {final_recon:.5f}  (threshold < {RECON_THRESHOLD})")

    loss_ok = final_loss < LOSS_THRESHOLD
    recon_ok = final_recon < RECON_THRESHOLD
    print()
    print(f"  Loss gate:  {'PASS ✓' if loss_ok else 'FAIL ✗'}")
    print(f"  Recon gate: {'PASS ✓' if recon_ok else 'FAIL ✗'}")

    if loss_ok and recon_ok:
        print()
        print("Phase 4 gate: PASS — pipeline is correctly wired.")
    else:
        print()
        print("Phase 4 gate: FAIL — check model, loss, or data pipeline for bugs.")
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 overfit sanity check.")
    parser.add_argument("--config", required=True, help="Path to base YAML config.")
    args, overrides = parser.parse_known_args()
    run_overfit_check(args.config, overrides or None)


if __name__ == "__main__":
    main()
