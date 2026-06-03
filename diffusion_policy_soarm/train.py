"""Training entry point for Diffusion Policy on SO-ARM101.

Usage:
    uv run python -m diffusion_policy_soarm.train \\
        --config diffusion_policy_soarm/configs/base.yaml \\
        [--override diffusion_policy_soarm/configs/ablations/transformer_backbone.yaml] \\
        [key=value overrides, e.g. training.batch_size=8]

The resolved config is saved to the run directory at startup.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning:torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from diffusion_policy_soarm.data.dataset import DiffusionDataset, frame_cache_dir
from diffusion_policy_soarm.data.normalization import build_normalizers
from diffusion_policy_soarm.models.factory import build_policy
from diffusion_policy_soarm.scripts.preextract_frames import preextract
from diffusion_policy_soarm.utils.config import (
    load_config,
    merge_config,
    resolve_run_dir,
    save_config,
)
from diffusion_policy_soarm.utils.seed import seed_everything
from diffusion_policy_soarm.utils.training import (
    EMAModel,
    load_checkpoint,
    make_lr_scheduler,
    move_batch,
    resolve_checkpoint,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg, ds: DiffusionDataset) -> tuple[nn.Module, dict]:
    """Construct the policy (diffusion or BC) from config and dataset."""
    normalizers = build_normalizers(ds.lerobot_dataset, cfg)
    model = build_policy(
        cfg,
        action_dim=ds.action_dim,
        state_dim=ds.state_dim,
        action_normalizer=normalizers["action"],
        state_normalizer=normalizers["state"],
    )
    return model, normalizers


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def run_training(
    cfg,
    run_dir: Path,
    resume_from: str | Path | None = None,
) -> nn.Module:
    """Main training loop.

    Args:
        cfg: Resolved OmegaConf config.
        run_dir: Directory to write logs, checkpoints, and config.
        resume_from: Path to a checkpoint ``.pt`` file or a run directory
            containing ``checkpoints/latest.pt``.  When provided, model /
            EMA / optimiser / scheduler states are restored and training
            continues from the saved epoch and step.

    Returns:
        The EMA model (shadow weights) for downstream evaluation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Frame cache check --------------------------------------------------
    cache_dir = frame_cache_dir(Path(cfg.dataset.path), tuple(cfg.dataset.image_size))
    if not cache_dir.exists():
        print("Frame cache not found. Running preextract_frames automatically...")
        preextract(cfg, num_workers=os.cpu_count())

    # --- Dataset & dataloader -----------------------------------------------
    ds = DiffusionDataset(cfg)
    print(f"Training samples: {len(ds)}")

    # persistent_workers=False: PyAV video decoding deadlocks with persistent workers.
    # Use persistent workers only when frame cache is active (no PyAV in workers).
    persistent = ds._use_cache and cfg.training.num_workers > 0
    loader = DataLoader(
        ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=persistent,
        prefetch_factor=2 if cfg.training.num_workers > 0 else None,
    )

    # --- Model --------------------------------------------------------------
    model, normalizers = build_model(cfg, ds)
    model = model.to(device)

    # Save normalizer stats so eval / infer can load them without the dataset.
    normalizers["action"].save(run_dir / "action_normalizer.json")
    normalizers["state"].save(run_dir / "state_normalizer.json")

    ema = EMAModel(model, decay=cfg.training.ema_decay).to(device)

    # --- Optimiser ----------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    steps_per_epoch = len(loader)
    total_steps = cfg.training.num_epochs * steps_per_epoch
    scheduler = make_lr_scheduler(
        optimizer,
        warmup_steps=cfg.training.warmup_steps,
        total_steps=total_steps,
        schedule=cfg.training.lr_scheduler,
    )

    # --- Resume -------------------------------------------------------------
    start_epoch = 0
    step = 0
    if resume_from is not None:
        ckpt_path = resolve_checkpoint(resume_from)
        print(f"Resuming from {ckpt_path}")
        start_epoch, step = load_checkpoint(ckpt_path, model, ema, optimizer, scheduler)
        start_epoch += 1  # saved epoch is the last completed one
        print(f"  Resumed at epoch {start_epoch}, global step {step}")

    # --- AMP ----------------------------------------------------------------
    use_amp: bool = cfg.training.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # --- Logging ------------------------------------------------------------
    writer = SummaryWriter(run_dir / "tb")
    best_loss = float("inf")

    # --- Training loop ------------------------------------------------------
    for epoch in range(start_epoch, cfg.training.num_epochs):
        model.train()
        epoch_losses: list[float] = []

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}", leave=False)
        for batch in pbar:
            batch = move_batch(batch, device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = model.compute_loss(batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)

            loss_val = loss.item()
            epoch_losses.append(loss_val)
            step += 1

            if step % cfg.training.log_every == 0:
                writer.add_scalar("train/loss", loss_val, step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)

            pbar.set_postfix(loss=f"{loss_val:.4f}")

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        writer.add_scalar("train/epoch_loss", mean_loss, epoch)
        print(f"Epoch {epoch+1:4d}  loss={mean_loss:.5f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if mean_loss < best_loss:
            best_loss = mean_loss
            save_checkpoint(run_dir, "best", model, ema, optimizer, scheduler, epoch, step)

        save_checkpoint(run_dir, "latest", model, ema, optimizer, scheduler, epoch, step)

    writer.close()
    return ema.shadow


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train Diffusion Policy on SO-ARM101.")
    parser.add_argument("--config", required=True, help="Path to base YAML config.")
    parser.add_argument("--override", default=None, help="Path to ablation override YAML.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a checkpoint .pt file or a run directory containing checkpoints/latest.pt.",
    )
    args, overrides = parser.parse_known_args(argv)

    cfg = load_config(args.config, overrides or None)
    if args.override:
        cfg = merge_config(cfg, args.override)

    seed_everything(cfg.training.seed)

    run_dir = resolve_run_dir(cfg)
    save_config(cfg, run_dir)
    print(f"Run directory: {run_dir}")

    run_training(cfg, run_dir, resume_from=args.resume)


if __name__ == "__main__":
    main()
