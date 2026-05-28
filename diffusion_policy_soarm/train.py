"""Training entry point for Diffusion Policy on SO-ARM101.

Usage:
    uv run python -m diffusion_policy_soarm.train \\
        --config diffusion_policy_soarm/configs/base.yaml \\
        [--override diffusion_policy_soarm/configs/ablations/transformer_backbone.yaml] \\
        [key=value overrides, e.g. training.batch_size=8]

The resolved config and git hash are saved to the run directory at startup.
"""

from __future__ import annotations

import argparse
import copy
import math
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

from diffusion_policy_soarm.data.dataset import DiffusionDataset
from diffusion_policy_soarm.data.normalization import build_normalizers
from diffusion_policy_soarm.models.diffusion import DiffusionModule, build_denoiser
from diffusion_policy_soarm.models.encoders import ObservationEncoder
from diffusion_policy_soarm.utils.config import (
    load_config,
    merge_config,
    resolve_run_dir,
    save_config,
)
from diffusion_policy_soarm.utils.seed import seed_everything


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

class EMAModel:
    """Exponential moving average of model parameters.

    Maintains a shadow copy of all parameters.  The shadow copy is used at
    eval/inference time for better generalisation.

    Args:
        model: Model whose parameters to track.
        decay: EMA decay rate (typical: 0.9999).
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module) -> None:
        """θ_ema ← decay · θ_ema + (1 − decay) · θ"""
        with torch.no_grad():
            for s_p, m_p in zip(self.shadow.parameters(), model.parameters()):
                s_p.mul_(self.decay).add_((1.0 - self.decay) * m_p.data)

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def to(self, device: torch.device) -> "EMAModel":
        self.shadow = self.shadow.to(device)
        return self


# ---------------------------------------------------------------------------
# LR scheduler with linear warm-up
# ---------------------------------------------------------------------------

def make_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    schedule: str = "cosine",
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine-decay LR scheduler with a linear warm-up phase."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if schedule == "cosine":
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return 1.0  # constant

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def move_batch(batch: dict, device: torch.device) -> dict:
    """Recursively move tensors (including nested image dicts) to device."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device, non_blocking=True) for kk, vv in v.items()}
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg, ds: DiffusionDataset) -> tuple[DiffusionModule, dict]:
    """Construct the full DiffusionModule from config and dataset."""
    normalizers = build_normalizers(ds.lerobot_dataset, cfg)
    camera_keys = list(cfg.dataset.camera_keys)
    encoder = ObservationEncoder(cfg, camera_keys=camera_keys, state_dim=ds.state_dim)
    denoiser = build_denoiser(cfg, action_dim=ds.action_dim, obs_cond_dim=encoder.output_dim)
    model = DiffusionModule(
        cfg, encoder, denoiser, normalizers["action"], normalizers["state"]
    )
    return model, normalizers


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    run_dir: Path,
    tag: str,
    model: nn.Module,
    ema: EMAModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    step: int,
) -> None:
    ckpt = {
        "epoch": epoch,
        "step": step,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    torch.save(ckpt, ckpt_dir / f"{tag}.pt")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    ema: EMAModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
) -> tuple[int, int]:
    ckpt = torch.load(path, weights_only=True)
    model.load_state_dict(ckpt["model"])
    ema.shadow.load_state_dict(ckpt["ema"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt["step"]


# ---------------------------------------------------------------------------
# Core training loop (shared between train.py and overfit_check.py)
# ---------------------------------------------------------------------------

def resolve_checkpoint(resume: str | Path) -> Path:
    """Return the checkpoint .pt path from a file or run-directory argument.

    Accepts either:
    - A direct path to a ``.pt`` file.
    - A run directory — auto-selects ``checkpoints/latest.pt``.
    """
    p = Path(resume)
    if p.is_file():
        return p
    latest = p / "checkpoints" / "latest.pt"
    if latest.exists():
        return latest
    raise FileNotFoundError(f"No checkpoint found at {resume}")


def run_training(
    cfg,
    run_dir: Path,
    episode_ids: list[int] | None = None,
    max_steps: int | None = None,
    resume_from: str | Path | None = None,
) -> DiffusionModule:
    """Main training loop.

    Args:
        cfg: Resolved OmegaConf config.
        run_dir: Directory to write logs, checkpoints, and config.
        episode_ids: If set, restrict training to these episode indices
            (used by the overfit sanity check).
        max_steps: Hard stop after this many gradient steps (overfit check).
        resume_from: Path to a checkpoint ``.pt`` file or a run directory
            containing ``checkpoints/latest.pt``.  When provided, model /
            EMA / optimiser / scheduler states are restored and training
            continues from the saved epoch and step.

    Returns:
        The EMA model (shadow weights) for downstream evaluation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Dataset & dataloader -----------------------------------------------
    ds = DiffusionDataset(cfg, episode_ids=episode_ids)
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

            if max_steps is not None and step >= max_steps:
                break

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        writer.add_scalar("train/epoch_loss", mean_loss, epoch)
        print(f"Epoch {epoch+1:4d}  loss={mean_loss:.5f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if mean_loss < best_loss:
            best_loss = mean_loss
            save_checkpoint(run_dir, "best", model, ema, optimizer, scheduler, epoch, step)

        save_checkpoint(run_dir, "latest", model, ema, optimizer, scheduler, epoch, step)

        if max_steps is not None and step >= max_steps:
            break

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
