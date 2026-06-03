"""Reusable training mechanics: EMA, LR schedule, batch device move, checkpoints.

These are generic, model-agnostic helpers consumed by ``train.py``. Keeping them
here leaves ``train.py`` as a readable orchestration layer (build_model ->
run_training -> main).
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

class EMAModel:
    """Exponential moving average of model parameters.

    Maintains a shadow copy of all model parameters (encoder + denoiser/head).
    The shadow copy is used at eval/inference time for better generalisation.

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
# Device helper
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
