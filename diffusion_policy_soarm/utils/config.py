"""Config loading, CLI override merging, and per-run directory setup."""

import datetime
import subprocess
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load a YAML config file and apply optional dot-notation CLI overrides.

    Args:
        config_path: Path to the base YAML config file.
        overrides: List of ``key=value`` or ``key.nested=value`` strings that
            are merged on top of the base config (last write wins).

    Returns:
        Fully-merged ``DictConfig``.
    """
    cfg = OmegaConf.load(config_path)
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)
    return cfg


def merge_config(base: DictConfig, override_path: str | Path) -> DictConfig:
    """Merge an ablation override YAML on top of a base config."""
    override = OmegaConf.load(override_path)
    return OmegaConf.merge(base, override)


def resolve_run_dir(cfg: DictConfig) -> Path:
    """Create a timestamped run directory.

    Layout:
    - With experiment name: ``<run_dir>/<experiment>/<YYYYMMDD_HHMMSS>/``
    - Without:              ``<run_dir>/<YYYYMMDD_HHMMSS>/``

    The git hash is saved inside the directory (``git_hash.txt``), not in the name.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment = getattr(cfg.training, "experiment", "") or ""
    base = Path(cfg.training.run_dir)
    run_dir = base / experiment / timestamp if experiment else base / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(cfg: DictConfig, run_dir: Path) -> None:
    """Persist the resolved config and git hash to *run_dir*.

    Files written:
    - ``config.yaml``: fully-resolved OmegaConf dump (all overrides applied).
    - ``git_hash.txt``: full commit SHA for exact reproduction.
    """
    OmegaConf.save(cfg, run_dir / "config.yaml")
    (run_dir / "git_hash.txt").write_text(_git_hash(short=False) + "\n")


def _git_hash(short: bool = True) -> str:
    flag = "--short" if short else None
    cmd = ["git", "rev-parse"] + ([flag] if flag else []) + ["HEAD"]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
