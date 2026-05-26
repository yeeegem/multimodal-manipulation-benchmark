"""Global seeding utility for reproducibility."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed all RNGs and set cuDNN to deterministic mode.

    Covers Python stdlib, NumPy, PyTorch (CPU + all GPUs), and cuDNN flags.
    Call once at program start, before any weight initialisation or data loading.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Makes convolution algorithms deterministic at the cost of some speed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
