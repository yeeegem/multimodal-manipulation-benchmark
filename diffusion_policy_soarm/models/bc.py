"""Behavior-cloning (BC) MSE baseline.

The baseline shares the exact observation encoder, data pipeline, normalization,
and receding-horizon chunked control of the Diffusion Policy. The only differences
are the action head (a plain MLP instead of an iterative denoiser) and the training
objective (mean-squared error instead of epsilon-prediction).

This isolates the multimodality question to the modeling choice: a single
deterministic regression onto the demonstration actions cannot represent a
multimodal conditional, so on the bimodal pick (left cube ~50%, right cube ~50%)
the MSE optimum is the average of the two valid action chunks, which reaches into
the empty gap between the cubes. Diffusion Policy, by contrast, samples one mode.

``BCModule`` mirrors ``DiffusionModule``'s public interface (``compute_loss`` and
``predict_actions``) so the shared training loop (``train.py``) and inference loop
(``infer.py``) work unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# MLP action head
# ---------------------------------------------------------------------------

class MLPActionHead(nn.Module):
    """Map the observation conditioning vector to a full action chunk.

    A stack of ``Linear -> SiLU (-> Dropout)`` layers followed by a final linear
    projection to ``pred_horizon * action_dim``, reshaped into the action chunk.
    The chunk length matches the Diffusion Policy denoiser output so the two
    methods are directly comparable under identical receding-horizon execution.

    Args:
        cfg: Resolved config; reads ``bc`` and ``model`` sub-trees.
        action_dim: Dimensionality of one action step (6 for SO-101).
        obs_cond_dim: ObservationEncoder output dimension.
    """

    def __init__(self, cfg: DictConfig, action_dim: int, obs_cond_dim: int) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.pred_horizon: int = cfg.model.pred_horizon
        hidden_dims: list[int] = list(cfg.bc.hidden_dims)
        dropout: float = float(getattr(cfg.bc, "dropout", 0.0))

        layers: list[nn.Module] = []
        prev_dim = obs_cond_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, self.pred_horizon * action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs_cond: torch.Tensor) -> torch.Tensor:
        """Predict an action chunk from the conditioning vector.

        Args:
            obs_cond: ``(B, obs_cond_dim)`` conditioning vector.

        Returns:
            ``(B, pred_horizon, action_dim)`` predicted (normalised) action chunk.
        """
        flat = self.net(obs_cond)                                   # (B, T_p * A)
        return flat.view(-1, self.pred_horizon, self.action_dim)    # (B, T_p, A)


# ---------------------------------------------------------------------------
# BCModule — top-level model
# ---------------------------------------------------------------------------

class BCModule(nn.Module):
    """Top-level BC model: encoder + MLP head + normalizers.

    Mirrors ``DiffusionModule`` so the training and inference loops are shared.
    There is no noise schedule and no sampler: prediction is a single forward
    pass.

    Args:
        cfg: Full resolved config.
        encoder: ObservationEncoder (identical to the diffusion model's).
        head: MLPActionHead producing the action chunk.
        action_normalizer: Normalizer for the action chunk.
        state_normalizer: Normalizer for the proprioceptive state.
    """

    def __init__(
        self,
        cfg: DictConfig,
        encoder: nn.Module,
        head: nn.Module,
        action_normalizer: nn.Module,
        state_normalizer: nn.Module,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.head = head
        self.action_normalizer = action_normalizer
        self.state_normalizer = state_normalizer

    def compute_loss(self, batch: dict) -> torch.Tensor:
        """Training forward pass.

        Normalises observations and actions, encodes observations, predicts the
        action chunk, and computes the MSE against the demonstration chunk (in
        normalised space, matching how the diffusion loss is computed).

        Args:
            batch: Dict with keys ``"images"``, ``"state"``, ``"actions"``.

        Returns:
            Scalar loss tensor (on the same device as the batch).
        """
        state_norm = self.state_normalizer(batch["state"])
        action_norm = self.action_normalizer(batch["actions"])
        obs_cond = self.encoder(batch["images"], state_norm)
        pred_norm = self.head(obs_cond)
        return F.mse_loss(pred_norm, action_norm)

    @torch.no_grad()
    def predict_actions(
        self,
        batch: dict,
        warm_start_actions_norm: torch.Tensor | None = None,
        return_normalized: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Inference forward pass.

        Encodes observations, runs the MLP head once, then denormalises the
        predicted action chunk before returning.

        Args:
            batch: Dict with keys ``"images"`` and ``"state"``.
            warm_start_actions_norm: Accepted for interface compatibility with
                ``DiffusionModule`` (and the shared inference loop) but ignored:
                a one-shot regressor has no iterative state to warm-start.
            return_normalized: If True, also return the normalised chunk.

        Returns:
            Denormalised action chunk ``(B, pred_horizon, action_dim)``, or a
            ``(denormalised, normalised)`` tuple when ``return_normalized``.
        """
        del warm_start_actions_norm  # no iterative refinement in BC
        state_norm = self.state_normalizer(batch["state"])
        obs_cond = self.encoder(batch["images"], state_norm)
        actions_norm = self.head(obs_cond)
        actions = self.action_normalizer.inverse(actions_norm)
        if return_normalized:
            return actions, actions_norm
        return actions
