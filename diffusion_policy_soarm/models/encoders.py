"""Observation encoder: per-camera ResNet18 + state embedding → conditioning vector.

Design decisions:
- Separate ResNet18 per camera (not shared across cameras): each viewpoint has
  different statistics and it is not clear that weight sharing helps.
- Weights are shared across the obs_horizon time axis: the same ResNet processes
  every frame in the observation window.  Temporal structure is provided to the
  denoiser via concatenation of the resulting feature vectors — the denoiser
  (1D U-Net) is the temporal model, not the encoder.
- ImageNet normalisation is applied inside the encoder so callers can pass raw
  [0, 1] float images without any preprocessing boilerplate.
- A linear projection head is added after global average pool.  Even with
  feature_dim == 512 (matching ResNet18's pool output), this gives a learned
  linear transformation that adapts the ImageNet features to the robot domain.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm
from omegaconf import DictConfig


# ImageNet normalisation constants (mean and std per channel)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class ImageEncoder(nn.Module):
    """Single-camera encoder: ResNet18 → feature vector.

    Applies ImageNet normalisation, runs ResNet18 (with global average pool),
    then projects to ``feature_dim``.

    Args:
        feature_dim: Output feature vector dimensionality.
        pretrained: If True, load ImageNet-1K weights (IMAGENET1K_V1).
        freeze_bn: If True, set BatchNorm layers to eval mode during training.
    """

    def __init__(self, feature_dim: int, pretrained: bool, freeze_bn: bool) -> None:
        super().__init__()
        self.freeze_bn = freeze_bn

        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tvm.resnet18(weights=weights)
        # Drop the final FC layer; keep everything through global average pool.
        # backbone.children(): conv1, bn1, relu, maxpool, layer1-4, avgpool, fc
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        # backbone output: (B, 512, 1, 1) after avgpool

        self.proj = nn.Linear(512, feature_dim)

        # Register ImageNet normalisation constants as non-trainable buffers so
        # they move with the module when .to(device) is called.
        mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean)
        self.register_buffer("_std", std)

    def train(self, mode: bool = True) -> "ImageEncoder":
        super().train(mode)
        if self.freeze_bn and mode:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images.

        Args:
            images: ``(B, 3, H, W)`` float tensor with pixel values in [0, 1].

        Returns:
            ``(B, feature_dim)`` feature vectors.
        """
        # ImageNet normalisation: (x - mean) / std
        x = (images - self._mean) / self._std
        x = self.backbone(x)          # (B, 512, 1, 1)
        x = x.flatten(1)              # (B, 512)
        return self.proj(x)           # (B, feature_dim)


class ObservationEncoder(nn.Module):
    """Encode a full observation history into a flat conditioning vector.

    Processes ``obs_horizon`` frames from each camera and the proprioceptive
    state, then concatenates all features into one vector.

    Output dimension (``output_dim``) is:
        obs_horizon × (n_cameras × feature_dim + state_embed_dim)

    Args:
        cfg: Resolved config; reads ``encoder`` and ``model`` sub-trees.
        camera_keys: Ordered list of camera feature names (determines concat order).
        state_dim: Dimensionality of the proprioceptive state vector (6 for SO-101).
    """

    def __init__(self, cfg: DictConfig, camera_keys: list[str], state_dim: int) -> None:
        super().__init__()
        self.camera_keys = camera_keys
        self.obs_horizon: int = cfg.model.obs_horizon
        feature_dim: int = cfg.encoder.feature_dim
        state_embed_dim: int = cfg.encoder.state_embed_dim

        # One ResNet18 encoder per camera (separate weights)
        self.image_encoders = nn.ModuleDict({
            key.replace(".", "_"): ImageEncoder(
                feature_dim=feature_dim,
                pretrained=cfg.encoder.pretrained,
                freeze_bn=cfg.encoder.freeze_bn,
            )
            for key in camera_keys
        })

        # Linear embedding of the proprioceptive state
        self.state_embed = nn.Sequential(
            nn.Linear(state_dim, state_embed_dim),
            nn.SiLU(),
        )

        self._output_dim = cfg.model.obs_horizon * (
            len(camera_keys) * feature_dim + state_embed_dim
        )

    @property
    def output_dim(self) -> int:
        """Dimensionality of the output conditioning vector."""
        return self._output_dim

    def forward(
        self,
        images: dict[str, torch.Tensor],
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Encode observations into a conditioning vector.

        Args:
            images: Dict mapping camera key → ``(B, obs_horizon, 3, H, W)`` tensor,
                pixel values in [0, 1].
            state: ``(B, obs_horizon, state_dim)`` normalised state tensor.

        Returns:
            ``(B, output_dim)`` conditioning vector.
        """
        B = state.shape[0]
        parts: list[torch.Tensor] = []

        for key in self.camera_keys:
            frames = images[key]                     # (B, T_o, 3, H, W)
            T_o = frames.shape[1]
            # Process all frames through the same encoder (shared weights over time)
            flat = frames.view(B * T_o, *frames.shape[2:])   # (B*T_o, 3, H, W)
            enc_key = key.replace(".", "_")
            feats = self.image_encoders[enc_key](flat)        # (B*T_o, feature_dim)
            feats = feats.view(B, -1)                         # (B, T_o * feature_dim)
            parts.append(feats)

        # State: (B, T_o, state_dim) → embed each timestep → (B, T_o * state_embed_dim)
        state_flat = state.view(B * self.obs_horizon, -1)     # (B*T_o, state_dim)
        state_feats = self.state_embed(state_flat)            # (B*T_o, state_embed_dim)
        parts.append(state_feats.view(B, -1))

        return torch.cat(parts, dim=-1)                       # (B, output_dim)
