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
- The encoder supports both the legacy average-pool/BatchNorm path used by old
  checkpoints and a paper-faithful path with GroupNorm + spatial softmax.
"""

from __future__ import annotations

import functools

import torch
import torch.nn as nn
import torchvision.models as tvm
from omegaconf import DictConfig


# ImageNet normalisation constants (mean and std per channel)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class SpatialSoftmax2d(nn.Module):
    """Spatial softmax pooling over a feature map.

    Returns the expected x/y coordinates for each channel, giving a compact
    location-aware representation of shape ``(B, 2 * C)``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool a feature map to channel-wise expected coordinates."""
        B, C, H, W = x.shape
        attn = torch.softmax(x.view(B, C, H * W), dim=-1)

        ys = torch.linspace(-1.0, 1.0, H, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid_x = grid_x.reshape(1, 1, H * W)
        grid_y = grid_y.reshape(1, 1, H * W)

        exp_x = (attn * grid_x).sum(dim=-1)
        exp_y = (attn * grid_y).sum(dim=-1)
        return torch.cat([exp_x, exp_y], dim=-1)


class ImageEncoder(nn.Module):
    """Single-camera encoder: ResNet18 → feature vector.

    Applies ImageNet normalisation, runs ResNet18 (with global average pool),
    then projects to ``feature_dim``.

    Args:
        feature_dim: Output feature vector dimensionality.
        pretrained: If True, load ImageNet-1K weights (IMAGENET1K_V1).
        freeze_bn: If True, set BatchNorm layers to eval mode during training.
    """

    def __init__(
        self,
        feature_dim: int,
        pretrained: bool,
        freeze_bn: bool,
        norm_type: str = "batch",
        pool_type: str = "avg",
        group_norm_groups: int = 32,
    ) -> None:
        super().__init__()
        self.freeze_bn = freeze_bn
        self.norm_type = norm_type
        self.pool_type = pool_type

        if norm_type == "group":
            if pretrained:
                raise ValueError("GroupNorm ResNet18 does not support torchvision ImageNet weights.")
            norm_layer = functools.partial(nn.GroupNorm, group_norm_groups)
            backbone = tvm.resnet18(weights=None, norm_layer=norm_layer)
        elif norm_type == "batch":
            weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = tvm.resnet18(weights=weights)
        else:
            raise ValueError(f"Unknown encoder norm_type: {norm_type!r}")

        # Keep the convolutional feature map. Pooling happens explicitly below.
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        if pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            proj_in_dim = 512
        elif pool_type == "spatial_softmax":
            self.pool = SpatialSoftmax2d()
            proj_in_dim = 1024
        else:
            raise ValueError(f"Unknown encoder pool_type: {pool_type!r}")

        self.proj = nn.Linear(proj_in_dim, feature_dim)

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
        x = self.backbone(x)          # (B, 512, H', W')
        x = self.pool(x)
        x = x.flatten(1)
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
        norm_type: str = getattr(cfg.encoder, "norm_type", "batch")
        pool_type: str = getattr(cfg.encoder, "pool_type", "avg")
        gn_groups: int = int(getattr(cfg.encoder, "group_norm_groups", 32))

        # One ResNet18 encoder per camera (separate weights)
        self.image_encoders = nn.ModuleDict({
            key.replace(".", "_"): ImageEncoder(
                feature_dim=feature_dim,
                pretrained=cfg.encoder.pretrained,
                freeze_bn=cfg.encoder.freeze_bn,
                norm_type=norm_type,
                pool_type=pool_type,
                group_norm_groups=gn_groups,
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
