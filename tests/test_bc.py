"""Phase 7 gate: unit tests for the behavior-cloning baseline.

Dataset-free and hardware-free: a tiny config, synthetic tensors, and
identity Normalizers exercise the full encoder -> MLP head -> MSE path and the
inference interface that ``infer.py`` relies on.

Run with:
    uv run pytest tests/test_bc.py -v
"""

import pytest
import torch
from omegaconf import OmegaConf

from diffusion_policy_soarm.data.normalization import Normalizer
from diffusion_policy_soarm.models.bc import BCModule, MLPActionHead
from diffusion_policy_soarm.models.factory import build_policy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ACTION_DIM = 6
STATE_DIM = 6
IMG_HW = 32
BATCH = 3
CAMERA_KEY = "observation.images.front"


@pytest.fixture
def cfg():
    """Tiny BC config: 1 camera, obs_horizon=1, pred_horizon=4, small head."""
    return OmegaConf.create({
        "dataset": {"camera_keys": [CAMERA_KEY]},
        "model": {"type": "bc", "obs_horizon": 1, "pred_horizon": 4},
        "encoder": {
            "backbone": "resnet18",
            "pretrained": False,
            "freeze_bn": False,
            "norm_type": "group",
            "group_norm_groups": 32,
            "pool_type": "avg",
            "feature_dim": 64,
            "state_embed_dim": 16,
        },
        "bc": {"hidden_dims": [32, 32], "dropout": 0.0},
    })


def identity_normalizer(dim: int) -> Normalizer:
    """Normalizer mapping [-1, 1] to itself (forward and inverse are identity)."""
    return Normalizer(mins=-torch.ones(dim), maxs=torch.ones(dim))


@pytest.fixture
def model(cfg):
    return build_policy(
        cfg,
        action_dim=ACTION_DIM,
        state_dim=STATE_DIM,
        action_normalizer=identity_normalizer(ACTION_DIM),
        state_normalizer=identity_normalizer(STATE_DIM),
    )


@pytest.fixture
def batch(cfg):
    T_o = cfg.model.obs_horizon
    H = cfg.model.pred_horizon
    return {
        "images": {CAMERA_KEY: torch.rand(BATCH, T_o, 3, IMG_HW, IMG_HW)},
        "state": torch.randn(BATCH, T_o, STATE_DIM),
        "actions": torch.randn(BATCH, H, ACTION_DIM),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_build_policy_returns_bc_module(model):
    assert isinstance(model, BCModule)


def test_mlp_head_output_shape(cfg):
    obs_cond_dim = 128
    head = MLPActionHead(cfg, action_dim=ACTION_DIM, obs_cond_dim=obs_cond_dim)
    out = head(torch.randn(BATCH, obs_cond_dim))
    assert out.shape == (BATCH, cfg.model.pred_horizon, ACTION_DIM)


def test_compute_loss_is_finite_scalar(model, batch):
    loss = model.compute_loss(batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_compute_loss_backward_produces_grads(model, batch):
    loss = model.compute_loss(batch)
    loss.backward()
    head_grads = [p.grad for p in model.head.parameters() if p.grad is not None]
    enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert head_grads, "MLP head received no gradients"
    assert enc_grads, "encoder received no gradients"


def test_predict_actions_shape(model, batch):
    model.eval()
    actions = model.predict_actions(batch)
    assert actions.shape == (BATCH, model.cfg.model.pred_horizon, ACTION_DIM)


def test_predict_actions_return_normalized(model, batch):
    model.eval()
    actions, actions_norm = model.predict_actions(batch, return_normalized=True)
    expected = (BATCH, model.cfg.model.pred_horizon, ACTION_DIM)
    assert actions.shape == expected
    assert actions_norm.shape == expected
    # Identity normalizer: denormalised equals normalised (up to fp round-trip).
    assert torch.allclose(actions, actions_norm, atol=1e-5)


def test_warm_start_is_ignored(model, batch):
    """BC has no iterative state; warm_start_actions_norm must not affect output."""
    model.eval()
    shape = (BATCH, model.cfg.model.pred_horizon, ACTION_DIM)
    base = model.predict_actions(batch)
    warm = model.predict_actions(batch, warm_start_actions_norm=torch.randn(shape))
    assert torch.equal(base, warm)
