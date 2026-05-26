# Multimodal Manipulation Benchmark

Diffusion Policy reimplemented from scratch (Chi et al., 2023) and evaluated on a
deliberately bimodal pick-and-place task with an SO-ARM101 arm. The headline result:
Diffusion Policy represents the multimodal action distribution and commits to one mode;
a behaviour-cloning MSE baseline regresses to the mean and grasps nothing.

---

## Repository layout

```
diffusion_policy_soarm/
  configs/            # base config + ablation override YAMLs
  data/               # dataset wrapper, normalization, batch visualizer
  models/
    encoders.py       # per-camera ResNet encoder + state embedding
    cnn_backbone.py   # 1-D temporal U-Net denoiser with FiLM conditioning
    transformer_backbone.py  # Transformer denoiser (ablation)
    diffusion.py      # noise schedule, q_sample, loss, DDPM + DDIM samplers
  train.py            # training entry point
  infer.py            # real-time receding-horizon inference loop
  eval/               # three-tier eval harness, failure logging, metrics
  scripts/            # overfit sanity check, plotting, config sweeps
  docs/
    how_it_works.md   # maps DDPM math to code; start here
    writeup.md        # mini-paper scaffold with result placeholders
recordings/           # LeRobotDataset v3.0 (not committed — symlink or set path)
```

---

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                     # creates .venv and installs all dependencies
uv sync --extra dev         # also installs pytest and ruff
```

---

## Reproduction

### 1. Verify the data pipeline

```bash
uv run python -m diffusion_policy_soarm.data.visualize_batch \
    --config diffusion_policy_soarm/configs/base.yaml
```

Expected output: tensor shapes printed to stdout, `batch_sample.png` saved showing
front-camera frames and action joint trajectories.

### 2. Run the Phase 4 overfit gate (5 demos)

```bash
uv run python -m diffusion_policy_soarm.scripts.overfit_check \
    --config diffusion_policy_soarm/configs/base.yaml
```

Expected: loss drops to < 0.001 and DDPM samples reproduce the 5 trajectories.

### 3. Full training

```bash
uv run python -m diffusion_policy_soarm.train \
    --config diffusion_policy_soarm/configs/base.yaml \
    training.run_dir=runs
```

Resolved config and git hash are saved to `runs/<timestamp>/`.

### 4. Ablations

```bash
# Transformer backbone
uv run python -m diffusion_policy_soarm.train \
    --config diffusion_policy_soarm/configs/base.yaml \
    --override diffusion_policy_soarm/configs/ablations/transformer_backbone.yaml

# DDIM 10-step sampler (weights from the base run)
uv run python -m diffusion_policy_soarm.train \
    --config diffusion_policy_soarm/configs/base.yaml \
    --override diffusion_policy_soarm/configs/ablations/ddim_sampler.yaml
```

### 5. Inference on the arm

```bash
uv run python -m diffusion_policy_soarm.infer \
    --checkpoint runs/<run>/checkpoints/ema_latest.pt \
    --config runs/<run>/config.yaml
```

---

## Design decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Noise schedule | Cosine | Better signal retention for low-dim actions vs. linear |
| Prediction target | ε (noise) | Cleaner gradient signal vs. x₀ prediction |
| Conditioning | FiLM on [t_emb; obs_emb] | Matches paper; no CFG needed |
| Normalisation | Min-max to [−1, 1] | Required by clip_sample and bounded noise schedule |
| Sampler | DDPM train / DDIM infer | DDIM ×10× faster with same quality at 10–20 steps |
| EMA | decay = 0.9999 | Standard for diffusion; always use EMA weights for eval |

---

## Citation

```bibtex
@inproceedings{chi2023diffusion,
  title     = {Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author    = {Chi, Cheng and Feng, Siyuan and Du, Yilun and Xu, Zhenjia and
               Cousineau, Eric and Burchfiel, Benjamin and Song, Shuran},
  booktitle = {Robotics: Science and Systems},
  year      = {2023}
}
```
