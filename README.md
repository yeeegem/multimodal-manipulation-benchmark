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

### 1. Pre-extract video frames (one-time, ~10 min)

Decodes all videos to numpy arrays so the training hot-path is pure RAM/disk reads —
no per-batch video decoding. Safe to interrupt and re-run (skips completed episodes).

```bash
uv run python -m diffusion_policy_soarm.scripts.preextract_frames \
    --config diffusion_policy_soarm/configs/base.yaml
```

Cache is written to `recordings/redcubes_bluecup/frame_cache/`.
At 96×96 (~0.9 GB) it is loaded fully into RAM at dataset init (`preload_cache: true`).

### 2. Full training

```bash
uv run python -m diffusion_policy_soarm.train \
    --config diffusion_policy_soarm/configs/base.yaml \
    training.experiment=main_96x96
```

Runs are saved to `runs/<experiment>/<YYYYMMDD_HHMMSS>/`.
Each run directory contains `config.yaml`, `git_hash.txt`, `tb/` (TensorBoard),
and `checkpoints/` with `best.pt` and `latest.pt`.

**Resume an interrupted run:**
```bash
uv run python -m diffusion_policy_soarm.train \
    --config diffusion_policy_soarm/configs/base.yaml \
    training.experiment=main_96x96 \
    --resume runs/main_96x96/<timestamp>
```

**Monitor training:**
```bash
uv run tensorboard --logdir runs/
# open http://localhost:6006
```

### 3. Ablations

```bash
# High-resolution (480x640) — re-extract cache first with high_res override
uv run python -m diffusion_policy_soarm.scripts.preextract_frames \
    --config diffusion_policy_soarm/configs/base.yaml \
    --override diffusion_policy_soarm/configs/ablations/high_res.yaml

uv run python -m diffusion_policy_soarm.train \
    --config diffusion_policy_soarm/configs/base.yaml \
    --override diffusion_policy_soarm/configs/ablations/high_res.yaml \
    training.experiment=ablation_high_res

# Transformer backbone
uv run python -m diffusion_policy_soarm.train \
    --config diffusion_policy_soarm/configs/base.yaml \
    --override diffusion_policy_soarm/configs/ablations/transformer_backbone.yaml \
    training.experiment=ablation_transformer

# DDIM fast sampler
uv run python -m diffusion_policy_soarm.train \
    --config diffusion_policy_soarm/configs/base.yaml \
    --override diffusion_policy_soarm/configs/ablations/ddim_sampler.yaml \
    training.experiment=ablation_ddim
```

### 4. Inference on the arm

**Smoke test without hardware (checks shapes and latency):**
```bash
uv run python -m diffusion_policy_soarm.infer \
    --checkpoint runs/<experiment>/<timestamp>/checkpoints/best.pt \
    --config runs/<experiment>/<timestamp>/config.yaml \
    --dry-run
```

**Full run on the arm:**
```bash
uv run python -m diffusion_policy_soarm.infer \
    --checkpoint runs/<experiment>/<timestamp>/checkpoints/best.pt \
    --config runs/<experiment>/<timestamp>/config.yaml
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
