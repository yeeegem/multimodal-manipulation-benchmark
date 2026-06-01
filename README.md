# Multimodal Manipulation Benchmark

[![Python](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Diffusion Policy](https://img.shields.io/badge/method-Diffusion%20Policy-8b5cf6)](https://diffusion-policy.cs.columbia.edu/)
[![Imitation Learning](https://img.shields.io/badge/task-Imitation%20Learning-0ea5e9)]()
[![Dataset](https://img.shields.io/badge/HuggingFace-yeeegem%2Fredcubes__bluecup-ffd21e?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/yeeegem/redcubes_bluecup)

A reproducible benchmark for evaluating multimodal imitation-learning policies on an SO-ARM101 arm. Reimplements Diffusion Policy (Chi et al., 2023) from scratch in PyTorch and compares it against behaviour-cloning and transformer-based action policies on a deliberately bimodal pick-and-place task, with quantitative metrics for success rate, mode commitment, midpoint collapse and robustness to layout perturbations.

---

## Research questions

1. Does Diffusion Policy actually represent the bimodal action distribution
   on a real arm, or does it collapse to one mode under realistic data sizes
   (about 100 episodes) and noisy proprioception?
2. How does it compare against a behaviour-cloning MSE baseline that is
   forced to regress through the mean, and against modern alternatives
   (ACT, flow matching) on the same hardware and the same eval protocol?
3. What is the latency vs success-rate trade-off of DDIM at deployment time
   (5 / 10 / 20 steps) versus full DDPM, when the closed-loop control budget
   is 33 ms at 30 fps?
4. Does the choice of denoiser backbone (1-D CNN U-Net vs Transformer)
   matter at this data scale, or is the encoder doing most of the work?
5. How sensitive is the policy to input choices: image resolution (96 x 96
   vs 480 x 640), observation horizon (1 / 2 / 4 frames), action chunk
   length (8 / 16 / 32 steps), and action representation (absolute angles
   vs deltas vs velocities)?
6. Can the model actually meet a real-time 30 Hz control budget end to end
   (camera capture, encode, sample, send_action) on a single commodity GPU?

Results live in "Results" at the end of this README. They are filled in
only after Phase 8 (eval harness) and Phase 9 (ablations) complete.

---

## Architecture

![Architecture](diffusion_policy_soarm/docs/architecture.png)

<details>
<summary>Same diagram as inline Mermaid (renders on GitHub)</summary>

```mermaid
flowchart LR
  subgraph IN["Observation window (T_o = 2 frames)"]
    Front["front camera<br/>480 x 640 x 3 uint8<br/>(resized to 96 x 96)"]
    Wrist["wrist camera<br/>480 x 640 x 3 uint8<br/>(resized to 96 x 96)"]
    State["state 6-D<br/>(joint angles deg)"]
  end

  Front -->|ResNet18 per camera| F["front features<br/>(2, 512)"]
  Wrist -->|ResNet18 per camera| W["wrist features<br/>(2, 512)"]
  State -->|Linear + SiLU| S["state embed<br/>(2, 64)"]

  F --> C(("concat<br/>2176-D"))
  W --> C
  S --> C

  C --> UNet["1-D temporal U-Net<br/>FiLM conditioning<br/>channels 256 / 512 / 1024<br/>47M params"]
  Noise["x_t noise (16, 6)"] --> UNet
  TS["t embedding<br/>(sinusoid -> MLP -> 512)"] --> UNet
  UNet --> EpsHat["epsilon-hat (16, 6)"]

  EpsHat --> DDIM["DDIM step (loops N times)"]
  DDIM -->|x_t -> x_t-1| UNet
  DDIM --> Out["actions (16, 6) deg<br/>saved main_96x96 executes first 4 then replan"]
```

</details>

The encoder runs **once per inference call**: each camera's two consecutive
frames go through their own ResNet18 (weights shared across the 2 obs
frames, separate across cameras), the proprioceptive state through a small
Linear + SiLU. All features concatenate into a **2176-D conditioning
vector**. The 47M-parameter 1-D temporal U-Net then denoises a random
(16, 6) tensor over **N DDIM steps** (default 10), conditioned on that
2176-D vector via FiLM. The output is denormalised back into degrees and
sent to the arm as **absolute joint-angle setpoints**; the Feetech motors'
firmware-side PID controllers do the actual position control.

For historical accuracy: the saved `main_96x96` run at
`runs/main_96x96/20260528_185334/` used a legacy vision encoder variant
(ImageNet-pretrained ResNet18 with average pooling and BatchNorm) and
`exec_horizon=4`. The current `configs/base.yaml` has been updated to the
paper-faithful vision stack (GroupNorm + spatial softmax + no pretraining)
for future runs.

For a step-by-step walk-through with code references and the inference
walkthrough, see [`diffusion_policy_soarm/docs/main_96x96_explained.ipynb`](diffusion_policy_soarm/docs/main_96x96_explained.ipynb)
and [`diffusion_policy_soarm/docs/how_it_works.md`](diffusion_policy_soarm/docs/how_it_works.md).

---

## What the model trains on

A random batch of (front, wrist, state, action) after normalisation:

![Batch sample](batch_sample.png)

Top row: front-camera frames. Middle row: wrist-camera frames. Bottom: the
normalised state and action chunks. This is exactly what the encoder and
denoiser see during training, after the same min-max normalisation that is
inverted at inference.

---

## Project plan and status

**Current step:** Phase 6 (inference on the arm), gate pending. The model
runs end-to-end at ~50 ms inference with DDIM at 10 steps; tuning
warm-started replanning and the Feetech `Acceleration` register, then
running the autonomous-success gate.

| # | Phase | Status | Output |
|---|---|---|---|
| 0 | Scaffold | done | repo layout, pyproject, config loader |
| 1 | Data pipeline | done | LeRobotDataset wrapper, normalizer, batch visualiser (`batch_sample.png`) |
| 2 | Diffusion core | done | cosine + linear schedules, `q_sample`, DDPM and DDIM samplers, 16/16 tests pass |
| 3 | Denoiser | done | ObservationEncoder 2176-D, ConditionalUNet1d 47M params, DiffusionModule end-to-end |
| 4 | Training loop | done | EMA, AMP, cosine + warmup LR, checkpointing, TensorBoard |
| 5 | Full training | done | 300 epochs at 96 x 96, final loss 0.00298, run `runs/main_96x96/20260528_185334` |
| 6 | Inference on arm | in progress | `infer.py` complete; tuning warm-start + motor acceleration; gate = one autonomous success |
| 7 | BC MSE baseline | todo | proves regression-to-mean failure on the bimodal task |
| 8 | Eval harness (3 tiers) | todo | tier A in-distribution, tier B distractors, tier C OOD; failure-category logging |
| 9 | Ablations | todo | see below |
| 10 | Writeup | todo | mini-paper PDF (`diffusion_policy_soarm/docs/writeup.md` is the scaffold) |

### Phase 9 ablations (planned)

Three families, each row a separate training run plus eval pass.

**Method comparisons** (Diffusion Policy vs alternatives on the same data
and eval protocol):

| Method | What it does differently | Hypothesis |
|---|---|---|
| BC MSE baseline (MLP) | direct regression to next action chunk | regresses to the mean on bimodal data; reaches into the gap between the cubes |
| ACT (Zhao et al. 2023) | CVAE + Transformer decoder on action chunks | competitive on unimodal sub-tasks but should also collapse modes since the prior is Gaussian |
| Flow matching (Lipman et al. 2023) | conditional flow matching instead of DDPM, same encoder and same chunked output | matches diffusion quality in 1 to 4 sampling steps, much lower inference latency |

**Architectural ablations of Diffusion Policy itself:**

| Axis | Default | Variants tested |
|---|---|---|
| Denoiser backbone | 1-D U-Net (CNN) | Transformer (`configs/ablations/transformer_backbone.yaml`) |
| Sampler | DDPM (T=100) | DDIM with N in {5, 10, 20} |
| Noise schedule | cosine | linear (Ho et al. 2020) |
| EMA at inference | on (decay 0.9999) | off (raw last weights) |
| Conditioning | FiLM | classifier-free guidance with 10% obs-dropout at training time |

**Control-loop and input ablations:**

| Axis | Default | Variants tested |
|---|---|---|
| Image resolution | 96 x 96 | 480 x 640 (`configs/ablations/high_res.yaml`, already trained: `runs/ablation_high_res/20260527_102159/`) |
| Action chunk length | T_p=16, T_a=4 in saved `main_96x96` | (T_p=8, T_a=4); (T_p=32, T_a=16) |
| Observation horizon | T_o=2 | T_o=1 (no implicit velocity); T_o=4 |
| Action representation | absolute joint angles (deg) | per-step deltas (deg/tick); joint velocities (deg/s) |

Each row produces one entry in the final eval table (Tier A success,
latency, parameter count). The Method-comparison rows are the headline
result; the architectural and control rows are the in-depth ablation
block.

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
  scripts/            # overfit sanity check, pre-extract frames, draw architecture
  docs/
    how_it_works.md             # maps DDPM math to code; start here
    main_96x96_explained.ipynb  # full walkthrough of the main run
    architecture.png            # diagram embedded above
    writeup.md                  # mini-paper scaffold
recordings/           # LeRobotDataset v3.0 (not committed; symlink or set path)
runs/                 # training run outputs (one timestamped dir per run)
```

---

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                     # creates .venv and installs all dependencies
uv sync --extra dev         # also installs pytest and ruff
```

---

## Reproduction

### 1. Pre-extract video frames (one-time, ~10 min)

Decodes all videos to numpy arrays so the training hot-path is pure RAM
and disk reads, with no per-batch video decoding. Safe to interrupt and
re-run (skips completed episodes).

```bash
uv run python -m diffusion_policy_soarm.scripts.preextract_frames \
    --config diffusion_policy_soarm/configs/base.yaml
```

Cache is written to `recordings/redcubes_bluecup/frame_cache/`. At 96 x 96
(about 0.9 GB) it is loaded fully into RAM at dataset init
(`preload_cache: true`).

### 2. Full training

```bash
uv run python -m diffusion_policy_soarm.train \
    --config diffusion_policy_soarm/configs/base.yaml \
    training.experiment=main_96x96
```

Runs are saved to `runs/<experiment>/<YYYYMMDD_HHMMSS>/`. Each run
directory contains `config.yaml`, `git_hash.txt`, `tb/` (TensorBoard),
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
# High-resolution (480x640): re-extract cache first with high_res override
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

The first time you connect: pass `id=<your_arm_id>` via `infer.robot_id`
in the config so LeRobot reads calibration from
`~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json`.
You should see a sequence of latency lines once the loop starts:

```
inference 47.3 ms | mean 49.1 | p95 56.0
```

The deployed loop warm-starts the next DDIM sample from the unexecuted
suffix of the previous action chunk. This improves temporal consistency
across replans compared with restarting each chunk from fresh Gaussian
noise.

---

## Training hardware and cost

All numbers below are wall-clock on a single NVIDIA RTX A5000 (24 GB VRAM,
RTX 3080-class compute, similar TFLOPs to a consumer RTX 3080).

| Run | Resolution | Epochs | Wall clock |
|---|---|---|---|
| Initial attempt (high-res) | 480 x 640 | 150 | over 24 hours |
| Main run (`main_96x96`) | 96 x 96 | 300 | 2 to 3 hours |
| High-res ablation (retained for Phase 9) | 480 x 640 | 150 | over 24 hours |

The first training pass was at native 480 x 640. After observing that a
single 150-epoch run took more than a day on the A5000, the main
resolution was switched to 96 x 96 (ResNet18 globally pools, so spatial
size is a free hyperparameter). 96 x 96 brings a 300-epoch run down to
2 to 3 hours, fits the entire `recordings/redcubes_bluecup` frame cache
in 0.9 GB of RAM, and keeps batch size at 256 inside 16 GB VRAM. The
high-res run is kept as a Phase 9 ablation to measure how much visual
fidelity actually matters for this task.

---

## Lessons from the build

A short retrospective of issues that actually slowed the project down and
how they were resolved. These shaped several of the design choices above.

### Phase 5: high-resolution training was too slow

The first full training pass was at native 480 x 640. On the RTX A5000 a
single 150-epoch run took over 24 hours, with the GPU mostly idle waiting
on the encoder's per-camera ResNet18 to chew through 307200-pixel frames
at batch size 16. ResNet18 globally pools, so spatial size is a free
hyperparameter: switching the main run to 96 x 96 dropped wall-clock to
2 to 3 hours for 300 epochs at batch size 256, with the full frame cache
in 0.9 GB of RAM. The 480 x 640 run is retained as a Phase 9 ablation to
measure how much fidelity matters on this task.

### Phase 6: arm jittered in place

First on-arm run: the model loaded, latency stayed in budget, but the arm
just twitched at the starting pose instead of moving toward the cubes.
Diagnosis came from printing the denormalised actions:
`shoulder_lift` and `elbow_flex` were pinned exactly at the normalizer's
min and max (-103.56 and +96.88 deg). That is the signature of
`clip_sample=True` saturating: the DDIM x-hat-zero estimate was outside
[-1, +1] at every step and the clip was hitting both rails. Root cause:
the arm was started in a pose far from the training distribution, so the
conditioning vector was out of distribution and the denoiser fell back
to extreme predictions. Fix: home the arm to the same neutral pose used
at the start of training episodes before running `infer.py`.

### Phase 6: arm moved much faster than during teleop

Even with reasonable predictions, the arm slammed to each new commanded
position. Teleop felt slow because the leader arm moved at human hand
speed and each step's delta was small; the diffusion model can predict
5 to 20 degree jumps between consecutive chunk steps because the data
distribution contains some fast motions. With the Feetech `Acceleration`
register at its near-max default, each delta executes at full motor
speed.

Fix landed in one place:
- A direct write to the Feetech `Acceleration` register
  (`infer.motor_acceleration`, default 64 of 255) right after
  `robot.connect()`. This lowers the firmware ramp rate so even an
  large commanded delta executes more gently. Persists until power cycle.

### Phase 6: serial bus dropped at 100 DDIM steps

Running with `infer.num_ddim_steps=100` (effectively full DDPM) produced
`ConnectionError: There is no status packet!` after a few hundred ms. The
serial bus to the Feetech motors has its own packet timeout, and an
inference call that took about 530 ms blew through it. DDIM at 5 to 10
steps brings inference to 26 to 50 ms and fixes this; the same number
also keeps the loop inside the 33 ms control budget.

### Phase 6: hardware integration footguns

Three small but real time sinks during the initial bring-up:

1. The motor driver shipped as `scservo_sdk` is not on PyPI under that
   name. It is bundled inside the `feetech-servo-sdk` PyPI package; that
   is the correct dependency to add.
2. `FeetechMotorsBus` exposes motor names via `robot.bus.motors.keys()`,
   not `robot.bus.motor_names` (the old API).
3. Calibration: instead of calling `write_calibration` manually, pass
   `id="<your_arm_id>"` to `SOFollowerRobotConfig`. LeRobot then loads
   the calibration JSON from
   `~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json`
   on `robot.connect()`, and `robot.bus.is_calibrated` is already true.

---

## Design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Noise schedule | Cosine | Better signal retention for low-dim actions vs linear |
| Prediction target | epsilon (noise) | Cleaner gradient signal vs x-zero prediction |
| Conditioning | FiLM on [t_emb; obs_emb] | Matches paper; no CFG needed at this scale |
| Normalisation | Min-max to [-1, +1] | Required by `clip_sample` and bounded noise schedule |
| Sampler | DDPM train, DDIM infer | DDIM is about 10x faster with same quality at 10 to 20 steps |
| EMA | decay = 0.9999 | Standard for diffusion; always use EMA weights for eval |
| Warm-start | shift previous action chunk suffix into next DDIM init | Improves temporal consistency across replans |
| Motor accel | Feetech `Acceleration` register lowered to 64 | Slows the firmware-side ramp without altering policy outputs |

---

## Results

Filled in after Phase 8 (eval harness) and Phase 9 (ablations) land. Until
then, treat this section as a placeholder.

- **Headline:** [pending] Diffusion Policy vs BC MSE baseline, Tier A
  success and mode-balance.
- **Method comparisons:** [pending] Diffusion Policy vs ACT vs flow
  matching on the same eval.
- **DDIM step count:** [pending] success and latency at N in {5, 10, 20,
  100}.
- **Architectural and input ablations:** [pending] table per row in
  "Phase 9 ablations" above.

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
