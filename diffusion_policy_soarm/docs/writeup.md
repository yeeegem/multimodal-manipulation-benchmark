# Diffusion Policy for Multimodal Manipulation: A Study on SO-ARM101

*Vasili Areshka — [DATE]*

---

## Abstract

We reimplement Diffusion Policy (Chi et al., 2023) from scratch in plain PyTorch and
evaluate it on a deliberately multimodal pick-and-place task: place either of two
identical red cubes into a blue cup using an SO-ARM101 arm. The demonstration data is
50/50 bimodal (left cube vs. right cube). We show that Diffusion Policy commits to one
mode and succeeds at the task, while a behaviour-cloning MSE baseline regresses to the
mean and fails to grasp either cube. We ablate denoiser backbone (CNN vs. Transformer),
sampler (DDPM vs. DDIM), and action chunk length, and report a three-tier evaluation
covering in-distribution, out-of-distribution, and distractor conditions.

---

## 1. Setup

### 1.1 Hardware

SO-ARM101 follower arm, two 640×480 RGB cameras (front top-down, wrist close-up),
running at 30 fps.

### 1.2 Task

Pick one of two identical red cubes (one left, one right of centre) and place it in a
blue cup. Either cube is a valid target. Demonstrations are teleoperated: **[PLACEHOLDER:
left-cube fraction]** pick the left cube, **[PLACEHOLDER: right-cube fraction]** pick the
right cube.

### 1.3 Dataset

- **[PLACEHOLDER: N]** teleoperated demonstration episodes.
- **[PLACEHOLDER: N]** total frames at 30 fps.
- Observations: two RGB cameras + 6-DOF joint state.
- Actions: target joint positions (6-DOF).

---

## 2. Method

### 2.1 Observation Encoder

Two ResNet18 encoders (one per camera, ImageNet-pretrained) produce 512-dim feature
vectors each. Proprioceptive state is projected to 64 dims. Features from
`obs_horizon = 2` consecutive frames are concatenated, giving a conditioning vector of
dimension **[PLACEHOLDER: 2 × (512 + 512 + 64) = 2176]**.

### 2.2 Denoiser

1-D temporal convolutional U-Net with FiLM conditioning on [timestep_emb; obs_emb].
Channels: 256 → 512 → 1024 → 512 → 256. Kernel size 5, GroupNorm (8 groups) inside
each residual block. Total parameters: **[PLACEHOLDER: M]**.

### 2.3 Diffusion

Cosine noise schedule, T = 100 training steps. Epsilon prediction. DDPM sampling at
inference (100 steps). Action chunks of length T_p = 16; execute first T_a = 8 steps
before replanning.

---

## 3. Results

### 3.1 Headline: Diffusion Policy vs. BC Baseline

| Model | Tier A success | Tier B success |
|-------|---------------|---------------|
| Diffusion Policy (CNN) | **[PLACEHOLDER]** | **[PLACEHOLDER]** |
| BC Baseline (MLP) | **[PLACEHOLDER]** | **[PLACEHOLDER]** |

The BC baseline reaches into the gap between the two cubes on
**[PLACEHOLDER: X%]** of trials. The Diffusion Policy commits to one cube on every
attempt.

### 3.2 Mode Balance

Among successful Diffusion Policy trials, **[PLACEHOLDER: X%]** chose the left cube and
**[PLACEHOLDER: Y%]** chose the right cube. Mode-balance score (|P(left) − 0.5|):
**[PLACEHOLDER]** (0 = perfect 50/50).

### 3.3 Failure Breakdown (Tier A, Diffusion Policy)

| Category | Fraction |
|----------|----------|
| grabbed_nothing | **[PLACEHOLDER]** |
| grasped_wrong_object | **[PLACEHOLDER]** |
| grasp_slip | **[PLACEHOLDER]** |
| missed_cup | **[PLACEHOLDER]** |
| knocked_cup_over | **[PLACEHOLDER]** |
| other | **[PLACEHOLDER]** |

---

## 4. Ablations

### 4.1 Denoiser Backbone

| Backbone | Tier A success | Parameters |
|----------|---------------|------------|
| CNN U-Net (default) | **[PLACEHOLDER]** | **[PLACEHOLDER]** |
| Transformer | **[PLACEHOLDER]** | **[PLACEHOLDER]** |

### 4.2 Sampler

| Sampler | Steps | Tier A success | Inference latency (p95) |
|---------|-------|---------------|------------------------|
| DDPM | 100 | **[PLACEHOLDER]** | **[PLACEHOLDER] ms** |
| DDIM | 10 | **[PLACEHOLDER]** | **[PLACEHOLDER] ms** |
| DDIM | 20 | **[PLACEHOLDER]** | **[PLACEHOLDER] ms** |

### 4.3 Action Chunk Length

| T_p | T_a | Tier A success |
|-----|-----|---------------|
| 8 | 4 | **[PLACEHOLDER]** |
| 16 (default) | 8 | **[PLACEHOLDER]** |
| 32 | 16 | **[PLACEHOLDER]** |

---

## 5. Limitations and Future Work

**Eval axes this object set cannot test:**
- **Size/shape variation:** Both target cubes are identical; we cannot measure whether
  the policy generalises to objects of different size or shape.
- **Mass and friction:** All objects are uniform; dynamic grasping under varying load
  is untested.
- **Transparency/reflectance:** The cup is opaque; transparent or highly-reflective
  containers would stress the visual encoder.
- **Multi-step tasks:** The task is a single pick-and-place. Composition and recovery
  from mid-task failures are untested.

**Modelling:**
- No classifier-free guidance; the policy cannot be steered by language or goal images
  at inference time.
- The cosine schedule was not tuned for 6-DOF joint space — a schedule tailored to the
  action range might improve sample quality.

---

## 6. References

- Chi, C., et al. (2023). *Diffusion Policy: Visuomotor Policy Learning via Action
  Diffusion.* RSS 2023.
- Ho, J., et al. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS 2020.
- Song, J., et al. (2021). *Denoising Diffusion Implicit Models.* ICLR 2021.
- Nichol, A., & Dhariwal, P. (2021). *Improved Denoising Diffusion Probabilistic Models.*
  ICML 2021.
