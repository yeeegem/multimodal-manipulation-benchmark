# How It Works: Diffusion Policy on SO-ARM101

A guide for readers who know diffusion models but not this codebase.

---

## 1. The Problem: Multimodal Imitation Learning

Standard behaviour cloning (BC) trains a policy π(a | o) by minimising the MSE between
predicted and demonstrated actions. When the training distribution is **multimodal**
(two equally-valid strategies exist) MSE regression averages across modes, producing an
action that is wrong for *all* modes. On our task (pick either red cube and place it in
the blue cup), the BC baseline reaches into the empty space between the two cubes.

Diffusion Policy avoids this by learning a distribution over actions rather than a point
estimate. It can represent multiple modes and, during inference, commit to one.

---

## 2. Diffusion Basics: The Forward Process

Given a clean action chunk x₀ ∈ ℝ^(T_p × action_dim), the forward (noising) process
gradually corrupts it over T steps:

    q(xₜ | x₀) = N(xₜ ; √ᾱₜ · x₀,  (1 - ᾱₜ) · I)

where ᾱₜ = ∏_{s=1}^{t} (1 - βₛ) is the cumulative noise multiplier.

**Code location:** `models/diffusion.py` → `make_noise_schedule`, `q_sample`.

At t = 0, xₜ ≈ x₀. At t = T, ᾱₜ ≈ 0, so xₜ ≈ N(0, I). This is the Phase 2 unit
test: x at t=0 should be near the clean action; x at t=T should be near unit Gaussian.

---

## 3. The Noise Schedule

We use the **cosine schedule** from Nichol & Dhariwal (2021):

    ᾱₜ = cos²( (t/T + s) / (1 + s) · π/2 ) / cos²( s / (1 + s) · π/2 )

with offset s = 0.008 to prevent β becoming too small near t = 0. This keeps signal
longer than the linear schedule and is better suited to low-dimensional continuous
actions (which have less redundancy than images).

**Code location:** `models/diffusion.py` → `make_noise_schedule` with `noise_schedule="cosine"`.

---

## 4. Training Objective: Epsilon Prediction

We train a neural network ε_θ(xₜ, t, c) to predict the noise ε that was added:

    L = E_{x₀, t, ε} [ ‖ε_θ(xₜ, t, c) − ε‖² ]

where c is the observation conditioning vector. This is equivalent to predicting x₀
but has better gradient behaviour.

**Why predict ε and not x₀?** Predicting x₀ directly causes the network to learn the
full clean signal, which can lead to compounding errors at small t. Predicting ε is
a cleaner supervised signal at every t.

**Code location:** `models/diffusion.py` → `diffusion_loss`.

---

## 5. Observation Conditioning

This is **conditional diffusion** on the observation c:

    ε_θ(xₜ, t, c)

c is the concatenation of:
- ResNet18 features from the `front` camera (obs_horizon frames).
- ResNet18 features from the `wrist` camera (obs_horizon frames).
- A linear embedding of the proprioceptive state (obs_horizon frames).

**Note:** This is *not* classifier-free guidance (CFG). CFG requires training an
unconditional model alongside the conditional one and interpolating at inference. The
original Diffusion Policy paper does not use CFG - it directly conditions the denoiser
on observations. We follow the paper here.

**Code location:** `models/encoders.py` → `ObservationEncoder`.

---

## 6. The Denoiser: 1-D Temporal U-Net with FiLM

The denoiser backbone is a 1-D temporal U-Net operating on the action time axis (T_p).

**FiLM conditioning** (Feature-wise Linear Modulation):
- At each residual block, two scalars (γ, β) per channel are derived from the
  concatenated [timestep_emb; obs_emb].
- The block output is transformed: h ← γ ⊙ h + β.
- This lets the network modulate its activations based on *what time step it is* and
  *what it observed*, without requiring the conditioning information to flow through
  standard skip connections.

**Code location:** `models/cnn_backbone.py` → `FiLMResBlock1d`, `ConditionalUNet1d`.

---

## 7. Action Chunking and Receding-Horizon Control

The policy predicts an action chunk of length T_p but only executes the first T_a
steps before replanning (receding-horizon control). This means:

- The network sees T_o = 2 past observation frames as context.
- It predicts T_p = 16 future actions at once.
- It executes T_a of them (default 8 in `base.yaml`; the saved `main_96x96` run
  used 4), then replans.

This is the temporal analogue of a model-predictive controller. Predicting multiple
steps avoids the discontinuities that arise from per-step replanning.

---

## 8. DDPM Sampling (Inference)

To sample a clean action at inference, we reverse the diffusion chain:

1. Start from xₜ ~ N(0, I).
2. For t = T, T-1, ..., 1:
   a. Predict ε̂ = ε_θ(xₜ, t, c).
   b. Compute the x₀ estimate: x̂₀ = (xₜ − √(1−ᾱₜ)·ε̂) / √ᾱₜ.
   c. Sample xₜ₋₁ ~ q(xₜ₋₁ | xₜ, x̂₀) (the posterior from Ho et al. Eq. 7).

**Code location:** `models/diffusion.py` → `DDPMSampler`.

---

## 9. DDIM Sampling (Fast Inference)

DDIM (Song et al. 2021) is a deterministic ODE solver for the reverse process. It
reaches the same quality as DDPM in 10-20 steps instead of T = 100.

At each step, instead of sampling from the posterior, DDIM follows:

    xₜ₋₁ = √ᾱₜ₋₁ · x̂₀ + √(1 − ᾱₜ₋₁) · ε̂

where x̂₀ is the same denoised estimate. No noise is added between steps, making it
deterministic.

**Code location:** `models/diffusion.py` → `DDIMSampler`.

---

## 10. Normalisation

Actions and states are normalised to [−1, 1] using per-dimension min/max statistics
computed from the training set. This matters because:
- The noise schedule and `clip_sample` flag assume the clean signal lives in [−1, 1].
- Different joints have very different value ranges (pan ≈ [−π, π], gripper ≈ [0, 1]).

Stats are saved to disk alongside each checkpoint so that the eval and inference code
uses identical scaling.

**Code location:** `data/normalization.py` → `Normalizer`.

---

## 11. EMA Weights

During training, an exponential moving average (EMA) of the model weights is
maintained:

    θ_ema ← decay · θ_ema + (1 − decay) · θ

Inference always uses θ_ema. EMA smooths out noise in the gradient updates and
consistently outperforms the raw weights on diffusion models.

**Code location:** `train.py` (Phase 4).

---

## 12. The Behavior-Cloning Baseline (Phase 7)

The BC baseline is the control that makes the multimodality result legible. It shares
the *exact* observation encoder, data pipeline, min-max normalisation, action chunking,
and receding-horizon execution of the diffusion policy. Only two things change:

- **Head:** a small MLP maps the 2176-D conditioning vector straight to a
  `(pred_horizon, action_dim)` chunk in a single forward pass, instead of the iterative
  FiLM U-Net denoiser.
- **Loss:** plain MSE against the demonstration chunk (in normalised space), instead of
  epsilon-prediction.

**Why it must fail on the bimodal pick.** For a conditioning vector c that matches both
a left-cube and a right-cube demonstration, the MSE-optimal prediction is the
*conditional mean* E[a | c]. With the two valid action chunks roughly mirrored about the
workspace centre, that mean is the midpoint: the gripper drives into the empty gap
between the cubes and grasps nothing. MSE regression collapses a multimodal target to a
single average; it has no mechanism to commit to one mode. Diffusion sampling does,
because each rollout draws a different initial noise and follows it to one mode.

This is the same encoder/c -> action contrast as the diffusion path; swapping only the
head and loss isolates the modeling choice as the cause of the behaviour.

**Code locations:** `models/bc.py` (`MLPActionHead`, `BCModule`),
`models/factory.py` (`build_policy` dispatches on `model.type`),
`configs/ablations/bc_baseline.yaml`. Train it with the same `train.py`; both policies
expose identical `compute_loss` / `predict_actions` interfaces, so eval and inference are
unchanged. A self-contained numerical demonstration of the mean-collapse lives in
`docs/bc_baseline_explained.ipynb`.

---

## 13. Ablation Axes

| Axis | Default | Ablation |
|------|---------|----------|
| Policy family | Diffusion Policy | BC MSE baseline (`model.type=bc`) |
| Denoiser backbone | CNN U-Net | Transformer |
| Sampler | DDPM (training, T=100) | DDIM (deployment, 10 steps) |
| Chunk length | T_p=16, T_a=4 in saved `main_96x96` | T_p=8/32, T_a=4/16 |

Config overrides live in `configs/ablations/`.
