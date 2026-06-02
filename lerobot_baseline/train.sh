#!/usr/bin/env bash
# Train LeRobot's built-in DiffusionPolicy on yeeegem/redcubes_bluecup.
#
# Usage:
#   bash lerobot_baseline/train.sh
#   bash lerobot_baseline/train.sh --steps=50000   # override any flag
#
# Checkpoints land in lerobot_baseline/runs/main/
# Check training progress: tensorboard --logdir lerobot_baseline/runs/main
#
# Tip: run `uv run lerobot-info --dataset.repo_id=yeeegem/redcubes_bluecup`
# to see total frame count, then set --steps accordingly.
# At batch_size=256, steps ≈ (total_frames / 256) * target_epochs.
set -euo pipefail

uv run lerobot-train \
  --dataset.repo_id=yeeegem/redcubes_bluecup \
  --dataset.root=recordings/redcubes_bluecup \
  --dataset.video_backend=pyav \
  --policy.type=diffusion \
  --policy.n_obs_steps=2 \
  --policy.horizon=16 \
  --policy.n_action_steps=8 \
  --policy.resize_shape="[96,96]" \
  --policy.down_dims="[256,512,1024]" \
  --policy.kernel_size=5 \
  --policy.n_groups=8 \
  --policy.use_group_norm=true \
  --policy.num_train_timesteps=100 \
  --policy.beta_schedule=squaredcos_cap_v2 \
  --policy.prediction_type=epsilon \
  --policy.clip_sample=true \
  --policy.optimizer_lr=1e-4 \
  --policy.optimizer_weight_decay=1e-6 \
  --policy.scheduler_name=cosine \
  --policy.scheduler_warmup_steps=500 \
  --policy.use_amp=true \
  --batch_size=128 \
  --num_workers=4 \
  --steps=100000 \
  --policy.push_to_hub=false \
  --seed=42 \
  --output_dir=lerobot_baseline/runs/main \
  "$@"
