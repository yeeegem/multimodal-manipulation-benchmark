# LeRobot Baseline

Trains LeRobot's built-in DiffusionPolicy on the same `yeeegem/redcubes_bluecup` dataset
as the custom implementation, for a side-by-side behaviour comparison on the SO-ARM101.

## Train

```bash
bash lerobot_baseline/train.sh
```

LeRobot downloads the dataset from HuggingFace Hub on first run. Checkpoints are saved
to `lerobot_baseline/runs/main/`.

Before training, check the dataset frame count to calibrate `--steps`:

```bash
uv run lerobot-info --dataset.repo_id=yeeegem/redcubes_bluecup
```

Then set steps in `train.sh` as: `steps = (total_frames / batch_size) * target_epochs`.
The default is 100,000 steps.

## Inference

```bash
uv run python lerobot_baseline/infer.py \
    --checkpoint lerobot_baseline/runs/main/checkpoints/step_100000
```

## Architecture comparison

| | LeRobot baseline | Custom implementation |
|---|---|---|
| Denoiser | 1-D temporal U-Net | 1-D temporal U-Net |
| Channels | [256, 512, 1024] | [256, 512, 1024] |
| Vision backbone | ResNet18, GroupNorm | ResNet18, GroupNorm |
| Pooling | spatial softmax | spatial softmax |
| Image size | 96x96 | 96x96 |
| Obs horizon | 2 | 2 |
| Pred horizon | 16 | 16 |
| Exec horizon | 8 | 8 |
| Noise schedule | squaredcos_cap_v2 | cosine |
| Diffusion steps | 100 | 100 |
| Prediction target | epsilon | epsilon |
| Batch size | 256 | 256 |
| Learning rate | 1e-4 | 1e-4 |
| Sampler | DDPM | DDIM (10 steps at inference) |
| Frame caching | none (OS page cache) | numpy preload into RAM |
