# METER Training & Evaluation Guide

> Full pipeline: LeJEPA self-supervised pre-training → METER depth fine-tuning → Evaluation

---

## Overview

The training pipeline has two phases:

1. **Pre-training (LeJEPA)**: Learn geometric representations from unlabeled RGB images using a self-supervised objective (SIGReg loss). Only the encoder is trained.
2. **Fine-tuning (METER)**: Train the full encoder-decoder model for monocular depth estimation using the Balanced Loss Function (BLF) on NYU Depth V2.

After training, the model is evaluated using standard depth estimation metrics on the NYU validation set.

---

## Phase 1: LeJEPA Pre-training

### Objective

Train the MobileViT encoder to produce spatially coherent feature maps that capture 3D geometry, without any depth labels. The learned representations transfer to downstream depth estimation.

### Method: LeJEPA with SIGReg

- **Input**: V augmented views of each image (default V=4)
- **Architecture**: MobileViT encoder → AdaptiveAvgPool → 3-layer MLP projector → 16-dim embeddings
- **Loss**: LeJEPA invariance loss + λ × SIGReg regularizer
  - **Invariance**: L2 distance between projected embeddings of different views (should decrease)
  - **SIGReg**: Prevents representation collapse by encouraging the covariance matrix of embeddings to have log-eigenvalues close to 0 (neither too large nor too small)
- **Key property**: No gradient clipping needed — LeJEPA is provably stable

### Hyperparameters (default)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Resolution | 128×128 | Square crops for SSL efficiency |
| Optimizer | AdamW | weight_decay=5e-2 |
| Learning rate | 2e-3 | Scale: lr = 2e-3 × bs/256 |
| Scheduler | LinearLR warmup (1 epoch) → CosineAnnealing | eta_min=1e-3 |
| λ (SIGReg weight) | 0.02 | Paper: 0.05 with 10 views |
| Views per image | 4 | 2 global + 2 local recommended |
| Projector dim | 16 | Low-dim projection head |
| Precision | bfloat16 | Mixed precision on CUDA |

### Augmentation Policy (Pre-training)

Multi-view augmentation for contrastive learning:
- Random resized crop (scale 0.2–1.0)
- Random horizontal flip
- Color jitter (brightness, contrast, saturation, hue)
- Random grayscale (p=0.1)
- Gaussian blur (p=0.5)
- ImageNet normalization

### Output

- Encoder weights: `checkpoints/lejepa_{variant}_epoch{N}.pth`
- PCA visualizations: `pca/pca_{variant}_epoch{N}.png`
- Loss curves: `plots/loss_curves_{variant}.png`

### Commands

```bash
# Quick sanity test (~2 min)
uv run python -m src.main +experiment=test

# Local GPU training
uv run python -m src.main +experiment=local

# Production (full dataset)
uv run python -m src.main +experiment=production

# Custom overrides
uv run python -m src.main epochs=100 bs=32 data.n_samples=5000
```

---

## Phase 2: METER Depth Fine-tuning

### Objective

Train the complete encoder-decoder model to predict dense depth maps from single RGB images.

### Architecture

```
Input RGB (3, H, W)
    │
    ▼
┌──────────────────────────┐
│   MobileViT Encoder      │  ← Pre-trained (or from scratch)
│   Outputs:                │
│   - feat (C, H/32, W/32) │
│   - skips [y0, y1, y2, y3]│
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│   METER Decoder           │
│   - 1×1 Conv (reduce ch) │
│   - UpSample + skip y3   │  → H/16
│   - UpSample + skip y2   │  → H/8
│   - UpSample + skip y1   │  → H/4
│   - 3×3 Conv → 1 ch      │  → H/4
│   - Bilinear upsample    │  → H, W
│   - ReLU (depth ≥ 0)     │
└──────────────────────────┘
    │
    ▼
Output: Depth map (1, H, W) in meters
```

### Decoder Details

Each UpSample block:
1. ConvTranspose2d (×2 spatial upsampling)
2. Concatenate with encoder skip connection
3. SeparableConv + BatchNorm + ReLU (fuse upsampled + skip features)

Decoder channel configurations:

| Variant | Input→Reduce | Upsample stages | Output conv |
|---------|-------------|-----------------|-------------|
| XXS | 160→64 | 64→32→16→8 | 8→1 |
| XS | 192→128 | 128→64→32→16 | 16→1 |
| S | 320→128 | 128→64→32→16 | 16→1 |

### Balanced Loss Function (BLF)

The training loss combines four complementary terms:

$$L_{BLF} = L_{depth} + \lambda_1 L_{grad} + \lambda_2 L_{norm} + \lambda_3 L_{SSIM}$$

| Term | Formula | Purpose |
|------|---------|---------|
| $L_{depth}$ | $\frac{1}{N}\sum \|d_i - \hat{d}_i\|_1$ | Point-wise depth accuracy (L1) |
| $L_{grad}$ | $\|\nabla_x(d - \hat{d})\|_1 + \|\nabla_y(d - \hat{d})\|_1$ | Edge sharpness via Sobel gradients |
| $L_{norm}$ | $1 - \cos(\mathbf{n}, \hat{\mathbf{n}})$ | Surface orientation via computed normals |
| $L_{SSIM}$ | $1 - SSIM(d, \hat{d})$ | Structural similarity (local statistics) |

Default weights: $\lambda_1 = 0.5$, $\lambda_2 = 1.0$, $\lambda_3 = 1.0$

All losses mask out invalid depth pixels (depth ≤ 0).

### METER Augmentation Policy (Fine-tuning)

Applied only during training (p=0.5 each):
- **Horizontal flip**: flip RGB and depth together
- **Vertical flip** (mirror): flip RGB and depth together
- **Channel swap**: randomly permute RGB channels
- **Shifting**: translate image by random offset, fill with reflection

### Hyperparameters (METER paper, NYU)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Resolution | 256×192 (H×W) | Official METER for NYU |
| Optimizer | AdamW | β1=0.9, β2=0.999, wd=0.01 |
| Learning rate | 0.001 | Fixed initial LR |
| Scheduler | StepLR | ×0.1 every 20 epochs |
| Epochs | 60 | |
| Batch size | 128 | |
| Encoder freezing | 0 epochs | End-to-end from start |
| Precision | bfloat16 | Mixed precision on CUDA |

### Encoder Initialization Options

1. **From scratch** (`pretrained_encoder: null`): Random initialization
2. **LeJEPA pre-trained** (`pretrained_encoder: path/to/checkpoint.pth`): Transfer learned geometry

### Output

- Model checkpoints: `checkpoints/meter_{variant}_epoch{N}.pth`
- Depth prediction grids: `plots/depth_{variant}_epoch{N}.png`
- Loss curves: `plots/depth_loss_curves_{variant}.png`
- Final model: `checkpoints/meter_{variant}_final.pth`

### Commands

```bash
# Quick fine-tuning test (~30 sec)
uv run python -m src.main +experiment=finetune_test

# Full METER fine-tuning (NYU, 60 epochs)
uv run python -m src.main +experiment=finetune_nyu

# With pre-trained encoder
uv run python -m src.main +experiment=finetune_nyu \
    finetune.pretrained_encoder=outputs/pretrain/xxs_2026-06-05/checkpoints/lejepa_xxs_final.pth

# Custom overrides
uv run python -m src.main +experiment=finetune_nyu finetune.epochs=30 finetune.bs=64

# Different variant
uv run python -m src.main +experiment=finetune_nyu variant=xs
```

---

## Phase 3: Evaluation

### Metrics

Standard monocular depth estimation metrics computed on the NYU validation set:

| Metric | Formula | Direction |
|--------|---------|-----------|
| δ1 | % of pixels where max(d/d̂, d̂/d) < 1.25 | ↑ higher is better |
| δ2 | % of pixels where max(d/d̂, d̂/d) < 1.25² | ↑ higher is better |
| δ3 | % of pixels where max(d/d̂, d̂/d) < 1.25³ | ↑ higher is better |
| RMSE | √(mean((d - d̂)²)) | ↓ lower is better |
| REL | mean(\|d - d̂\| / d) | ↓ lower is better |
| log10 | mean(\|log10(d) - log10(d̂)\|) | ↓ lower is better |

### During Training

Validation metrics are computed:
- Every `val_every` epochs (default: 5)
- At the final epoch
- At every checkpoint save

Depth prediction grids (RGB | GT | Predicted | Error) are generated at every checkpoint.

### Standalone Evaluation

```bash
# Evaluate a checkpoint (auto-detects variant from filename)
uv run python -m src.evaluation --checkpoint path/to/meter_xxs_final.pth

# Full validation set evaluation (654 images)
uv run python -m src.evaluation --checkpoint path/to/meter_xxs_final.pth --full-val

# Custom options
uv run python -m src.evaluation \
    --checkpoint path/to/model.pth \
    --variant xxs \
    --n-images 8 \
    --output results/my_eval.png
```

### Evaluation Output

1. **Console**: Prints all 6 metrics (δ1, δ2, δ3, RMSE, REL, log10)
2. **Depth grid image**: 4-column visualization:
   - Column 1: Input RGB
   - Column 2: Ground truth depth (plasma colormap, 0–10m)
   - Column 3: Predicted depth (plasma colormap, 0–10m)
   - Column 4: Absolute error |GT − Pred| (hot colormap)

### PCA Feature Visualization

Visualize encoder representations without depth prediction:

```bash
uv run python -m src.visualize --checkpoint path/to/lejepa_xxs_final.pth
```

Generates PCA projections of skip connection features at 3 scales (H/4, H/8, H/16).

---

## Output Directory Structure

Each run produces a timestamped output folder:

```
outputs/
├── pretrain/
│   └── xxs_2026-06-08_18-00-00/
│       ├── .hydra/              # saved config + overrides
│       ├── checkpoints/
│       │   ├── lejepa_xxs_epoch10.pth
│       │   └── lejepa_xxs_final.pth
│       ├── pca/
│       │   ├── pca_xxs_epoch10.png
│       │   └── pca_xxs_epoch50.png
│       ├── plots/
│       │   └── loss_curves_xxs.png
│       └── main.log
│
└── finetune/
    └── xxs_2026-06-08_20-00-00/
        ├── .hydra/
        ├── checkpoints/
        │   ├── meter_xxs_epoch5.pth
        │   └── meter_xxs_final.pth
        ├── plots/
        │   ├── depth_xxs_epoch5.png
        │   ├── depth_xxs_epoch10.png
        │   └── depth_loss_curves_xxs.png
        └── main.log
```

---

## Expected Results (METER paper, NYU Depth V2)

| Variant | δ1 ↑ | RMSE ↓ | REL ↓ | Params |
|---------|-------|--------|-------|--------|
| XXS | 0.820 | 0.442 | 0.153 | 0.71M |
| XS | 0.853 | 0.398 | 0.138 | 1.45M |
| S | 0.878 | 0.367 | 0.124 | 3.29M |

*Note: Results depend on training duration, data volume, and whether LeJEPA pre-training is used.*

---

## References

- Papa, L., Russo, P., & Amerini, I. (2024). *METER: A Mobile Vision Transformer Architecture for Monocular Depth Estimation.*
- Balestriero, R., & LeCun, Y. (2025). *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics.*
- Silberman, N., et al. (2012). *Indoor Segmentation and Support Inference from RGBD Images.* (NYU Depth V2)
