# LeJEPA + METER — Monocular Depth Estimation

Self-supervised pre-training (LeJEPA/SIGReg) + supervised fine-tuning (METER decoder) for monocular depth estimation on NYU Depth V2, using lightweight MobileViT encoder variants.

[nyu kaggle](https://www.kaggle.com/datasets/awsaf49/nyuv2-official-split-dataset/data)

## Setup

### 1. Clone and enter the repository

```bash
git clone https://github.com/anton/cv-test.git
cd cv-test
```

### 2. Install dependencies

Requires [uv](https://docs.astral.sh/uv/) (Python package manager) and Python 3.13+.

```bash
uv sync
```

This installs PyTorch with CUDA 12.4 support, Hydra, and all other dependencies.

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set:

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace token (optional, avoids rate limits) |
| `HF_OFFLINE` | `true` to use local cache only, `false` to stream from HF |
| `NYU_DATASET_PATH` | Path to local NYU tar archives (e.g. `datasets/nyu`). If set, loads from local `.tar/.h5` files instead of HuggingFace. Relative paths resolve from project root. |
| `NYU_MMAP_DIR` | Path to preprocessed memory-mapped `.npy` files (default: `datasets/nyu_mmap`). Created by `uv run python -m src.preprocess`. |

On first run with HuggingFace, set `HF_OFFLINE=false` so the dataset gets cached. After that, set it to `true` for offline operation.

Alternatively, place the NYU Depth V2 `.tar` shards (`train-000000.tar`, `train-000001.tar`, ...) in `datasets/nyu/` and set `NYU_DATASET_PATH=datasets/nyu` to skip HuggingFace entirely.

## Data Preprocessing (Memory-Mapped Dataset)

### Why

The default data pipeline loads all images into RAM as full-resolution PIL objects (~4.7 MB per sample). This works for small runs (1000 samples ≈ 4.7 GB RAM) but becomes impossible for the full NYU dataset (45k samples ≈ 211 GB RAM).

**Memory-mapped preprocessing** solves this by:
1. Resizing all images **once** to the training resolution (192×256) and saving them as flat `.npy` arrays on disk
2. At training time, opening the file with `mmap_mode='r'` — the OS pages in **only the accessed samples**, so RAM usage is near-zero regardless of dataset size
3. Eliminating all per-sample overhead (no tar decompression, no h5 decoding, no PIL resize) — giving **~10× faster data loading** compared to the tar/h5 pipeline

| Approach | 45k samples RAM | I/O speed | Setup |
|----------|----------------|-----------|-------|
| Load into RAM (default) | ~211 GB ❌ | Instant (already in memory) | None |
| Memory-mapped `.npy` | **~50 MB** ✅ | Near-instant (OS page cache) | One-time preprocessing |
| Lazy tar/h5 reading | ~50 MB | Very slow (decompress per batch) | None |

### How to preprocess

```bash
# Default: resize to 192×256 (the METER paper resolution for NYU)
uv run python -m src.preprocess

# Explicit resolution (height width)
uv run python -m src.preprocess 192 256
```

This reads all tar/h5 shards from `NYU_DATASET_PATH` (or streams from HuggingFace if not set), resizes every image, and writes two files per split:

```
datasets/nyu_mmap/
├── nyu_train_rgb_192x256.npy    # (N, 192, 256, 3) uint8  — ~3.4 GB for 45k
├── nyu_train_depth_192x256.npy  # (N, 192, 256)    float32 — ~3.5 GB for 45k
├── nyu_val_rgb_192x256.npy      # (654, 192, 256, 3) uint8
└── nyu_val_depth_192x256.npy    # (654, 192, 256)    float32
```

Total disk: **~7 GB** for the full NYU dataset (train + val).

The output directory is controlled by `NYU_MMAP_DIR` in `.env` (defaults to `datasets/nyu_mmap`).

### Auto-detection

Once preprocessed, the training pipeline **automatically detects** the mmap files and uses them — no config changes needed. The priority order is:

1. Memory-mapped `.npy` files exist → use them (fastest, zero RAM)
2. `data.use_cache: true` in config → use per-sample `.pt` cache
3. Neither → load from tar/h5 into RAM (default for small runs)

You can verify the files were created correctly:

```bash
uv run python -c "
import numpy as np
rgb = np.load('datasets/nyu_mmap/nyu_train_rgb_192x256.npy', mmap_mode='r')
depth = np.load('datasets/nyu_mmap/nyu_train_depth_192x256.npy', mmap_mode='r')
print(f'RGB: {rgb.shape} {rgb.dtype}')    # (N, 192, 256, 3) uint8
print(f'Depth: {depth.shape} {depth.dtype}')  # (N, 192, 256) float32
"
```

### Training with the full dataset

After preprocessing, you can train on all 45k samples with minimal RAM:

```bash
uv run python -m src.main +experiment=finetune_small data.n_samples=45000
```

## Training

All hyperparameters are managed via [Hydra](https://hydra.cc/) configs in `configs/`.

### Pre-training (LeJEPA)

Trains the encoder with self-supervised SIGReg loss on unlabeled RGB images.

```bash
# Quick sanity test (~2 min)
uv run python -m src.main +experiment=test

# Meaningful local run (~2h on MX450)
uv run python -m src.main +experiment=local

# Full production run (~6h on T4)
uv run python -m src.main +experiment=production

# CLI overrides
uv run python -m src.main +experiment=local epochs=100 bs=8

# Force CPU (default: auto-detects CUDA)
uv run python -m src.main +experiment=test device=cpu
```

### Fine-tuning (METER Depth)

Trains encoder + decoder for monocular depth prediction with Balanced Loss Function.

```bash
# Quick fine-tuning test (~30 sec)
uv run python -m src.main +experiment=finetune_test

# Full METER training on NYU (60 epochs, bs=128)
uv run python -m src.main +experiment=finetune_nyu

# With pre-trained encoder from LeJEPA
uv run python -m src.main +experiment=finetune_nyu \
    finetune.pretrained_encoder=outputs/pretrain/xxs_.../checkpoints/lejepa_xxs_final.pth

# Custom variant
uv run python -m src.main +experiment=finetune_nyu variant=xs
```

### Device selection

The `device` option in `configs/config.yaml` controls hardware:

| Value | Behavior |
|-------|----------|
| `auto` (default) | Uses CUDA if available, otherwise CPU |
| `cuda` | Force GPU (fails if no CUDA) |
| `cpu` | Force CPU |

### Stopping training (Ctrl+C)

Training handles interrupts gracefully:

- **First Ctrl+C**: finishes the current epoch, saves an `_interrupted` checkpoint, plots loss curves for completed epochs, then exits cleanly.
- **Second Ctrl+C**: force-quits immediately with a best-effort checkpoint of the last completed epoch.

Each run creates a timestamped folder in `outputs/<task>/` containing:

```
outputs/pretrain/xxs_2026-06-05_17-41-30/
├── .hydra/          # saved config + overrides
├── checkpoints/     # encoder weights
├── pca/             # PCA feature visualizations
├── plots/           # loss curves
└── main.log         # full training log

outputs/finetune/xxs_2026-06-08_20-00-00/
├── .hydra/          # saved config + overrides
├── checkpoints/     # full model weights (encoder + decoder)
├── plots/           # loss curves + depth prediction grids
└── main.log         # full training log
```

### Verification (optional)

Run a forward-pass sanity check before committing to a long training:

```bash
uv run python -m src.verify
```

## PCA Visualization

After training, visualize learned features with PCA on validation images:

```bash
# Point to any checkpoint (variant is auto-detected from filename)
uv run python -m src.visualize --checkpoint outputs/pretrain/xxs_.../checkpoints/lejepa_xxs_final.pth

# Explicitly set variant and number of images
uv run python -m src.visualize --checkpoint path/to/checkpoint.pth --variant xs --n-images 8
```

The output PCA grid is saved to `outputs/pca_{variant}.png` in the current directory.

## Depth Evaluation

Evaluate a trained METER model on the NYU validation set:

```bash
# Quick evaluation (4 images, prints metrics + saves depth grid)
uv run python -m src.evaluation --checkpoint path/to/meter_xxs_final.pth

# Full validation set (654 images)
uv run python -m src.evaluation --checkpoint path/to/meter_xxs_final.pth --full-val

# Custom output path and more images
uv run python -m src.evaluation --checkpoint path/to/model.pth --n-images 8 --output results/eval.png
```

Outputs:
- **Console**: δ1, δ2, δ3, RMSE, REL, log10 metrics
- **Image**: 4-column grid (RGB | GT Depth | Predicted | Error map)

## Experiment configs

| Config | Task | Use case |
|--------|------|----------|
| `+experiment=test` | pretrain | 5 epochs, 100 samples — CI/sanity check |
| `+experiment=local` | pretrain | 50 epochs, 5k samples — local GPU |
| `+experiment=production` | pretrain | 200 epochs, full dataset — Colab T4 |
| `+experiment=production_a100` | pretrain | 200 epochs, BS=256 — A100 |
| `+experiment=finetune_test` | finetune | 2 epochs, 50 samples — sanity check |
| `+experiment=finetune_nyu` | finetune | 60 epochs, full NYU — production |

## Project structure

```
├── configs/
│   ├── config.yaml              # base defaults
│   └── experiment/              # override profiles
├── src/
│   ├── main.py                  # Hydra entry point (pretrain + finetune)
│   ├── config.py                # device + HF secrets + dataset paths
│   ├── data.py                  # NYU dataset + augmentation + mmap loader
│   ├── preprocess.py            # one-time tar/h5 → memory-mapped .npy conversion
│   ├── model.py                 # MobileViT encoder + METER decoder
│   ├── loss.py                  # SIGReg + Balanced Depth Loss + metrics
│   ├── train.py                 # training loops (pretrain + finetune + resume)
│   ├── evaluation.py            # standalone depth evaluation CLI
│   ├── verify.py                # sanity checks
│   └── visualize.py             # PCA + depth prediction visualization
├── docs/
│   ├── architecture.md          # encoder + decoder architecture details
│   └── training_and_evaluation.md  # full training & eval methodology
├── .env.example                 # template for secrets
└── pyproject.toml               # dependencies (uv)
```