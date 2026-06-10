# LeJEPA + METER — Monocular Depth Estimation

Self-supervised pre-training (LeJEPA/SIGReg) + supervised fine-tuning (METER decoder) for monocular depth estimation on NYU Depth V2, using lightweight METER encoder variants (xxs/xs/s).

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

# Force rebuild even if files already exist
uv run python -m src.preprocess --force
```

The preprocessor uses a **streaming + parallel** architecture:
- Samples are read one-at-a-time from tar/h5 shards (never loaded all at once into RAM)
- Resize operations are parallelized across multiple CPU workers (~190 img/s on 8 cores)
- Results are written directly into pre-allocated memory-mapped files
- Peak RAM usage stays under **1 GB** regardless of dataset size

Output files include the sample count in the filename for incremental processing:

```
datasets/nyu_mmap/
├── nyu_train_rgb_192x256_N12881.npy    # (12881, 192, 256, 3) uint8
├── nyu_train_depth_192x256_N12881.npy  # (12881, 192, 256)    float32
├── nyu_val_rgb_192x256_N654.npy        # (654, 192, 256, 3)   uint8
└── nyu_val_depth_192x256_N654.npy      # (654, 192, 256)      float32
```

Total disk: **~4.3 GB** for the full NYU dataset (train + val). Processing time: ~75 seconds.

The output directory is controlled by `NYU_MMAP_DIR` in `.env` (defaults to `datasets/nyu_mmap`).

### Incremental behavior

Re-running `uv run python -m src.preprocess` is safe and fast:
- If mmap files exist with the correct sample count → **skipped** (instant)
- If sample count differs (e.g., new tar shards added) → **rebuilt** automatically
- Use `--force` to rebuild unconditionally

### Auto-detection

Once preprocessed, the training pipeline **automatically detects** the mmap files via glob pattern matching and uses them. The loader supports both old (`nyu_train_rgb_192x256.npy`) and new (`nyu_train_rgb_192x256_N12881.npy`) filename formats. All experiment configs have `data.use_mmap: true` enabled by default.

You can verify the files were created correctly:

```bash
uv run python -c "
import numpy as np
from src.data import _find_mmap_file
rgb_path = _find_mmap_file('train', 'rgb', 192, 256)
rgb = np.load(str(rgb_path), mmap_mode='r')
print(f'RGB: {rgb.shape} {rgb.dtype}')    # (N, 192, 256, 3) uint8
"
```

### Training with the full dataset

After preprocessing, you can train on all samples with minimal RAM:

```bash
uv run python -m src.main +experiment=finetune_production
```

## Training

All hyperparameters are managed via [Hydra](https://hydra.cc/) configs in `configs/`.

### Pre-training (LeJEPA)

Trains the encoder with self-supervised SIGReg loss on unlabeled RGB images.

```bash
# Quick sanity test (~2 min)
uv run python -m src.main +experiment=pretrain_test

# Meaningful local run (~2h on MX450)
uv run python -m src.main +experiment=pretrain_local

# Full production run (~2h on A100)
uv run python -m src.main +experiment=pretrain_production

# CLI overrides
uv run python -m src.main +experiment=pretrain_local epochs=100 bs=8

# Force CPU (default: auto-detects CUDA)
uv run python -m src.main +experiment=pretrain_test device=cpu
```

### Fine-tuning (METER Depth)

Trains encoder + decoder for monocular depth prediction with Balanced Loss Function.

```bash
# Quick fine-tuning test (~30 sec)
uv run python -m src.main +experiment=finetune_test

# Full METER training on NYU (60 epochs, bs=128)
uv run python -m src.main +experiment=finetune_production

# Custom variant
uv run python -m src.main +experiment=finetune_production variant=xs
```

#### Using a pre-trained LeJEPA encoder

After pre-training with LeJEPA/SIGReg, you can initialize the METER encoder with those weights instead of training from scratch:

```bash
uv run python -m src.main +experiment=finetune_production \
    finetune.pretrained_encoder=outputs/pretrain/xxs_.../checkpoints/lejepa_xxs_final.pth
```

The encoder weights are loaded from the LeJEPA checkpoint and used as the starting point for fine-tuning. The decoder is always randomly initialized.

#### Freezing the encoder

When using a pre-trained encoder, you can optionally **freeze** it for the first N epochs so only the decoder trains initially:

```bash
uv run python -m src.main +experiment=finetune_production \
    finetune.pretrained_encoder=outputs/pretrain/xxs_.../checkpoints/lejepa_xxs_final.pth \
    finetune.freeze_encoder_epochs=10
```

After epoch N, the encoder is automatically unfrozen and the optimizer is re-created to include all parameters. This is a form of staged/gradual training.

**When to freeze (`freeze_encoder_epochs > 0`):**

| Pros | Cons |
|------|------|
| Prevents catastrophic forgetting of learned SSL features in early epochs when the decoder produces random gradients | Limits capacity — encoder can't adapt to the depth task during frozen phase |
| Decoder converges faster since it trains against stable features | Requires tuning the freeze duration (too short = no benefit, too long = wasted epochs) |
| Useful when pre-training data is much larger/richer than fine-tuning data | The optimizer/scheduler resets at unfreeze, which can cause a learning rate discontinuity |
| Acts as implicit regularization — reduces effective model capacity early on | With small encoders (xxs/xs), features may not be rich enough on their own — joint training helps |

**When to train end-to-end from the start (`freeze_encoder_epochs=0`, default):**

| Pros | Cons |
|------|------|
| Encoder features adapt to the depth task from epoch 1 | Pre-trained features may degrade before decoder catches up |
| Simpler — one continuous training phase, no hyperparameter to tune | Higher risk of overfitting with small datasets if encoder + decoder overfit together |
| Best for small encoders where learned features need refinement | — |

**Rule of thumb**: Freeze for 5–15 epochs when you have a strong pre-trained encoder (trained for many epochs on large data). Skip freezing for quick experiments or when the encoder is small (xxs/xs) and benefits from joint adaptation.

#### Resuming interrupted training

Fine-tuning saves **full-state checkpoints** (model + optimizer + scheduler + epoch + loss history + RNG state) at regular intervals and on interrupt. To resume from where training stopped:

```bash
uv run python -m src.main +experiment=finetune_production \
    finetune.resume=outputs/finetune/xxs_2026-06-08_21-09-28/checkpoints/meter_xxs_epoch10.pth
```

Resume restores:
- Model weights (encoder + decoder)
- Optimizer state (momentum buffers, adaptive learning rates)
- Scheduler state (correct LR for the resumed epoch)
- Training history (loss curves continue seamlessly)
- RNG state (reproducible data ordering)

Training continues from `epoch + 1` as if it was never interrupted. This is useful for:
- Recovering from crashes or Colab disconnects
- Extending training beyond the original epoch count (set a higher `finetune.epochs`)
- Checkpoints saved on interrupt (`_interrupted` suffix) work the same way

**Note**: Resume and `pretrained_encoder` serve different purposes. Use `pretrained_encoder` to initialize a fresh fine-tuning run with SSL weights. Use `resume` to continue a previously started fine-tuning run from its exact state.

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
uv run python -m src.pca_visualization --checkpoint outputs/pretrain/xxs_.../checkpoints/lejepa_xxs_final.pth

# Explicitly set variant, number of images, and resolution
uv run python -m src.pca_visualization --checkpoint path/to/checkpoint.pth --variant xs --n-images 8 --resolution 192 256
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
| `+experiment=pretrain_test` | pretrain | 40 epochs, 100 samples — sanity check |
| `+experiment=pretrain_local` | pretrain | 200 epochs, 5k samples — local GPU |
| `+experiment=pretrain_production` | pretrain | 200 epochs, full dataset, BS=256 — A100 |
| `+experiment=finetune_test` | finetune | 2 epochs, 50 samples — sanity check |
| `+experiment=finetune_local` | finetune | 60 epochs, 10k samples — local GPU |
| `+experiment=finetune_production` | finetune | 60 epochs, full NYU — production |

All configs use `compile: true`, `data.use_mmap: true`, and `data.use_cache: false` by default.

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
│   ├── model.py                 # METER encoder + decoder + LeJEPA wrapper
│   ├── utils.py                 # SIGReg + Balanced Depth Loss + metrics + shared helpers
│   ├── train.py                 # training loops (pretrain + finetune + resume)
│   ├── evaluation.py            # depth evaluation + depth visualization
│   ├── verify.py                # sanity checks
│   └── pca_visualization.py     # PCA feature visualization (LeJEPA probing)
├── docs/
│   ├── architecture.md          # encoder + decoder architecture details
│   └── training_and_evaluation.md  # full training & eval methodology
├── .env.example                 # template for secrets
└── pyproject.toml               # dependencies (uv)
```