# LeJEPA Pre-training for METER (MobileViT)

Self-supervised pre-training of the METER monocular depth encoder using LeJEPA (SIGReg) on NYU Depth V2 RGB images.

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

On first run with HuggingFace, set `HF_OFFLINE=false` so the dataset gets cached. After that, set it to `true` for offline operation.

Alternatively, place the NYU Depth V2 `.tar` shards (`train-000000.tar`, `train-000001.tar`, ...) in `datasets/nyu/` and set `NYU_DATASET_PATH=datasets/nyu` to skip HuggingFace entirely.

## Training

All hyperparameters are managed via [Hydra](https://hydra.cc/) configs in `configs/`.

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

Each run creates a timestamped folder in `outputs/` containing:

```
outputs/xxs_2026-06-05_17-41-30/
├── .hydra/          # saved config + overrides
├── checkpoints/     # model weights
├── plots/           # loss curves
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
uv run python -m src.visualize --checkpoint outputs/xxs_2026-06-05_17-41-30/checkpoints/lejepa_xxs_final.pth

# Explicitly set variant and number of images
uv run python -m src.visualize --checkpoint path/to/checkpoint.pth --variant xs --n-images 8
```

The output PCA grid is saved to `outputs/pca_{variant}.png` in the current directory.

## Experiment configs

| Config | Use case |
|--------|----------|
| `+experiment=test` | 5 epochs, 100 samples — CI/sanity check |
| `+experiment=local` | 50 epochs, 5k samples — local GPU |
| `+experiment=production` | 200 epochs, full dataset — Colab T4 |
| `+experiment=production_a100` | 200 epochs, BS=256 — A100 |

## Project structure

```
├── configs/
│   ├── config.yaml              # base defaults
│   └── experiment/              # override profiles
├── src/
│   ├── main.py                  # Hydra entry point (training)
│   ├── config.py                # device + HF secrets
│   ├── data.py                  # NYU dataset + augmentation
│   ├── model.py                 # MobileViT + LeJEPA projector
│   ├── loss.py                  # SIGReg loss
│   ├── train.py                 # training loop
│   ├── verify.py                # sanity checks
│   └── visualize.py             # PCA visualization
├── docs/                        # architecture + training guides
├── .env.example                 # template for secrets
└── pyproject.toml               # dependencies (uv)
```