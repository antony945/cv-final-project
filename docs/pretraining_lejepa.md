# LeJEPA Pre-training — Hyperparameter Guide

This document explains every configurable parameter, its purpose, the theory behind it, and how to set it depending on your hardware, time budget, and goals.

---

## How to configure

All hyperparameters are managed via Hydra YAML configs in `configs/`. The base config is `configs/config.yaml`, with named experiment profiles in `configs/experiment/`.

```powershell
# Run with base config
uv run python -m src.main

# Use a named experiment profile
uv run python -m src.main +experiment=pretrain_test
uv run python -m src.main +experiment=pretrain_local
uv run python -m src.main +experiment=pretrain_production

# Override any parameter from CLI
uv run python -m src.main +experiment=pretrain_local epochs=100 bs=8
```

---

## The one true LeJEPA hyperparameter

### `LAMBDA` — SIGReg weight (default: `0.02`)

This is the **only hyperparameter that the LeJEPA paper introduces**. Everything else (LR, batch size, epochs) is standard deep learning practice.

**What it does**: balances the two loss components:
```
total_loss = LAMBDA × SIGReg_loss  +  (1 - LAMBDA) × invariance_loss
```

- The **invariance loss** pushes all views of the same image toward the same embedding (the "learning" signal)
- The **SIGReg loss** pushes the embedding distribution toward a standard Gaussian, preventing collapse (the "don't cheat" signal)

**Why 0.02**: The paper showed this is robust across 60+ architectures and 10+ datasets. It keeps SIGReg as a light regularizer (2% of the loss) while letting invariance dominate. The official example uses exactly `0.02` and so do we.

**Should you change it?** Almost never. Only experiment with it if:
- Loss becomes unstable (try 0.01)
- Invariance collapses to 0 too fast and SIGReg explodes (try 0.05)

---

## Data parameters

### `N_SAMPLES` — Number of training images (default: `1000`)

**What it does**: how many NYU RGB images are pulled from HuggingFace via streaming and stored in memory.

NYU Depth V2 has **47,584 training images** total.

| N_SAMPLES | Memory | Loading time | Use case |
|-----------|--------|--------------|----------|
| 100 | ~50 MB | ~10 sec | Quick sanity test |
| 1000 | ~500 MB | ~1 min | Debug / verify pipeline |
| 5000 | ~2.5 GB | ~5 min | Lightweight training run |
| 47584 | ~24 GB | ~45 min | Full dataset (Colab) |

**Key insight**: LeJEPA is designed to work with small datasets. The paper shows strong results even with very limited data, because SIGReg provides a rich learning signal without needing labels.

**What to use**: 
- Local laptop (≤ 8 GB RAM): `N_SAMPLES=5000`
- Local laptop (≥ 16 GB RAM): `N_SAMPLES=10000`
- Colab: `N_SAMPLES=47584`

---

## Training budget parameters

These three interact. Think of the training budget as:

```
total_gradient_steps = (N_SAMPLES / PRETRAIN_BS) × PRETRAIN_EPOCHS
```

The LeJEPA paper on ImageNet (10k images, BS=256): 800 epochs → ~31,000 steps.
To match that on NYU (47k images, BS=64): ~40 epochs → ~29,500 steps.

### Do I need to increase batch size when I increase epochs/samples?

**No.** Batch size and epochs/samples serve different purposes:

- **Batch size** controls *gradient quality per step*: larger BS = smoother, more stable gradients, fewer steps to converge, but more VRAM. Smaller BS = noisier gradients, more steps needed, but less VRAM.
- **Epochs** controls *how many times the model sees the data*: more epochs = more optimization steps, more time for the model to converge.
- **N_SAMPLES** controls *data diversity*: more samples = richer training signal per epoch, slower plateau.

**What happens if you run more epochs with the same BS/samples?** The model keeps training with new random augmentations each pass. Loss decreases until it plateaus. With few samples (e.g., 100) you'll plateau fast (limited image diversity). With 5000+ samples, each epoch still provides novel combinations of crops and augmentations.

**Why the recommended configs increase BS with scale**: it's about *training efficiency*, not correctness. With 5000 samples and BS=4 you'd get 1250 batches/epoch — each epoch is slow with noisy gradients. BS=16 → 312 batches/epoch — faster epochs, smoother training. The LR is scaled proportionally (linear scaling rule).

**Rule of thumb**: pick the largest BS that fits in your VRAM, scale LR proportionally, then choose epochs to reach ~30,000 total steps.

### `PRETRAIN_EPOCHS` — Number of full passes over data (default: `50`)

**What it does**: how many times the model sees all N_SAMPLES images.

| Epochs | Steps (N=5000, BS=32) | Quality | Time (MX450) | Time (T4) |
|--------|----------------------|---------|--------------|-----------|
| 5 | ~780 | sanity check | ~5 min | ~1 min |
| 50 | ~7800 | early features visible | ~50 min | ~10 min |
| 200 | ~31000 | comparable to paper | ~3.5 h | ~40 min |

**Rule of thumb**: aim for ~30,000 total gradient steps for a meaningful run.

### `PRETRAIN_BS` — Batch size (default: `64`)

**What it does**: number of *images* per gradient step. Each image produces `N_VIEWS` views, so the actual tensor going through the encoder is `PRETRAIN_BS × N_VIEWS`.

**VRAM cost** (xxs variant, 128×128 input):

| BS | Views | Encoder inputs/step | Approx VRAM |
|----|-------|---------------------|-------------|
| 4 | 4 | 16 | ~0.5 GB |
| 16 | 4 | 64 | ~1.0 GB |
| 32 | 4 | 128 | ~1.5 GB |
| 64 | 4 | 256 | ~2.5 GB |
| 128 | 4 | 512 | ~4.5 GB |

**For your MX450 (2 GB VRAM)**: use `PRETRAIN_BS=16` max.  
**For Colab T4 (16 GB VRAM)**: use `PRETRAIN_BS=64` or `128`.  
**For Colab A100 (40 GB VRAM)**: use `PRETRAIN_BS=256` (matches paper).

**Important**: if you change BS, adjust LR proportionally (linear scaling rule):
```
new_LR = base_LR × (new_BS / base_BS)
```
E.g., if you go from BS=64 to BS=16: `LR = 2e-3 × (16/64) = 5e-4`

### `PRETRAIN_LR` — Learning rate (default: `2e-3`)

**What it does**: step size for AdamW optimizer.

The default `2e-3` is from the official LeJEPA example, calibrated for BS=256. Scale linearly with batch size as above.

### `WEIGHT_DECAY` — AdamW weight decay (default: `0.01`)

**What it does**: L2 regularization applied to all parameters by AdamW. It shrinks weights toward zero each step, preventing overfitting and improving generalization.

The LeJEPA paper recommends searching over `{1e-1, 1e-2, 1e-5}`:
- `1e-1`: aggressive regularization — use when overfitting (low N_SAMPLES, many epochs)
- `1e-2`: balanced (our default) — works well in most cases
- `1e-5`: near-zero regularization — use for very short runs where overfitting isn't a risk

**Important**: there is **no scheduler on weight decay** — it stays constant throughout training. Only the learning rate is annealed.

### `LR_MIN` — Cosine annealing floor (default: `0.001`)

**What it does**: the minimum learning rate that the cosine scheduler decays *to*. After warmup, LR follows a cosine curve from `lr` down to `lr_min`.

- `lr_min=1e-3` with `lr=2e-3`: mild decay (lr only halves). Keeps the model learning throughout. This is what the LeJEPA minimal example uses.
- `lr_min=1e-5` with `lr=2e-3`: aggressive full annealing (lr drops to ~0). More standard in the SSL literature (DINO, MAE, etc).

**Recommendation**: keep `1e-3` for our use case. LeJEPA is provably stable and benefits from continued learning at moderate LR. Only lower it for very long runs (400+ epochs) where you want fine-grained convergence at the end.

---

## Learning Rate Schedule — How it works

Our pre-training uses a **2-phase learning rate schedule** via PyTorch's `SequentialLR`:

```
Phase 1: Linear Warmup (1 epoch)
   lr goes from 0.01 × lr  →  lr

Phase 2: Cosine Annealing (remaining epochs)
   lr goes from lr  →  lr_min, following a cosine curve
```

### The code

```python
warmup_steps = len(loader)                           # = batches_per_epoch
total_steps = len(loader) * epochs                   # = total batches
s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
s2 = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=cfg.lr_min)
scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])
```

The scheduler is **stepped every batch** (not every epoch). This gives smooth transitions.

### Concrete example

With our default config: `lr=2e-3`, `lr_min=1e-3`, `epochs=100`, `BS=128`, `N_SAMPLES=12881`:

```
batches_per_epoch = 12881 / 128 = ~101
warmup_steps = 101 (= 1 epoch)
cosine_steps = 101 × 100 - 101 = 9999

Step 0:     lr = 0.01 × 2e-3 = 2e-5      (cold start)
Step 50:    lr = 1.0e-3                    (mid-warmup)
Step 101:   lr = 2e-3                      (warmup complete, peak)
Step 5000:  lr ≈ 1.5e-3                    (mid-cosine)
Step 10100: lr = 1e-3                      (final, = lr_min)
```

### Visual shape

```
lr
2e-3 ┤          ╭─────╮
     │         ╱       ╲
     │        ╱         ╲
1.5e-3 ┤      ╱           ╲
     │     ╱             ╲
     │    ╱               ╲─────────
1e-3 ┤   ╱                         ← lr_min (floor)
     │  ╱
     │ ╱ ← warmup
2e-5 ┤╱
     └──────────────────────────────── steps
      0   101        5000       10100
      │←1 ep→│←── cosine decay ──→│
```

### Why warmup?

Without warmup, the optimizer starts with full LR on randomly initialized weights. This causes:
- Large, unstable gradient updates in the first few hundred steps
- Risk of divergence (NaN loss), especially with mixed precision
- SIGReg can spike because the embedding space starts degenerate

Linear warmup gives the model time to "find its footing" — gradients stabilize, batch normalization layers accumulate useful statistics, and the embedding space spreads out gently.

**1 epoch is enough** because LeJEPA is provably stable (no momentum teacher, no stop-gradient), so it doesn't need the 5-10 epoch warmups common in DINO/BYOL.

### Why cosine (not step/constant)?

Cosine annealing provides a smooth, continuous decay without the sudden "cliff" of StepLR. Benefits:
- No hyperparameter for step timing (just total steps + floor)
- Gradual decrease matches the diminishing-returns nature of training
- The paper's minimal example uses exactly this schedule

---

## Model parameters

### `VARIANT` — METER encoder size (default: `xxs`)

| Variant | Params | emb_dim | VRAM (+projector) | Notes |
|---------|--------|---------|-------------------|-------|
| `xxs` | 0.71M | 160 | baseline | Start here always |
| `xs` | 1.45M | 192 | +30% | Use when xxs looks good |
| `s` | 3.29M | 320 | +80% | Full experiment on Colab |

For the project, `xxs` is the correct starting point. The METER paper shows all three variants, so ideally you run all three eventually (each as a separate experiment).

### `N_VIEWS` — Augmented views per image (default: `4`)

**What it does**: how many random crops/augmentations of each image are generated per step.

The official LeJEPA example uses `V=4`. More views = stronger invariance signal but higher VRAM.

- `N_VIEWS=2`: minimum viable (like SimCLR). Less diverse signal.
- `N_VIEWS=4`: paper default. Best quality/cost tradeoff.
- `N_VIEWS=8`: stronger but doubles VRAM and computation.

**Don't change this** unless you're doing an ablation study.

### `PROJ_DIM` — Projector output dimension (default: `16`)

**What it does**: the dimensionality of the projected embedding that SIGReg operates on.

`16` is the official LeJEPA recommendation — surprisingly small, but the paper shows this works. SIGReg is designed to work in low dimensions efficiently. The projector MLP is `emb_dim → 2048 → 2048 → 16`.

- Smaller = faster SIGReg computation, slightly less expressive
- Larger (e.g. 64, 128) = more expressive but no proven benefit

**Don't change** unless doing ablation.

### `PRETRAIN_RES` — Input resolution for pre-training (default: `128`)

**What it does**: the size of the square crop fed to the encoder during SSL.

`128×128` is the official LeJEPA recommendation (same as their ViT example). The METER encoder is fully convolutional, so the backbone is resolution-agnostic — this doesn't affect downstream fine-tuning quality.

- `128`: fast, standard SSL practice ✓
- `192` or `256`: higher-res crops, slightly richer features, ~2.25–4× more VRAM
- `64`: very fast, but crops may be too small to contain useful structure

#### Should I match the fine-tuning resolution (192×256)?

This is a natural question since our fine-tuning uses `192×256`. Here's the tradeoff:

**Pros of higher pretrain resolution (192×192 or 256×256)**:
- Closer distribution match between pretrain and finetune — the encoder sees similar spatial scales during both phases, reducing the "resolution gap" at transfer
- More spatial tokens per image → richer PCA maps, more fine-grained features
- The encoder learns to represent small objects that are invisible at 128×128
- Better feature maps at the H/16 and H/32 levels (12×12 vs 8×8 at 192, or 16×16 vs 8×8 at 256)

**Cons of higher pretrain resolution**:
- **VRAM scales quadratically**: 192×192 is 2.25× more pixels → ~2× more VRAM per image. 256×256 is 4× more pixels → ~3.5× more VRAM. With BS=128 and 4 views, you may need to halve BS (fewer steps/epoch)
- **Slower training**: each forward/backward pass takes proportionally longer. A 200-epoch run at 256×256 takes ~3-4× longer than at 128×128
- **The paper uses 128**: the LeJEPA minimal example trains at 128×128 on ImageNette and achieves SOTA. Higher resolution was not shown to help significantly for self-supervised learning
- **Diminishing returns for SSL**: the invariance objective doesn't inherently benefit from seeing more pixels — it learns crop-invariant features regardless. The additional pixels mostly add computation cost
- **Resolution mismatch is normal in SSL**: DINO pretrains at 224 but finetunes at 518. MAE pretrains at 224, finetunes at 448. The community consensus is that SSL benefits more from data diversity than pixel count

**Recommendation**:
- **Default**: keep `128×128` for pretraining. It's fast, proven, and the encoder adapts to higher resolution at fine-tuning time (fully convolutional)
- **If you have spare VRAM** (Colab A100): try `192×192` as an ablation — compare PCA maps and downstream depth metrics vs. 128×128 baseline
- **Don't go to 256×256** unless you halve batch size — the VRAM cost is prohibitive and benefits are marginal for SSL

| Resolution | Tokens (H/16 × H/32) | VRAM (BS=128, 4 views, xxs) | Training speed | Notes |
|------------|----------------------|----------------------------|----------------|-------|
| 64×64 | 4×4 + 2×2 | ~1.5 GB | very fast | Too small, poor features |
| 128×128 | 8×8 + 4×4 | ~4.5 GB | baseline | **Recommended** |
| 192×192 | 12×12 + 6×6 | ~9 GB | ~2.2× slower | Ablation on A100 |
| 256×256 | 16×16 + 8×8 | ~16 GB | ~4× slower | Requires halved BS |

**Keep at 128** unless you have abundant VRAM and want to ablate.

---

## Infrastructure parameters

### `CKPT_EVERY` — Checkpoint save frequency (default: `10`)

Saves backbone weights every N epochs to `checkpoints/lejepa_xxs_epochN.pth`.

Set to `5` for short runs (50 epochs), `25` for long runs (200+ epochs).

### `USE_WANDB` — Enable wandb logging (default: `true`)

Set `false` for quick tests. When `true`, logs per-batch losses to your wandb dashboard. Requires `wandb login` first (see `docs/wandb_setup.txt`).

### Automatic PCA visualization

PCA probing is run **automatically at every checkpoint** during training. Results are saved inside the hydra run folder:

```
outputs/pretrain/xxs_2026-06-05_17-41-30/
├── checkpoints/
│   ├── lejepa_xxs_epoch10.pth
│   └── lejepa_xxs_final.pth
├── plots/
│   ├── loss_curves_xxs.png
│   ├── pca_xxs_epoch10.png      ← auto-generated
│   └── pca_xxs_epoch50.png      ← auto-generated (final)
└── main.log
```

This lets you visually track how features evolve during training without manually running `pca_visualization.py`. The auto-PCA uses 4 validation images (fewer than the standalone 6) for speed.

If PCA fails for any reason (e.g., dataset not cached), training continues normally — it's wrapped in a try/except and logs a warning.

---

## Recommended configurations

### Quick test (laptop, CPU or MX450)
```
VARIANT=xxs
N_SAMPLES=100
PRETRAIN_EPOCHS=5
PRETRAIN_BS=4
PRETRAIN_LR=2e-3
WEIGHT_DECAY=0.01
LR_MIN=0.001
USE_WANDB=false
```
~2 min. Just check the pipeline works.

### Meaningful local run (MX450, 2 GB VRAM)
```
VARIANT=xxs
N_SAMPLES=5000
PRETRAIN_EPOCHS=50
PRETRAIN_BS=16
PRETRAIN_LR=5e-4
WEIGHT_DECAY=0.01
LR_MIN=5e-4
CKPT_EVERY=10
USE_WANDB=true
```
~2 hours. Should produce visually coherent PCA maps.

### Full experiment (Colab T4)
```
VARIANT=xxs
N_SAMPLES=47584
PRETRAIN_EPOCHS=200
PRETRAIN_BS=64
PRETRAIN_LR=2e-3
WEIGHT_DECAY=0.01
LR_MIN=0.001
CKPT_EVERY=25
USE_WANDB=true
```
~6 hours. Production-quality pre-training. Run xs and s variants after.

### Full experiment (Colab A100)
```
VARIANT=xxs
N_SAMPLES=47584
PRETRAIN_EPOCHS=200
PRETRAIN_BS=256
PRETRAIN_LR=2e-3
WEIGHT_DECAY=0.01
LR_MIN=0.001
CKPT_EVERY=25
USE_WANDB=true
```
~2 hours. Matches original LeJEPA paper scale.

---

## What to look for during training

| Signal | Healthy | Problem |
|--------|---------|---------|
| `inv` loss | Decreases steadily toward 0 | Stuck high = LR too low |
| `sigreg` loss | Rises slightly then plateaus around 1.5–2.0 | Keeps exploding = LAMBDA too low |
| `lejepa` total | Smooth decrease | Spiky = LR too high, try halving |
| PCA maps | Spatially coherent regions | Random noise = not enough training |

The invariance loss should drop fastest in the first 20% of training. If it's still near the initial value after 20 epochs, increase epochs or increase N_SAMPLES.

---

## Production run — best possible configuration

This is the configuration that maximizes pre-training quality for downstream depth estimation. Run this on Colab A100 (recommended) or T4.

### Target: ~30,000 gradient steps

This is the benchmark from the LeJEPA paper (800 epochs × ~39 steps/epoch on ImageNette with BS=256). The formula is:
```
steps = (N_SAMPLES / PRETRAIN_BS) × PRETRAIN_EPOCHS
```

With the full NYU dataset and A100:
```
(47584 / 256) × 200 = ~37,000 steps  ✓
```

### Colab A100 (best quality)
```
VARIANT=xxs
N_SAMPLES=47584
PRETRAIN_EPOCHS=200
PRETRAIN_BS=256
PRETRAIN_LR=2e-3
WEIGHT_DECAY=0.01
LR_MIN=0.001
N_VIEWS=4
PROJ_DIM=16
PRETRAIN_RES=128
CKPT_EVERY=25
USE_WANDB=true
WANDB_PROJECT=lejepa-meter
```
Estimated time: ~2–3 hours. This matches the exact scale of the official LeJEPA example.

### Colab T4 (good quality, slower)
```
VARIANT=xxs
N_SAMPLES=47584
PRETRAIN_EPOCHS=200
PRETRAIN_BS=64
PRETRAIN_LR=5e-4
WEIGHT_DECAY=0.01
LR_MIN=0.001
N_VIEWS=4
PROJ_DIM=16
PRETRAIN_RES=128
CKPT_EVERY=25
USE_WANDB=true
```
Estimated time: ~6–8 hours. Fewer steps than A100 config per epoch, but still converges well.

### After xxs, run xs and s for the ablation table
Once xxs is trained, re-run with `VARIANT=xs` then `VARIANT=s` — all other parameters stay the same. Each subsequent run takes longer proportionally to the parameter count.

---

## When to stop — convergence criteria

**The key insight about LeJEPA**: unlike most SSL methods, the training loss is *informative* — the paper demonstrates **94%+ Spearman correlation** between training loss and downstream task performance. This means you can use the loss curve itself to decide when to stop, without needing labeled validation data.

### Concrete stopping rules

**1. Invariance loss flattens**

Plot `inv` over epochs. When the curve looks like this:
```
0.09 → 0.03 → 0.01 → 0.005 → 0.004 → 0.004 → 0.004
```
...and hasn't meaningfully decreased for 20+ epochs — you've converged on invariance. More epochs won't help the encoder learn new invariances.

**2. Total LeJEPA loss flattens**

The total loss should decrease monotonically and smoothly. When the relative improvement per epoch drops below ~0.1% for 20 consecutive epochs, stop.

Practically: if epoch 180 and epoch 200 have the same total loss to 3 decimal places, epoch 201 won't help.

**3. SIGReg loss has stabilized**

SIGReg rises early (as the encoder spreads representations out) then plateaus. A plateau around 1.5–2.0 means the embedding space is well-distributed and in equilibrium with the invariance objective. If it's still rising steeply at epoch 200, train longer.

**4. PCA maps are stable between checkpoints**

Run `visualize` on `epoch_150.pth` and `epoch_200.pth`. If the PCA maps look identical (same colored regions, same depth-aligned structure), training has converged. If they still look different, train more.

### Rule of thumb: 30,000 steps is the safe minimum

Below 10,000 steps: PCA maps will look noisy and unstructured.  
10,000–20,000 steps: coarse structure starts emerging (sky vs. ground, large objects).  
20,000–35,000 steps: fine-grained structure (object boundaries, depth layers).  
Beyond 40,000 steps: diminishing returns unless using a larger variant.

### Should you run indefinitely?

**No.** Two reasons:
1. The loss has a floor — below a certain value, the invariance loss cannot decrease further because the augmentations are genuinely extreme (e.g. heavy solarize makes two views look fundamentally different, even in a perfect embedding space).
2. There is mild overfitting risk when N_SAMPLES is small. With 5,000 images and 200+ epochs, the model may start memorizing specific crops rather than generalizing. Monitor with `inv`: if it decreases to essentially 0 and then wandb shows an uptick, you overfit.

**Safe upper bounds by dataset size:**

| N_SAMPLES | Max useful epochs | Total steps |
|-----------|------------------|-------------|
| 1,000 | 200 | ~25,000 (BS=8) |
| 5,000 | 150 | ~35,000 (BS=16) |
| 47,584 | 200 | ~37,000 (BS=256) |

The table assumes the batch sizes from the recommended configs above. More steps than these upper bounds are not harmful but give marginal returns.
