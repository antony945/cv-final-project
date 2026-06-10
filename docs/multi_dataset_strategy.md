# Multi-Dataset Training Strategy: NYU + KITTI

> Analysis of how the METER paper uses two datasets and how our architecture can support it.

---

## 1. What the Paper Does

The METER paper trains and evaluates **separately** on NYU Depth v2 and KITTI — they are **not mixed**. This is the standard practice in monocular depth estimation literature:

| Aspect | NYU Depth v2 | KITTI |
|--------|-------------|-------|
| Domain | Indoor | Outdoor |
| Input resolution | 256×192 (H×W) | 636×192 (H×W) |
| Depth range | 0–10 m | 0–80 m |
| Training samples | 50K subset | ~23K (Eigen split) |
| Test samples | 654 | 697 (Eigen test) |
| Depth map type | Dense (Kinect) | Sparse (LiDAR projected) |
| Evaluation crop | Full image | Garg/Eigen crop (bottom center) |

The paper reports results as **two separate tables** — one for NYU, one for KITTI — with the same architecture but potentially different trained weights.

---

## 2. How They Likely Train

Based on standard MDE practice and the paper's reporting style:

### Approach: **Separate training per dataset → Two independent models**

```
LeJEPA Pre-training (unlabeled RGB from both/either)
    │
    ├─→ Fine-tune on NYU (256×192) → NYU MODEL (weights A) → Evaluate on NYU test
    │
    └─→ Fine-tune on KITTI (636×192) → KITTI MODEL (weights B) → Evaluate on KITTI test
```

> **Key clarification:** The end result is **two separate trained models**, not one unified model. They share the same architecture (same code, same layer structure) but have **different learned weights** after fine-tuning on different datasets. This is analogous to training the same ResNet architecture on ImageNet vs CIFAR — same code, different `.pth` files.

### Why two models, not one?

The paper never claims a single model that handles both datasets. The reasons two models are necessary:

1. **Depth ranges are incompatible** — NYU maxes at 10m, KITTI at 80m. A ReLU output head can't produce both scales from the same weights without explicit conditioning.
2. **Domains are fundamentally different** — indoor furniture vs outdoor roads. The decoder needs to learn very different depth priors.
3. **Every MDE paper does this** — Eigen et al., BTS, AdaBins, PixelFormer, DPT, MiDaS (the only exception is MiDaS which uses relative depth + affine-invariant loss, a different paradigm).
4. **The paper reports separate tables** with no shared evaluation, confirming independent training.

### Does fine-tuning order matter?

**No — because the two fine-tuning runs are completely independent.** They both start from the same pre-trained encoder checkpoint and diverge from there. There is no sequential dependency:

```
                    Pre-trained encoder (shared starting point)
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
     Fine-tune on NYU                Fine-tune on KITTI
     (independent run)               (independent run)
              │                               │
              ▼                               ▼
      NYU model weights              KITTI model weights
      (meter_xxs_nyu.pth)           (meter_xxs_kitti.pth)
```

These are two separate `uv run python -m src.main` invocations that happen to use the same `pretrained_encoder` checkpoint. You can run them in any order, in parallel, or months apart — the result is identical.

### Pre-training may be shared

LeJEPA pre-training is self-supervised (no depth labels) — it can use RGB images from **both** datasets combined, or from ImageNet, or from either one alone. The encoder learns general visual features that transfer to both domains.

---

## 3. Can Our Architecture Handle Different Resolutions?

**Yes — fully.** Here's why:

### Encoder: Resolution-agnostic

- Uses only **convolutions** and **MobileViT blocks** — no fixed positional embeddings.
- The MobileViT patch-based attention operates on `(ph×pw)` local patches with relative positions encoded implicitly via the folding pattern. No learned absolute position.
- The `image_size` constructor parameter is only used for an **assertion** (`H % patch_size == 0 and W % patch_size == 0`) — the forward pass works with any resolution divisible by 32.

### Decoder: Adapts via skip connections + bilinear

- UpSample blocks use `ConvTranspose2d` (stride=2) — works at any spatial size.
- Skip connections have explicit **padding logic** to handle odd-size mismatches.
- Final output uses `nn.functional.interpolate(..., size=target_size)` — the `target_size` is passed at forward time, not baked into weights.

### What this means concretely:

```python
# Same weights, different resolutions:
model = METERModel(variant="xxs", resolution=(192, 256))   # NYU
model = METERModel(variant="xxs", resolution=(192, 636))   # KITTI

# The encoder weights are 100% shared
# The decoder weights are 100% shared
# Only the resolution parameter changes the output interpolation target
```

**Constraint:** Input H and W must each be divisible by 32 (due to 5 stride-2 downsampling stages). Both 192×256 and 192×636 fail this for width:
- 256 / 32 = 8 ✓
- 636 / 32 = 19.875 ✗

**Fix for KITTI:** Use 192×640 (640/32 = 20 ✓) — a negligible 4-pixel crop/pad.

---

## 4. Three Possible Multi-Dataset Strategies

### Strategy A: Separate Training (Paper's approach) ⭐ Recommended

```
Pre-train encoder (shared) → Fine-tune NYU model → Evaluate NYU
                           → Fine-tune KITTI model → Evaluate KITTI
```

| Pros | Cons |
|------|------|
| Exactly matches paper's methodology | Requires training twice |
| Each model specializes for its domain | Two sets of weights to maintain |
| Simple implementation | — |
| Easy to compare with published results | — |
| Depth range/bias tuned per dataset | — |

**Implementation effort:** Medium — need KITTI dataset class + KITTI-specific eval (Eigen crop, 80m cap).

### Strategy B: Joint/Mixed Training

Train on a combined dataset, sampling from both NYU and KITTI in each batch.

```
Pre-train encoder (shared)
    │
    └─→ Fine-tune on NYU+KITTI (mixed batches) → Evaluate both
```

| Pros | Cons |
|------|------|
| Single model handles both domains | Depth ranges conflict (10m vs 80m) |
| More training data → potentially better generalization | Resolution mismatch in same batch (need padding or separate batch sizes) |
| Simpler deployment (one model) | Hard to reproduce paper results |
| | Domain imbalance (50K NYU vs 23K KITTI) |
| | Different augmentation policies needed |

**Implementation complexity:** High — requires:
- Depth normalization (log-depth or per-dataset scaling)
- Resolution handling (pad KITTI images or resize, or alternate batch resolution)
- Domain-balanced sampling
- Separate evaluation pipelines
- Not comparable to published results

### Strategy C: Sequential Fine-tuning (Transfer between domains)

Train on one dataset first, then fine-tune on the other.

```
Pre-train → Fine-tune NYU (60 epochs) → Fine-tune KITTI (60 epochs)
```

| Pros | Cons |
|------|------|
| Transfer learning between domains | Catastrophic forgetting of first domain |
| Can use NYU features for KITTI | Final model only works well on last domain |
| Interesting research question | Order matters (NYU→KITTI ≠ KITTI→NYU) |
| | Not standard practice |

**When useful:** If KITTI is your target but you have limited KITTI data — NYU pre-fine-tuning teaches depth estimation basics that transfer.

---

## 5. Recommendation for Our Project

### Primary goal: **Reproduce paper results → Strategy A**

The paper reports separate results. To validate our implementation:

1. **Phase 1:** Get NYU working well (already done ✓)
2. **Phase 2:** Implement KITTI data pipeline + evaluation
3. **Phase 3:** Train a separate KITTI model using same encoder architecture

### Shared components between NYU and KITTI:

| Component | Shared? | Notes |
|-----------|---------|-------|
| Encoder architecture | ✓ Same | Identical weights structure |
| Encoder pre-training | ✓ Can share | LeJEPA on combined RGB is fine |
| Decoder architecture | ✓ Same | Weights shared, output interpolation differs |
| Decoder bias init | ✗ Different | 3.0 for NYU, ~15.0 for KITTI |
| Loss function (BLF) | ✓ Same | Same 4-term loss works for both |
| Evaluation metrics | ~Partial | Same formulas, different protocols |
| Evaluation crop | ✗ Different | Full image (NYU) vs Eigen crop (KITTI) |
| Depth cap | ✗ Different | 10m (NYU) vs 80m (KITTI) |
| Resolution | ✗ Different | 192×256 (NYU) vs 192×640 (KITTI) |

---

## 6. What Needs to Change in Our Code

### Already supports multi-resolution:
- `METERModel(resolution=...)` takes target output size at construction
- Encoder is fully convolutional (no positional embeddings)
- Decoder has skip-connection padding for odd sizes
- Config already has `data.dataset: kitti` routing

### Needs implementation:

| Component | Current state | What to do |
|-----------|---------------|------------|
| `KITTIDepthDataset` | Stub (raises NotImplementedError) | Implement: load RGB + sparse depth from Eigen split |
| `MmapKITTIDepthDataset` | Stub | Implement mmap loader (same pattern as NYU) |
| KITTI preprocessing | None | `preprocess.py` for KITTI (fill sparse depth, resize to 192×640) |
| KITTI evaluation | None | Eigen crop, cap at 80m, same metrics |
| Decoder bias init | Hardcoded 3.0 | Make configurable per dataset |
| KITTI experiment configs | None | `finetune_production_kitti.yaml` |
| KITTI augmentation | None | May differ (no vertical flip for driving scenes) |

---

## 7. Key Technical Decisions

### 7.1 Sparse Depth Handling (KITTI)

KITTI depth maps are **sparse** (~5% valid pixels from LiDAR projection). Options:

| Option | Description | Paper's approach |
|--------|-------------|-----------------|
| Train on sparse GT directly | Mask loss to valid pixels only | ✓ Standard |
| Densify first (interpolation) | Fill holes before training | Not recommended |
| Semi-dense (accumulate frames) | Use multiple scans | Some papers do this |

**Recommendation:** Mask all loss terms to valid pixels. Our `BalancedDepthLoss` already masks `depth > 0` — this naturally handles sparse KITTI maps if we set invalid pixels to 0.

### 7.2 Eigen Split

The standard KITTI depth benchmark uses the Eigen et al. split:
- **Train:** ~23,488 images from 28 scenes
- **Val:** ~888 images from 4 scenes  
- **Test:** 697 images from 29 scenes

We already have `datasets/kitti/eigen_test_files_only_rgb/` — need to verify format and add the train split.

### 7.3 Evaluation Protocol (KITTI)

KITTI evaluation differs from NYU:
- **Depth cap:** Clamp predictions and GT to 80m
- **Eigen crop:** Evaluate only in the region `[0.40810811*H : 0.99189189*H, 0.03594771*W : 0.96405229*W]`
- **Min depth:** Ignore pixels with GT depth < 1e-3 (or < 1m in some protocols)
- **Same metrics:** δ1, δ2, δ3, RMSE, REL, log10 (same formulas as NYU)

### 7.4 Resolution Choice

The paper uses 636×192 but this isn't divisible by 32. Practical options:
- **640×192** (pad 4 pixels) — cleanly divisible, minimal distortion
- **608×192** (crop 28 pixels) — also divisible by 32
- **640×192** is more common in literature (used by Godard et al., BTS, etc.)

---

## 8. Proposed Training Pipeline (Strategy A)

```
┌─────────────────────────────────────────────────────────────────┐
│  SHARED: LeJEPA Pre-training                                     │
│  - Dataset: NYU RGB (50K) + KITTI RGB (23K) = 73K unlabeled     │
│  - Resolution: 128×128 (square crops for SSL)                    │
│  - Output: encoder weights                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌──────────────────────────┐   ┌──────────────────────────────┐
│  NYU Fine-tuning          │   │  KITTI Fine-tuning            │
│  - Resolution: 192×256    │   │  - Resolution: 192×640        │
│  - Depth range: 0–10m     │   │  - Depth range: 0–80m         │
│  - Decoder bias: 3.0      │   │  - Decoder bias: 15.0         │
│  - Dense GT               │   │  - Sparse GT (masked loss)    │
│  - 60 epochs, BS=128      │   │  - 60 epochs, BS=128          │
│  - Full-image eval        │   │  - Eigen crop eval            │
└──────────────────────────┘   └──────────────────────────────────┘
              │                               │
              ▼                               ▼
┌──────────────────────────┐   ┌──────────────────────────────┐
│  NYU Evaluation           │   │  KITTI Evaluation             │
│  - 654 test images        │   │  - 697 test images            │
│  - Cap: 10m               │   │  - Cap: 80m                   │
│  - Metrics: δ1, RMSE, REL │   │  - Eigen crop                 │
└──────────────────────────┘   └──────────────────────────────────┘
```

---

## 9. Summary

| Question | Answer |
|----------|--------|
| How does the paper use two datasets? | **Separate training and evaluation** — not mixed |
| Can our architecture handle different resolutions? | **Yes** — fully convolutional, no positional embeddings |
| What's the best strategy? | **Strategy A (separate)** — matches paper, simpler, comparable results |
| What's the biggest implementation gap? | KITTI dataset loading + sparse depth masking + Eigen eval protocol |
| Can pre-training be shared? | **Yes** — LeJEPA is label-free, can use both RGB sources |
| Do we need different model weights? | **Yes** — one NYU model, one KITTI model (same architecture) |

---

## 10. Next Steps (Implementation Order)

1. **KITTI data pipeline** — Implement `KITTIDepthDataset` using Eigen split files
2. **KITTI preprocessing** — Sparse depth loading from velodyne/depth_annotated
3. **KITTI evaluation** — Add Eigen crop + 80m cap to evaluation module
4. **Config** — Create `finetune_production_kitti.yaml` with KITTI-specific params
5. **Decoder bias** — Make `conv_out.bias` init configurable (3.0 vs 15.0)
6. **Train + evaluate** — Run KITTI fine-tuning and report metrics
7. **(Optional)** Mixed pre-training — Add KITTI RGB to LeJEPA dataset pool

---

## 11. Decoder Bias Initialization — Design Decision

### The Problem

Currently in `model.py` (`METERDecoder.__init__`):

```python
# Hardcoded NYU-specific value
nn.init.constant_(self.conv_out.bias, 3.0)
```

This initializes the output bias to ~3m (NYU mean depth), which helps the model start predicting reasonable depth values instead of zeros. But:
- For KITTI, mean depth is ~15m → bias should be ~15.0
- For a future custom dataset, the mean could be anything
- Hardcoding ties the architecture to a single dataset

### Why bias init matters

The decoder output goes through `ReLU` (depth ≥ 0). Without a positive bias:
- Initial predictions are all ~0 (random weights → near-zero output → ReLU clips)
- Gradients are tiny because predictions are far from GT → slow convergence
- The model can get "stuck" predicting constant depth (collapse)

With correct bias:
- Initial predictions start near the dataset mean → reasonable initial loss
- Gradients flow well from epoch 1 → faster convergence
- Acts as a "warm start" for the depth head

### Proposed Options (DECISION NEEDED)

#### Option A: Pass `depth_bias` to the decoder constructor

```python
class METERDecoder(nn.Module):
    def __init__(self, variant: str = "xxs", depth_bias: float = 3.0):
        ...
        nn.init.constant_(self.conv_out.bias, depth_bias)

class METERModel(nn.Module):
    def __init__(self, variant="xxs", resolution=(256, 192), depth_bias=3.0):
        ...
        self.decoder = METERDecoder(variant, depth_bias=depth_bias)
```

Config would have: `finetune.depth_bias: 3.0` (NYU) or `finetune.depth_bias: 15.0` (KITTI)

**Pros:** Simple, explicit, easy to understand.
**Cons:** You need to manually set it per dataset.

#### Option B: Dataset-aware lookup table in config

```yaml
# In config.yaml or per-experiment config:
dataset_defaults:
  nyu:
    depth_bias: 3.0
    max_depth: 10.0
  kitti:
    depth_bias: 15.0
    max_depth: 80.0
```

The training loop reads `depth_bias` from the dataset defaults automatically.

**Pros:** Set once, always correct for known datasets.
**Cons:** Slightly more abstraction; custom datasets still need manual entry.

#### Option C: Compute from data at runtime

Before training, scan the first N samples and compute `mean(depth[depth > 0])`, use that as bias.

```python
# In finetune_depth() before model creation:
depth_bias = compute_mean_depth(train_loader, n_samples=100)
model = METERModel(variant=variant, resolution=resolution, depth_bias=depth_bias)
```

**Pros:** Fully automatic, works for any dataset without config changes.
**Cons:** Adds a pre-scan step (~5 sec); result varies slightly per run if dataset is shuffled; harder to reproduce exact initialization.

#### Option D: Keep hardcoded per-dataset, no config (current approach extended)

Have two decoder variants or an if/else:

```python
DEPTH_BIAS = {"nyu": 3.0, "kitti": 15.0}
nn.init.constant_(self.conv_out.bias, DEPTH_BIAS[dataset])
```

**Pros:** Minimal change, no config overhead.
**Cons:** New datasets require code change; less flexible.

### Recommendation

**Option A** is the cleanest balance of simplicity and flexibility. The bias is a single float in the config, easy to understand, and works for any future dataset. No magic, no runtime computation, no hidden lookups.

**✅ IMPLEMENTED** — `depth_bias` is now a config parameter:
```yaml
# finetune_production.yaml (NYU)
finetune:
  depth_bias: 3.0

# finetune_production_kitti.yaml (future)
finetune:
  depth_bias: 15.0
```

Code: `METERDecoder(variant, depth_bias=...)` → `METERModel(variant, resolution, depth_bias=...)` → read from `cfg.finetune.depth_bias` (defaults to 3.0).
