# METER Architecture — Detailed Overview

> **Reference**: Papa, L., Russo, P., & Amerini, I. (2024). *METER: A Mobile Vision Transformer Architecture for Monocular Depth Estimation.*

## High-Level Design

METER is a lightweight encoder-decoder model for monocular depth estimation (MDE), designed to infer on embedded/edge devices. It achieves real-time performance through:

1. A **hybrid CNN-Transformer encoder** (modified MobileViT)
2. A **fully convolutional decoder** with skip connections

The model produces a dense depth map (per-pixel distance) from a single RGB image.

---

## Encoder: Modified MobileViT

The encoder combines the efficiency of **MobileNetV2 inverted residual blocks** (CNN) with the global reasoning of **Vision Transformers** (attention), in a carefully optimized hybrid:

### Architecture Flow

```
Input (3, H, W)
│
├─ conv1: 3×3 SeparableConv, stride=2, ReLU     → (C₀, H/2, W/2)    ← skip y₀
│
├─ MV2Block (stride=1)                           → (C₁, H/2, W/2)
├─ MV2Block (stride=2)                           → (C₂, H/4, W/4)    ← skip y₁
├─ MV2Block × 2 (stride=1)                       → (C₃, H/4, W/4)
│
├─ MV2Block (stride=2)                           → (C₄, H/8, W/8)    ← skip y₂
├─ MobileViTBlock (1 transformer, depth=1)        → (C₅, H/8, W/8)
│
├─ MV2Block (stride=2)                           → (C₆, H/16, W/16)  ← skip y₃
├─ MobileViTBlock (1 transformer, depth=1)        → (C₇, H/16, W/16)
│
├─ MV2Block (stride=2)                           → (C₈, H/32, W/32)
├─ MobileViTBlock (1 transformer, depth=1)        → (C₉, H/32, W/32)
├─ conv2: 1×1 Conv, ReLU                         → (C_final, H/32, W/32)
│
└─ Output: (feat, [y₀, y₁, y₂, y₃])
```

### Key METER Modifications vs. Original MobileViT

| Aspect | Original MobileViT | METER |
|--------|-------------------|-------|
| Transformer depth per block | [2, 4, 3] (cascaded) | **[1, 1, 1]** (single) |
| Activation function | SiLU (Swish) | **ReLU** |
| Final output channels | ~320/384/640 | **160/192/320** (halved) |
| Block design | 4 convs + N transformers | **3 convs + 1 transformer** |

**Rationale** (from the paper):
- Single transformer reduces latency dramatically while retaining attention's global reasoning
- ReLU works better than SiLU for depth-data distribution
- Halved channels reduce memory footprint for edge deployment

### MobileNetV2 Block (MV2Block)

Standard inverted residual with depthwise-separable convolution:
```
Input → 1×1 expand → 3×3 depthwise → 1×1 project → (+residual if same shape)
```
Expansion factor: 2 (xxs), 4 (xs, s).

### MobileViT Block (METER Block)

The core innovation — combines local convolution features with global transformer attention:

```
Input x
│
├─ 3×3 SeparableConv + BN + ReLU     (local features)
├─ 1×1 Conv + BN + ReLU              (channel projection to transformer dim)
│
├─ Unfold: (B, D, H, W) → (B, P², HW/P², D)   (spatial patches → sequence)
├─ Transformer: MultiHeadAttention + FFN         (global attention, 1 layer)
├─ Fold:   (B, P², HW/P², D) → (B, D, H, W)   (sequence → spatial)
│
├─ 1×1 Conv + BN + ReLU              (project back to channel dim)
├─ Concat with original input x       (skip connection within block)
├─ 3×3 SeparableConv + BN + ReLU     (fuse local + global)
│
└─ Output (same spatial dims as input)
```

The unfold/fold with patch_size=(2,2) means each 2×2 spatial patch becomes one token in the transformer sequence. This is computationally lighter than treating every pixel as a token.

### Transformer Block (inside MobileViTBlock)

```
Input → LayerNorm → MultiHeadAttention (4 heads, dim_head=8) → +residual
      → LayerNorm → FFN (Linear → ReLU → Linear)             → +residual
```

No positional embeddings — the model is fully resolution-agnostic.

### Three Variants

| Variant | Channels Config | Transformer Dims | Final Emb | Params | Use Case |
|---------|----------------|------------------|-----------|--------|----------|
| **XXS** | [16,16,24,24,48,48,64,64,80,80,160] | [64, 80, 96] | 160 | 0.71M | Extreme edge (IoT) |
| **XS** | [16,32,48,48,64,64,80,80,96,96,192] | [96, 120, 144] | 192 | 1.45M | Mobile phones |
| **S** | [16,32,64,64,96,96,128,128,160,160,320] | [144, 192, 240] | 320 | 3.29M | Embedded GPU |

---

## Decoder (for fine-tuning — not used in LeJEPA pre-training)

The decoder recovers spatial resolution using transposed convolutions and skip connections:

```
Encoder output (C_final, H/32, W/32)
│
├─ 1×1 Conv (reduce channels)
│
├─ UpSampleBlock 1: ConvTranspose2d ×2 + concat skip y₃ + SepConv  → (H/16)
├─ UpSampleBlock 2: ConvTranspose2d ×2 + concat skip y₂ + SepConv  → (H/8)
├─ UpSampleBlock 3: ConvTranspose2d ×2 + concat skip y₁ + SepConv  → (H/4)
│
├─ 3×3 Conv → 1 channel                                             → (H/4)
├─ Bilinear interpolate to input resolution                          → (H, W)
├─ ReLU (depth must be ≥ 0)
│
└─ Output: depth map (1, H, W)
```

---

## Our Implementation

### What we implement now (Phase 1: LeJEPA pre-training)

```
┌──────────────────────────────────────────────────────┐
│  METERLeJEPA Wrapper                                 │
│                                                      │
│  Input: (B, V, 3, 128, 128) — V augmented views     │
│                                                      │
│  ┌────────────────────┐                              │
│  │ METER Encoder      │  ← exact METER encoder      │
│  │ (any resolution)   │                              │
│  └────────┬───────────┘                              │
│           │ feat: (B*V, C_final, 4, 4)               │
│           ▼                                          │
│  AdaptiveAvgPool2d(1)                                │
│           │ emb: (B*V, emb_dim)                      │
│           ▼                                          │
│  Projector MLP:                                      │
│  emb_dim → 2048 → BN → ReLU                         │
│         → 2048 → BN → ReLU                          │
│         → proj_dim (16)                              │
│           │ proj: (V, B, proj_dim)                   │
│           ▼                                          │
│  LeJEPA Loss:                                        │
│  invariance + λ × SIGReg                             │
│                                                      │
│  Only encoder weights saved for downstream           │
└──────────────────────────────────────────────────────┘
```

### What comes later (Phase 2: supervised fine-tuning)

The saved encoder weights are loaded into the full METER model (encoder + decoder), and fine-tuned end-to-end for depth estimation on NYU/KITTI with the BalancedDepthLoss (L1 + gradient + normals + SSIM).

### Resolution flexibility

The encoder is **fully convolutional** with no positional embeddings:
- Pre-training: 128×128 (fast iteration, standard SSL practice)
- PCA visualization: 256×192 or 480×640 (any resolution divisible by 32)
- Fine-tuning: 480×640 (NYU) or 192×640 (KITTI)

No architecture change is needed between these — the same weights transfer directly.

---

## References

- Papa, L., Russo, P., & Amerini, I. (2024). METER: A Mobile Vision Transformer Architecture for Monocular Depth Estimation.
- Mehta, S., & Rastegari, M. (2022). MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer. ICLR 2022.
- Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR 2018.
- Balestriero, R., & LeCun, Y. (2025). LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics.
