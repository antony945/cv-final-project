# Plan: LeJEPA-METER Integration — Project 7
*Lean Geometry: Boosting Mobile MDE with LeJEPA*
**Team**: Franzoso & Liberti | Course: Computer Vision 2025–2026

---

## TL;DR
Pre-train the MobileViT backbone (from METER) using the LeJEPA self-supervised objective (SIGReg + invariance loss) on NYU/KITTI RGB images *without depth labels*. Then attach the METER decoder and fine-tune end-to-end with supervised depth loss. Compare against the original ImageNet-supervised METER.

---

## Architecture

### What we KEEP from METER (unchanged)
- `MobileViT` encoder — `conv1` + `MV2Block` stack + `MobileViTBlock` stack
- `decoder` class — U-Net style upsampling with skip connections [y1, y2, y3]
- `balanced_loss_function` — SSIM + Sobel gradient + depth L1 loss

### What we ADD from LeJEPA
- `SIGReg` module (≈20 lines, from `src/lejepa/MINIMAL.md`)
- `nn.AdaptiveAvgPool2d(1)` global pooling head on the backbone
- Projector MLP: `emb_dim → 2048 → 2048 → proj_dim`
- Multi-view augmentation (V=4 random crops per image, no depth needed)
- Pre-training loop: `λ·SIGReg + (1−λ)·inv_loss`

---

## Architecture Diagrams

### Phase 1 — LeJEPA Pre-training (no depth labels)
```
RGB image
  │
  ├─ Aug(·) → view_1 ─┐
  ├─ Aug(·) → view_2 ─┤  stacked: (B, V, 3, H, W)
  ├─ Aug(·) → view_3 ─┤
  └─ Aug(·) → view_4 ─┘
             │
             ▼
  ┌──────────────────────────────────────┐
  │       MobileViT Backbone             │  ← randomly initialized
  │  conv1 → MV2×4 → MViTBlock×3 → conv2│
  │  output: (B*V, C_last, h, w)         │  C_last = 160(xxs) / 192(xs) / 320(s)
  └──────────────────────────────────────┘
             │
             ▼
  AdaptiveAvgPool2d(1) → flatten
  emb: (B*V, C_last)
             │
             ▼
  Projector MLP:  C_last → 2048 → 2048 → proj_dim
  proj: reshaped to (V, B, proj_dim)
             │
       ┌─────┴──────┐
       ▼            ▼
  SIGReg(proj)   inv_loss = (proj.mean(0) − proj)²
       │            │
       └─────┬──────┘
             ▼
  LeJEPA loss = λ·SIGReg + (1−λ)·inv_loss
```

### Phase 2 — Fine-tuning for Depth Estimation (supervised)
```
RGB image (B, 3, H, W)
  │
  ▼
┌────────────────────────────────────────────────────────────┐
│                 MobileViT Backbone                         │
│                (LeJEPA pre-trained weights)                │
│                                                            │
│  conv1 → MV2[0] → MV2[1] ─────────────────────→  y1       │  (B, C2, H/4,  W/4)
│                      │                                     │
│                   MV2[2,3] → MV2[4] ──────────→  y2       │  (B, C4, H/8,  W/8)
│                                  │                         │
│                              MViT[0] → MV2[5] →  y3       │  (B, C6, H/16, W/16)
│                                             │              │
│                                        MViT[1] → MV2[6] → MViT[2] → conv2
└──────────────────────────────────────────────────────────────────────────┘
       x: (B, C_last, H/32, W/32)          skip: y1, y2, y3
       │
       ▼
┌────────────────────────────────────────────────────────────┐
│              METER Decoder  (trained from scratch)         │
│                                                            │
│  conv2d_in ──→ ups_block_1 ←───── y3                      │
│                    │                                       │
│               ups_block_2 ←───── y2                       │
│                    │                                       │
│               ups_block_3 ←───── y1                       │
│                    │                                       │
│               conv2d_out (1 channel)                       │
└────────────────────────────────────────────────────────────┘
       │
       ▼
  Depth Map (B, 1, H, W)
       │
       ▼
  METER loss: λ₁·L1 + λ₂·SSIM + λ₃·Sobel-gradient
```

### Phase 3 — Zero-shot PCA Probing (analysis)
```
Pre-trained backbone (frozen)
  → extract y3 (B, C, H/16, W/16)
  → reshape to (B·H/16·W/16, C)
  → PCA(n_components=3)
  → reshape to (B, H/16, W/16, 3)
  → visualize as RGB

Compare side-by-side:
  RGB input | GT depth map | PCA map
Goal: depth edges ≈ PCA segment boundaries?
```

---

## Implementation Phases

### Phase 0 — Environment & Data Setup
1. `pip install torch torchvision einops pytorch-ssim scikit-learn wandb tqdm`
2. Download NYU Depth V2 and KITTI Eigen split (see Dataset section below)
3. Implement `NYUDataset` and `KITTIDataset` with `return_depth` toggle

### Phase 1 — Supervised Baseline (reference numbers)
4. Copy & fix METER source (remove `globals` import, fix `device='cuda:0'` bug)
5. Load provided METER weights (`src/METER/models/`) for xxs/xs/s variants
6. Evaluate on NYU + KITTI test splits → record RMSE, AbsRel, δ1

### Phase 2 — LeJEPA Pre-training
7. Build `MobileViTLeJEPA` wrapper (backbone + GAP + projector)
8. Train with multi-view SSL dataloader on NYU RGB (no depth), V=4, λ=0.02
9. Save backbone weights: `checkpoints/lejepa_{variant}_{dataset}.pth`

### Phase 3 — Zero-shot PCA Probing
10. Load saved backbone, extract y3 feature maps on test images
11. Apply PCA(n=3), visualize vs RGB+GT depth — assess geometric structure

### Phase 4 — Fine-tuning LeJEPA-METER
12. Load LeJEPA backbone, attach fresh METER decoder
13. Fine-tune with differential LR: backbone 1e-4, decoder 1e-3
14. Also train random-init baseline (same setup, no pre-training) for ablation
15. Save: `checkpoints/meter_{variant}_{dataset}_{pretrain}.pth`

### Phase 5 — Evaluation & Comparison
16. Evaluate all models: RMSE, AbsRel, δ1 on NYU + KITTI test splits
17. Plot convergence curves: LeJEPA-METER vs. random-init
18. Qualitative side-by-side depth predictions

---

## Training Matrix

| Variant | Pre-training | NYU | KITTI | Priority |
|---------|-------------|-----|-------|----------|
| **xxs** | Supervised (provided) | eval only | eval only | Day 1–2 |
| **xxs** | Random init (ablation) | ✓ | ✓ | Day 3–6 |
| **xxs** | **LeJEPA** | ✓ | ✓ | Day 3–6 |
| **xs** | Supervised (provided) | eval only | eval only | Day 7–10 |
| **xs** | Random init | ✓ | ✓ | Day 7–10 |
| **xs** | **LeJEPA** | ✓ | ✓ | Day 7–10 |
| **s** | Supervised (provided) | eval only | eval only | Day 11–14 |
| **s** | Random init | ✓ | ✓ | Day 11–14 |
| **s** | **LeJEPA** | ✓ | ✓ | Day 11–14 |

**Parallelisation strategy**: run LeJEPA pre-training on Google Colab (longer jobs)
while running fine-tuning + evaluation on RTX 5060 16 GB locally.

---

## Datasets

### NYU Depth V2 (~2.8 GB)
```bash
mkdir -p dataset/nyu
wget https://s3-eu-west-1.amazonaws.com/densedepth/nyu_data.zip -P dataset/nyu/
# unzip in place — the loader reads directly from the zip
```
Format: DenseDepth preprocessed. Zip contains `data/nyu2_train.csv` and
`data/nyu2_test.csv` with paths to paired RGB (JPEG) + depth (16-bit PNG, ÷1000 = metres).
50 449 train pairs + 654 test pairs.

### KITTI Eigen Split (~10 GB)
```bash
mkdir -p dataset/kitti
wget https://s3-eu-west-1.amazonaws.com/densedepth/KITTI.zip -P dataset/kitti/
# unzip in place
```
Format: DenseDepth preprocessed Eigen split. 23 158 train + 697 test.
Depth stored as 16-bit PNG, ÷256 = metres (KITTI velodyne projection convention).

---

## Time Budget

| Run type | xxs | xs | s |
|----------|-----|----|---|
| LeJEPA pre-train (200 ep) | ~2 h | ~4 h | ~8 h |
| Fine-tune (100 ep) | ~2 h | ~4 h | ~8 h |

Total (all variants, both datasets, LeJEPA + ablation) ≈ 90 GPU-hours.
With two GPUs in parallel: ≈ 45 h ≈ 3 days of wall time.

---

## Bugs Fixed vs Original METER Source

| Bug | Original | Fixed |
|-----|---------|-------|
| Missing module | `from globals import RGB_img_res` | replaced with `IMG_RES` dict in Globals |
| Hardcoded device | `SeparableConv2d(..., device='cuda:0')` | `device` param removed; use `.to(DEVICE)` at model level |
| Syntax error | `mobilevit_xs()` has mismatched parenthesis | fixed |

---

## Key References

- Balestriero, R. & LeCun, Y. (2025). *LeJEPA: Provable and Scalable SSL Without the Heuristics.* [arXiv:2511.08544](https://arxiv.org/abs/2511.08544)
- Papa, L., Russo, P. & Amerini, I. (2024). *METER: A Mobile Vision Transformer for MDE.*
- Mur-Labadia, L. et al. (2026). *V-JEPA 2.1: Unlocking Dense Features in Video SSL.*
