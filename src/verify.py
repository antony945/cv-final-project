"""Verification script — sanity check that everything works before training.

Shows:
1. Sample images from the dataset (with augmentation views)
2. Model forward pass test (correct shapes)
3. One training step (loss computes and backprop works)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import DEVICE, EMB_DIM
from src.data import get_pretrain_loader
from src.model import METERLeJEPA
from src.utils import SIGReg, compute_lejepa_loss, denorm_imagenet
from omegaconf import OmegaConf


def run_verify(variant: str = "xxs", dataset: str = "nyu",
               resolution: int = 128, n_views: int = 4,
               proj_dim: int = 16, lamb: float = 0.02):
    """Run full verification pipeline."""
    print("\n" + "=" * 55)
    print("  VERIFICATION: Dataset + Model + Loss")
    print("=" * 55)

    # ── 1. Dataset check ──────────────────────────────────────────────
    print(f"\n[1/3] Loading dataset '{dataset}' and showing samples...")
    verify_cfg = OmegaConf.create({
        "data": {"datasets": {dataset: {"n_samples": 5000}}, "use_mmap": True},
        "n_views": n_views,
        "resolution": resolution,
        "bs": 4,
    })
    loader = get_pretrain_loader(verify_cfg)

    batch_views, _ = next(iter(loader))
    B, V, C, H, W = batch_views.shape
    print(f"  Batch shape: ({B}, {V}, {C}, {H}, {W})")
    print(f"  → {B} images × {V} views, {H}×{W} resolution")

    # Get views directly from the dataset so originals and views match
    ds = loader.dataset
    n_show = min(4, len(ds))
    # Pick random well-spread indices for visual variety
    rng = np.random.default_rng(42)
    show_indices = sorted(rng.choice(len(ds), size=n_show, replace=False))

    # Show original + all views for first image
    idx = show_indices[0]
    views_0, _ = ds[idx]  # (V, 3, H, W)
    original_pil = ds._get_image(idx)
    original_resized = original_pil.resize((W, H))

    fig_single, axes_single = plt.subplots(1, V + 1, figsize=(3 * (V + 1), 3))
    axes_single[0].imshow(original_resized)
    axes_single[0].set_title("Original")
    axes_single[0].axis("off")
    for v in range(V):
        axes_single[v + 1].imshow(denorm_imagenet(views_0[v]))
        axes_single[v + 1].set_title(f"View {v}")
        axes_single[v + 1].axis("off")
    plt.suptitle(f"Original + {V} augmented views ({dataset.upper()})",
                 fontsize=13)
    plt.tight_layout()
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    single_path = out_dir / f"verify_single_image_views_{dataset}.png"
    plt.savefig(single_path, dpi=100, bbox_inches="tight")
    print(f"  ✓ Saved single-image views → {single_path}")
    plt.show()

    # Show n_show images: original + V views per row
    fig, axes = plt.subplots(n_show, V + 1, figsize=(3 * (V + 1), 3 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(show_indices):
        orig_pil = ds._get_image(idx)
        views_i, _ = ds[idx]  # (V, 3, H, W)
        axes[row, 0].imshow(orig_pil.resize((W, H)))
        axes[row, 0].set_title("Original" if row == 0 else "")
        axes[row, 0].axis("off")
        for v in range(V):
            axes[row, v + 1].imshow(denorm_imagenet(views_i[v]))
            axes[row, v + 1].set_title(f"View {v}" if row == 0 else "")
            axes[row, v + 1].axis("off")

    plt.suptitle(f"Dataset Augmented Views — {dataset.upper()} (each row = same image)",
                 fontsize=13)
    plt.tight_layout()
    out_path = out_dir / f"verify_dataset_{dataset}.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    print(f"  ✓ Saved dataset preview → {out_path}")
    plt.show()

    # ── 2. Model forward pass ─────────────────────────────────────────
    print(f"\n[2/3] Testing model forward pass (METER-{variant.upper()})...")
    net = METERLeJEPA(variant=variant, proj_dim=proj_dim,
                          resolution=resolution).to(DEVICE)
    backbone_params = sum(p.numel() for p in net.backbone.parameters())
    total_params = sum(p.numel() for p in net.parameters())
    print(f"  Backbone (encoder) parameters: {backbone_params:,} ({backbone_params/1e6:.2f}M)")
    print(f"  Total (encoder + projector):   {total_params:,} ({total_params/1e6:.2f}M)")

    # Use a batch from the loader for the forward pass test
    batch_views, _ = next(iter(loader))
    with torch.no_grad():
        test_input = batch_views[:2].to(DEVICE)
        emb, proj = net(test_input)
        print(f"  Input:  {tuple(test_input.shape)}")
        print(f"  Emb:    {tuple(emb.shape)} — expected ({2*V}, {EMB_DIM[variant]})")
        print(f"  Proj:   {tuple(proj.shape)} — expected ({V}, 2, {proj_dim})")
        assert emb.shape == (2 * V, EMB_DIM[variant]), "Embedding shape mismatch!"
        assert proj.shape == (V, 2, proj_dim), "Projection shape mismatch!"
    print("  ✓ Forward pass correct!")

    # ── 3. Training step test ─────────────────────────────────────────
    print("\n[3/3] Testing one training step (loss + backward)...")
    sigreg = SIGReg().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3)

    net.train()
    views = batch_views.to(DEVICE)
    _, proj = net(views)
    loss, components = compute_lejepa_loss(proj, sigreg, lamb)

    opt.zero_grad()
    loss.backward()
    opt.step()

    print(f"  LeJEPA loss : {components['lejepa']:.4f}")
    print(f"  SIGReg loss : {components['sigreg']:.4f}")
    print(f"  Invariance  : {components['inv']:.4f}")
    print("  ✓ Backward pass + optimizer step successful!")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'─' * 55}")
    print("  ALL CHECKS PASSED ✓")
    print(f"{'─' * 55}")
    print(f"\n  Ready to train! Run:")
    print(f"    uv run python -m src.main +experiment=pretrain_test")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LeJEPA verification")
    parser.add_argument("--variant", default="xxs", choices=["xxs", "xs", "s"])
    parser.add_argument("--dataset", default="nyu", choices=["nyu", "kitti"])
    args = parser.parse_args()
    run_verify(variant=args.variant, dataset=args.dataset)
