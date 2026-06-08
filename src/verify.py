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
from src.model import MobileViTLeJEPA
from src.loss import SIGReg, compute_lejepa_loss


def _denorm_view(t: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization for display."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (t.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def run_verify(variant: str = "xxs", resolution: int = 128,
               n_views: int = 4, proj_dim: int = 16, lamb: float = 0.02):
    """Run full verification pipeline."""
    print("\n" + "=" * 55)
    print("  VERIFICATION: Dataset + Model + Loss")
    print("=" * 55)

    # ── 1. Dataset check ──────────────────────────────────────────────
    print("\n[1/3] Loading dataset and showing samples...")
    loader = get_pretrain_loader(batch_size=4, n_samples=50,
                                 n_views=n_views, resolution=resolution)

    batch_views, _ = next(iter(loader))
    B, V, C, H, W = batch_views.shape
    print(f"  Batch shape: ({B}, {V}, {C}, {H}, {W})")
    print(f"  → {B} images × {V} views, {H}×{W} resolution")

    # Show 4 images with their V views
    n_show = min(4, B)
    fig, axes = plt.subplots(n_show, V + 1, figsize=(3 * (V + 1), 3 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_show):
        # First column: label
        axes[i, 0].text(0.5, 0.5, f"Image {i}", ha="center", va="center",
                        fontsize=12, transform=axes[i, 0].transAxes)
        axes[i, 0].axis("off")
        # Remaining columns: augmented views
        for v in range(V):
            axes[i, v + 1].imshow(_denorm_view(batch_views[i, v]))
            axes[i, v + 1].set_title(f"View {v}" if i == 0 else "")
            axes[i, v + 1].axis("off")

    # Adjust layout: remove first column empty space
    for i in range(n_show):
        axes[i, 0].set_visible(False)

    plt.suptitle("Dataset Augmented Views (each row = same image)", fontsize=13)
    plt.tight_layout()
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "verify_dataset.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    print(f"  ✓ Saved dataset preview → {out_path}")
    plt.show()

    # ── 2. Model forward pass ─────────────────────────────────────────
    print(f"\n[2/3] Testing model forward pass (MobileViT-{variant.upper()})...")
    net = MobileViTLeJEPA(variant=variant, proj_dim=proj_dim,
                          resolution=resolution).to(DEVICE)
    params = sum(p.numel() for p in net.backbone.parameters())
    print(f"  Backbone parameters: {params:,} ({params/1e6:.2f}M)")

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
    print(f"    uv run python -m src.main +experiment=test")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LeJEPA verification")
    parser.add_argument("--variant", default="xxs", choices=["xxs", "xs", "s"])
    args = parser.parse_args()
    run_verify(variant=args.variant)
