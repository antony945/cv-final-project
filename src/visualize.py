"""PCA visualization of encoder features — zero-shot geometric probing.

After LeJEPA pre-training, we extract spatial feature maps from the encoder's
skip connections and project them to 3 principal components for visualization.
If the pre-training learned useful geometry, PCA maps should show spatial
coherence (e.g., separating foreground objects from background, aligning with
depth discontinuities).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision.transforms import v2
from pathlib import Path

from src.config import DEVICE, EMB_DIM
from src.model import MobileViT, _BACKBONE_FN


# ── Visualization resolution (must be divisible by 32) ────────────────
VIS_RES = (192, 256)  # (H, W) — reasonable for display without being huge


def _load_backbone(checkpoint: str | Path, variant: str = "xxs") -> MobileViT:
    """Load pre-trained backbone weights from a checkpoint."""
    ckpt_path = Path(checkpoint)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    backbone = _BACKBONE_FN[variant](VIS_RES)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    backbone.load_state_dict(state)
    backbone.to(DEVICE).eval()
    print(f"Loaded backbone: {ckpt_path.name} ({variant})")
    return backbone


def _get_vis_transform():
    """Transform for visualization (no augmentation, just resize + normalize)."""
    return v2.Compose([
        v2.Resize(VIS_RES),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _denorm(t: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization for display."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (t.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def pca_feature_map(feat: torch.Tensor, n_components: int = 3) -> np.ndarray:
    """Project a spatial feature map to n_components via PCA.

    Args:
        feat: (B, C, H, W) tensor.
    Returns:
        (B, H, W, n_components) array, each component normalized to [0, 1].
    """
    B, C, H, W = feat.shape
    X = feat.permute(0, 2, 3, 1).reshape(B * H * W, C).cpu().float().numpy()
    pca = PCA(n_components=n_components)
    proj = pca.fit_transform(X).reshape(B, H, W, n_components)
    for c in range(n_components):
        mn, mx = proj[..., c].min(), proj[..., c].max()
        proj[..., c] = (proj[..., c] - mn) / (mx - mn + 1e-8)
    return proj


def _infer_variant(checkpoint: str | Path) -> str | None:
    """Try to infer variant (xxs/xs/s) from checkpoint filename."""
    name = Path(checkpoint).stem.lower()
    for v in ("xxs", "xs", "s"):
        if f"_{v}_" in name or name.endswith(f"_{v}"):
            return v
    return None


@torch.no_grad()
def visualize_pca(checkpoint: str | Path, variant: str | None = None,
                  n_images: int = 6):
    """Generate PCA visualization of encoder skip features.

    Creates a grid: RGB | PCA(y1, H/4) | PCA(y2, H/8) | PCA(y3, H/16)
    Saves to outputs/pca_{variant}.png
    """
    from datasets import load_dataset
    from src.config import HF_TOKEN, HF_OFFLINE
    import os

    # Auto-detect variant from filename if not provided
    if variant is None:
        variant = _infer_variant(checkpoint)
        if variant is None:
            raise ValueError(
                f"Cannot infer variant from '{Path(checkpoint).name}'. "
                "Pass --variant explicitly (xxs/xs/s).")
        print(f"Auto-detected variant: {variant}")

    print(f"Device: {DEVICE}")

    backbone = _load_backbone(checkpoint, variant)
    transform = _get_vis_transform()

    if HF_OFFLINE:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        print(f"Loading {n_images} NYU images from local cache (offline)...")
    else:
        os.environ.pop("HF_DATASETS_OFFLINE", None)
        print(f"Loading {n_images} NYU images for visualization...")
    ds = load_dataset("sayakpaul/nyu_depth_v2", split="validation",
                      streaming=True, trust_remote_code=True, token=HF_TOKEN)

    # Prepare batch
    imgs = []
    rgb_display = []
    for i, row in enumerate(ds):
        if i >= n_images:
            break
        img = row["image"].convert("RGB")
        imgs.append(transform(img))
        # Also save un-normalized version for display
        rgb_display.append(
            np.array(img.resize((VIS_RES[1], VIS_RES[0]))) / 255.0
        )
    batch = torch.stack(imgs).to(DEVICE)

    # Extract features
    feat, skips = backbone(batch)
    # skips: [y0 (H/2), y1 (H/4), y2 (H/8), y3 (H/16)]

    # PCA on skip features at 3 scales
    skip_labels = [
        ("y₁ (H/4)", skips[1]),
        ("y₂ (H/8)", skips[2]),
        ("y₃ (H/16)", skips[3]),
    ]

    n_cols = 1 + len(skip_labels)  # RGB + 3 PCA maps
    fig, axes = plt.subplots(n_images, n_cols, figsize=(4 * n_cols, 4 * n_images))
    if n_images == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_images):
        # RGB
        axes[i, 0].imshow(rgb_display[i])
        axes[i, 0].set_title("RGB Input" if i == 0 else "")
        axes[i, 0].axis("off")

        # PCA for each skip scale
        for j, (label, skip_feat) in enumerate(skip_labels):
            pca_map = pca_feature_map(skip_feat[i:i+1])  # (1, H, W, 3)
            axes[i, j + 1].imshow(pca_map[0])
            axes[i, j + 1].set_title(f"PCA {label}" if i == 0 else "")
            axes[i, j + 1].axis("off")

    plt.suptitle(
        f"LeJEPA Zero-Shot PCA Probing — MobileViT-{variant.upper()}",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pca_{variant}.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    print(f"✓ PCA visualization saved → {out_path}")
    plt.show()

    return fig


@torch.no_grad()
def visualize_pca_inline(state_dict: dict, variant: str, epoch: int,
                         out_dir: str | Path = "pca", device: str = DEVICE,
                         n_images: int = 4):
    """Run PCA visualization from an in-memory state_dict (called during training).

    Saves to out_dir/pca_{variant}_epoch{epoch}.png. Uses fewer images than
    the standalone version for speed.
    """
    from datasets import load_dataset
    from src.config import HF_TOKEN, HF_OFFLINE
    import os

    # Build backbone from state_dict
    backbone = _BACKBONE_FN[variant](VIS_RES)
    backbone.load_state_dict(state_dict)
    backbone.to(device).eval()

    transform = _get_vis_transform()

    if HF_OFFLINE:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_DATASETS_OFFLINE", None)

    ds = load_dataset("sayakpaul/nyu_depth_v2", split="validation",
                      streaming=True, trust_remote_code=True, token=HF_TOKEN)

    imgs = []
    rgb_display = []
    for i, row in enumerate(ds):
        if i >= n_images:
            break
        img = row["image"].convert("RGB")
        imgs.append(transform(img))
        rgb_display.append(
            np.array(img.resize((VIS_RES[1], VIS_RES[0]))) / 255.0
        )
    batch = torch.stack(imgs).to(device)

    feat, skips = backbone(batch)
    skip_labels = [
        ("y1 (H/4)", skips[1]),
        ("y2 (H/8)", skips[2]),
        ("y3 (H/16)", skips[3]),
    ]

    n_cols = 1 + len(skip_labels)
    fig, axes = plt.subplots(n_images, n_cols, figsize=(4 * n_cols, 3.5 * n_images))
    if n_images == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_images):
        axes[i, 0].imshow(rgb_display[i])
        axes[i, 0].set_title("RGB" if i == 0 else "")
        axes[i, 0].axis("off")
        for j, (label, skip_feat) in enumerate(skip_labels):
            pca_map = pca_feature_map(skip_feat[i:i+1])
            axes[i, j + 1].imshow(pca_map[0])
            axes[i, j + 1].set_title(f"PCA {label}" if i == 0 else "")
            axes[i, j + 1].axis("off")

    plt.suptitle(f"PCA — MobileViT-{variant.upper()} — Epoch {epoch}", fontsize=13)
    plt.tight_layout()

    save_dir = Path(out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"pca_{variant}_epoch{epoch}.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PCA visualization of LeJEPA features")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to backbone checkpoint (.pth)")
    parser.add_argument("--variant", default=None, choices=["xxs", "xs", "s"],
                        help="Model variant (auto-detected from filename if omitted)")
    parser.add_argument("--n-images", type=int, default=6)
    args = parser.parse_args()
    visualize_pca(checkpoint=args.checkpoint, variant=args.variant,
                  n_images=args.n_images)
