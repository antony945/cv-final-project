"""PCA visualization of METER encoder features — zero-shot geometric probing.

After LeJEPA pre-training, we extract spatial feature maps from the encoder
and project them to 3 principal components for visualization (mapped to RGB).
If the pre-training learned useful geometry, PCA maps should show spatial
coherence — warm colors (red/magenta/pink) capturing foreground objects,
cool colors (cyan/green/yellow) representing backgrounds and foliage.

Layout: RGB | PCA(y3, H/16) | PCA(feat, H/32)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from pathlib import Path
from PIL import Image

from src.config import DEVICE
from src.utils import DEFAULT_VIS_RES, pca_feature_map, infer_variant


def _load_backbone(checkpoint: str | Path, variant: str = "xxs",
                   resolution: tuple[int, int] = DEFAULT_VIS_RES):
    """Load pre-trained backbone weights from a checkpoint."""
    from src.model import METEREncoder, _BACKBONE_FN

    ckpt_path = Path(checkpoint)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    backbone = _BACKBONE_FN[variant](resolution)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    backbone.load_state_dict(state)
    backbone.to(DEVICE).eval()
    print(f"Loaded backbone: {ckpt_path.name} ({variant})")
    return backbone


def _get_vis_transform(resolution: tuple[int, int] = DEFAULT_VIS_RES):
    """Transform for visualization (no augmentation, just resize + normalize)."""
    return v2.Compose([
        v2.Resize(resolution),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _upsample_pca(pca_map: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Bilinearly upsample a PCA map (H, W, 3) to target display size."""
    # Use PIL for clean bilinear upsampling
    img = Image.fromarray((pca_map * 255).astype(np.uint8))
    img_resized = img.resize((target_w, target_h), Image.BILINEAR)
    return np.array(img_resized).astype(np.float32) / 255.0


@torch.no_grad()
def visualize_pca(checkpoint: str | Path, variant: str | None = None,
                  n_images: int = 6, resolution: tuple[int, int] = DEFAULT_VIS_RES):
    """Generate PCA visualization of encoder features (paper-style).

    Creates a grid: RGB | PCA(y3, H/16) | PCA(feat, H/32)
    For each image, features are independently projected to RGB using the
    first 3 principal components.

    Saves to outputs/pca_{variant}.png
    """
    from datasets import load_dataset
    from src.config import HF_TOKEN, HF_OFFLINE
    from src.model import _BACKBONE_FN
    import os

    # Auto-detect variant from filename if not provided
    if variant is None:
        variant = infer_variant(checkpoint)
        if variant is None:
            raise ValueError(
                f"Cannot infer variant from '{Path(checkpoint).name}'. "
                "Pass --variant explicitly (xxs/xs/s).")
        print(f"Auto-detected variant: {variant}")

    print(f"Device: {DEVICE}")
    print(f"Resolution: {resolution[0]}×{resolution[1]}")

    backbone = _load_backbone(checkpoint, variant, resolution)
    transform = _get_vis_transform(resolution)
    H, W = resolution

    # Try mmap first (fastest, consistent with training)
    from src.data import _mmap_files_exist, _find_mmap_file
    if _mmap_files_exist(resolution, "val"):
        print(f"Loading {n_images} NYU images from mmap...")
        rgb_path = _find_mmap_file("val", "rgb", H, W)
        rgb_mmap = np.load(str(rgb_path), mmap_mode="r")
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        imgs = []
        rgb_display = []
        for i in range(min(n_images, rgb_mmap.shape[0])):
            rgb_np = np.array(rgb_mmap[i])  # (H, W, 3) uint8
            rgb_display.append(rgb_np.astype(np.float32) / 255.0)
            rgb_f = rgb_np.astype(np.float32) / 255.0
            rgb_f = (rgb_f - mean) / std
            imgs.append(torch.from_numpy(rgb_f.transpose(2, 0, 1)).float())
    else:
        # Fallback: HuggingFace streaming
        if HF_OFFLINE:
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            print(f"Loading {n_images} NYU images from local cache (offline)...")
        else:
            os.environ.pop("HF_DATASETS_OFFLINE", None)
            print(f"Loading {n_images} NYU images for visualization...")
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
                np.array(img.resize((W, H))) / 255.0
            )
    batch = torch.stack(imgs).to(DEVICE)

    # Extract features
    feat, skips = backbone(batch)
    # feat: (B, C_final, H/32, W/32) — deepest, most semantic
    # skips[3]: y3 (B, C, H/16, W/16) — post-transformer skip

    # PCA on two feature levels
    feat_labels = [
        ("PCA y₃ (H/16)", skips[3]),
        ("PCA feat (H/32)", feat),
    ]

    # Fit PCA on full batch (all images together) for consistent color axes
    pca_results = []
    for label, feature in feat_labels:
        pca_map, _ = pca_feature_map(feature)  # (B, h, w, 3)
        pca_results.append((label, pca_map))

    n_cols = 1 + len(feat_labels)  # RGB + 2 PCA maps
    fig, axes = plt.subplots(n_images, n_cols, figsize=(4.5 * n_cols, 4 * n_images))
    if n_images == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_images):
        # RGB
        axes[i, 0].imshow(rgb_display[i])
        axes[i, 0].set_title("RGB" if i == 0 else "")
        axes[i, 0].axis("off")

        # PCA for each feature level
        for j, (label, pca_map) in enumerate(pca_results):
            # Upsample to display resolution
            pca_upsampled = _upsample_pca(pca_map[i], H, W)
            axes[i, j + 1].imshow(pca_upsampled)
            axes[i, j + 1].set_title(label if i == 0 else "")
            axes[i, j + 1].axis("off")

    plt.suptitle(
        f"LeJEPA PCA Probing — METER-{variant.upper()} | {H}×{W}",
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
                         out_dir: str | Path = "plots", device: str = DEVICE,
                         n_images: int = 4,
                         resolution: tuple[int, int] = DEFAULT_VIS_RES):
    """Run PCA visualization from an in-memory state_dict (called during training).

    Saves to out_dir/pca_{variant}_epoch{epoch}.png. Uses fewer images than
    the standalone version for speed.

    Layout: RGB | PCA(y3, H/16) | PCA(feat, H/32)
    """
    from datasets import load_dataset
    from src.config import HF_TOKEN, HF_OFFLINE
    from src.model import _BACKBONE_FN
    import os

    # Build backbone from state_dict
    backbone = _BACKBONE_FN[variant](resolution)
    backbone.load_state_dict(state_dict)
    backbone.to(device).eval()

    transform = _get_vis_transform(resolution)
    H, W = resolution

    # Try mmap first (fastest, consistent with training)
    from src.data import _mmap_files_exist, _find_mmap_file
    if _mmap_files_exist(resolution, "val"):
        rgb_path = _find_mmap_file("val", "rgb", H, W)
        rgb_mmap = np.load(str(rgb_path), mmap_mode="r")
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        imgs = []
        rgb_display = []
        for i in range(min(n_images, rgb_mmap.shape[0])):
            rgb_np = np.array(rgb_mmap[i])  # (H, W, 3) uint8
            rgb_display.append(rgb_np.astype(np.float32) / 255.0)
            rgb_f = rgb_np.astype(np.float32) / 255.0
            rgb_f = (rgb_f - mean) / std
            imgs.append(torch.from_numpy(rgb_f.transpose(2, 0, 1)).float())
    else:
        # Fallback: HuggingFace streaming
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
                np.array(img.resize((W, H))) / 255.0
            )
    batch = torch.stack(imgs).to(device)

    feat, skips = backbone(batch)

    feat_labels = [
        ("PCA y₃ (H/16)", skips[3]),
        ("PCA feat (H/32)", feat),
    ]

    # Fit PCA on full batch for consistent color axes
    pca_results = []
    for label, feature in feat_labels:
        pca_map, _ = pca_feature_map(feature)
        pca_results.append((label, pca_map))

    n_cols = 1 + len(feat_labels)
    fig, axes = plt.subplots(n_images, n_cols, figsize=(4.5 * n_cols, 3.5 * n_images))
    if n_images == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_images):
        axes[i, 0].imshow(rgb_display[i])
        axes[i, 0].set_title("RGB" if i == 0 else "")
        axes[i, 0].axis("off")
        for j, (label, pca_map) in enumerate(pca_results):
            pca_upsampled = _upsample_pca(pca_map[i], H, W)
            axes[i, j + 1].imshow(pca_upsampled)
            axes[i, j + 1].set_title(label if i == 0 else "")
            axes[i, j + 1].axis("off")

    plt.suptitle(f"PCA — METER-{variant.upper()} — Epoch {epoch}", fontsize=13)
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
    parser.add_argument("--resolution", type=int, nargs=2, default=[192, 256],
                        metavar=("H", "W"),
                        help="Input resolution for the model (default: 192 256)")
    args = parser.parse_args()
    visualize_pca(checkpoint=args.checkpoint, variant=args.variant,
                  n_images=args.n_images, resolution=tuple(args.resolution))
