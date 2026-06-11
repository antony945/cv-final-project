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
from pathlib import Path
from PIL import Image

from src.config import DEVICE
from src.utils import DATASET_DEFAULTS, VIS_SEED, pca_feature_map, infer_variant, load_val_samples


def _upsample_pca(pca_map: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Bilinearly upsample a PCA map (H, W, 3) to target display size."""
    # Use PIL for clean bilinear upsampling
    img = Image.fromarray((pca_map * 255).astype(np.uint8))
    img_resized = img.resize((target_w, target_h), Image.BILINEAR)
    return np.array(img_resized).astype(np.float32) / 255.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PCA Visualization (inline + standalone)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def visualize_pca_inline(state_dict: dict, variant: str, epoch: int,
                         out_dir: str | Path = "plots", device: str = DEVICE,
                         n_images: int = 4,
                         resolution: tuple[int, int] | None = None,
                         dataset: str = "nyu"):
    """Run PCA visualization from an in-memory state_dict (called during training).

    Saves to out_dir/pca_{variant}_epoch{epoch}.png. Uses fewer images than
    the standalone version for speed.

    Layout: RGB | PCA(y3, H/16) | PCA(feat, H/32)
    """
    from src.model import _BACKBONE_FN

    ds_defaults = DATASET_DEFAULTS[dataset]
    ds_resolution = resolution if resolution else ds_defaults["resolution"]
    H, W = ds_resolution

    # Build backbone from state_dict
    backbone = _BACKBONE_FN[variant](ds_resolution)
    backbone.load_state_dict(state_dict)
    backbone.to(device).eval()

    # Load val images
    rgb_batch, _, rgb_display = load_val_samples(
        n_images, ds_resolution, dataset=dataset, seed=VIS_SEED)
    batch = rgb_batch.to(device)

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


@torch.no_grad()
def visualize_pca_standalone(checkpoint: str | Path, variant: str | None = None,
                  n_images: int = 4, resolution: tuple[int, int] | None = None,
                  datasets: list[str] | None = None, seed: int = VIS_SEED) -> list[Path]:
    """Generate PCA visualization of encoder features (paper-style).

    Creates one PNG per dataset with grid: RGB | PCA(y3, H/16) | PCA(feat, H/32)
    For each image, features are independently projected to RGB using the
    first 3 principal components.

    Args:
        checkpoint: Path to backbone checkpoint.
        variant: Model variant (xxs/xs/s). Auto-detected if None.
        n_images: Number of images per dataset.
        resolution: Override resolution for all datasets. If None, uses dataset default.
        datasets: List of datasets to visualize (default: ["nyu"]).
        seed: Random seed for reproducible image selection.
    Returns:
        List of output paths.
    """
    from src.model import _BACKBONE_FN

    if datasets is None:
        datasets = ["nyu"]

    # Auto-detect variant from filename if not provided
    if variant is None:
        variant = infer_variant(checkpoint)
        if variant is None:
            raise ValueError(
                f"Cannot infer variant from '{Path(checkpoint).name}'. "
                "Pass --variant explicitly (xxs/xs/s).")
        print(f"Auto-detected variant: {variant}")

    print(f"Device: {DEVICE}")
    print(f"Datasets: {', '.join(d.upper() for d in datasets)}")

    ckpt_path = Path(checkpoint)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    output_paths = []

    for ds_name in datasets:
        ds_defaults = DATASET_DEFAULTS[ds_name]
        ds_resolution = resolution if resolution else ds_defaults["resolution"]
        H, W = ds_resolution

        print(f"\n{'='*50}")
        print(f"  {ds_name.upper()} — {H}×{W}")
        print(f"{'='*50}")

        # Load backbone at this resolution
        backbone = _BACKBONE_FN[variant](ds_resolution)
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        backbone.load_state_dict(state)
        backbone.to(DEVICE).eval()

        # Load val images (random selection with seed)
        rgb_batch, _, rgb_display = load_val_samples(
            n_images, ds_resolution, dataset=ds_name, seed=seed)
        batch = rgb_batch.to(DEVICE)

        # Extract features
        feat, skips = backbone(batch)

        # PCA on two feature levels
        feat_labels = [
            ("PCA y₃ (H/16)", skips[3]),
            ("PCA feat (H/32)", feat),
        ]

        pca_results = []
        for label, feature in feat_labels:
            pca_map, _ = pca_feature_map(feature)
            pca_results.append((label, pca_map))

        # Compute figure size based on aspect ratio
        aspect = W / H  # e.g. 640/192 ≈ 3.3 for KITTI, 256/192 ≈ 1.3 for NYU
        col_width = 4.5
        row_height = col_width / aspect + 0.5  # scale height to aspect ratio
        n_cols = 1 + len(feat_labels)
        fig, axes = plt.subplots(n_images, n_cols,
                                 figsize=(col_width * n_cols, row_height * n_images))
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

        plt.suptitle(
            f"LeJEPA PCA Probing — METER-{variant.upper()} | {ds_name.upper()} {H}×{W}",
            fontsize=14, y=1.01,
        )
        plt.tight_layout()

        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"pca_{variant}_{ds_name}.png"
        plt.savefig(out_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        print(f"  ✓ Saved → {out_path}")
        output_paths.append(out_path)

    return output_paths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PCA visualization of LeJEPA features")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to backbone checkpoint (.pth)")
    parser.add_argument("--variant", default=None, choices=["xxs", "xs", "s"],
                        help="Model variant (auto-detected from filename if omitted)")
    parser.add_argument("--n-images", type=int, default=4)
    parser.add_argument("--resolution", type=int, nargs=2, default=None,
                        metavar=("H", "W"),
                        help="Override resolution (default: dataset-specific)")
    parser.add_argument("--dataset", type=str, nargs="+", default=["nyu"],
                        choices=["nyu", "kitti"],
                        help="Dataset(s) to visualize (default: nyu). "
                             "Pass multiple: --dataset nyu kitti")
    parser.add_argument("--seed", type=int, default=VIS_SEED,
                        help=f"Random seed for image selection (default: {VIS_SEED})")
    args = parser.parse_args()
    res = tuple(args.resolution) if args.resolution else None
    paths = visualize_pca_standalone(checkpoint=args.checkpoint, variant=args.variant,
                          n_images=args.n_images, resolution=res,
                          datasets=args.dataset, seed=args.seed)

    # Show images interactively
    for p in paths:
        img = plt.imread(str(p))
        plt.figure(figsize=(14, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
    plt.show()
