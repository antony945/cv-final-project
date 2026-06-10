"""METER depth evaluation — compute metrics and visualize depth predictions.

Usage:
    uv run python -m src.evaluation --checkpoint checkpoints/meter_xxs_final.pth
    uv run python -m src.evaluation --checkpoint path/to/model.pth --variant xxs --n-images 6
    uv run python -m src.evaluation --checkpoint path/to/model.pth --output results/eval.png
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import DEVICE
from src.utils import (
    compute_depth_metrics, infer_variant, load_val_samples,
    DEFAULT_VIS_RES, DEPTH_VMIN, DEPTH_VMAX,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Depth Visualization (inline + standalone)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@torch.no_grad()
def visualize_depth_inline(model, variant: str, epoch: int,
                           out_dir: str | Path = "plots", device: str = DEVICE,
                           n_images: int = 4) -> Path:
    """Generate depth prediction grid during training.

    Grid layout: RGB | GT Depth | Predicted Depth | Diff Map
    Saves to {out_dir}/depth_{variant}_epoch{epoch}.png.

    Args:
        model: METERModel (already on device). Will be set to eval mode.
        variant: Model variant string (xxs/xs/s).
        epoch: Current epoch number.
        out_dir: Output directory for the image.
        device: Device string.
        n_images: Number of rows in the grid.
    Returns:
        Path to saved image.
    """
    model.eval()
    resolution = model.resolution  # (H, W)

    rgb_batch, depth_gt_batch, rgb_display = load_val_samples(n_images, resolution)
    rgb_batch = rgb_batch.to(device)

    # Predict
    depth_pred = model(rgb_batch)  # (N, 1, H, W)
    depth_pred_np = depth_pred.cpu().numpy()[:, 0]  # (N, H, W)
    depth_gt_np = depth_gt_batch.numpy()[:, 0]       # (N, H, W)

    # Plot grid: 4 columns
    fig, axes = plt.subplots(n_images, 4, figsize=(16, 4 * n_images))
    if n_images == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["RGB", "GT Depth", "Predicted Depth", "Error |GT - Pred|"]

    for i in range(n_images):
        axes[i, 0].imshow(rgb_display[i])
        axes[i, 0].axis("off")

        axes[i, 1].imshow(depth_gt_np[i], cmap="plasma",
                          vmin=DEPTH_VMIN, vmax=DEPTH_VMAX)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(depth_pred_np[i], cmap="plasma",
                          vmin=DEPTH_VMIN, vmax=DEPTH_VMAX)
        axes[i, 2].axis("off")

        diff = np.abs(depth_gt_np[i] - depth_pred_np[i])
        axes[i, 3].imshow(diff, cmap="hot", vmin=0, vmax=DEPTH_VMAX / 2)
        axes[i, 3].axis("off")

        if i == 0:
            for j, title in enumerate(col_titles):
                axes[i, j].set_title(title, fontsize=11)

    plt.suptitle(f"METER Depth — METER-{variant.upper()} — Epoch {epoch}",
                 fontsize=13)
    plt.tight_layout()

    save_dir = Path(out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"depth_{variant}_epoch{epoch}.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    return out_path


@torch.no_grad()
def visualize_depth_standalone(checkpoint: str | Path, variant: str | None = None,
                               n_images: int = 4, output: str | Path | None = None):
    """Standalone depth visualization — loads model from checkpoint.

    Runs inference on val images and saves the depth prediction grid.

    Returns:
        (metrics_dict, output_path)
    """
    from src.model import METERModel

    ckpt_path = Path(checkpoint)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    # Auto-detect variant
    if variant is None:
        variant = infer_variant(checkpoint)
        if variant is None:
            raise ValueError(
                f"Cannot infer variant from '{ckpt_path.name}'. "
                "Pass --variant explicitly (xxs/xs/s).")
        print(f"Auto-detected variant: {variant}")

    print(f"Device: {DEVICE}")

    # Build model and load weights
    resolution = DEFAULT_VIS_RES  # (192, 256)
    model = METERModel(variant=variant, resolution=resolution).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded METER model: {ckpt_path.name} ({variant})")

    # Visualization grid
    rgb_batch, depth_gt_batch, rgb_display = load_val_samples(n_images, resolution)
    rgb_batch = rgb_batch.to(DEVICE)

    depth_pred = model(rgb_batch)
    depth_pred_np = depth_pred.cpu().numpy()[:, 0]
    depth_gt_np = depth_gt_batch.numpy()[:, 0]

    fig, axes = plt.subplots(n_images, 4, figsize=(16, 4 * n_images))
    if n_images == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["RGB", "GT Depth", "Predicted Depth", "Error |GT - Pred|"]
    for i in range(n_images):
        axes[i, 0].imshow(rgb_display[i])
        axes[i, 0].axis("off")
        axes[i, 1].imshow(depth_gt_np[i], cmap="plasma",
                          vmin=DEPTH_VMIN, vmax=DEPTH_VMAX)
        axes[i, 1].axis("off")
        axes[i, 2].imshow(depth_pred_np[i], cmap="plasma",
                          vmin=DEPTH_VMIN, vmax=DEPTH_VMAX)
        axes[i, 2].axis("off")
        diff = np.abs(depth_gt_np[i] - depth_pred_np[i])
        axes[i, 3].imshow(diff, cmap="hot", vmin=0, vmax=DEPTH_VMAX / 2)
        axes[i, 3].axis("off")
        if i == 0:
            for j, title in enumerate(col_titles):
                axes[i, j].set_title(title, fontsize=11)

    plt.suptitle(f"METER Depth Evaluation — METER-{variant.upper()}", fontsize=13)
    plt.tight_layout()

    if output is None:
        output = Path("outputs") / f"depth_eval_{variant}.png"
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"Depth grid saved: {out_path}")

    # Compute full-batch metrics on the vis images
    depth_pred_t = depth_pred.to(DEVICE)
    depth_gt_t = depth_gt_batch.to(DEVICE)
    metrics = compute_depth_metrics(depth_pred_t, depth_gt_t)

    return metrics, out_path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Full Validation Set Evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def evaluate_full(checkpoint: str | Path, variant: str | None = None,
                  device: str = DEVICE) -> dict:
    """Run evaluation on the full NYU validation set and return metrics.

    Args:
        checkpoint: Path to METER model checkpoint.
        variant: Model variant (xxs/xs/s). Auto-detected if None.
        device: Device for inference.
    Returns:
        dict with keys: delta1, delta2, delta3, rmse, rel, log10
    """
    from src.model import METERModel
    from src.data import get_depth_loader
    from omegaconf import OmegaConf

    ckpt_path = Path(checkpoint)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    if variant is None:
        variant = infer_variant(checkpoint)
        if variant is None:
            raise ValueError(
                f"Cannot infer variant from '{ckpt_path.name}'. "
                "Pass --variant explicitly (xxs/xs/s).")

    # Build model
    resolution = DEFAULT_VIS_RES  # (192, 256)
    model = METERModel(variant=variant, resolution=resolution).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Build a minimal config for get_depth_loader
    cfg = OmegaConf.create({
        "data": {"dataset": "nyu", "n_samples": 654},  # NYU val has 654 images
        "finetune": {"resolution": list(resolution), "bs": 8},
    })
    val_loader = get_depth_loader(cfg, device=device, split="val")

    all_metrics = []
    with torch.no_grad():
        for rgb, depth_gt in val_loader:
            rgb = rgb.to(device, non_blocking=True)
            depth_gt = depth_gt.to(device, non_blocking=True)
            depth_pred = model(rgb)
            metrics = compute_depth_metrics(depth_pred, depth_gt)
            all_metrics.append(metrics)

    # Average
    avg = {}
    for key in all_metrics[0]:
        avg[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
    return avg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="METER Depth Evaluation — compute metrics and visualize predictions")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to METER model checkpoint (.pth)")
    parser.add_argument("--variant", default=None, choices=["xxs", "xs", "s"],
                        help="Model variant (auto-detected from filename if omitted)")
    parser.add_argument("--n-images", type=int, default=4,
                        help="Number of images in the visualization grid")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for depth grid image (default: outputs/depth_eval_{variant}.png)")
    parser.add_argument("--full-val", action="store_true", default=False,
                        help="Run evaluation on full validation set (654 images). "
                             "Without this flag, only evaluates on --n-images samples.")
    args = parser.parse_args()

    print("=" * 60)
    print("  METER Depth Evaluation")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device: {DEVICE}")
    print()

    # Generate visualization grid + quick metrics on N images
    metrics_vis, out_path = visualize_depth_standalone(
        checkpoint=args.checkpoint,
        variant=args.variant,
        n_images=args.n_images,
        output=args.output,
    )

    # Optionally run full validation
    if args.full_val:
        print("\nRunning full validation (654 images)...")
        metrics = evaluate_full(args.checkpoint, args.variant)
    else:
        metrics = metrics_vis

    # Print results
    print("\n" + "=" * 60)
    print("  Evaluation Metrics")
    print("=" * 60)
    print(f"  d1 (delta < 1.25)   : {metrics['delta1']:.4f}")
    print(f"  d2 (delta < 1.25^2) : {metrics['delta2']:.4f}")
    print(f"  d3 (delta < 1.25^3) : {metrics['delta3']:.4f}")
    print(f"  RMSE                : {metrics['rmse']:.4f}")
    print(f"  REL (AbsRel)        : {metrics['rel']:.4f}")
    print(f"  log10               : {metrics['log10']:.4f}")
    print("=" * 60)
    print(f"\n  Depth grid saved: {out_path}")


if __name__ == "__main__":
    main()
