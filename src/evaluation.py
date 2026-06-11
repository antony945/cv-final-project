"""METER depth evaluation — compute metrics and visualize depth predictions.

Usage:
    uv run python -m src.evaluation --checkpoint checkpoints/meter_xxs_final.pth
    uv run python -m src.evaluation --checkpoint path/to/model.pth --variant xxs --n-images 6
    uv run python -m src.evaluation --checkpoint path/to/model.pth --dataset nyu kitti
    uv run python -m src.evaluation --checkpoint path/to/model.pth --dataset kitti --skip-full-val
    uv run python -m src.evaluation --checkpoint path/to/model.pth --seed 123
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import DEVICE
from src.utils import (
    compute_depth_metrics, infer_variant, load_val_samples, DATASET_DEFAULTS, VIS_SEED,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Depth Visualization (inline + standalone)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@torch.no_grad()
def visualize_depth_inline(model, variant: str, epoch: int,
                           out_dir: str | Path = "plots", device: str = DEVICE,
                           n_images: int = 4, dataset: str = "nyu",
                           seed: int = VIS_SEED) -> Path:
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
        dataset: "nyu" or "kitti" — controls resolution and colormap range.
    Returns:
        Path to saved image.
    """
    model.eval()
    ds = DATASET_DEFAULTS[dataset]
    resolution = ds["resolution"]
    depth_vmin, depth_vmax = ds["depth_vmin"], ds["depth_vmax"]

    # Temporarily override model resolution for inference
    orig_resolution = model.resolution
    model.resolution = resolution

    rgb_batch, depth_gt_batch, rgb_display = load_val_samples(
        n_images, resolution, dataset=dataset, seed=seed)
    rgb_batch = rgb_batch.to(device)

    # Predict
    depth_pred = model(rgb_batch)  # (N, 1, H, W)
    depth_pred_np = depth_pred.cpu().numpy()[:, 0]  # (N, H, W)
    depth_gt_np = depth_gt_batch.numpy()[:, 0]       # (N, H, W)

    # Restore original resolution
    model.resolution = orig_resolution

    # Plot grid: 4 columns
    fig, axes = plt.subplots(n_images, 4, figsize=(16, 4 * n_images))
    if n_images == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["RGB", "GT Depth", "Predicted Depth", "Error |GT - Pred|"]

    for i in range(n_images):
        axes[i, 0].imshow(rgb_display[i])
        axes[i, 0].axis("off")

        axes[i, 1].imshow(depth_gt_np[i], cmap="plasma",
                          vmin=depth_vmin, vmax=depth_vmax)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(depth_pred_np[i], cmap="plasma",
                          vmin=depth_vmin, vmax=depth_vmax)
        axes[i, 2].axis("off")

        diff = np.abs(depth_gt_np[i] - depth_pred_np[i])
        axes[i, 3].imshow(diff, cmap="hot", vmin=0, vmax=depth_vmax / 2)
        axes[i, 3].axis("off")

        if i == 0:
            for j, title in enumerate(col_titles):
                axes[i, j].set_title(title, fontsize=11)

    plt.suptitle(f"METER Depth — METER-{variant.upper()} — Epoch {epoch} ({dataset.upper()})",
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
                               n_images: int = 4, dataset: str = "nyu",
                               seed: int = VIS_SEED):
    """Standalone depth visualization — loads model from checkpoint.

    Runs inference on val images and saves the depth prediction grid.
    Output: outputs/depth_eval_{variant}_{dataset}.png

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

    ds = DATASET_DEFAULTS[dataset]
    resolution = ds["resolution"]
    depth_vmin, depth_vmax = ds["depth_vmin"], ds["depth_vmax"]

    print(f"Device: {DEVICE}")
    print(f"Dataset: {dataset.upper()} ({resolution[0]}x{resolution[1]})")

    # Build model and load weights (override resolution for target dataset)
    model = METERModel(variant=variant, resolution=resolution).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded METER model: {ckpt_path.name} ({variant})")

    # Visualization grid
    rgb_batch, depth_gt_batch, rgb_display = load_val_samples(
        n_images, resolution, dataset=dataset, seed=seed)
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
                          vmin=depth_vmin, vmax=depth_vmax)
        axes[i, 1].axis("off")
        axes[i, 2].imshow(depth_pred_np[i], cmap="plasma",
                          vmin=depth_vmin, vmax=depth_vmax)
        axes[i, 2].axis("off")
        diff = np.abs(depth_gt_np[i] - depth_pred_np[i])
        axes[i, 3].imshow(diff, cmap="hot", vmin=0, vmax=depth_vmax / 2)
        axes[i, 3].axis("off")
        if i == 0:
            for j, title in enumerate(col_titles):
                axes[i, j].set_title(title, fontsize=11)

    plt.suptitle(f"METER Depth Evaluation — METER-{variant.upper()} ({dataset.upper()})", fontsize=13)
    plt.tight_layout()

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"depth_eval_{variant}_{dataset}.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  Depth grid saved: {out_path}")

    # Compute full-batch metrics on the vis images
    depth_pred_t = depth_pred.to(DEVICE)
    depth_gt_t = depth_gt_batch.to(DEVICE)
    metrics = compute_depth_metrics(depth_pred_t, depth_gt_t)

    return metrics, out_path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Full Validation Set Evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@torch.no_grad()
def validate_depth(model, cfg, device: str = DEVICE) -> dict:
    """Run validation on the full val set and return averaged metrics.

    This is the shared validation loop used by both the training pipeline
    and the standalone CLI evaluation.

    Args:
        model: METERModel (already on device, will be set to eval mode).
        cfg: DictConfig with data.datasets and finetune.resolution.
        device: Device string.
    Returns:
        dict with keys: delta1, delta2, delta3, rmse, rel, log10, n_images
    """
    from src.data import get_depth_loader

    model.eval()
    val_loader = get_depth_loader(cfg, device=device, split="val")

    # Dataset-specific evaluation params
    dataset_name = next(iter(cfg.data.datasets))
    ds_cfg = cfg.data.datasets[dataset_name]
    eval_crop = ds_cfg.get("eval_crop", "none")
    min_depth = ds_cfg.get("min_depth", 1e-3)
    max_depth = ds_cfg.get("max_depth", 10.0)

    all_metrics = []
    n_images_total = 0
    for rgb, depth_gt in val_loader:
        rgb = rgb.to(device, non_blocking=True)
        depth_gt = depth_gt.to(device, non_blocking=True)
        depth_pred = model(rgb)
        n_images_total += rgb.shape[0]
        metrics = compute_depth_metrics(
            depth_pred, depth_gt,
            min_depth=min_depth,
            max_depth=max_depth,
            eval_crop=eval_crop,
        )
        all_metrics.append(metrics)

    # Average metrics across batches
    avg = {}
    for key in all_metrics[0]:
        avg[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
    avg["n_images"] = n_images_total
    return avg


def evaluate_full(checkpoint: str | Path, variant: str | None = None,
                  device: str = DEVICE, dataset: str = "nyu") -> dict:
    """Load model from checkpoint and run full validation.

    Args:
        checkpoint: Path to METER model checkpoint.
        variant: Model variant (xxs/xs/s). Auto-detected if None.
        device: Device for inference.
        dataset: "nyu" or "kitti".
    Returns:
        dict with keys: delta1, delta2, delta3, rmse, rel, log10, n_images
    """
    from src.model import METERModel
    from omegaconf import OmegaConf

    ckpt_path = Path(checkpoint)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    if variant is None:
        variant = infer_variant(checkpoint)
        if variant is None:
            raise ValueError(
                f"Cannot infer variant from '{ckpt_path.name}'. "
                "Pass --variant explicitly (xxs/xs/s).")

    ds = DATASET_DEFAULTS[dataset]
    resolution = ds["resolution"]

    # Build model (override resolution for target dataset)
    model = METERModel(variant=variant, resolution=resolution).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    # Build a minimal config for get_depth_loader
    ds_cfg = {
        "n_samples": 999_999,
        "depth_shift": 0.1 if dataset == "nyu" else 1.0,
        "min_depth": ds["min_depth"],
        "max_depth": ds["max_depth"],
        "eval_crop": ds["eval_crop"],
    }
    if dataset == "kitti":
        ds_cfg["resolution"] = list(resolution)

    cfg = OmegaConf.create({
        "data": {
            "datasets": {dataset: ds_cfg},
            "use_mmap": True,
        },
        "finetune": {"resolution": list(resolution), "bs": 8},
    })

    return validate_depth(model, cfg, device)


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
    parser.add_argument("--dataset", type=str, nargs="+", default=["nyu"],
                        choices=["nyu", "kitti"],
                        help="Dataset(s) to evaluate (default: nyu). "
                             "Pass multiple: --dataset nyu kitti")
    parser.add_argument("--skip-full-val", action="store_true", default=False,
                        help="Skip full validation set evaluation (only evaluate on --n-images samples).")
    parser.add_argument("--seed", type=int, default=VIS_SEED,
                        help=f"Random seed for image selection (default: {VIS_SEED})")
    args = parser.parse_args()

    print("=" * 60)
    print("  METER Depth Evaluation")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Datasets: {', '.join(d.upper() for d in args.dataset)}")
    print(f"  Device: {DEVICE}")
    print()

    out_paths = []

    for dataset in args.dataset:
        print(f"\n{'─' * 60}")
        print(f"  {dataset.upper()}")
        print(f"{'─' * 60}")

        # Generate visualization grid + quick metrics on N images
        metrics_vis, out_path = visualize_depth_standalone(
            checkpoint=args.checkpoint,
            variant=args.variant,
            n_images=args.n_images,
            dataset=dataset,
            seed=args.seed,
        )
        out_paths.append(out_path)

        # Optionally run full validation
        if not args.skip_full_val:
            print(f"\n  Running full {dataset.upper()} validation...")
            metrics = evaluate_full(args.checkpoint, args.variant, dataset=dataset)
            print(f"  Evaluated on {metrics['n_images']} images.")
        else:
            metrics = metrics_vis

        # Print results
        print(f"\n  {dataset.upper()} Evaluation Metrics")
        print(f"  {'─' * 40}")
        print(f"  d1 (delta < 1.25)   : {metrics['delta1']:.4f}")
        print(f"  d2 (delta < 1.25^2) : {metrics['delta2']:.4f}")
        print(f"  d3 (delta < 1.25^3) : {metrics['delta3']:.4f}")
        print(f"  RMSE                : {metrics['rmse']:.4f}")
        print(f"  REL (AbsRel)        : {metrics['rel']:.4f}")
        print(f"  log10               : {metrics['log10']:.4f}")

    print("\n" + "=" * 60)
    print("  Saved depth grids:")
    for p in out_paths:
        print(f"    {p}")
    print("=" * 60)

    # Show images interactively
    for p in out_paths:
        img = plt.imread(str(p))
        plt.figure(figsize=(16, 4 * args.n_images))
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
