"""Standalone depth evaluation — compute metrics and visualize predictions.

Usage:
    uv run python -m src.evaluation --checkpoint checkpoints/meter_xxs_final.pth
    uv run python -m src.evaluation --checkpoint path/to/model.pth --variant xxs --n-images 6
    uv run python -m src.evaluation --checkpoint path/to/model.pth --output results/eval.png
"""

import torch
import numpy as np
from pathlib import Path

from src.config import DEVICE
from src.visualize import visualize_depth_standalone, _load_val_samples, _infer_variant


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
    from src.loss import compute_depth_metrics
    from src.data import get_depth_loader
    from omegaconf import OmegaConf

    ckpt_path = Path(checkpoint)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    if variant is None:
        variant = _infer_variant(checkpoint)
        if variant is None:
            raise ValueError(
                f"Cannot infer variant from '{ckpt_path.name}'. "
                "Pass --variant explicitly (xxs/xs/s).")

    # Build model
    from src.visualize import VIS_RES
    resolution = VIS_RES  # (192, 256)
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
