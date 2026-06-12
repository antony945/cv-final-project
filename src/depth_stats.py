"""Compute depth statistics from memory-mapped dataset files.

Usage:
    uv run python -m src.depth_stats                   # all datasets
    uv run python -m src.depth_stats --dataset nyu     # NYU only
    uv run python -m src.depth_stats --dataset kitti   # KITTI only
    uv run python -m src.depth_stats --max-samples 500 # limit samples processed
"""

import argparse
import numpy as np
from pathlib import Path

from src.config import ROOT, get_nyu_mmap_dir, get_kitti_mmap_dir


def compute_stats(depth_path: Path, max_samples: int | None = None) -> dict:
    """Compute depth statistics from a memory-mapped depth file.

    Args:
        depth_path: Path to .npy depth mmap file (N, H, W) float32.
        max_samples: Max samples to process (None = all).

    Returns:
        Dictionary with statistics.
    """
    d = np.load(str(depth_path), mmap_mode="r")
    n_total = d.shape[0]
    n = min(max_samples, n_total) if max_samples else n_total

    all_valid = []
    n_empty = 0
    for i in range(n):
        sample = np.array(d[i])
        valid = sample[sample > 0]
        if valid.size == 0:
            n_empty += 1
        else:
            all_valid.append(valid)

    if not all_valid:
        return {"error": "No valid depth pixels found", "n_empty": n_empty}

    all_valid = np.concatenate(all_valid)
    total_pixels = n * d.shape[1] * d.shape[2]

    return {
        "file": depth_path.name,
        "shape": d.shape,
        "samples_processed": n,
        "empty_samples": n_empty,
        "valid_pixels": all_valid.size,
        "total_pixels": total_pixels,
        "fill_rate": all_valid.size / total_pixels,
        "mean": float(all_valid.mean()),
        "median": float(np.median(all_valid)),
        "std": float(all_valid.std()),
        "min": float(all_valid.min()),
        "max": float(all_valid.max()),
        "p10": float(np.percentile(all_valid, 10)),
        "p25": float(np.percentile(all_valid, 25)),
        "p50": float(np.percentile(all_valid, 50)),
        "p75": float(np.percentile(all_valid, 75)),
        "p90": float(np.percentile(all_valid, 90)),
    }


def print_stats(stats: dict):
    """Pretty-print depth statistics."""
    if "error" in stats:
        print(f"  ERROR: {stats['error']} ({stats['n_empty']} empty samples)")
        return

    print(f"  File: {stats['file']}")
    print(f"  Shape: {stats['shape']}")
    print(f"  Samples processed: {stats['samples_processed']}")
    if stats["empty_samples"] > 0:
        print(f"  Empty samples (all zeros): {stats['empty_samples']}")
    print(f"  Fill rate: {stats['fill_rate']*100:.1f}% valid pixels")
    print(f"  Mean:   {stats['mean']:.2f} m")
    print(f"  Median: {stats['median']:.2f} m")
    print(f"  Std:    {stats['std']:.2f} m")
    print(f"  Min:    {stats['min']:.3f} m")
    print(f"  Max:    {stats['max']:.2f} m")
    print(f"  Percentiles:")
    print(f"    10th: {stats['p10']:.2f} m")
    print(f"    25th: {stats['p25']:.2f} m")
    print(f"    50th: {stats['p50']:.2f} m (median)")
    print(f"    75th: {stats['p75']:.2f} m")
    print(f"    90th: {stats['p90']:.2f} m")
    print(f"  Suggested depth_bias: {stats['median']:.1f} (median)")


def find_depth_mmaps(dataset: str) -> list[Path]:
    """Find all depth mmap files for a dataset."""
    if dataset == "nyu":
        mmap_dir = Path(get_nyu_mmap_dir())
        if not mmap_dir.is_absolute():
            mmap_dir = ROOT / mmap_dir
        return sorted(mmap_dir.glob("nyu_*_depth_*.npy"))
    elif dataset == "kitti":
        mmap_dir = Path(get_kitti_mmap_dir())
        if not mmap_dir.is_absolute():
            mmap_dir = ROOT / mmap_dir
        return sorted(mmap_dir.glob("kitti_*_depth_*.npy"))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def main():
    parser = argparse.ArgumentParser(description="Compute depth statistics from mmap files")
    parser.add_argument("--dataset", nargs="*", default=["nyu", "kitti"],
                        choices=["nyu", "kitti"], help="Datasets to analyze")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to process per file (default: all)")
    args = parser.parse_args()

    for ds in args.dataset:
        print(f"\n{'='*60}")
        print(f"  {ds.upper()} Depth Statistics")
        print(f"{'='*60}")

        files = find_depth_mmaps(ds)
        if not files:
            print(f"  No mmap files found for {ds}.")
            continue

        for f in files:
            print(f"\n--- {f.name} ---")
            stats = compute_stats(f, max_samples=args.max_samples)
            print_stats(stats)


if __name__ == "__main__":
    main()
