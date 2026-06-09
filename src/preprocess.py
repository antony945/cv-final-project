"""One-time preprocessing: convert NYU tar/h5 data to memory-mapped .npy files.

Creates two files per split (train/val):
  - nyu_{split}_rgb_{H}x{W}.npy   → (N, H, W, 3) uint8
  - nyu_{split}_depth_{H}x{W}.npy → (N, H, W) float32

Usage:
  uv run python -m src.preprocess [resolution_h] [resolution_w]
  uv run python -m src.preprocess              # default: 192 256
  uv run python -m src.preprocess 192 256      # explicit

The output directory is controlled by NYU_MMAP_DIR env var or defaults to datasets/nyu_mmap.
"""

import glob
import io
import logging
import os
import sys
import tarfile
from pathlib import Path

import h5py
import numpy as np
import tqdm
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _get_output_dir() -> Path:
    """Resolve output directory for mmap files."""
    from src.config import ROOT, get_nyu_mmap_dir
    mmap_dir = get_nyu_mmap_dir()
    p = Path(mmap_dir)
    if not p.is_absolute():
        p = ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_all_from_tars(tar_dir: Path, split: str) -> tuple[list, list]:
    """Load ALL samples from tar/h5 shards for a given split."""
    pattern = f"{split}-*.tar"
    tar_files = sorted(glob.glob(str(tar_dir / pattern)))
    if not tar_files:
        raise FileNotFoundError(
            f"No {pattern} files in {tar_dir}. "
            f"Expected NYU tar archives with .h5 files inside."
        )

    log.info(f"Reading {len(tar_files)} tar shards for split='{split}' from {tar_dir}...")

    rgbs = []
    depths = []
    for tar_path in tar_files:
        with tarfile.open(tar_path, "r") as tf:
            members = [m for m in tf if m.name.endswith(".h5")]
            for member in tqdm.tqdm(members, desc=f"  {Path(tar_path).name}",
                                    unit="img"):
                f = tf.extractfile(member)
                if f is None:
                    continue
                h5_bytes = f.read()
                f.close()
                with h5py.File(io.BytesIO(h5_bytes), "r") as h5f:
                    rgb = np.array(h5f["rgb"])      # (3, H, W) uint8
                    depth = np.array(h5f["depth"])  # (H, W) float32
                rgb = np.transpose(rgb, (1, 2, 0))  # (H, W, 3)
                rgbs.append(rgb)
                depths.append(depth)

    log.info(f"  Total: {len(rgbs)} samples")

    return rgbs, depths


def _load_all_from_hf(split: str) -> tuple[list, list]:
    """Load ALL samples from HuggingFace for a given split."""
    from datasets import load_dataset
    from src.config import HF_TOKEN

    hf_split = "validation" if split == "val" else split
    log.info(f"Loading NYU '{hf_split}' split from HuggingFace (this may take a while)...")

    ds = load_dataset(
        "sayakpaul/nyu_depth_v2",
        split=hf_split,
        trust_remote_code=True,
        token=HF_TOKEN,
    )

    rgbs = []
    depths = []
    for i, row in enumerate(tqdm.tqdm(ds, desc=f"  Loading {hf_split}", unit="img")):
        rgb = np.array(row["image"].convert("RGB"))  # (H, W, 3)
        depth = np.array(row["depth_map"])            # (H, W)
        rgbs.append(rgb)
        depths.append(depth)

    return rgbs, depths


def preprocess_split(split: str, h: int, w: int, out_dir: Path,
                     tar_dir: Path | None = None):
    """Preprocess one split into memory-mapped .npy files."""
    rgb_path = out_dir / f"nyu_{split}_rgb_{h}x{w}.npy"
    depth_path = out_dir / f"nyu_{split}_depth_{h}x{w}.npy"

    if rgb_path.exists() and depth_path.exists():
        existing = np.load(str(rgb_path), mmap_mode="r")
        log.info(f"Already exists: {rgb_path.name} ({existing.shape[0]} samples). Skipping.")
        return

    # Load all samples
    if tar_dir and tar_dir.exists():
        rgbs, depths = _load_all_from_tars(tar_dir, split)
    else:
        rgbs, depths = _load_all_from_hf(split)

    n = len(rgbs)
    log.info(f"Loaded {n} samples for split='{split}'. Resizing to {h}x{w}...")

    # Create memory-mapped output files
    rgb_mmap = np.lib.format.open_memmap(
        str(rgb_path), mode="w+", dtype=np.uint8, shape=(n, h, w, 3))
    depth_mmap = np.lib.format.open_memmap(
        str(depth_path), mode="w+", dtype=np.float32, shape=(n, h, w))

    # Resize and write
    for i, (rgb, dep) in enumerate(tqdm.tqdm(zip(rgbs, depths),
                                              total=n, desc="  Resizing",
                                              unit="img")):
        # RGB: resize with PIL (high quality)
        rgb_pil = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
        rgb_resized = rgb_pil.resize((w, h), Image.BILINEAR)
        rgb_mmap[i] = np.array(rgb_resized)

        # Depth: resize with PIL in float mode
        dep_pil = Image.fromarray(dep.astype(np.float32), mode="F")
        dep_resized = dep_pil.resize((w, h), Image.BILINEAR)
        depth_mmap[i] = np.array(dep_resized)

    # Flush to disk
    rgb_mmap.flush()
    depth_mmap.flush()

    size_gb = (rgb_path.stat().st_size + depth_path.stat().st_size) / (1024**3)
    log.info(f"Saved: {rgb_path.name} + {depth_path.name} "
             f"({n} samples, {size_gb:.2f} GB total)")


def main():
    """Entry point: preprocess train + val splits."""
    from src.config import ROOT, get_nyu_dataset_path

    # Parse resolution from CLI args
    if len(sys.argv) >= 3:
        h, w = int(sys.argv[1]), int(sys.argv[2])
    else:
        h, w = 192, 256

    out_dir = _get_output_dir()
    log.info(f"Output directory: {out_dir}")
    log.info(f"Target resolution: {h}x{w}")

    # Resolve tar directory
    nyu_path = get_nyu_dataset_path()
    tar_dir = None
    if nyu_path:
        tar_dir = Path(nyu_path)
        if not tar_dir.is_absolute():
            tar_dir = ROOT / tar_dir

    # Process both splits
    preprocess_split("train", h, w, out_dir, tar_dir)
    preprocess_split("val", h, w, out_dir, tar_dir)

    log.info("Done! You can now train with data.use_mmap=true")


if __name__ == "__main__":
    main()
