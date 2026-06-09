"""One-time preprocessing: convert NYU tar/h5 data to memory-mapped .npy files.

Streaming architecture — never loads all samples into RAM at once.
Uses multiprocess workers for parallel resize. Supports 45k+ samples
on machines with as little as 2 GB free RAM.

Creates two files per split (train/val):
  - nyu_{split}_rgb_{H}x{W}_N{count}.npy   → (N, H, W, 3) uint8
  - nyu_{split}_depth_{H}x{W}_N{count}.npy → (N, H, W) float32

Usage:
  uv run python -m src.preprocess [resolution_h] [resolution_w]
  uv run python -m src.preprocess              # default: 192 256
  uv run python -m src.preprocess 192 256      # explicit
  uv run python -m src.preprocess --force      # rebuild even if files exist

The output directory is controlled by NYU_MMAP_DIR env var or defaults to datasets/nyu_mmap.
"""

import glob
import io
import logging
import os
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from tqdm.auto import tqdm
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Cap workers to avoid disk I/O contention
_MAX_WORKERS = min(os.cpu_count() or 1, 8)


def _get_output_dir() -> Path:
    """Resolve output directory for mmap files."""
    from src.config import ROOT, get_nyu_mmap_dir
    mmap_dir = get_nyu_mmap_dir()
    p = Path(mmap_dir)
    if not p.is_absolute():
        p = ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 1: Count samples (fast, no data read)
# ═══════════════════════════════════════════════════════════════════════════


def _count_samples_in_tars(tar_dir: Path, split: str) -> tuple[int, list[str]]:
    """Count total .h5 members across all tar shards. Returns (count, tar_paths)."""
    pattern = f"{split}-*.tar"
    tar_files = sorted(glob.glob(str(tar_dir / pattern)))
    if not tar_files:
        raise FileNotFoundError(
            f"No {pattern} files in {tar_dir}. "
            f"Expected NYU tar archives with .h5 files inside."
        )

    total = 0
    for tar_path in tar_files:
        with tarfile.open(tar_path, "r") as tf:
            total += sum(1 for m in tf if m.name.endswith(".h5"))

    return total, tar_files


def _count_samples_in_hf(split: str) -> int:
    """Count samples in HuggingFace dataset (requires download/cache)."""
    from datasets import load_dataset
    from src.config import HF_TOKEN

    hf_split = "validation" if split == "val" else split
    ds = load_dataset(
        "sayakpaul/nyu_depth_v2",
        split=hf_split,
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    return len(ds)


# ═══════════════════════════════════════════════════════════════════════════
#  Worker function (stateless, picklable for multiprocessing)
# ═══════════════════════════════════════════════════════════════════════════


def _resize_sample(h5_bytes: bytes, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode h5 bytes and resize to target resolution. Runs in worker process."""
    with h5py.File(io.BytesIO(h5_bytes), "r") as h5f:
        rgb = np.array(h5f["rgb"])      # (3, H, W) uint8
        depth = np.array(h5f["depth"])  # (H, W) float32

    rgb = np.transpose(rgb, (1, 2, 0))  # (H, W, 3)

    # RGB resize
    rgb_pil = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    rgb_resized = np.array(rgb_pil.resize((w, h), Image.BILINEAR))

    # Depth resize
    dep_pil = Image.fromarray(depth.astype(np.float32), mode="F")
    dep_resized = np.array(dep_pil.resize((w, h), Image.BILINEAR))

    return rgb_resized, dep_resized


def _resize_sample_from_pil(rgb_np: np.ndarray, depth_np: np.ndarray,
                            h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Resize already-decoded arrays. Used for HuggingFace path."""
    rgb_pil = Image.fromarray(rgb_np.astype(np.uint8), mode="RGB")
    rgb_resized = np.array(rgb_pil.resize((w, h), Image.BILINEAR))

    dep_pil = Image.fromarray(depth_np.astype(np.float32), mode="F")
    dep_resized = np.array(dep_pil.resize((w, h), Image.BILINEAR))

    return rgb_resized, dep_resized


# ═══════════════════════════════════════════════════════════════════════════
#  Check for existing files (glob-based)
# ═══════════════════════════════════════════════════════════════════════════


def _find_existing(out_dir: Path, split: str, h: int, w: int
                   ) -> tuple[Path | None, Path | None, int | None]:
    """Find existing mmap files and extract sample count from filename."""
    rgb_matches = sorted(out_dir.glob(f"nyu_{split}_rgb_{h}x{w}_N*.npy"))
    depth_matches = sorted(out_dir.glob(f"nyu_{split}_depth_{h}x{w}_N*.npy"))

    if rgb_matches and depth_matches:
        rgb_path = rgb_matches[0]
        depth_path = depth_matches[0]
        # Parse N from filename: nyu_train_rgb_192x256_N4433.npy
        try:
            n_str = rgb_path.stem.split("_N")[-1]
            n = int(n_str)
        except (ValueError, IndexError):
            n = None
        return rgb_path, depth_path, n

    # Also check old format (without N)
    old_rgb = out_dir / f"nyu_{split}_rgb_{h}x{w}.npy"
    old_depth = out_dir / f"nyu_{split}_depth_{h}x{w}.npy"
    if old_rgb.exists() and old_depth.exists():
        existing = np.load(str(old_rgb), mmap_mode="r")
        return old_rgb, old_depth, existing.shape[0]

    return None, None, None


# ═══════════════════════════════════════════════════════════════════════════
#  Main preprocessing logic
# ═══════════════════════════════════════════════════════════════════════════


def preprocess_split(split: str, h: int, w: int, out_dir: Path,
                     tar_dir: Path | None = None, force: bool = False):
    """Preprocess one split into memory-mapped .npy files (streaming, parallel)."""

    # ── Count samples ─────────────────────────────────────────────────
    if tar_dir and tar_dir.exists():
        n_total, tar_files = _count_samples_in_tars(tar_dir, split)
        source = "tar"
    else:
        n_total = _count_samples_in_hf(split)
        tar_files = []
        source = "hf"

    log.info(f"Split '{split}': {n_total} samples found ({source})")

    # ── Check existing ────────────────────────────────────────────────
    existing_rgb, existing_depth, existing_n = _find_existing(out_dir, split, h, w)
    if existing_rgb and existing_n == n_total and not force:
        log.info(f"  Already exists with {existing_n} samples. Skipping. "
                 f"(Use --force to rebuild)")
        return

    # Remove old files if count differs or forcing
    if existing_rgb and existing_rgb.exists():
        log.info(f"  Removing old file: {existing_rgb.name}")
        existing_rgb.unlink()
    if existing_depth and existing_depth.exists():
        log.info(f"  Removing old file: {existing_depth.name}")
        existing_depth.unlink()

    # ── Create mmap files at exact size ───────────────────────────────
    rgb_path = out_dir / f"nyu_{split}_rgb_{h}x{w}_N{n_total}.npy"
    depth_path = out_dir / f"nyu_{split}_depth_{h}x{w}_N{n_total}.npy"

    log.info(f"  Creating: {rgb_path.name} + {depth_path.name}")

    rgb_mmap = np.lib.format.open_memmap(
        str(rgb_path), mode="w+", dtype=np.uint8, shape=(n_total, h, w, 3))
    depth_mmap = np.lib.format.open_memmap(
        str(depth_path), mode="w+", dtype=np.float32, shape=(n_total, h, w))

    # ── Stream + parallel resize ──────────────────────────────────────
    if source == "tar":
        _stream_from_tars(tar_files, rgb_mmap, depth_mmap, h, w, n_total)
    else:
        _stream_from_hf(split, rgb_mmap, depth_mmap, h, w, n_total)

    # Flush to disk
    rgb_mmap.flush()
    depth_mmap.flush()

    size_gb = (rgb_path.stat().st_size + depth_path.stat().st_size) / (1024**3)
    log.info(f"  Done: {n_total} samples, {size_gb:.2f} GB total")


def _stream_from_tars(tar_files: list[str], rgb_mmap: np.ndarray,
                      depth_mmap: np.ndarray, h: int, w: int, n_total: int):
    """Stream samples from tar files with parallel resize workers."""
    n_workers = _MAX_WORKERS
    log.info(f"  Streaming from {len(tar_files)} tars with {n_workers} resize workers...")

    idx = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit in batches to control memory (queue at most 2*workers futures)
        batch_size = n_workers * 4
        futures = {}
        pbar = tqdm(total=n_total, desc="  Processing", unit="img")

        for tar_path in tar_files:
            with tarfile.open(tar_path, "r") as tf:
                members = [m for m in tf if m.name.endswith(".h5")]
                for member in members:
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    h5_bytes = f.read()
                    f.close()

                    future = executor.submit(_resize_sample, h5_bytes, h, w)
                    futures[future] = idx
                    idx += 1

                    # Drain completed futures when batch is full
                    if len(futures) >= batch_size:
                        _drain_futures(futures, rgb_mmap, depth_mmap, pbar)

        # Drain remaining
        _drain_futures(futures, rgb_mmap, depth_mmap, pbar, wait_all=True)
        pbar.close()


def _drain_futures(futures: dict, rgb_mmap: np.ndarray, depth_mmap: np.ndarray,
                   pbar, wait_all: bool = False):
    """Collect completed futures and write results to mmap."""
    if wait_all:
        done = list(as_completed(futures))
    else:
        # Wait for at least half to complete
        done = []
        for future in as_completed(futures):
            done.append(future)
            if len(done) >= len(futures) // 2:
                break

    for future in done:
        i = futures.pop(future)
        rgb_resized, dep_resized = future.result()
        rgb_mmap[i] = rgb_resized
        depth_mmap[i] = dep_resized
        pbar.update(1)


def _stream_from_hf(split: str, rgb_mmap: np.ndarray, depth_mmap: np.ndarray,
                    h: int, w: int, n_total: int):
    """Stream samples from HuggingFace with parallel resize workers."""
    from datasets import load_dataset
    from src.config import HF_TOKEN

    hf_split = "validation" if split == "val" else split
    log.info(f"  Streaming from HuggingFace ({hf_split}) with {_MAX_WORKERS} workers...")

    ds = load_dataset(
        "sayakpaul/nyu_depth_v2",
        split=hf_split,
        trust_remote_code=True,
        token=HF_TOKEN,
    )

    n_workers = _MAX_WORKERS
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        batch_size = n_workers * 4
        futures = {}
        pbar = tqdm(total=n_total, desc="  Processing", unit="img")

        for i, row in enumerate(ds):
            rgb_np = np.array(row["image"].convert("RGB"))
            depth_np = np.array(row["depth_map"])

            future = executor.submit(_resize_sample_from_pil, rgb_np, depth_np, h, w)
            futures[future] = i

            if len(futures) >= batch_size:
                _drain_futures(futures, rgb_mmap, depth_mmap, pbar)

        _drain_futures(futures, rgb_mmap, depth_mmap, pbar, wait_all=True)
        pbar.close()


# ═══════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════════


def main():
    """Entry point: preprocess train + val splits."""
    from src.config import ROOT, get_nyu_dataset_path

    # Parse CLI args
    force = "--force" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if len(args) >= 2:
        h, w = int(args[0]), int(args[1])
    else:
        h, w = 192, 256

    out_dir = _get_output_dir()
    log.info(f"Output directory: {out_dir}")
    log.info(f"Target resolution: {h}x{w}")
    log.info(f"Workers: {_MAX_WORKERS}")
    if force:
        log.info("Force mode: will rebuild even if files exist")

    # Resolve tar directory
    nyu_path = get_nyu_dataset_path()
    tar_dir = None
    if nyu_path:
        tar_dir = Path(nyu_path)
        if not tar_dir.is_absolute():
            tar_dir = ROOT / tar_dir

    # Process both splits
    preprocess_split("train", h, w, out_dir, tar_dir, force=force)
    preprocess_split("val", h, w, out_dir, tar_dir, force=force)

    log.info("\nDone! You can now train with data.use_mmap=true")


if __name__ == "__main__":
    main()
