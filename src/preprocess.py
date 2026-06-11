"""One-time preprocessing: convert dataset data to memory-mapped .npy files.

Supports NYU (tar/h5 or HuggingFace) and KITTI (Eigen split + depth zips).
Streaming architecture — never loads all samples into RAM at once.
Uses multiprocess workers for parallel resize.

Creates two files per split (train/val):
  NYU:   nyu_{split}_rgb_{H}x{W}_N{count}.npy   → (N, H, W, 3) uint8
         nyu_{split}_depth_{H}x{W}_N{count}.npy → (N, H, W) float32
  KITTI: kitti_{split}_rgb_{H}x{W}_N{count}.npy   → (N, H, W, 3) uint8
         kitti_{split}_depth_{H}x{W}_N{count}.npy → (N, H, W) float32

Usage:
  uv run python -m src.preprocess                                        # NYU default
  uv run python -m src.preprocess --dataset nyu 192 256                  # explicit NYU
  uv run python -m src.preprocess --dataset kitti                        # KITTI default 192x640
  uv run python -m src.preprocess --dataset kitti 192 640                # explicit KITTI
  uv run python -m src.preprocess --dataset nyu kitti                    # both datasets
  uv run python -m src.preprocess --force                                # rebuild

The output directory is controlled by NYU_MMAP_DIR / KITTI_MMAP_DIR env vars.
"""

import glob
import io
import logging
import os
import sys
import tarfile
import zipfile
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
    """Resolve output directory for NYU mmap files."""
    from src.config import ROOT, get_nyu_mmap_dir
    mmap_dir = get_nyu_mmap_dir()
    p = Path(mmap_dir)
    if not p.is_absolute():
        p = ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def _get_kitti_output_dir() -> Path:
    """Resolve output directory for KITTI mmap files."""
    from src.config import ROOT, get_kitti_mmap_dir
    mmap_dir = get_kitti_mmap_dir()
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
        log.warning(f"No {pattern} files in {tar_dir}. Skipping {split} split.")
        return 0, []

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

    if n_total == 0:
        return

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
#  KITTI Preprocessing — stream from Eigen split files + depth zip
# ═══════════════════════════════════════════════════════════════════════════


def _parse_eigen_split(split_file: Path) -> list[str]:
    """Parse Eigen split file, return list of image_02 relative paths.

    Each line: '{date}/{drive}/image_02/data/{frame}.jpg {date}/{drive}/image_03/data/{frame}.jpg'
    We use only the left camera (image_02).
    """
    paths = []
    with open(split_file) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                paths.append(parts[0])  # image_02 path
    return paths


def _rgb_to_depth_path(rgb_rel: str, depth_split: str = "train") -> str:
    """Map an Eigen RGB path to the corresponding depth ground truth path.

    RGB:   {date}/{date}_drive_{id}_sync/image_02/data/{frame}.jpg
    Depth: {depth_split}/{date}_drive_{id}_sync/proj_depth/groundtruth/image_02/{frame}.png
    """
    parts = rgb_rel.split("/")
    # parts: [date, drive_name, "image_02", "data", "frame.jpg"]
    drive_name = parts[1]
    frame = parts[-1].replace(".jpg", ".png")
    return f"{depth_split}/{drive_name}/proj_depth/groundtruth/image_02/{frame}"


def _resize_kitti_sample(rgb_bytes: bytes, depth_bytes: bytes,
                         h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode and resize KITTI RGB + depth. Uses NEAREST for depth (sparse)."""
    # RGB
    rgb_pil = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
    rgb_resized = np.array(rgb_pil.resize((w, h), Image.BILINEAR))

    # Depth: 16-bit PNG, decode as float32 / 256.0, pixel=0 is invalid
    depth_pil = Image.open(io.BytesIO(depth_bytes))
    depth_np = np.array(depth_pil, dtype=np.float32) / 256.0
    # Resize with NEAREST to preserve sparsity (no interpolation of invalid pixels)
    dep_pil = Image.fromarray(depth_np, mode="F")
    dep_resized = np.array(dep_pil.resize((w, h), Image.NEAREST))

    return rgb_resized, dep_resized


def _find_kitti_existing(out_dir: Path, split: str, h: int, w: int
                         ) -> tuple[Path | None, Path | None, int | None]:
    """Find existing KITTI mmap files and extract sample count from filename."""
    rgb_matches = sorted(out_dir.glob(f"kitti_{split}_rgb_{h}x{w}_N*.npy"))
    depth_matches = sorted(out_dir.glob(f"kitti_{split}_depth_{h}x{w}_N*.npy"))

    if rgb_matches and depth_matches:
        rgb_path = rgb_matches[0]
        depth_path = depth_matches[0]
        try:
            n_str = rgb_path.stem.split("_N")[-1]
            n = int(n_str)
        except (ValueError, IndexError):
            n = None
        return rgb_path, depth_path, n

    return None, None, None


def preprocess_kitti_split(split: str, h: int, w: int, out_dir: Path,
                           kitti_dir: Path, force: bool = False):
    """Preprocess one KITTI split into memory-mapped .npy files.

    Args:
        split: "train" or "val" (mapped to eigen_train_files.txt / eigen_test_files.txt)
        h, w: Target resolution
        out_dir: Output directory for mmap files
        kitti_dir: Path to datasets/kitti directory
        force: Rebuild even if files exist
    """
    # Map our split names to Eigen files
    if split == "train":
        split_file = kitti_dir / "eigen_train_files.txt"
    else:
        split_file = kitti_dir / "eigen_test_files.txt"

    if not split_file.exists():
        log.warning(f"  Split file not found: {split_file}. Skipping.")
        return

    rgb_paths = _parse_eigen_split(split_file)
    log.info(f"KITTI split '{split}': {len(rgb_paths)} RGB paths in Eigen split")

    # Locate depth zip
    depth_zip_path = kitti_dir / "data_depth_annotated.zip"
    if not depth_zip_path.exists():
        raise FileNotFoundError(
            f"KITTI depth zip not found: {depth_zip_path}\n"
            f"Download from: https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip"
        )

    # Locate RGB source: zip or extracted directory
    rgb_zip_path = None
    rgb_dir = None
    if split == "train":
        candidate_zip = kitti_dir / "eigen_train_files.zip"
        if candidate_zip.exists():
            rgb_zip_path = candidate_zip
        else:
            log.warning(f"  KITTI train RGB zip not found: {candidate_zip}. Skipping train split.")
            return
    else:
        # Test split: check for zip first, then extracted directory
        candidate_zip = kitti_dir / "eigen_test_files.zip"
        candidate_dir = kitti_dir / "eigen_test_files_only_rgb"
        if candidate_zip.exists():
            rgb_zip_path = candidate_zip
        elif candidate_dir.exists():
            rgb_dir = candidate_dir
        else:
            log.warning(
                f"  KITTI test RGB not found (checked {candidate_zip.name} and "
                f"{candidate_dir.name}/). Skipping val split."
            )
            return

    # Check existing
    existing_rgb, existing_depth, existing_n = _find_kitti_existing(out_dir, split, h, w)
    if existing_rgb and not force:
        log.info(f"  Already exists with {existing_n} samples. Skipping. "
                 f"(Use --force to rebuild)")
        return

    # Remove old files
    if existing_rgb and existing_rgb.exists():
        existing_rgb.unlink()
    if existing_depth and existing_depth.exists():
        existing_depth.unlink()

    # Open depth zip and find valid pairs (RGB + depth both exist)
    log.info("  Scanning for valid RGB+depth pairs...")
    depth_zf = zipfile.ZipFile(str(depth_zip_path), "r")
    depth_namelist = set(depth_zf.namelist())

    # For each RGB path, try to find depth in both train/ and val/ of depth zip
    valid_pairs: list[tuple[str, str]] = []  # (rgb_rel_path, depth_zip_path)
    for rgb_rel in rgb_paths:
        # Try both sub-directories in depth zip
        for depth_sub in ("train", "val"):
            dep_path = _rgb_to_depth_path(rgb_rel, depth_sub)
            if dep_path in depth_namelist:
                valid_pairs.append((rgb_rel, dep_path))
                break

    n_total = len(valid_pairs)
    log.info(f"  Found {n_total} valid pairs (out of {len(rgb_paths)} RGB paths)")

    if n_total == 0:
        depth_zf.close()
        log.warning("  No valid pairs found. Skipping.")
        return

    # Create mmap files
    rgb_mmap_path = out_dir / f"kitti_{split}_rgb_{h}x{w}_N{n_total}.npy"
    depth_mmap_path = out_dir / f"kitti_{split}_depth_{h}x{w}_N{n_total}.npy"

    log.info(f"  Creating: {rgb_mmap_path.name} + {depth_mmap_path.name}")

    rgb_mmap = np.lib.format.open_memmap(
        str(rgb_mmap_path), mode="w+", dtype=np.uint8, shape=(n_total, h, w, 3))
    depth_mmap = np.lib.format.open_memmap(
        str(depth_mmap_path), mode="w+", dtype=np.float32, shape=(n_total, h, w))

    # Open RGB source
    rgb_zf = None
    if rgb_zip_path:
        rgb_zf = zipfile.ZipFile(str(rgb_zip_path), "r")

    # Stream and process
    n_workers = _MAX_WORKERS
    log.info(f"  Processing with {n_workers} workers...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        batch_size = n_workers * 4
        futures = {}
        pbar = tqdm(total=n_total, desc=f"  KITTI {split}", unit="img")

        for idx, (rgb_rel, dep_zip_path) in enumerate(valid_pairs):
            # Read RGB bytes
            if rgb_zf:
                rgb_bytes = rgb_zf.read(rgb_rel)
            else:
                # Read from extracted directory
                rgb_file = rgb_dir / rgb_rel
                rgb_bytes = rgb_file.read_bytes()

            # Read depth bytes
            depth_bytes = depth_zf.read(dep_zip_path)

            future = executor.submit(_resize_kitti_sample, rgb_bytes, depth_bytes, h, w)
            futures[future] = idx

            if len(futures) >= batch_size:
                _drain_futures(futures, rgb_mmap, depth_mmap, pbar)

        _drain_futures(futures, rgb_mmap, depth_mmap, pbar, wait_all=True)
        pbar.close()

    # Cleanup
    if rgb_zf:
        rgb_zf.close()
    depth_zf.close()

    rgb_mmap.flush()
    depth_mmap.flush()

    size_gb = (rgb_mmap_path.stat().st_size + depth_mmap_path.stat().st_size) / (1024**3)
    log.info(f"  Done: {n_total} samples, {size_gb:.2f} GB total")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════════


def main():
    """Entry point: preprocess train + val splits."""
    from src.config import ROOT, get_nyu_dataset_path

    # Parse CLI args
    force = "--force" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    # Determine datasets (support multiple: --dataset nyu kitti)
    datasets = ["nyu"]
    if "--dataset" in sys.argv:
        ds_idx = sys.argv.index("--dataset") + 1
        datasets = []
        while ds_idx < len(sys.argv) and not sys.argv[ds_idx].startswith("--"):
            datasets.append(sys.argv[ds_idx])
            ds_idx += 1
        # Remove dataset values from positional args
        args = [a for a in args if a not in datasets]
    if not datasets:
        datasets = ["nyu"]

    log.info(f"Datasets: {', '.join(d.upper() for d in datasets)}")
    log.info(f"Workers: {_MAX_WORKERS}")
    if force:
        log.info("Force mode: will rebuild even if files exist")

    for ds in datasets:
        if ds == "kitti":
            if len(args) >= 2:
                h, w = int(args[0]), int(args[1])
            else:
                h, w = 192, 640

            out_dir = _get_kitti_output_dir()
            from src.config import get_kitti_dataset_path
            kitti_path = get_kitti_dataset_path()
            kitti_dir = Path(kitti_path)
            if not kitti_dir.is_absolute():
                kitti_dir = ROOT / kitti_dir

            log.info(f"\n{'='*50}")
            log.info(f"  KITTI — {h}x{w}")
            log.info(f"  Output: {out_dir}")
            log.info(f"{'='*50}")

            preprocess_kitti_split("train", h, w, out_dir, kitti_dir, force=force)
            preprocess_kitti_split("val", h, w, out_dir, kitti_dir, force=force)

        elif ds == "nyu":
            if len(args) >= 2:
                h, w = int(args[0]), int(args[1])
            else:
                h, w = 192, 256

            out_dir = _get_output_dir()

            log.info(f"\n{'='*50}")
            log.info(f"  NYU — {h}x{w}")
            log.info(f"  Output: {out_dir}")
            log.info(f"{'='*50}")

            # Resolve tar directory
            nyu_path = get_nyu_dataset_path()
            tar_dir = None
            if nyu_path:
                tar_dir = Path(nyu_path)
                if not tar_dir.is_absolute():
                    tar_dir = ROOT / tar_dir

            preprocess_split("train", h, w, out_dir, tar_dir, force=force)
            preprocess_split("val", h, w, out_dir, tar_dir, force=force)

        else:
            log.warning(f"Unknown dataset '{ds}'. Skipping.")

    log.info("\nDone! You can now train with data.use_mmap=true")


if __name__ == "__main__":
    main()
