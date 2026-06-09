"""Dataset loading for LeJEPA pre-training (RGB multi-view) and depth fine-tuning.

Supports:
- NYU Depth V2: local tar/h5 files (set NYU_DATASET_PATH) or HuggingFace streaming
- KITTI: placeholder (TO IMPLEMENT)

Environment variables:
- NYU_DATASET_PATH: path to directory with train-*.tar files (each containing .h5 samples).
                    If set, loads locally. If unset, falls back to HuggingFace streaming.
"""

import abc
import glob
import io
import logging
import os
import random
import tarfile

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm 
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from omegaconf import DictConfig

from src.config import HF_TOKEN, HF_OFFLINE, get_nyu_dataset_path, DEVICE

log = logging.getLogger(__name__)

# Windows spawn-based multiprocessing copies the full dataset per worker,
# causing OOM with large in-memory datasets. Use 0 on Windows.
import sys
_NUM_WORKERS = 0 if sys.platform == "win32" else min(4, os.cpu_count() or 1)


# ═══════════════════════════════════════════════════════════════════════════
#  Transforms
# ═══════════════════════════════════════════════════════════════════════════


def _get_pretrain_transforms(resolution: int) -> v2.Compose:
    """Multi-view augmentation pipeline for LeJEPA pre-training (RGB only)."""
    return v2.Compose([
        v2.RandomResizedCrop(resolution, scale=(0.08, 1.0)),
        v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),
        v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _get_depth_transforms(resolution: tuple[int, int] | int, train: bool = True):
    """Get transforms for depth fine-tuning.

    In validation mode: just resize + normalize.
    In training mode: full METER augmentation is applied in __getitem__ (numpy-level),
    so transforms here only handle resize + ToTensor + normalize.

    Args:
        resolution: (H, W) tuple or single int for square crop.
        train: whether to include training augmentations (handled externally).
    """
    if isinstance(resolution, int):
        resolution = (resolution, resolution)

    rgb_transform = v2.Compose([
        v2.Resize(resolution),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    depth_transform = v2.Compose([
        v2.Resize(resolution),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=False),
    ])
    return rgb_transform, depth_transform


def _meter_augmentation(img: np.ndarray, depth: np.ndarray, p: float = 0.5,
                        depth_shift_range: int = 10
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Full METER data augmentation policy (numpy-level, before ToTensor).

    Applied to both RGB image (H,W,3 uint8/float) and depth map (H,W float).
    All transforms applied with probability p.

    Includes:
    - Vertical flip
    - Horizontal mirror
    - Channel swap (RGB only)
    - Shifting strategy: gamma/brightness/color on RGB + depth shift
    """
    # Ensure float for image processing
    img = img.astype(np.float64) / 255.0 if img.dtype == np.uint8 else img.astype(np.float64)

    # Random vertical flip
    if random.random() < p:
        img = img[::-1, :, :].copy()
        depth = depth[::-1, :].copy()

    # Random horizontal mirror
    if random.random() < p:
        img = img[:, ::-1, :].copy()
        depth = depth[:, ::-1].copy()

    # Channel swap (RGB only)
    if random.random() < p:
        perm = list(np.random.permutation(3))
        img = img[:, :, perm]

    # Shifting strategy
    if random.random() < p:
        # Gamma + brightness augmentation
        gamma = random.uniform(0.9, 1.1)
        brightness = random.uniform(0.9, 1.1)
        img = np.clip(img, 0, None)  # Ensure non-negative before power
        img = brightness * (img ** gamma)

        # Color augmentation (per-channel scaling)
        colors = np.random.uniform(0.9, 1.1, size=3)
        img = img * colors[np.newaxis, np.newaxis, :]
        img = np.clip(img, 0, 1.0)

        # Depth shift (±10cm = ±0.1m for meters unit)
        shift = random.uniform(-depth_shift_range, depth_shift_range) / 100.0
        depth = depth + shift

    # Convert back to uint8 for PIL
    img = (np.clip(img, 0, 1.0) * 255).astype(np.uint8)
    return img, depth


# ═══════════════════════════════════════════════════════════════════════════
#  Abstract Base Classes
# ═══════════════════════════════════════════════════════════════════════════


class BasePretrainDataset(Dataset, abc.ABC):
    """Base class for pre-training datasets (RGB multi-view augmentation).

    Contract: __getitem__ returns (views: Tensor[V, 3, H, W], label: int).
    Subclasses implement _load_images() -> list[PIL.Image].
    """

    def __init__(self, n_samples: int, n_views: int = 4, resolution: int = 128):
        self.n_views = n_views
        self.resolution = resolution
        self.aug = _get_pretrain_transforms(resolution)
        self._images = self._load_images(n_samples)
        log.info(f"{len(self._images)} pretrain samples loaded.")

    @abc.abstractmethod
    def _load_images(self, n_samples: int) -> list:
        """Load up to n_samples RGB PIL images."""
        ...

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = self._images[idx]
        views = torch.stack([self.aug(img) for _ in range(self.n_views)])
        return views, 0  # label unused, kept for API consistency


class BaseDepthDataset(Dataset, abc.ABC):
    """Base class for depth fine-tuning datasets (RGB + depth pairs).

    Contract: __getitem__ returns (rgb: Tensor[3, H, W], depth: Tensor[1, H, W]).
    Subclasses implement _load_samples() -> list[tuple[PIL.Image, np.ndarray]].
    """

    def __init__(self, n_samples: int, resolution: tuple[int, int] | int = 128,
                 train: bool = True, augment: bool = True):
        self.resolution = resolution
        self.train = train
        self.augment = augment and train
        self.rgb_transform, self.depth_transform = _get_depth_transforms(
            resolution, train=train)
        self._samples = self._load_samples(n_samples)
        log.info(f"{len(self._samples)} depth samples loaded ({'train' if train else 'val'}).")

    @abc.abstractmethod
    def _load_samples(self, n_samples: int) -> list:
        """Load up to n_samples (PIL.Image, np.ndarray) tuples."""
        ...

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, depth = self._samples[idx]

        # Convert PIL to numpy for augmentation
        img_np = np.array(img)  # (H, W, 3) uint8
        depth_np = depth.copy()

        # Apply METER augmentation (numpy-level, before tensor conversion)
        if self.augment:
            img_np, depth_np = _meter_augmentation(img_np, depth_np)

        # Convert back to PIL for torchvision transforms
        img_pil = Image.fromarray(img_np.astype(np.uint8), mode="RGB")
        rgb = self.rgb_transform(img_pil)

        # Depth: convert to PIL for resize, then to tensor
        depth_pil = Image.fromarray(depth_np.astype(np.float32), mode="F")
        depth_tensor = self.depth_transform(depth_pil)  # (1, H, W)
        return rgb, depth_tensor


# ═══════════════════════════════════════════════════════════════════════════
#  NYU Depth V2 — Local tar/h5 loading
# ═══════════════════════════════════════════════════════════════════════════


def _load_nyu_local(path: str, n_samples: int, include_depth: bool = False,
                    split: str = "train"):
    """Load NYU samples from local tar archives containing .h5 files.

    Each .h5 has keys: "rgb" (3, H, W) uint8 and "depth" (H, W) float.
    Relative paths are resolved against the project root.
    """
    from src.config import ROOT
    from pathlib import Path

    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved

    pattern = f"{split}-*.tar"
    tar_files = sorted(glob.glob(str(resolved / pattern)))
    if not tar_files:
        raise FileNotFoundError(
            f"No {pattern} files found in {resolved}. "
            f"Expected NYU Depth V2 tar archives with .h5 samples inside."
        )
    log.info(f"Loading NYU locally from {len(tar_files)} tar shards in {resolved}...")

    results = []
    for tar_path in tar_files:
        if len(results) >= n_samples:
            break
        with tarfile.open(tar_path, "r") as tf:
            members = [m for m in tf if m.name.endswith(".h5")]
            cap = min(len(members), n_samples - len(results))
            for member in tqdm(members[:cap],
                                    desc=f"  Loading {Path(tar_path).name}",
                                    unit="img"):
                f = tf.extractfile(member)
                if f is None:
                    continue
                h5_bytes = f.read()
                f.close()
                h5f = h5py.File(io.BytesIO(h5_bytes), "r")
                rgb = np.array(h5f["rgb"])            # (3, H, W) uint8
                rgb = np.transpose(rgb, (1, 2, 0))   # (H, W, 3)
                rgb_pil = Image.fromarray(rgb.astype(np.uint8), mode="RGB")

                if include_depth:
                    depth = np.array(h5f["depth"])    # (H, W) float
                    results.append((rgb_pil, depth))
                else:
                    results.append(rgb_pil)
                h5f.close()

    log.info(f"Loaded {len(results)} NYU samples from local tar files.")
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  NYU Depth V2 — HuggingFace streaming
# ═══════════════════════════════════════════════════════════════════════════


def _load_nyu_hf(n_samples: int, include_depth: bool = False,
                 token: str | None = HF_TOKEN, offline: bool = HF_OFFLINE,
                 split: str = "train"):
    """Load NYU samples from HuggingFace (streaming/cached)."""
    import os
    from datasets import load_dataset

    if offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        log.info(f"Loading {n_samples} NYU {split} samples from local cache (offline)...")
    else:
        os.environ.pop("HF_DATASETS_OFFLINE", None)
        log.info(f"Loading {n_samples} NYU {split} samples from HuggingFace (streaming)...")

    hf_split = "validation" if split == "val" else split
    stream = load_dataset(
        "sayakpaul/nyu_depth_v2",
        split=hf_split,
        streaming=True,
        trust_remote_code=True,
        token=token,
    )

    results = []
    pbar = tqdm(total=n_samples, desc=f"  Loading NYU {hf_split} (HF)",
                     unit="img")
    for i, row in enumerate(stream):
        if i >= n_samples:
            break
        rgb_pil = row["image"].convert("RGB")
        if include_depth:
            depth = np.array(row["depth_map"])  # PIL → ndarray
            results.append((rgb_pil, depth))
        else:
            results.append(rgb_pil)
        pbar.update(1)
    pbar.close()

    log.info(f"Loaded {len(results)} NYU samples from HuggingFace.")
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  NYU Dataset Classes
# ═══════════════════════════════════════════════════════════════════════════


class NYUPretrainDataset(BasePretrainDataset):
    """NYU Depth V2 — RGB images with multi-view augmentation for LeJEPA.

    Loads from local tar/h5 (if NYU_DATASET_PATH set) or HuggingFace streaming.
    Depth maps are ignored — only RGB is used for self-supervised pre-training.
    """

    def _load_images(self, n_samples: int) -> list:
        nyu_path = get_nyu_dataset_path()
        if nyu_path:
            return _load_nyu_local(nyu_path, n_samples, include_depth=False)
        return _load_nyu_hf(n_samples, include_depth=False)


class NYUDepthDataset(BaseDepthDataset):
    """NYU Depth V2 — RGB + depth pairs for monocular depth fine-tuning.

    Loads from local tar/h5 (if NYU_DATASET_PATH set) or HuggingFace streaming.
    Supports train/val splits and METER augmentation policy.
    """

    def __init__(self, n_samples: int, resolution: tuple[int, int] | int = 128,
                 train: bool = True, augment: bool = True, split: str = "train"):
        self._split = split
        super().__init__(n_samples=n_samples, resolution=resolution,
                         train=train, augment=augment)

    def _load_samples(self, n_samples: int) -> list:
        nyu_path = get_nyu_dataset_path()
        if nyu_path:
            return _load_nyu_local(nyu_path, n_samples, include_depth=True,
                                   split=self._split)
        return _load_nyu_hf(n_samples, include_depth=True, split=self._split)


# ═══════════════════════════════════════════════════════════════════════════
#  Cached NYU Dataset (pre-resized .pt tensors for fast loading)
# ═══════════════════════════════════════════════════════════════════════════


class CachedNYUDepthDataset(Dataset):
    """NYU Depth V2 with pre-processed tensor cache for fast training.

    On first use, converts tar/h5 data to pre-resized .pt files.
    Subsequent runs load directly from cache — no tar/h5/PIL overhead.

    Cache structure: {cache_dir}/{split}_{H}x{W}/sample_{idx}.pt
    Each .pt file contains: {"rgb": Tensor[3,H,W], "depth": Tensor[1,H,W]}
    """

    def __init__(self, n_samples: int, resolution: tuple[int, int] | int = 128,
                 train: bool = True, augment: bool = True, split: str = "train",
                 cache_dir: str | None = None):
        from pathlib import Path

        self.train = train
        self.augment = augment and train
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        self.resolution = resolution
        self._split = split

        # Determine cache directory
        if cache_dir is None:
            from src.config import ROOT
            cache_dir = str(ROOT / "datasets" / "nyu_cache")
        res_str = f"{resolution[0]}x{resolution[1]}"
        self._cache_path = Path(cache_dir) / f"{split}_{res_str}"

        # Build or load cache
        if not self._cache_path.exists() or len(list(self._cache_path.glob("*.pt"))) == 0:
            self._build_cache(n_samples)

        # Load file list (cap to n_samples)
        all_files = sorted(self._cache_path.glob("*.pt"))
        self._files = all_files[:n_samples] if n_samples < len(all_files) else all_files
        log.info(f"{len(self._files)} cached depth samples loaded "
                 f"({'train' if train else 'val'}, {res_str}).")

        # ImageNet normalization (applied on cached tensors)
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def _build_cache(self, n_samples: int):
        """One-time preprocessing: load from tar/h5, resize, save as .pt."""
        from pathlib import Path
        import torch.nn.functional as F

        log.info(f"Building tensor cache at {self._cache_path} ...")
        self._cache_path.mkdir(parents=True, exist_ok=True)

        # Load raw samples
        nyu_path = get_nyu_dataset_path()
        if nyu_path:
            samples = _load_nyu_local(nyu_path, n_samples, include_depth=True,
                                      split=self._split)
        else:
            samples = _load_nyu_hf(n_samples, include_depth=True, split=self._split)

        h, w = self.resolution
        for i, (img_pil, depth_np) in enumerate(samples):
            # RGB: PIL → tensor [0,1] → resize
            img_np = np.array(img_pil).astype(np.float32) / 255.0
            rgb = torch.from_numpy(img_np).permute(2, 0, 1)  # (3, H_orig, W_orig)
            rgb = F.interpolate(rgb.unsqueeze(0), size=(h, w),
                                mode="bilinear", align_corners=False).squeeze(0)

            # Depth: numpy → tensor → resize
            depth = torch.from_numpy(depth_np.astype(np.float32)).unsqueeze(0)
            depth = F.interpolate(depth.unsqueeze(0), size=(h, w),
                                  mode="bilinear", align_corners=False).squeeze(0)

            torch.save({"rgb": rgb, "depth": depth},
                       self._cache_path / f"sample_{i:06d}.pt")

        log.info(f"Cache built: {len(samples)} samples saved to {self._cache_path}")

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = torch.load(self._files[idx], weights_only=True)
        rgb = data["rgb"]    # (3, H, W) float32 in [0, 1]
        depth = data["depth"]  # (1, H, W) float32

        # Fast tensor-space augmentation (no PIL/numpy conversion)
        if self.augment:
            rgb, depth = self._tensor_augment(rgb, depth)

        # Normalize RGB (ImageNet stats)
        rgb = (rgb - self._mean) / self._std
        return rgb, depth

    def _tensor_augment(self, rgb: torch.Tensor, depth: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor]:
        """METER augmentation in tensor space (fast, no numpy)."""
        # Random vertical flip
        if random.random() < 0.5:
            rgb = rgb.flip(-2)
            depth = depth.flip(-2)

        # Random horizontal flip
        if random.random() < 0.5:
            rgb = rgb.flip(-1)
            depth = depth.flip(-1)

        # Channel swap
        if random.random() < 0.5:
            perm = torch.randperm(3)
            rgb = rgb[perm]

        # Brightness/gamma shift
        if random.random() < 0.5:
            gamma = random.uniform(0.9, 1.1)
            brightness = random.uniform(0.9, 1.1)
            rgb = (brightness * rgb.clamp(min=0).pow(gamma)).clamp(0, 1)
            # Per-channel color
            colors = torch.empty(3, 1, 1).uniform_(0.9, 1.1)
            rgb = (rgb * colors).clamp(0, 1)
            # Depth shift
            shift = random.uniform(-0.1, 0.1)
            depth = depth + shift

        return rgb, depth


# ═══════════════════════════════════════════════════════════════════════════
#  NYU Depth V2 — Memory-Mapped Dataset (zero RAM overhead)
# ═══════════════════════════════════════════════════════════════════════════


class MmapNYUPretrainDataset(Dataset):
    """NYU Depth V2 — memory-mapped RGB for LeJEPA pre-training.

    Loads RGB images from preprocessed .npy files (mmap) and applies
    multi-view augmentation on-the-fly. Depth is ignored.

    Uses the finetune-resolution mmap files as source; the pretrain
    augmentation pipeline (RandomResizedCrop) handles final sizing.
    """

    def __init__(self, n_samples: int, n_views: int = 4, resolution: int = 128,
                 mmap_resolution: tuple[int, int] = (192, 256)):
        from pathlib import Path
        from src.config import ROOT, get_nyu_mmap_dir

        self.n_views = n_views
        self.aug = _get_pretrain_transforms(resolution)

        h, w = mmap_resolution
        mmap_dir = get_nyu_mmap_dir()
        p = Path(mmap_dir)
        if not p.is_absolute():
            p = ROOT / p

        rgb_path = p / f"nyu_train_rgb_{h}x{w}.npy"
        if not rgb_path.exists():
            raise FileNotFoundError(
                f"Memory-mapped RGB file not found at {rgb_path}. "
                f"Run preprocessing first:\n"
                f"  uv run python -m src.preprocess {h} {w}"
            )

        self._rgb = np.load(str(rgb_path), mmap_mode="r")  # (N, H, W, 3) uint8
        total = self._rgb.shape[0]
        self._n = min(n_samples, total)

        log.info(f"{self._n}/{total} mmap pretrain samples loaded ({h}x{w}).")

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rgb_np = np.array(self._rgb[idx])  # (H, W, 3) uint8 — copy from mmap
        img = Image.fromarray(rgb_np, mode="RGB")
        views = torch.stack([self.aug(img) for _ in range(self.n_views)])
        return views, 0


class MmapNYUDepthDataset(Dataset):
    """NYU Depth V2 using memory-mapped .npy files for near-zero RAM usage.

    Requires preprocessing via `uv run python -m src.preprocess`.
    Files expected:
      {mmap_dir}/nyu_{split}_rgb_{H}x{W}.npy   → (N, H, W, 3) uint8
      {mmap_dir}/nyu_{split}_depth_{H}x{W}.npy → (N, H, W) float32

    The OS pages in only the accessed samples — supports 45k+ samples
    with effectively zero RAM overhead beyond the current batch.
    """

    def __init__(self, n_samples: int, resolution: tuple[int, int] | int = (192, 256),
                 train: bool = True, augment: bool = True, split: str = "train"):
        from pathlib import Path
        from src.config import ROOT, get_nyu_mmap_dir

        self.train = train
        self.augment = augment and train
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        self.resolution = resolution
        h, w = resolution

        # Resolve mmap directory
        mmap_dir = get_nyu_mmap_dir()
        p = Path(mmap_dir)
        if not p.is_absolute():
            p = ROOT / p

        rgb_path = p / f"nyu_{split}_rgb_{h}x{w}.npy"
        depth_path = p / f"nyu_{split}_depth_{h}x{w}.npy"

        if not rgb_path.exists() or not depth_path.exists():
            raise FileNotFoundError(
                f"Memory-mapped files not found at {p}. "
                f"Run preprocessing first:\n"
                f"  uv run python -m src.preprocess {h} {w}"
            )

        # Open as memory-mapped (read-only)
        self._rgb = np.load(str(rgb_path), mmap_mode="r")    # (N, H, W, 3)
        self._depth = np.load(str(depth_path), mmap_mode="r")  # (N, H, W)

        # Cap to n_samples
        total = self._rgb.shape[0]
        self._n = min(n_samples, total)

        # ImageNet normalization constants
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        log.info(f"{self._n}/{total} mmap depth samples loaded "
                 f"({'train' if train else 'val'}, {h}x{w}).")

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Read from mmap (triggers OS page-in for this sample only)
        rgb_np = np.array(self._rgb[idx])      # (H, W, 3) uint8 — copy from mmap
        depth_np = np.array(self._depth[idx])   # (H, W) float32 — copy from mmap

        # Apply METER augmentation (numpy-level)
        if self.augment:
            rgb_np, depth_np = _meter_augmentation(
                rgb_np, depth_np)
            # _meter_augmentation returns rgb as uint8

        # Convert to tensors
        rgb = torch.from_numpy(rgb_np.copy()).permute(2, 0, 1).float() / 255.0  # (3, H, W)
        depth = torch.from_numpy(depth_np.copy()).unsqueeze(0)  # (1, H, W)

        # Normalize RGB
        rgb = (rgb - self._mean) / self._std
        return rgb, depth


# ═══════════════════════════════════════════════════════════════════════════
#  KITTI — Stub (TO IMPLEMENT)
# ═══════════════════════════════════════════════════════════════════════════


class KITTIPretrainDataset(BasePretrainDataset):
    """KITTI — RGB images for pre-training. TO IMPLEMENT.

    Expected format: TBD (likely PNG images in directory structure).
    Set KITTI_DATASET_PATH env var when implemented.
    """

    def _load_images(self, n_samples: int) -> list:
        raise NotImplementedError(
            "KITTI pre-training dataset loading not yet implemented. "
            "Contributions welcome — see src/data.py for the interface."
        )


class KITTIDepthDataset(BaseDepthDataset):
    """KITTI — RGB + depth/LiDAR pairs for fine-tuning. TO IMPLEMENT.

    Expected format: TBD (likely PNG RGB + sparse depth maps).
    Set KITTI_DATASET_PATH env var when implemented.
    """

    def _load_samples(self, n_samples: int) -> list:
        raise NotImplementedError(
            "KITTI depth dataset loading not yet implemented. "
            "Contributions welcome — see src/data.py for the interface."
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Dataset Registry & Loader Factory
# ═══════════════════════════════════════════════════════════════════════════

_PRETRAIN_DATASETS = {
    "nyu": NYUPretrainDataset,
    "kitti": KITTIPretrainDataset,
}

_DEPTH_DATASETS = {
    "nyu": NYUDepthDataset,
    "kitti": KITTIDepthDataset,
}


def get_pretrain_loader(cfg: DictConfig, device: str | None = None) -> DataLoader:
    """Build the pre-training DataLoader from config.

    Dataset selection priority (same as fine-tuning):
    1. Memory-mapped .npy files (if use_mmap=true and files exist)
    2. Load from tar/h5 into RAM (default)

    Args:
        cfg: Hydra DictConfig (needs cfg.data.dataset, cfg.data.n_samples,
             cfg.n_views, cfg.resolution, cfg.bs).
        device: Override device for pin_memory decision.
    """
    dev = device or DEVICE
    dataset_name = cfg.data.dataset
    n_samples = cfg.data.n_samples

    if dataset_name not in _PRETRAIN_DATASETS:
        raise ValueError(f"Unknown pretrain dataset: {dataset_name}. "
                         f"Available: {list(_PRETRAIN_DATASETS.keys())}")

    # Check if mmap files are available (stored at finetune resolution)
    use_mmap = cfg.get("data", {}).get("use_mmap", False)
    mmap_resolution = tuple(cfg.get("finetune", {}).get("resolution", [192, 256]))

    if use_mmap and dataset_name == "nyu" and _mmap_files_exist(mmap_resolution, "train"):
        ds = MmapNYUPretrainDataset(
            n_samples=n_samples, n_views=cfg.n_views,
            resolution=cfg.resolution, mmap_resolution=mmap_resolution)
    else:
        ds_cls = _PRETRAIN_DATASETS[dataset_name]
        ds = ds_cls(n_samples=n_samples, n_views=cfg.n_views, resolution=cfg.resolution)

    return DataLoader(
        ds,
        batch_size=cfg.bs,
        shuffle=True,
        drop_last=True,
        num_workers=_NUM_WORKERS,
        persistent_workers=(_NUM_WORKERS > 0),
        pin_memory=(dev == "cuda"),
    )


def _mmap_files_exist(resolution: tuple[int, int] | int, split: str) -> bool:
    """Check if preprocessed memory-mapped .npy files exist for given resolution."""
    from pathlib import Path
    from src.config import ROOT, get_nyu_mmap_dir

    if isinstance(resolution, int):
        resolution = (resolution, resolution)
    h, w = resolution

    mmap_dir = get_nyu_mmap_dir()
    p = Path(mmap_dir)
    if not p.is_absolute():
        p = ROOT / p

    rgb_path = p / f"nyu_{split}_rgb_{h}x{w}.npy"
    depth_path = p / f"nyu_{split}_depth_{h}x{w}.npy"
    return rgb_path.exists() and depth_path.exists()


def get_depth_loader(cfg: DictConfig, device: str | None = None,
                     split: str = "train") -> DataLoader:
    """Build the depth fine-tuning DataLoader from config.

    Args:
        cfg: Hydra DictConfig (needs cfg.data.dataset, cfg.data.n_samples,
             cfg.finetune.resolution, cfg.bs or cfg.finetune.bs).
        device: Override device for pin_memory decision.
        split: "train" or "val".
    """
    dev = device or DEVICE
    dataset_name = cfg.data.dataset
    n_samples = cfg.data.n_samples

    # For validation, load all available samples (ignore n_samples cap)
    if split == "val":
        n_samples = 999_999

    if dataset_name not in _DEPTH_DATASETS:
        raise ValueError(f"Unknown depth dataset: {dataset_name}. "
                         f"Available: {list(_DEPTH_DATASETS.keys())}")

    # Resolution: prefer finetune.resolution, fallback to cfg.resolution
    resolution = cfg.get("finetune", {}).get("resolution", cfg.get("resolution", 128))
    if isinstance(resolution, (list, tuple)):
        resolution = tuple(resolution)

    batch_size = cfg.get("finetune", {}).get("bs", cfg.get("bs", 8))
    is_train = (split == "train")

    # Dataset selection priority:
    # 1. Memory-mapped .npy files (fastest, zero RAM overhead)
    # 2. Cached .pt tensors (fast, one file per sample)
    # 3. Load from tar/h5 into RAM (slowest, high RAM)
    use_mmap = cfg.get("data", {}).get("use_mmap", False)
    use_cache = cfg.get("data", {}).get("use_cache", False)

    if use_mmap and dataset_name == "nyu" and _mmap_files_exist(resolution, split):
        ds = MmapNYUDepthDataset(
            n_samples=n_samples, resolution=resolution,
            train=is_train, augment=is_train, split=split)
    elif use_cache and dataset_name == "nyu":
        cache_dir = cfg.get("data", {}).get("cache_dir", None)
        ds = CachedNYUDepthDataset(
            n_samples=n_samples, resolution=resolution,
            train=is_train, augment=is_train, split=split,
            cache_dir=cache_dir)
    else:
        ds_cls = _DEPTH_DATASETS[dataset_name]
        ds = ds_cls(n_samples=n_samples, resolution=resolution,
                    train=is_train, augment=is_train, split=split)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=is_train,
        drop_last=is_train,
        num_workers=_NUM_WORKERS,
        persistent_workers=(_NUM_WORKERS > 0),
        pin_memory=(dev == "cuda"),
    )
