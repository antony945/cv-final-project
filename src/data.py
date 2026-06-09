"""Dataset loading for LeJEPA pre-training (RGB multi-view) and depth fine-tuning.

Supports:
- NYU Depth V2: local tar/h5 files (set NYU_DATASET_PATH) or HuggingFace streaming
- KITTI: placeholder (TO IMPLEMENT)

Storage backends (selected via config):
- Normal: loads all data into RAM at startup (fast access, high memory)
- Mmap: memory-mapped .npy files (near-zero RAM, requires preprocessing)
- Cached: per-sample .pt files on disk (moderate speed, low RAM)

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
import sys
import tarfile

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from omegaconf import DictConfig

from src.config import HF_TOKEN, HF_OFFLINE, get_nyu_dataset_path, DEVICE

log = logging.getLogger(__name__)

# Windows spawn-based multiprocessing copies the full dataset per worker,
# causing OOM with large in-memory datasets. Use 0 on Windows.
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


# ═══════════════════════════════════════════════════════════════════════════
#  Abstract Base Classes
# ═══════════════════════════════════════════════════════════════════════════


class BasePretrainDataset(Dataset, abc.ABC):
    """Base class for pre-training datasets (RGB multi-view augmentation).

    Contract:
      - __getitem__ returns (views: Tensor[V, 3, H, W], label: int)
      - Subclasses implement _get_image(idx) and __len__()
      - _get_image(idx) returns a PIL.Image at any resolution;
        the augmentation pipeline (RandomResizedCrop) handles sizing.
    """

    def __init__(self, n_views: int = 4, resolution: int = 128):
        self.n_views = n_views
        self.resolution = resolution
        self.aug = _get_pretrain_transforms(resolution)

    @abc.abstractmethod
    def _get_image(self, idx: int) -> Image.Image:
        """Return a single RGB PIL image for the given index."""
        ...

    @abc.abstractmethod
    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = self._get_image(idx)
        views = torch.stack([self.aug(img) for _ in range(self.n_views)])
        return views, 0  # label unused, kept for DataLoader API consistency


class BaseDepthDataset(Dataset, abc.ABC):
    """Base class for depth fine-tuning datasets (RGB + depth pairs).

    Contract:
      - __getitem__ returns (rgb: Tensor[3, H, W], depth: Tensor[1, H, W])
      - Subclasses implement _get_sample(idx) and __len__()
      - _get_sample(idx) returns (rgb [3,H,W] float32 [0,1], depth [1,H,W] float32)
        already at the target resolution.
      - Augmentation and normalization are applied by the base class.
    """

    # ImageNet normalization constants (shared across all instances)
    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    _STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, resolution: tuple[int, int] | int, train: bool = True,
                 augment: bool = True):
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        self.resolution = resolution
        self.train = train
        self.augment = augment and train

    @abc.abstractmethod
    def _get_sample(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (rgb [3,H,W] float32 [0,1], depth [1,H,W] float32) at target resolution."""
        ...

    @abc.abstractmethod
    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rgb, depth = self._get_sample(idx)

        if self.augment:
            rgb, depth = self._augment(rgb, depth)

        # Normalize RGB (ImageNet stats)
        rgb = (rgb - self._MEAN) / self._STD
        return rgb, depth

    @staticmethod
    def _augment(rgb: torch.Tensor, depth: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """Unified METER augmentation in tensor space (fast, no numpy)."""
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

        # Brightness/gamma/color shift + depth shift
        if random.random() < 0.5:
            gamma = random.uniform(0.9, 1.1)
            brightness = random.uniform(0.9, 1.1)
            rgb = (brightness * rgb.clamp(min=0).pow(gamma)).clamp(0, 1)
            colors = torch.empty(3, 1, 1).uniform_(0.9, 1.1)
            rgb = (rgb * colors).clamp(0, 1)
            shift = random.uniform(-0.1, 0.1)
            depth = depth + shift

        return rgb, depth


# ═══════════════════════════════════════════════════════════════════════════
#  NYU Depth V2 — Raw data loaders (tar/h5 and HuggingFace)
# ═══════════════════════════════════════════════════════════════════════════


def _load_nyu_local(path: str, n_samples: int, include_depth: bool = False,
                    split: str = "train"):
    """Load NYU samples from local tar archives containing .h5 files.

    Each .h5 has keys: "rgb" (3, H, W) uint8 and "depth" (H, W) float.
    Relative paths are resolved against the project root.
    """
    from src.config import ROOT

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


def _load_nyu_hf(n_samples: int, include_depth: bool = False,
                 token: str | None = HF_TOKEN, offline: bool = HF_OFFLINE,
                 split: str = "train"):
    """Load NYU samples from HuggingFace (streaming/cached)."""
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
#  NYU — Normal (in-RAM) Datasets
# ═══════════════════════════════════════════════════════════════════════════


class NYUPretrainDataset(BasePretrainDataset):
    """NYU Depth V2 — RGB images loaded into RAM with multi-view augmentation.

    Loads from local tar/h5 (if NYU_DATASET_PATH set) or HuggingFace streaming.
    All images are held in memory as PIL objects.
    """

    def __init__(self, n_samples: int, n_views: int = 4, resolution: int = 128):
        super().__init__(n_views=n_views, resolution=resolution)
        nyu_path = get_nyu_dataset_path()
        if nyu_path:
            self._images = _load_nyu_local(nyu_path, n_samples, include_depth=False)
        else:
            self._images = _load_nyu_hf(n_samples, include_depth=False)
        log.info(f"{len(self._images)} pretrain samples loaded (in-RAM).")

    def __len__(self) -> int:
        return len(self._images)

    def _get_image(self, idx: int) -> Image.Image:
        return self._images[idx]


class NYUDepthDataset(BaseDepthDataset):
    """NYU Depth V2 — RGB + depth pairs loaded into RAM.

    All samples (PIL + depth numpy) are held in memory.
    Resize happens per-access in _get_sample via F.interpolate.
    """

    def __init__(self, n_samples: int, resolution: tuple[int, int] | int = (192, 256),
                 train: bool = True, augment: bool = True, split: str = "train"):
        super().__init__(resolution=resolution, train=train, augment=augment)
        nyu_path = get_nyu_dataset_path()
        if nyu_path:
            self._samples = _load_nyu_local(nyu_path, n_samples,
                                            include_depth=True, split=split)
        else:
            self._samples = _load_nyu_hf(n_samples, include_depth=True, split=split)
        log.info(f"{len(self._samples)} depth samples loaded "
                 f"({'train' if train else 'val'}, in-RAM).")

    def __len__(self) -> int:
        return len(self._samples)

    def _get_sample(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_pil, depth_np = self._samples[idx]
        h, w = self.resolution

        # RGB: PIL → tensor [0,1] → resize
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        rgb = torch.from_numpy(img_np).permute(2, 0, 1)  # (3, H_orig, W_orig)
        rgb = F.interpolate(rgb.unsqueeze(0), size=(h, w),
                            mode="bilinear", align_corners=False).squeeze(0)

        # Depth: numpy → tensor → resize
        depth = torch.from_numpy(depth_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        depth = F.interpolate(depth, size=(h, w),
                              mode="bilinear", align_corners=False).squeeze(0)

        return rgb, depth


# ═══════════════════════════════════════════════════════════════════════════
#  NYU — Memory-Mapped Datasets (zero RAM overhead)
# ═══════════════════════════════════════════════════════════════════════════


class MmapNYUPretrainDataset(BasePretrainDataset):
    """NYU Depth V2 — memory-mapped RGB for LeJEPA pre-training.

    Reads from preprocessed .npy mmap files (at finetune resolution).
    The pretrain augmentation pipeline (RandomResizedCrop) handles final sizing.
    Near-zero RAM regardless of dataset size.
    """

    def __init__(self, n_samples: int, n_views: int = 4, resolution: int = 128,
                 mmap_resolution: tuple[int, int] = (192, 256)):
        super().__init__(n_views=n_views, resolution=resolution)

        h, w = mmap_resolution
        rgb_path = _find_mmap_file("train", "rgb", h, w)
        if rgb_path is None:
            raise FileNotFoundError(
                f"Memory-mapped RGB file not found for {h}x{w}. "
                f"Run: uv run python -m src.preprocess {h} {w}"
            )

        self._rgb = np.load(str(rgb_path), mmap_mode="r")  # (N, H, W, 3) uint8
        self._n = min(n_samples, self._rgb.shape[0])
        log.info(f"{self._n}/{self._rgb.shape[0]} mmap pretrain samples ({h}x{w}).")

    def __len__(self) -> int:
        return self._n

    def _get_image(self, idx: int) -> Image.Image:
        rgb_np = np.array(self._rgb[idx])  # copy from mmap
        return Image.fromarray(rgb_np, mode="RGB")


class MmapNYUDepthDataset(BaseDepthDataset):
    """NYU Depth V2 — memory-mapped RGB + depth for fine-tuning.

    Reads from preprocessed .npy mmap files (already at target resolution).
    Near-zero RAM regardless of dataset size.
    """

    def __init__(self, n_samples: int, resolution: tuple[int, int] | int = (192, 256),
                 train: bool = True, augment: bool = True, split: str = "train"):
        super().__init__(resolution=resolution, train=train, augment=augment)

        h, w = self.resolution
        rgb_path = _find_mmap_file(split, "rgb", h, w)
        depth_path = _find_mmap_file(split, "depth", h, w)

        if rgb_path is None or depth_path is None:
            raise FileNotFoundError(
                f"Memory-mapped files not found for {split} {h}x{w}. "
                f"Run: uv run python -m src.preprocess {h} {w}"
            )

        self._rgb = np.load(str(rgb_path), mmap_mode="r")      # (N, H, W, 3) uint8
        self._depth = np.load(str(depth_path), mmap_mode="r")   # (N, H, W) float32
        self._n = min(n_samples, self._rgb.shape[0])
        log.info(f"{self._n}/{self._rgb.shape[0]} mmap depth samples "
                 f"({'train' if train else 'val'}, {h}x{w}).")

    def __len__(self) -> int:
        return self._n

    def _get_sample(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rgb_np = np.array(self._rgb[idx])      # (H, W, 3) uint8
        depth_np = np.array(self._depth[idx])   # (H, W) float32

        rgb = torch.from_numpy(rgb_np.copy()).permute(2, 0, 1).float() / 255.0
        depth = torch.from_numpy(depth_np.copy()).unsqueeze(0)
        return rgb, depth


# ═══════════════════════════════════════════════════════════════════════════
#  NYU — Cached Datasets (per-sample .pt files on disk)
# ═══════════════════════════════════════════════════════════════════════════


def _get_cache_path(cache_dir: str | None, split: str,
                    resolution: tuple[int, int], suffix: str = "") -> Path:
    """Resolve cache directory path for a given split and resolution."""
    from src.config import ROOT

    if cache_dir is None:
        cache_dir = str(ROOT / "datasets" / "nyu_cache")
    h, w = resolution
    return Path(cache_dir) / f"{split}_{h}x{w}{suffix}"


class CachedNYUPretrainDataset(BasePretrainDataset):
    """NYU Depth V2 — cached RGB .pt files for pre-training.

    On first use, loads tar/h5, resizes to mmap_resolution, saves as .pt.
    Subsequent runs load directly from cache — no tar/h5 decompression overhead.
    """

    def __init__(self, n_samples: int, n_views: int = 4, resolution: int = 128,
                 mmap_resolution: tuple[int, int] = (192, 256),
                 cache_dir: str | None = None):
        super().__init__(n_views=n_views, resolution=resolution)
        self._mmap_resolution = mmap_resolution
        self._cache_path = _get_cache_path(cache_dir, "train",
                                           mmap_resolution, suffix="_rgb")

        if not self._cache_path.exists() or not any(self._cache_path.glob("*.pt")):
            self._build_cache(n_samples)

        all_files = sorted(self._cache_path.glob("*.pt"))
        self._files = all_files[:n_samples] if n_samples < len(all_files) else all_files
        log.info(f"{len(self._files)} cached pretrain samples loaded.")

    def _build_cache(self, n_samples: int):
        """One-time: load from tar/h5, resize, save as .pt (uint8 tensor)."""
        log.info(f"Building pretrain cache at {self._cache_path} ...")
        self._cache_path.mkdir(parents=True, exist_ok=True)

        nyu_path = get_nyu_dataset_path()
        if nyu_path:
            images = _load_nyu_local(nyu_path, n_samples, include_depth=False)
        else:
            images = _load_nyu_hf(n_samples, include_depth=False)

        h, w = self._mmap_resolution
        for i, img_pil in enumerate(images):
            img_np = np.array(img_pil).astype(np.float32) / 255.0
            rgb = torch.from_numpy(img_np).permute(2, 0, 1)  # (3, H_orig, W_orig)
            rgb = F.interpolate(rgb.unsqueeze(0), size=(h, w),
                                mode="bilinear", align_corners=False).squeeze(0)
            # Store as uint8 to save disk space
            rgb_uint8 = (rgb.clamp(0, 1) * 255).byte()
            torch.save(rgb_uint8, self._cache_path / f"sample_{i:06d}.pt")

        log.info(f"Cache built: {len(images)} samples at {self._cache_path}")

    def __len__(self) -> int:
        return len(self._files)

    def _get_image(self, idx: int) -> Image.Image:
        rgb_uint8 = torch.load(self._files[idx], weights_only=True)  # (3, H, W) uint8
        rgb_np = rgb_uint8.permute(1, 2, 0).numpy()  # (H, W, 3)
        return Image.fromarray(rgb_np, mode="RGB")


class CachedNYUDepthDataset(BaseDepthDataset):
    """NYU Depth V2 — cached RGB+depth .pt files for fine-tuning.

    On first use, loads tar/h5, resizes, saves as .pt.
    Subsequent runs load directly from cache.
    Each .pt contains: {"rgb": Tensor[3,H,W] float32, "depth": Tensor[1,H,W] float32}
    """

    def __init__(self, n_samples: int, resolution: tuple[int, int] | int = (192, 256),
                 train: bool = True, augment: bool = True, split: str = "train",
                 cache_dir: str | None = None):
        super().__init__(resolution=resolution, train=train, augment=augment)
        self._cache_path = _get_cache_path(cache_dir, split, self.resolution)

        if not self._cache_path.exists() or not any(self._cache_path.glob("*.pt")):
            self._build_cache(n_samples, split)

        all_files = sorted(self._cache_path.glob("*.pt"))
        self._files = all_files[:n_samples] if n_samples < len(all_files) else all_files
        log.info(f"{len(self._files)} cached depth samples "
                 f"({'train' if train else 'val'}, "
                 f"{self.resolution[0]}x{self.resolution[1]}).")

    def _build_cache(self, n_samples: int, split: str):
        """One-time: load, resize, save as .pt (float32 tensors)."""
        log.info(f"Building depth cache at {self._cache_path} ...")
        self._cache_path.mkdir(parents=True, exist_ok=True)

        nyu_path = get_nyu_dataset_path()
        if nyu_path:
            samples = _load_nyu_local(nyu_path, n_samples,
                                      include_depth=True, split=split)
        else:
            samples = _load_nyu_hf(n_samples, include_depth=True, split=split)

        h, w = self.resolution
        for i, (img_pil, depth_np) in enumerate(samples):
            img_np = np.array(img_pil).astype(np.float32) / 255.0
            rgb = torch.from_numpy(img_np).permute(2, 0, 1)
            rgb = F.interpolate(rgb.unsqueeze(0), size=(h, w),
                                mode="bilinear", align_corners=False).squeeze(0)

            depth = torch.from_numpy(depth_np.astype(np.float32)).unsqueeze(0)
            depth = F.interpolate(depth.unsqueeze(0), size=(h, w),
                                  mode="bilinear", align_corners=False).squeeze(0)

            torch.save({"rgb": rgb, "depth": depth},
                       self._cache_path / f"sample_{i:06d}.pt")

        log.info(f"Cache built: {len(samples)} samples at {self._cache_path}")

    def __len__(self) -> int:
        return len(self._files)

    def _get_sample(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = torch.load(self._files[idx], weights_only=True)
        return data["rgb"], data["depth"]  # already (3,H,W) and (1,H,W) float32


# ═══════════════════════════════════════════════════════════════════════════
#  KITTI — Stubs (TO IMPLEMENT)
# ═══════════════════════════════════════════════════════════════════════════


class KITTIPretrainDataset(BasePretrainDataset):
    """KITTI — RGB images for pre-training. TO IMPLEMENT."""

    def __init__(self, n_samples: int, n_views: int = 4, resolution: int = 128):
        super().__init__(n_views=n_views, resolution=resolution)
        raise NotImplementedError("KITTI pretrain dataset not yet implemented.")

    def __len__(self) -> int:
        return 0

    def _get_image(self, idx: int) -> Image.Image:
        raise NotImplementedError


class MmapKITTIPretrainDataset(BasePretrainDataset):
    """KITTI — Memory-mapped RGB for pre-training. TO IMPLEMENT."""

    def __init__(self, n_samples: int, n_views: int = 4, resolution: int = 128,
                 mmap_resolution: tuple[int, int] = (192, 256)):
        super().__init__(n_views=n_views, resolution=resolution)
        raise NotImplementedError("KITTI mmap pretrain dataset not yet implemented.")

    def __len__(self) -> int:
        return 0

    def _get_image(self, idx: int) -> Image.Image:
        raise NotImplementedError


class CachedKITTIPretrainDataset(BasePretrainDataset):
    """KITTI — Cached .pt RGB for pre-training. TO IMPLEMENT."""

    def __init__(self, n_samples: int, n_views: int = 4, resolution: int = 128,
                 mmap_resolution: tuple[int, int] = (192, 256),
                 cache_dir: str | None = None):
        super().__init__(n_views=n_views, resolution=resolution)
        raise NotImplementedError("KITTI cached pretrain dataset not yet implemented.")

    def __len__(self) -> int:
        return 0

    def _get_image(self, idx: int) -> Image.Image:
        raise NotImplementedError


class KITTIDepthDataset(BaseDepthDataset):
    """KITTI — RGB + depth for fine-tuning. TO IMPLEMENT."""

    def __init__(self, n_samples: int, resolution: tuple[int, int] | int = (192, 256),
                 train: bool = True, augment: bool = True, split: str = "train"):
        super().__init__(resolution=resolution, train=train, augment=augment)
        raise NotImplementedError("KITTI depth dataset not yet implemented.")

    def __len__(self) -> int:
        return 0

    def _get_sample(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class MmapKITTIDepthDataset(BaseDepthDataset):
    """KITTI — Memory-mapped RGB + depth for fine-tuning. TO IMPLEMENT."""

    def __init__(self, n_samples: int, resolution: tuple[int, int] | int = (192, 256),
                 train: bool = True, augment: bool = True, split: str = "train"):
        super().__init__(resolution=resolution, train=train, augment=augment)
        raise NotImplementedError("KITTI mmap depth dataset not yet implemented.")

    def __len__(self) -> int:
        return 0

    def _get_sample(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class CachedKITTIDepthDataset(BaseDepthDataset):
    """KITTI — Cached .pt RGB + depth for fine-tuning. TO IMPLEMENT."""

    def __init__(self, n_samples: int, resolution: tuple[int, int] | int = (192, 256),
                 train: bool = True, augment: bool = True, split: str = "train",
                 cache_dir: str | None = None):
        super().__init__(resolution=resolution, train=train, augment=augment)
        raise NotImplementedError("KITTI cached depth dataset not yet implemented.")

    def __len__(self) -> int:
        return 0

    def _get_sample(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers — Mmap file discovery (glob-based, supports sample count in name)
# ═══════════════════════════════════════════════════════════════════════════


def _get_mmap_dir() -> Path:
    """Resolve the mmap directory path."""
    from src.config import ROOT, get_nyu_mmap_dir
    p = Path(get_nyu_mmap_dir())
    if not p.is_absolute():
        p = ROOT / p
    return p


def _find_mmap_file(split: str, kind: str, h: int, w: int) -> Path | None:
    """Find a mmap .npy file by glob, supporting both old and new naming.

    Naming formats:
      Old: nyu_{split}_{kind}_{H}x{W}.npy
      New: nyu_{split}_{kind}_{H}x{W}_N{count}.npy

    Returns the path if found, None otherwise.
    """
    p = _get_mmap_dir()
    # Try new format first (with sample count)
    matches = sorted(p.glob(f"nyu_{split}_{kind}_{h}x{w}_N*.npy"))
    if matches:
        return matches[0]
    # Fall back to old format (without count)
    old = p / f"nyu_{split}_{kind}_{h}x{w}.npy"
    if old.exists():
        return old
    return None


def _mmap_rgb_exists(resolution: tuple[int, int] | int, split: str) -> bool:
    """Check if preprocessed memory-mapped RGB file exists (pretrain only needs RGB)."""
    if isinstance(resolution, int):
        resolution = (resolution, resolution)
    h, w = resolution
    return _find_mmap_file(split, "rgb", h, w) is not None


def _mmap_files_exist(resolution: tuple[int, int] | int, split: str) -> bool:
    """Check if preprocessed memory-mapped RGB + depth files exist."""
    if isinstance(resolution, int):
        resolution = (resolution, resolution)
    h, w = resolution
    return (_find_mmap_file(split, "rgb", h, w) is not None and
            _find_mmap_file(split, "depth", h, w) is not None)


# ═══════════════════════════════════════════════════════════════════════════
#  Loader Factory Functions
# ═══════════════════════════════════════════════════════════════════════════


def get_pretrain_loader(cfg: DictConfig, device: str | None = None) -> DataLoader:
    """Build the pre-training DataLoader from config.

    Dataset selection priority:
    1. Memory-mapped .npy (if use_mmap=true and RGB file exists)
    2. Cached .pt (if use_cache=true)
    3. Load from tar/h5 into RAM (default)

    Args:
        cfg: Hydra DictConfig (needs cfg.data.dataset, cfg.data.n_samples,
             cfg.n_views, cfg.resolution, cfg.bs).
        device: Override device for pin_memory decision.
    """
    dev = device or DEVICE
    dataset_name = cfg.data.dataset
    n_samples = cfg.data.n_samples
    n_views = cfg.n_views
    resolution = cfg.resolution

    use_mmap = cfg.get("data", {}).get("use_mmap", False)
    use_cache = cfg.get("data", {}).get("use_cache", False)
    mmap_resolution = tuple(cfg.get("finetune", {}).get("resolution", [192, 256]))

    if dataset_name == "nyu":
        if use_mmap and _mmap_rgb_exists(mmap_resolution, "train"):
            ds = MmapNYUPretrainDataset(
                n_samples=n_samples, n_views=n_views,
                resolution=resolution, mmap_resolution=mmap_resolution)
        elif use_cache:
            cache_dir = cfg.get("data", {}).get("cache_dir", None)
            ds = CachedNYUPretrainDataset(
                n_samples=n_samples, n_views=n_views,
                resolution=resolution, mmap_resolution=mmap_resolution,
                cache_dir=cache_dir)
        else:
            ds = NYUPretrainDataset(
                n_samples=n_samples, n_views=n_views, resolution=resolution)
    elif dataset_name == "kitti":
        ds = KITTIPretrainDataset(
            n_samples=n_samples, n_views=n_views, resolution=resolution)
    else:
        raise ValueError(f"Unknown pretrain dataset: {dataset_name}. "
                         f"Available: nyu, kitti")

    return DataLoader(
        ds,
        batch_size=cfg.bs,
        shuffle=True,
        drop_last=True,
        num_workers=_NUM_WORKERS,
        persistent_workers=(_NUM_WORKERS > 0),
        pin_memory=(dev == "cuda"),
    )


def get_depth_loader(cfg: DictConfig, device: str | None = None,
                     split: str = "train") -> DataLoader:
    """Build the depth fine-tuning DataLoader from config.

    Dataset selection priority:
    1. Memory-mapped .npy (if use_mmap=true and files exist)
    2. Cached .pt (if use_cache=true)
    3. Load from tar/h5 into RAM (default)

    Args:
        cfg: Hydra DictConfig (needs cfg.data.dataset, cfg.data.n_samples,
             cfg.finetune.resolution, cfg.bs or cfg.finetune.bs).
        device: Override device for pin_memory decision.
        split: "train" or "val".
    """
    dev = device or DEVICE
    dataset_name = cfg.data.dataset
    n_samples = cfg.data.n_samples

    # For validation, load all available samples
    if split == "val":
        n_samples = 999_999

    # Resolution: prefer finetune.resolution, fallback to cfg.resolution
    resolution = cfg.get("finetune", {}).get("resolution", cfg.get("resolution", 128))
    if isinstance(resolution, (list, tuple)):
        resolution = tuple(resolution)

    batch_size = cfg.get("finetune", {}).get("bs", cfg.get("bs", 8))
    is_train = (split == "train")

    use_mmap = cfg.get("data", {}).get("use_mmap", False)
    use_cache = cfg.get("data", {}).get("use_cache", False)

    if dataset_name == "nyu":
        if use_mmap and _mmap_files_exist(resolution, split):
            ds = MmapNYUDepthDataset(
                n_samples=n_samples, resolution=resolution,
                train=is_train, augment=is_train, split=split)
        elif use_cache:
            cache_dir = cfg.get("data", {}).get("cache_dir", None)
            ds = CachedNYUDepthDataset(
                n_samples=n_samples, resolution=resolution,
                train=is_train, augment=is_train, split=split,
                cache_dir=cache_dir)
        else:
            ds = NYUDepthDataset(
                n_samples=n_samples, resolution=resolution,
                train=is_train, augment=is_train, split=split)
    elif dataset_name == "kitti":
        ds = KITTIDepthDataset(
            n_samples=n_samples, resolution=resolution,
            train=is_train, augment=is_train, split=split)
    else:
        raise ValueError(f"Unknown depth dataset: {dataset_name}. "
                         f"Available: nyu, kitti")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=is_train,
        drop_last=is_train,
        num_workers=_NUM_WORKERS,
        persistent_workers=(_NUM_WORKERS > 0),
        pin_memory=(dev == "cuda"),
    )
