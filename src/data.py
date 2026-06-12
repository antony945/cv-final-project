"""Dataset loading for LeJEPA pre-training (RGB multi-view) and depth fine-tuning.

Supports:
- NYU Depth V2: local tar/h5 files (NYU_DATASET_PATH, default: datasets/nyu)
- KITTI: memory-mapped .npy files (requires preprocessing)

Storage backends (selected via config):
- Mmap: memory-mapped .npy files (near-zero RAM, requires preprocessing)
- Normal: loads all data into RAM at startup (fast access, high memory — NYU only fallback)

Environment variables:
- NYU_DATASET_PATH: path to directory with train-*.tar files (each containing .h5 samples).
                    Defaults to 'datasets/nyu'.
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

from src.config import get_nyu_dataset_path, DEVICE

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
                 augment: bool = True, depth_shift: float = 0.1):
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        self.resolution = resolution
        self.train = train
        self.augment = augment and train
        self.depth_shift = depth_shift

    @abc.abstractmethod
    def _get_sample(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (rgb [3,H,W] float32 [0,1], depth [1,H,W] float32) at target resolution."""
        ...

    @abc.abstractmethod
    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rgb, depth = self._get_sample(idx)

        if self.augment:
            rgb, depth = self._augment(rgb, depth, self.depth_shift)

        # Normalize RGB (ImageNet stats)
        rgb = (rgb - self._MEAN) / self._STD
        return rgb, depth

    @staticmethod
    def _augment(rgb: torch.Tensor, depth: torch.Tensor,
                 depth_shift: float = 0.1
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
            shift = random.uniform(-depth_shift, depth_shift)
            depth = depth + shift

        return rgb, depth


# ═══════════════════════════════════════════════════════════════════════════
#  NYU Depth V2 — Raw data loaders (tar/h5)
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


# ═══════════════════════════════════════════════════════════════════════════
#  NYU — Normal (in-RAM) Datasets
# ═══════════════════════════════════════════════════════════════════════════


class NYUPretrainDataset(BasePretrainDataset):
    """NYU Depth V2 — RGB images loaded into RAM with multi-view augmentation.

    Loads from local tar/h5 files. All images are held in memory as PIL objects.
    """

    def __init__(self, n_samples: int, n_views: int = 4, resolution: int = 128):
        super().__init__(n_views=n_views, resolution=resolution)
        nyu_path = get_nyu_dataset_path()
        self._images = _load_nyu_local(nyu_path, n_samples, include_depth=False)
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
                 train: bool = True, augment: bool = True, split: str = "train",
                 depth_shift: float = 0.1):
        super().__init__(resolution=resolution, train=train, augment=augment,
                         depth_shift=depth_shift)
        nyu_path = get_nyu_dataset_path()
        self._samples = _load_nyu_local(nyu_path, n_samples,
                                        include_depth=True, split=split)
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
                 train: bool = True, augment: bool = True, split: str = "train",
                 depth_shift: float = 0.1):
        super().__init__(resolution=resolution, train=train, augment=augment,
                         depth_shift=depth_shift)

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
#  KITTI — Stubs (TO IMPLEMENT)
# ═══════════════════════════════════════════════════════════════════════════


class KITTIPretrainDataset(BasePretrainDataset):
    """KITTI — RGB images for pre-training. TO IMPLEMENT (needs raw zip loading)."""

    def __init__(self, n_samples: int, n_views: int = 4, resolution: int = 128):
        super().__init__(n_views=n_views, resolution=resolution)
        raise NotImplementedError(
            "KITTI pretrain from raw zips not yet implemented. "
            "Run 'uv run python -m src.preprocess --dataset kitti' first, "
            "then use MmapKITTIPretrainDataset via data.use_mmap=true."
        )

    def __len__(self) -> int:
        return 0

    def _get_image(self, idx: int) -> Image.Image:
        raise NotImplementedError


class MmapKITTIPretrainDataset(BasePretrainDataset):
    """KITTI — Memory-mapped RGB for LeJEPA pre-training.

    Reads from preprocessed .npy mmap files (at finetune resolution).
    The pretrain augmentation pipeline (RandomResizedCrop) handles final sizing.
    Near-zero RAM regardless of dataset size.
    """

    def __init__(self, n_samples: int, n_views: int = 4, resolution: int = 128,
                 mmap_resolution: tuple[int, int] = (192, 640)):
        super().__init__(n_views=n_views, resolution=resolution)

        h, w = mmap_resolution
        rgb_path = _find_mmap_file("train", "rgb", h, w, dataset="kitti")
        if rgb_path is None:
            raise FileNotFoundError(
                f"KITTI memory-mapped RGB file not found for {h}x{w}. "
                f"Run: uv run python -m src.preprocess --dataset kitti {h} {w}"
            )

        self._rgb = np.load(str(rgb_path), mmap_mode="r")  # (N, H, W, 3) uint8
        self._n = min(n_samples, self._rgb.shape[0])
        log.info(f"{self._n}/{self._rgb.shape[0]} KITTI mmap pretrain samples ({h}x{w}).")

    def __len__(self) -> int:
        return self._n

    def _get_image(self, idx: int) -> Image.Image:
        rgb_np = np.array(self._rgb[idx])  # copy from mmap
        return Image.fromarray(rgb_np, mode="RGB")


class KITTIDepthDataset(BaseDepthDataset):
    """KITTI — RGB + depth for fine-tuning. TO IMPLEMENT (needs raw zip loading)."""

    def __init__(self, n_samples: int, resolution: tuple[int, int] | int = (192, 640),
                 train: bool = True, augment: bool = True, split: str = "train",
                 depth_shift: float = 1.0):
        super().__init__(resolution=resolution, train=train, augment=augment,
                         depth_shift=depth_shift)
        raise NotImplementedError(
            "KITTI depth from raw zips not yet implemented. "
            "Run 'uv run python -m src.preprocess --dataset kitti' first, "
            "then use MmapKITTIDepthDataset via data.use_mmap=true."
        )

    def __len__(self) -> int:
        return 0

    def _get_sample(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class MmapKITTIDepthDataset(BaseDepthDataset):
    """KITTI — Memory-mapped RGB + depth for fine-tuning.

    Reads from preprocessed .npy mmap files (already at target resolution).
    Near-zero RAM regardless of dataset size.
    """

    def __init__(self, n_samples: int, resolution: tuple[int, int] | int = (192, 640),
                 train: bool = True, augment: bool = True, split: str = "train",
                 depth_shift: float = 1.0):
        super().__init__(resolution=resolution, train=train, augment=augment,
                         depth_shift=depth_shift)

        h, w = self.resolution
        rgb_path = _find_mmap_file(split, "rgb", h, w, dataset="kitti")
        depth_path = _find_mmap_file(split, "depth", h, w, dataset="kitti")

        if rgb_path is None or depth_path is None:
            raise FileNotFoundError(
                f"KITTI memory-mapped files not found for {split} {h}x{w}. "
                f"Run: uv run python -m src.preprocess --dataset kitti {h} {w}"
            )

        self._rgb = np.load(str(rgb_path), mmap_mode="r")      # (N, H, W, 3) uint8
        self._depth = np.load(str(depth_path), mmap_mode="r")   # (N, H, W) float32
        self._n = min(n_samples, self._rgb.shape[0])
        log.info(f"{self._n}/{self._rgb.shape[0]} KITTI mmap depth samples "
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
#  Helpers — Mmap file discovery (glob-based, supports sample count in name)
# ═══════════════════════════════════════════════════════════════════════════


def _get_mmap_dir(dataset: str = "nyu") -> Path:
    """Resolve the mmap directory path for a given dataset."""
    from src.config import ROOT, get_nyu_mmap_dir, get_kitti_mmap_dir
    if dataset == "kitti":
        p = Path(get_kitti_mmap_dir())
    else:
        p = Path(get_nyu_mmap_dir())
    if not p.is_absolute():
        p = ROOT / p
    return p


def _find_mmap_file(split: str, kind: str, h: int, w: int,
                    dataset: str = "nyu") -> Path | None:
    """Find a mmap .npy file by glob, supporting both old and new naming.

    Naming formats:
      Old: {dataset}_{split}_{kind}_{H}x{W}.npy
      New: {dataset}_{split}_{kind}_{H}x{W}_N{count}.npy

    Returns the path if found, None otherwise.
    """
    p = _get_mmap_dir(dataset)
    # Try new format first (with sample count)
    matches = sorted(p.glob(f"{dataset}_{split}_{kind}_{h}x{w}_N*.npy"))
    if matches:
        return matches[0]
    # Fall back to old format (without count)
    old = p / f"{dataset}_{split}_{kind}_{h}x{w}.npy"
    if old.exists():
        return old
    return None


def _mmap_rgb_exists(resolution: tuple[int, int] | int, split: str,
                     dataset: str = "nyu") -> bool:
    """Check if preprocessed memory-mapped RGB file exists (pretrain only needs RGB)."""
    if isinstance(resolution, int):
        resolution = (resolution, resolution)
    h, w = resolution
    return _find_mmap_file(split, "rgb", h, w, dataset) is not None


def _mmap_files_exist(resolution: tuple[int, int] | int, split: str,
                      dataset: str = "nyu") -> bool:
    """Check if preprocessed memory-mapped RGB + depth files exist."""
    if isinstance(resolution, int):
        resolution = (resolution, resolution)
    h, w = resolution
    return (_find_mmap_file(split, "rgb", h, w, dataset) is not None and
            _find_mmap_file(split, "depth", h, w, dataset) is not None)


# ═══════════════════════════════════════════════════════════════════════════
#  Loader Factory Functions
# ═══════════════════════════════════════════════════════════════════════════


def get_pretrain_loader(cfg: DictConfig, device: str | None = None) -> DataLoader:
    """Build the pre-training DataLoader from config.

    Iterates all datasets in cfg.data.datasets and wraps them in ConcatDataset
    for mixed pre-training. Dataset selection priority per dataset:
    1. Memory-mapped .npy (if use_mmap=true — errors if files missing)
    2. Load from tar/h5 into RAM (if use_mmap=false — NYU only)

    Args:
        cfg: Hydra DictConfig (needs cfg.data.datasets dict, cfg.n_views,
             cfg.resolution, cfg.bs).
        device: Override device for pin_memory decision.
    """
    dev = device or DEVICE
    n_views = cfg.n_views
    resolution = cfg.resolution

    use_mmap = cfg.data.get("use_mmap", False)

    datasets_cfg = cfg.data.datasets
    dataset_list = []

    for ds_name, ds_cfg in datasets_cfg.items():
        n_samples = ds_cfg.get("n_samples", 1000)

        if ds_name == "nyu":
            # TODO: Now it will search for fixed res mmap files
            mmap_res = (192, 256)
            if use_mmap:
                if not _mmap_rgb_exists(mmap_res, "train", dataset="nyu"):
                    raise FileNotFoundError(
                        f"data.use_mmap=true but NYU mmap RGB file not found for "
                        f"{mmap_res[0]}x{mmap_res[1]}. "
                        f"Run 'uv run python -m src.preprocess' or set "
                        f"data.use_mmap=false. See README §5 for details."
                    )
                ds = MmapNYUPretrainDataset(
                    n_samples=n_samples, n_views=n_views,
                    resolution=resolution, mmap_resolution=mmap_res)
            else:
                ds = NYUPretrainDataset(
                    n_samples=n_samples, n_views=n_views, resolution=resolution)
            dataset_list.append(ds)

        elif ds_name == "kitti":
            # TODO: Now it will search for fixed res mmap files
            mmap_res = (192, 640)
            if use_mmap:
                if not _mmap_rgb_exists(mmap_res, "train", dataset="kitti"):
                    raise FileNotFoundError(
                        f"data.use_mmap=true but KITTI mmap RGB file not found for "
                        f"{mmap_res[0]}x{mmap_res[1]}. "
                        f"Run 'uv run python -m src.preprocess --dataset kitti' or set "
                        f"data.use_mmap=false. See README §5 for details."
                    )
                ds = MmapKITTIPretrainDataset(
                    n_samples=n_samples, n_views=n_views,
                    resolution=resolution, mmap_resolution=mmap_res)
            else:
                ds = KITTIPretrainDataset(
                    n_samples=n_samples, n_views=n_views, resolution=resolution)
            dataset_list.append(ds)
        else:
            raise ValueError(f"Unknown pretrain dataset: {ds_name}. "
                             f"Available: nyu, kitti")

    if not dataset_list:
        raise ValueError("No datasets configured in cfg.data.datasets.")

    # Single dataset → use directly; multiple → ConcatDataset
    if len(dataset_list) == 1:
        final_ds = dataset_list[0]
    else:
        from torch.utils.data import ConcatDataset
        final_ds = ConcatDataset(dataset_list)

    return DataLoader(
        final_ds,
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

    Uses the first (and typically only) dataset in cfg.data.datasets for
    fine-tuning. Dataset selection priority:
    1. Memory-mapped .npy (if use_mmap=true — errors if files missing)
    2. Load from tar/h5 into RAM (if use_mmap=false — NYU only)

    Args:
        cfg: Hydra DictConfig (needs cfg.data.datasets dict,
             cfg.finetune.resolution, cfg.bs or cfg.finetune.bs).
        device: Override device for pin_memory decision.
        split: "train" or "val".
    """
    dev = device or DEVICE

    # Get the first (only) dataset from config
    datasets_cfg = cfg.data.datasets
    dataset_name = next(iter(datasets_cfg))
    ds_cfg = datasets_cfg[dataset_name]

    n_samples = ds_cfg.get("n_samples", 1000)
    depth_shift = ds_cfg.get("depth_shift", 0.1)

    # For validation, load all available samples
    if split == "val":
        n_samples = 999_999

    # Resolution: prefer finetune.resolution, fallback to cfg.resolution
    # TODO: Fix this inconsistency in the future
    resolution = cfg.get("finetune", {}).get("resolution", cfg.get("resolution", 128))
    if isinstance(resolution, (list, tuple)):
        resolution = tuple(resolution)

    batch_size = cfg.get("finetune", {}).get("bs", cfg.get("bs", 8))
    is_train = (split == "train")

    use_mmap = cfg.data.get("use_mmap", False)

    if dataset_name == "nyu":
        if use_mmap:
            if not _mmap_files_exist(resolution, split, dataset="nyu"):
                raise FileNotFoundError(
                    f"data.use_mmap=true but NYU mmap files not found for "
                    f"{split} {resolution[0]}x{resolution[1]}. "
                    f"Run 'uv run python -m src.preprocess' or set "
                    f"data.use_mmap=false. See README §5 for details."
                )
            ds = MmapNYUDepthDataset(
                n_samples=n_samples, resolution=resolution,
                train=is_train, augment=is_train, split=split,
                depth_shift=depth_shift)
        else:
            ds = NYUDepthDataset(
                n_samples=n_samples, resolution=resolution,
                train=is_train, augment=is_train, split=split,
                depth_shift=depth_shift)
    elif dataset_name == "kitti":
        # KITTI uses finetune.resolution (same pattern as NYU)
        if use_mmap:
            if not _mmap_files_exist(resolution, split, dataset="kitti"):
                raise FileNotFoundError(
                    f"data.use_mmap=true but KITTI mmap files not found for "
                    f"{split} {resolution[0]}x{resolution[1]}. "
                    f"Run 'uv run python -m src.preprocess --dataset kitti' or set "
                    f"data.use_mmap=false. See README §5 for details."
                )
            ds = MmapKITTIDepthDataset(
                n_samples=n_samples, resolution=resolution,
                train=is_train, augment=is_train, split=split,
                depth_shift=depth_shift)
        else:
            ds = KITTIDepthDataset(
                n_samples=n_samples, resolution=resolution,
                train=is_train, augment=is_train, split=split,
                depth_shift=depth_shift)
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
