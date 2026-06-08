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
import tarfile

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from omegaconf import DictConfig

from src.config import HF_TOKEN, HF_OFFLINE, get_nyu_dataset_path, DEVICE

log = logging.getLogger(__name__)


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


def _get_depth_transforms(resolution: int):
    """Synchronized transforms for RGB+depth pairs (fine-tuning).

    Returns (rgb_transform, depth_transform) that apply consistent spatial ops.
    Color jitter is only applied to RGB; depth gets resize + normalize only.
    """
    rgb_transform = v2.Compose([
        v2.Resize((resolution, resolution)),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    depth_transform = v2.Compose([
        v2.Resize((resolution, resolution)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=False),
    ])
    return rgb_transform, depth_transform


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

    def __init__(self, n_samples: int, resolution: int = 128):
        self.resolution = resolution
        self.rgb_transform, self.depth_transform = _get_depth_transforms(resolution)
        self._samples = self._load_samples(n_samples)
        log.info(f"{len(self._samples)} depth samples loaded.")

    @abc.abstractmethod
    def _load_samples(self, n_samples: int) -> list:
        """Load up to n_samples (PIL.Image, np.ndarray) tuples."""
        ...

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, depth = self._samples[idx]

        # Apply transforms
        rgb = self.rgb_transform(img)

        # Depth: convert to PIL for torchvision transforms, then to tensor
        depth_pil = Image.fromarray(depth.astype(np.float32), mode="F")
        depth_tensor = self.depth_transform(depth_pil)  # (1, H, W)
        return rgb, depth_tensor


# ═══════════════════════════════════════════════════════════════════════════
#  NYU Depth V2 — Local tar/h5 loading
# ═══════════════════════════════════════════════════════════════════════════


def _load_nyu_local(path: str, n_samples: int, include_depth: bool = False):
    """Load NYU samples from local tar archives containing .h5 files.

    Each .h5 has keys: "rgb" (3, H, W) uint8 and "depth" (H, W) float.
    Relative paths are resolved against the project root.
    """
    from src.config import ROOT
    from pathlib import Path

    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved

    tar_files = sorted(glob.glob(str(resolved / "train-*.tar")))
    if not tar_files:
        raise FileNotFoundError(
            f"No train-*.tar files found in {resolved}. "
            f"Expected NYU Depth V2 tar archives with .h5 samples inside."
        )
    log.info(f"Loading NYU locally from {len(tar_files)} tar shards in {resolved}...")

    results = []
    for tar_path in tar_files:
        if len(results) >= n_samples:
            break
        with tarfile.open(tar_path, "r") as tf:
            for member in tf:
                if len(results) >= n_samples:
                    break
                if not member.name.endswith(".h5"):
                    continue
                f = tf.extractfile(member)
                if f is None:
                    continue
                h5_bytes = f.read()
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
                 token: str | None = HF_TOKEN, offline: bool = HF_OFFLINE):
    """Load NYU samples from HuggingFace (streaming/cached)."""
    import os
    from datasets import load_dataset

    if offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        log.info(f"Loading {n_samples} NYU samples from local cache (offline)...")
    else:
        os.environ.pop("HF_DATASETS_OFFLINE", None)
        log.info(f"Loading {n_samples} NYU samples from HuggingFace (streaming)...")

    stream = load_dataset(
        "sayakpaul/nyu_depth_v2",
        split="train",
        streaming=True,
        trust_remote_code=True,
        token=token,
    )

    results = []
    for i, row in enumerate(stream):
        if i >= n_samples:
            break
        rgb_pil = row["image"].convert("RGB")
        if include_depth:
            depth = np.array(row["depth_map"])  # PIL → ndarray
            results.append((rgb_pil, depth))
        else:
            results.append(rgb_pil)

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
    """

    def _load_samples(self, n_samples: int) -> list:
        nyu_path = get_nyu_dataset_path()
        if nyu_path:
            return _load_nyu_local(nyu_path, n_samples, include_depth=True)
        return _load_nyu_hf(n_samples, include_depth=True)


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

    ds_cls = _PRETRAIN_DATASETS[dataset_name]
    ds = ds_cls(n_samples=n_samples, n_views=cfg.n_views, resolution=cfg.resolution)

    return DataLoader(
        ds,
        batch_size=cfg.bs,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=(dev == "cuda"),
    )


def get_depth_loader(cfg: DictConfig, device: str | None = None,
                     shuffle: bool = True) -> DataLoader:
    """Build the depth fine-tuning DataLoader from config.

    Args:
        cfg: Hydra DictConfig (needs cfg.data.dataset, cfg.data.n_samples,
             cfg.resolution, cfg.bs).
        device: Override device for pin_memory decision.
        shuffle: Whether to shuffle (True for train, False for val).
    """
    dev = device or DEVICE
    dataset_name = cfg.data.dataset
    n_samples = cfg.data.n_samples

    if dataset_name not in _DEPTH_DATASETS:
        raise ValueError(f"Unknown depth dataset: {dataset_name}. "
                         f"Available: {list(_DEPTH_DATASETS.keys())}")

    ds_cls = _DEPTH_DATASETS[dataset_name]
    ds = ds_cls(n_samples=n_samples, resolution=cfg.resolution)

    return DataLoader(
        ds,
        batch_size=cfg.bs,
        shuffle=shuffle,
        drop_last=True,
        num_workers=0,
        pin_memory=(dev == "cuda"),
    )
