"""NYU Depth V2 dataset for LeJEPA pre-training (RGB only, multi-view augmentation)."""

import logging
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

from src.config import HF_TOKEN, HF_OFFLINE, DEVICE

log = logging.getLogger(__name__)


class NYUPretrainDataset(Dataset):
    """NYU Depth V2 — RGB images with multi-view augmentation for LeJEPA.

    Downloads N_SAMPLES images from HuggingFace on first use (cached after).
    Depth maps are ignored — only RGB is used for self-supervised pre-training.
    """

    def __init__(self, n_samples: int, n_views: int = 4,
                 resolution: int = 128, token: str | None = HF_TOKEN,
                 offline: bool = HF_OFFLINE):
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
        # Collect only n_samples images into memory (PIL RGB)
        self._images = []
        for i, row in enumerate(stream):
            if i >= n_samples:
                break
            self._images.append(row["image"].convert("RGB"))
        self.n_views = n_views
        log.info(f"{len(self._images)} samples loaded.")

        # ── Augmentation: verbatim from official LeJEPA minimal example ──
        self.aug = v2.Compose([
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

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = self._images[idx]
        views = torch.stack([self.aug(img) for _ in range(self.n_views)])
        return views, 0  # label unused, kept for API consistency


def get_pretrain_loader(batch_size: int, n_samples: int,
                        n_views: int = 4, resolution: int = 128,
                        num_workers: int | None = None,
                        device: str | None = None) -> DataLoader:
    """Build the pre-training DataLoader."""
    dev = device or DEVICE
    # num_workers=0 because images are already in RAM (PIL objects);
    # multiprocessing would pickle them to each worker, wasting memory.
    if num_workers is None:
        num_workers = 0
    ds = NYUPretrainDataset(n_samples=n_samples, n_views=n_views,
                            resolution=resolution)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(dev == "cuda"),
    )
