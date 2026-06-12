"""Configuration — secrets from .env + constants. Hyperparams live in configs/*.yaml."""

import os
import torch
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env from project root ───────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# ── Device ─────────────────────────────────────────────────────────────
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# ── Dataset paths ──────────────────────────────────────────────────────
NYU_DATASET_PATH: str = os.getenv("NYU_DATASET_PATH", "datasets/nyu")
NYU_MMAP_DIR: str = os.getenv("NYU_MMAP_DIR", "datasets/nyu_mmap")
KITTI_DATASET_PATH: str = os.getenv("KITTI_DATASET_PATH", "datasets/kitti")
KITTI_MMAP_DIR: str = os.getenv("KITTI_MMAP_DIR", "datasets/kitti_mmap")


def get_nyu_dataset_path() -> str:
    """Read NYU_DATASET_PATH at call time (supports dynamic env var changes)."""
    return os.getenv("NYU_DATASET_PATH", "datasets/nyu")


def get_kitti_dataset_path() -> str:
    """Read KITTI_DATASET_PATH at call time (supports dynamic env var changes)."""
    return os.getenv("KITTI_DATASET_PATH", "datasets/kitti")


def get_nyu_mmap_dir() -> str:
    """Read NYU_MMAP_DIR at call time (supports dynamic env var changes)."""
    return os.getenv("NYU_MMAP_DIR", "datasets/nyu_mmap")


def get_kitti_mmap_dir() -> str:
    """Read KITTI_MMAP_DIR at call time (supports dynamic env var changes)."""
    return os.getenv("KITTI_MMAP_DIR", "datasets/kitti_mmap")


# ── Model constants (architecture-defined, not hyperparams) ────────────
EMB_DIM: dict = {"xxs": 160, "xs": 192, "s": 320}
