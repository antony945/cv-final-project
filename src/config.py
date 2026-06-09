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

# ── HuggingFace secrets ────────────────────────────────────────────────
HF_TOKEN: str | None = os.getenv("HF_TOKEN")
HF_OFFLINE: bool = str(os.getenv("HF_OFFLINE", "false")).lower() in ("1", "true", "yes")

# ── Dataset paths (set to use local tar files instead of HuggingFace) ──
NYU_DATASET_PATH: str | None = os.getenv("NYU_DATASET_PATH")
NYU_MMAP_DIR: str = os.getenv("NYU_MMAP_DIR", "datasets/nyu_mmap")


def get_nyu_dataset_path() -> str | None:
    """Read NYU_DATASET_PATH at call time (supports dynamic env var changes)."""
    return os.getenv("NYU_DATASET_PATH")


def get_nyu_mmap_dir() -> str:
    """Read NYU_MMAP_DIR at call time (supports dynamic env var changes)."""
    return os.getenv("NYU_MMAP_DIR", "datasets/nyu_mmap")


# ── Model constants (architecture-defined, not hyperparams) ────────────
EMB_DIM: dict = {"xxs": 160, "xs": 192, "s": 320}
