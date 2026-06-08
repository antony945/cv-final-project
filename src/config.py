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

# ── Model constants (architecture-defined, not hyperparams) ────────────
EMB_DIM: dict = {"xxs": 160, "xs": 192, "s": 320}
