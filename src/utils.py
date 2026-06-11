"""Losses, metrics, and shared utilities for LeJEPA + METER.

Contains:
- SIGReg: LeJEPA collapse-prevention regularizer
- compute_lejepa_loss: Full LeJEPA loss (invariance + SIGReg)
- BalancedDepthLoss: METER BLF (L1 + gradient + normals + SSIM)
- compute_depth_metrics: Standard MDE evaluation metrics
- Shared helpers: denorm_imagenet, infer_variant, pca_feature_map, load_val_samples
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from sklearn.decomposition import PCA
from torchvision.transforms import v2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Per-dataset defaults for evaluation and visualization
DATASET_DEFAULTS: dict[str, dict] = {
    "nyu": {
        "resolution": (192, 256),
        "min_depth": 1e-3,
        "max_depth": 10.0,
        "eval_crop": "none",
        "depth_vmin": 0.0,
        "depth_vmax": 10.0,
    },
    "kitti": {
        "resolution": (192, 640),
        "min_depth": 1e-3,
        "max_depth": 80.0,
        "eval_crop": "eigen",
        "depth_vmin": 0.0,
        "depth_vmax": 50.0,
    },
}

DEFAULT_VIS_RES: tuple[int, int] = DATASET_DEFAULTS["nyu"]["resolution"]  # (H, W) — must be divisible by 32
VIS_SEED: int = 42  # Default seed for reproducible image selection in visualizations


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Shared Utilities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def denorm_imagenet(t: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization for display.

    Args:
        t: (3, H, W) tensor, ImageNet-normalized.
    Returns:
        (H, W, 3) numpy array in [0, 1].
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (t.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def infer_variant(checkpoint: str | Path) -> str | None:
    """Try to infer variant (xxs/xs/s) from checkpoint filename."""
    name = Path(checkpoint).stem.lower()
    for v in ("xxs", "xs", "s"):
        if f"_{v}_" in name or name.endswith(f"_{v}"):
            return v
    return None


def pca_feature_map(feat: torch.Tensor, n_components: int = 3,
                    pca_model: PCA | None = None) -> tuple[np.ndarray, PCA]:
    """Project a spatial feature map to n_components via PCA.

    Fits PCA across ALL spatial tokens in the batch so that principal
    components are shared (consistent color axes across images).

    Args:
        feat: (B, C, H, W) tensor.
        pca_model: If provided, transform with this PCA (don't re-fit).
    Returns:
        proj: (B, H, W, n_components) array, each component normalized to [0, 1].
        pca_model: The fitted PCA object (for reuse across feature levels).
    """
    B, C, H, W = feat.shape
    X = feat.permute(0, 2, 3, 1).reshape(B * H * W, C).cpu().float().numpy()
    if pca_model is None:
        pca_model = PCA(n_components=n_components)
        proj = pca_model.fit_transform(X).reshape(B, H, W, n_components)
    else:
        proj = pca_model.transform(X).reshape(B, H, W, n_components)
    # Normalize globally (across all images) per component
    for c in range(n_components):
        mn, mx = proj[..., c].min(), proj[..., c].max()
        proj[..., c] = (proj[..., c] - mn) / (mx - mn + 1e-8)
    return proj, pca_model


def load_val_samples(n_images: int, resolution: tuple[int, int] = DEFAULT_VIS_RES,
                     dataset: str = "nyu", seed: int | None = None):
    """Load N samples from validation set (RGB + depth GT).

    By default loads the first N samples. If seed is provided, selects N random
    indices (deterministic given the seed) for better dataset coverage.

    Args:
        n_images: Number of samples to load.
        resolution: (H, W) target resolution.
        dataset: "nyu" or "kitti".
        seed: If provided, randomly sample indices instead of taking the first N.
    Returns:
        rgb_batch: (N, 3, H, W) tensor, ImageNet-normalized.
        depth_batch: (N, 1, H, W) tensor, meters.
        rgb_display: list of (H, W, 3) numpy arrays in [0, 1].
    """
    from src.config import HF_TOKEN, HF_OFFLINE, get_nyu_dataset_path
    from src.data import _mmap_files_exist, _find_mmap_file
    import os

    H, W = resolution

    # Try mmap first (fastest, consistent with training)
    if _mmap_files_exist(resolution, "val", dataset=dataset):
        rgb_path = _find_mmap_file("val", "rgb", H, W, dataset=dataset)
        depth_path = _find_mmap_file("val", "depth", H, W, dataset=dataset)
        rgb_mmap = np.load(str(rgb_path), mmap_mode="r")
        depth_mmap = np.load(str(depth_path), mmap_mode="r")

        # Determine which indices to load
        total = rgb_mmap.shape[0]
        n = min(n_images, total)
        if seed is not None:
            rng = np.random.default_rng(seed)
            indices = sorted(rng.choice(total, size=n, replace=False))
        else:
            indices = list(range(n))

        rgb_tensors = []
        depth_tensors = []
        rgb_display = []

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        for i in indices:
            rgb_np = np.array(rgb_mmap[i])        # (H, W, 3) uint8
            depth_np = np.array(depth_mmap[i])    # (H, W) float32

            rgb_display.append(rgb_np.astype(np.float32) / 255.0)

            # Normalize for model input
            rgb_f = rgb_np.astype(np.float32) / 255.0
            rgb_f = (rgb_f - mean) / std
            rgb_tensors.append(torch.from_numpy(rgb_f.transpose(2, 0, 1)).float())
            depth_tensors.append(torch.from_numpy(depth_np).unsqueeze(0))

        rgb_batch = torch.stack(rgb_tensors)
        depth_batch = torch.stack(depth_tensors)
        return rgb_batch, depth_batch, rgb_display

    # Fallback: load from tar/h5 or HuggingFace (NYU only)
    if dataset != "nyu":
        raise FileNotFoundError(
            f"No mmap val files found for {dataset} at {H}x{W}. "
            f"Run: uv run python -m src.preprocess --dataset {dataset} {H} {W}"
        )

    nyu_path = get_nyu_dataset_path()
    if nyu_path:
        from src.data import _load_nyu_local
        samples = _load_nyu_local(nyu_path, n_images, include_depth=True, split="val")
    else:
        from datasets import load_dataset
        if HF_OFFLINE:
            os.environ["HF_DATASETS_OFFLINE"] = "1"
        else:
            os.environ.pop("HF_DATASETS_OFFLINE", None)
        ds = load_dataset("sayakpaul/nyu_depth_v2", split="validation",
                          streaming=True, trust_remote_code=True, token=HF_TOKEN)
        samples = []
        for i, row in enumerate(ds):
            if i >= n_images:
                break
            rgb_pil = row["image"].convert("RGB")
            depth = np.array(row["depth_map"])
            samples.append((rgb_pil, depth))

    # Prepare tensors
    transform = v2.Compose([
        v2.Resize((H, W)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    rgb_tensors = []
    depth_tensors = []
    rgb_display = []

    for rgb_pil, depth_np in samples:
        if isinstance(rgb_pil, np.ndarray):
            rgb_pil = Image.fromarray(rgb_pil.transpose(1, 2, 0) if rgb_pil.shape[0] == 3
                                      else rgb_pil)
        rgb_tensors.append(transform(rgb_pil))
        rgb_display.append(np.array(rgb_pil.resize((W, H))) / 255.0)
        # Resize depth to match resolution
        depth_resized = np.array(
            Image.fromarray(depth_np.astype(np.float32)).resize((W, H), Image.BILINEAR)
        )
        depth_tensors.append(torch.from_numpy(depth_resized).unsqueeze(0))  # (1, H, W)

    rgb_batch = torch.stack(rgb_tensors)       # (N, 3, H, W)
    depth_batch = torch.stack(depth_tensors)    # (N, 1, H, W)
    return rgb_batch, depth_batch, rgb_display


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LeJEPA Loss (SIGReg)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SIGReg(nn.Module):
    """SIGReg: Sketched Isotropic Gaussian Regularization.

    Compares the empirical characteristic function of the projected embeddings
    to a standard Gaussian's characteristic function using numerical quadrature.
    Single hyperparameter: λ (weight in total loss, configured externally).
    """

    def __init__(self, knots: int = 17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """Compute SIGReg statistic.

        Args:
            proj: (V, B, proj_dim) — projected embeddings from V views.
        Returns:
            Scalar loss (the SIGReg statistic).
        """
        A = torch.randn(proj.size(-1), 256, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))          # unit-norm random projections
        x_t = (proj @ A).unsqueeze(-1) * self.t  # (V, B, 256, knots)
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


def compute_lejepa_loss(proj: torch.Tensor, sigreg: SIGReg,
                        lamb: float = 0.02) -> tuple[torch.Tensor, dict]:
    """Compute the full LeJEPA loss.

    Loss = sigreg_loss * λ + invariance_loss * (1 - λ)

    Args:
        proj: (V, B, proj_dim) — projected embeddings.
        sigreg: SIGReg module.
        lamb: λ weight for SIGReg (default from config).
    Returns:
        total_loss: scalar tensor.
        components: dict with individual loss values for logging.
    """
    # Invariance: each view's projection should match the mean across views
    inv_loss = (proj.mean(0) - proj).square().mean()
    sigreg_loss = sigreg(proj)
    total = sigreg_loss * lamb + inv_loss * (1 - lamb)

    return total, {
        "lejepa": total.item(),
        "sigreg": sigreg_loss.item(),
        "inv": inv_loss.item(),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  METER Balanced Depth Loss (BLF)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _sobel_gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Sobel gradients along x and y for a (B, 1, H, W) tensor."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=x.dtype, device=x.device).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=x.dtype, device=x.device).reshape(1, 1, 3, 3)
    grad_x = F.conv2d(x, sobel_x, padding=1)
    grad_y = F.conv2d(x, sobel_y, padding=1)
    return grad_x, grad_y


def _ssim(pred: torch.Tensor, target: torch.Tensor,
           window_size: int = 11, C1: float = 0.01**2,
           C2: float = 0.03**2) -> torch.Tensor:
    """Compute mean SSIM between pred and target (B, 1, H, W)."""
    # Gaussian window
    coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    window = (g.unsqueeze(1) @ g.unsqueeze(0))
    window = window / window.sum()
    window = window.reshape(1, 1, window_size, window_size)

    pad = window_size // 2
    mu_pred = F.conv2d(pred, window, padding=pad)
    mu_target = F.conv2d(target, window, padding=pad)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_cross = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred ** 2, window, padding=pad) - mu_pred_sq
    sigma_target_sq = F.conv2d(target ** 2, window, padding=pad) - mu_target_sq
    sigma_cross = F.conv2d(pred * target, window, padding=pad) - mu_cross

    ssim_map = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
    return ssim_map.mean()


class BalancedDepthLoss(nn.Module):
    """METER Balanced Loss Function (BLF) for monocular depth estimation.

    L = L_depth + λ1*L_grad + λ2*L_norm + λ3*L_SSIM

    All components are computed only on valid pixels (depth > 0).
    """

    def __init__(self, lamb1: float = 0.5, lamb2: float = 1.0,
                 lamb3: float = 1.0):
        super().__init__()
        self.lamb1 = lamb1
        self.lamb2 = lamb2
        self.lamb3 = lamb3

    def forward(self, pred: torch.Tensor, target: torch.Tensor
                ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            pred: (B, 1, H, W) predicted depth map
            target: (B, 1, H, W) ground truth depth map
        Returns:
            total_loss, components dict
        """
        # Mask for valid depth pixels
        mask = (target > 0).float()
        n_valid = mask.sum().clamp(min=1)

        # L_depth: masked L1
        abs_diff = (pred - target).abs() * mask
        l_depth = abs_diff.sum() / n_valid

        # L_grad: gradient of absolute error (Sobel)
        grad_x, grad_y = _sobel_gradients(abs_diff)
        l_grad = (grad_x.abs() + grad_y.abs()).sum() / n_valid

        # L_norm: cosine similarity of surface normals
        pred_gx, pred_gy = _sobel_gradients(pred * mask)
        target_gx, target_gy = _sobel_gradients(target * mask)

        # Surface normals: n = [-∂z/∂x, -∂z/∂y, 1]
        pred_normal = torch.cat([-pred_gx, -pred_gy,
                                  torch.ones_like(pred_gx)], dim=1)
        target_normal = torch.cat([-target_gx, -target_gy,
                                    torch.ones_like(target_gx)], dim=1)

        cos_sim = F.cosine_similarity(pred_normal, target_normal, dim=1,
                                      eps=1e-6)
        # Mask and average: 1 - cos_sim
        l_norm = ((1 - cos_sim) * mask.squeeze(1)).sum() / n_valid

        # L_SSIM: 1 - SSIM
        # Apply mask by zeroing invalid regions
        l_ssim = 1.0 - _ssim(pred * mask, target * mask)

        total = l_depth + self.lamb1 * l_grad + self.lamb2 * l_norm + \
                self.lamb3 * l_ssim

        return total, {
            "total": total.item(),
            "depth": l_depth.item(),
            "grad": l_grad.item(),
            "norm": l_norm.item(),
            "ssim": l_ssim.item(),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Depth Evaluation Metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@torch.no_grad()
def compute_depth_metrics(pred: torch.Tensor, target: torch.Tensor,
                          min_depth: float = 1e-3, max_depth: float = 10.0,
                          eval_crop: str = "none"
                          ) -> dict[str, float]:
    """Compute standard monocular depth estimation metrics.

    Args:
        pred: (B, 1, H, W) predicted depth
        target: (B, 1, H, W) ground truth depth
        min_depth: Minimum valid depth (meters). Predictions/targets below are clamped.
        max_depth: Maximum valid depth (meters). Pixels with target > max_depth are masked.
        eval_crop: Crop mode before evaluation:
            "none" — no crop (NYU default, already center-cropped in data)
            "eigen" — Eigen crop for KITTI (top 8.2%, bottom 6.5%, left 4.4%, right 1.5%)
            "garg" — Garg crop (more aggressive, not commonly used)
    Returns:
        dict with δ1, δ2, δ3, rmse, rel, log10
    """
    # Apply evaluation crop
    if eval_crop == "eigen":
        _, _, h, w = pred.shape
        crop_h = (int(0.3324324 * h), int(0.91351351 * h))
        crop_w = (int(0.0359477 * w), int(0.96405229 * w))
        pred = pred[:, :, crop_h[0]:crop_h[1], crop_w[0]:crop_w[1]]
        target = target[:, :, crop_h[0]:crop_h[1], crop_w[0]:crop_w[1]]
    elif eval_crop == "garg":
        _, _, h, w = pred.shape
        crop_h = (int(0.40810811 * h), int(0.99189189 * h))
        crop_w = (int(0.03594771 * w), int(0.96405229 * w))
        pred = pred[:, :, crop_h[0]:crop_h[1], crop_w[0]:crop_w[1]]
        target = target[:, :, crop_h[0]:crop_h[1], crop_w[0]:crop_w[1]]

    # Mask: valid depth between min and max
    mask = (target > min_depth) & (target <= max_depth)
    pred_valid = pred[mask].clamp(min=min_depth, max=max_depth)
    target_valid = target[mask]

    if pred_valid.numel() == 0:
        return {"delta1": 0., "delta2": 0., "delta3": 0.,
                "rmse": 0., "rel": 0., "log10": 0.}

    # Threshold accuracy (δ)
    ratio = torch.max(pred_valid / target_valid, target_valid / pred_valid)
    delta1 = (ratio < 1.25).float().mean().item()
    delta2 = (ratio < 1.25 ** 2).float().mean().item()
    delta3 = (ratio < 1.25 ** 3).float().mean().item()

    # RMSE
    rmse = ((pred_valid - target_valid) ** 2).mean().sqrt().item()

    # AbsRel
    rel = ((pred_valid - target_valid).abs() / target_valid).mean().item()

    # log10
    log10_err = (pred_valid.log10() - target_valid.log10()).abs().mean().item()

    return {
        "delta1": delta1, "delta2": delta2, "delta3": delta3,
        "rmse": rmse, "rel": rel, "log10": log10_err,
    }
