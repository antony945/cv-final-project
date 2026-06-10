"""Loss functions for LeJEPA pre-training and METER depth fine-tuning.

Contains:
- SIGReg: LeJEPA collapse-prevention regularizer
- compute_lejepa_loss: Full LeJEPA loss (invariance + SIGReg)
- BalancedDepthLoss: METER BLF (L1 + gradient + normals + SSIM)
- compute_depth_metrics: Standard MDE evaluation metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
def compute_depth_metrics(pred: torch.Tensor, target: torch.Tensor
                          ) -> dict[str, float]:
    """Compute standard monocular depth estimation metrics.

    Args:
        pred: (B, 1, H, W) predicted depth
        target: (B, 1, H, W) ground truth depth
    Returns:
        dict with δ1, δ2, δ3, rmse, rel, log10
    """
    mask = target > 0
    pred_valid = pred[mask]
    target_valid = target[mask]

    if pred_valid.numel() == 0:
        return {"delta1": 0., "delta2": 0., "delta3": 0.,
                "rmse": 0., "rel": 0., "log10": 0.}

    # Clamp predictions to avoid division by zero / log(0)
    pred_valid = pred_valid.clamp(min=1e-3)
    target_valid = target_valid.clamp(min=1e-3)

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
