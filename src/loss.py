"""SIGReg loss — the core component of LeJEPA.

Sketched Isotropic Gaussian Regularization forces projected embeddings toward
an isotropic Gaussian distribution, preventing representation collapse without
stop-gradients or teacher networks.

Implementation: verbatim from the official LeJEPA minimal example
(Balestriero & LeCun, 2025).
"""

import torch
import torch.nn as nn


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
