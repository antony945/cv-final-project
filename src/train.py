"""LeJEPA pre-training loop for MobileViT encoder.

Follows the official LeJEPA minimal example structure:
- AdamW optimizer, weight_decay=5e-2
- LinearLR warmup (1 epoch) → CosineAnnealingLR (eta_min=1e-3)
- Mixed precision (bf16 on CUDA)
- NO gradient clipping (LeJEPA is provably stable)
- wandb logging per batch (if enabled)
- Checkpoint saving every ckpt_every epochs
"""

import logging
import signal
import torch
import warnings
from pathlib import Path
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import tqdm
from omegaconf import DictConfig

log = logging.getLogger(__name__)

from src.config import DEVICE as _AUTO_DEVICE, EMB_DIM
from src.model import MobileViTLeJEPA
from src.loss import SIGReg, compute_lejepa_loss
from src.data import get_pretrain_loader
from src.visualize import visualize_pca_inline


def _resolve_device(cfg: DictConfig) -> str:
    """Resolve device from config: auto picks CUDA if available."""
    if cfg.device == "auto":
        return _AUTO_DEVICE
    return cfg.device


def _init_wandb(cfg: DictConfig):
    """Initialize wandb if enabled. Returns True if active."""
    if not cfg.use_wandb:
        return False
    try:
        import wandb
        wandb.init(
            project=cfg.wandb_project,
            name=f"lejepa_{cfg.variant}",
            config={
                "variant": cfg.variant,
                "emb_dim": EMB_DIM[cfg.variant],
                "epochs": cfg.epochs,
                "batch_size": cfg.bs,
                "lr": cfg.lr,
                "lambda": cfg.lamb,
                "proj_dim": cfg.proj_dim,
                "resolution": cfg.resolution,
                "device": cfg.device,
            },
        )
        return True
    except Exception as e:
        print(f"wandb init failed ({e}), falling back to console logging.")
        return False


def pretrain_lejepa(cfg: DictConfig) -> dict:
    """Run LeJEPA self-supervised pre-training.

    Args:
        cfg: Hydra DictConfig with all training hyperparameters.
    Returns:
        dict with training history (epoch losses).
    """
    variant = cfg.variant
    epochs = cfg.epochs
    device = _resolve_device(cfg)

    log.info(f"LeJEPA Pre-training: MobileViT-{variant.upper()}")
    log.info(f"Epochs: {epochs} | BS: {cfg.bs} | lambda: {cfg.lamb} | Device: {device}")

    wandb_active = _init_wandb(cfg)

    # ── Model, loss, optimizer ────────────────────────────────────────
    net = MobileViTLeJEPA(variant=variant, proj_dim=cfg.proj_dim,
                          resolution=cfg.resolution).to(device)
    sigreg = SIGReg().to(device)

    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr,
                            weight_decay=5e-2)

    loader = get_pretrain_loader(cfg, device=device)

    # ── Scheduler: 1-epoch warmup → cosine decay ─────────────────────
    warmup_steps = len(loader)
    total_steps = len(loader) * epochs
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-3)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

    scaler = GradScaler(enabled=(device == "cuda"))
    history = {"lejepa": [], "sigreg": [], "inv": []}

    # Suppress harmless SequentialLR deprecation warning from PyTorch internals
    warnings.filterwarnings("ignore", category=UserWarning,
                            message=".*epoch parameter.*scheduler.step.*")

    # ── Graceful interrupt handling ───────────────────────────────────
    _stop_requested = False
    _original_sigint = signal.getsignal(signal.SIGINT)

    def _graceful_handler(signum, frame):
        nonlocal _stop_requested
        _stop_requested = True
        log.info("Ctrl+C received — finishing current epoch. "
                 "Press Ctrl+C again to force quit.")
        # Second Ctrl+C will raise KeyboardInterrupt immediately
        signal.signal(signal.SIGINT, _original_sigint)

    signal.signal(signal.SIGINT, _graceful_handler)
    last_completed_epoch = 0

    # ── Training loop ─────────────────────────────────────────────────
    try:
        for epoch in range(1, epochs + 1):
            net.train()
            ep_lejepa = ep_sig = ep_inv = 0.0

            pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
            for views, _ in pbar:
                views = views.to(device, non_blocking=True)

                with autocast(device, dtype=torch.bfloat16):
                    _, proj = net(views)
                    loss, components = compute_lejepa_loss(proj, sigreg, cfg.lamb)

                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                scheduler.step()

                ep_lejepa += components["lejepa"]
                ep_sig += components["sigreg"]
                ep_inv += components["inv"]

                pbar.set_postfix(loss=f"{components['lejepa']:.4f}")

                if wandb_active:
                    import wandb
                    wandb.log({
                        "train/lejepa": components["lejepa"],
                        "train/sigreg": components["sigreg"],
                        "train/inv": components["inv"],
                    })

            # ── Epoch summary ─────────────────────────────────────────
            n = len(loader)
            ep_lejepa /= n
            ep_sig /= n
            ep_inv /= n
            history["lejepa"].append(ep_lejepa)
            history["sigreg"].append(ep_sig)
            history["inv"].append(ep_inv)
            last_completed_epoch = epoch

            log.info(f"Epoch {epoch:>3} | lejepa={ep_lejepa:.4f} "
                     f"sigreg={ep_sig:.4f} inv={ep_inv:.4f}")

            # ── Checkpoint ────────────────────────────────────────────
            if epoch % cfg.ckpt_every == 0:
                ckpt_dir = Path("checkpoints")
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir / f"lejepa_{variant}_epoch{epoch}.pth"
                torch.save(net.backbone.state_dict(), ckpt_path)
                log.info(f"Checkpoint saved: {ckpt_path}")

                # Auto PCA visualization
                try:
                    pca_path = visualize_pca_inline(
                        net.backbone.state_dict(), variant, epoch,
                        out_dir="pca", device=device)
                    log.info(f"PCA visualization saved: {pca_path}")
                except Exception as e:
                    log.warning(f"PCA visualization failed at epoch {epoch}: {e}")

            # ── Check graceful stop ───────────────────────────────────
            if _stop_requested:
                log.info(f"Graceful stop after epoch {epoch}/{epochs}")
                ckpt_dir = Path("checkpoints")
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                stop_path = ckpt_dir / f"lejepa_{variant}_interrupted_epoch{epoch}.pth"
                torch.save(net.backbone.state_dict(), stop_path)
                log.info(f"Interrupted checkpoint saved: {stop_path}")
                break

    except KeyboardInterrupt:
        log.warning(f"Force interrupted during epoch {last_completed_epoch + 1}")
        if last_completed_epoch > 0:
            ckpt_dir = Path("checkpoints")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            stop_path = ckpt_dir / f"lejepa_{variant}_interrupted_epoch{last_completed_epoch}.pth"
            torch.save(net.backbone.state_dict(), stop_path)
            log.info(f"Best-effort checkpoint saved: {stop_path}")

    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, _original_sigint)

    # ── Save final backbone weights (if completed normally) ───────────
    if not _stop_requested and last_completed_epoch == epochs:
        ckpt_dir = Path("checkpoints")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        final_path = ckpt_dir / f"lejepa_{variant}_final.pth"
        torch.save(net.backbone.state_dict(), final_path)
        log.info(f"Final backbone saved: {final_path}")

        try:
            pca_path = visualize_pca_inline(
                net.backbone.state_dict(), variant, epochs,
                out_dir="pca", device=device)
            log.info(f"Final PCA visualization saved: {pca_path}")
        except Exception as e:
            log.warning(f"Final PCA visualization failed: {e}")

    if wandb_active:
        import wandb
        wandb.finish()

    # ── Plot loss curves (if any epochs completed) ────────────────────
    if history["lejepa"]:
        _plot_loss_curves(history, variant)

    return history


def _plot_loss_curves(history: dict, variant: str):
    """Save a loss curve plot to current working directory (hydra output dir)."""
    import matplotlib.pyplot as plt

    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs_range = range(1, len(history["lejepa"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(epochs_range, history["lejepa"], "b-", linewidth=2)
    axes[0].set_title("Total LeJEPA Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, history["inv"], "r-", linewidth=2)
    axes[1].set_title("Invariance Loss (should ↓)")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs_range, history["sigreg"], "g-", linewidth=2)
    axes[2].set_title("SIGReg Loss (should stay bounded)")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f"LeJEPA Training — MobileViT-{variant.upper()}", fontsize=13)
    plt.tight_layout()
    out_path = plots_dir / f"loss_curves_{variant}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    log.info(f"Loss curves saved: {out_path}")
    plt.close(fig)
