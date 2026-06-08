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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  METER Fine-tuning (Depth Estimation)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def finetune_depth(cfg: DictConfig) -> dict:
    """Run METER depth estimation fine-tuning.

    Trains encoder + decoder for monocular depth prediction using the
    Balanced Loss Function (BLF) from the METER paper.

    Args:
        cfg: Hydra DictConfig with all training hyperparameters.
    Returns:
        dict with training history and best metrics.
    """
    from src.model import METERModel
    from src.loss import BalancedDepthLoss, compute_depth_metrics
    from src.data import get_depth_loader

    variant = cfg.variant
    device = _resolve_device(cfg)
    ft_cfg = cfg.finetune
    resolution = tuple(ft_cfg.resolution)  # (H, W)
    epochs = ft_cfg.epochs
    bs = ft_cfg.get("bs", cfg.get("bs", 8))

    log.info(f"METER Fine-tuning: MobileViT-{variant.upper()}")
    log.info(f"Resolution: {resolution} | Epochs: {epochs} | BS: {bs} | Device: {device}")

    # ── Model ─────────────────────────────────────────────────────────
    model = METERModel(variant=variant, resolution=resolution).to(device)

    # Load pretrained encoder if specified
    pretrained = ft_cfg.get("pretrained_encoder")
    if pretrained:
        from src.config import ROOT
        ckpt_path = Path(pretrained)
        if not ckpt_path.is_absolute():
            ckpt_path = ROOT / ckpt_path
        model.load_pretrained_encoder(str(ckpt_path))
        log.info(f"Loaded pretrained encoder from: {ckpt_path}")
    else:
        log.info("Training from scratch (no pretrained encoder).")

    params = sum(p.numel() for p in model.parameters())
    log.info(f"Total parameters: {params:,} ({params/1e6:.2f}M)")

    # ── Encoder freezing ──────────────────────────────────────────────
    freeze_epochs = ft_cfg.get("freeze_encoder_epochs", 0)
    if freeze_epochs > 0:
        for param in model.encoder.parameters():
            param.requires_grad = False
        log.info(f"Encoder frozen for first {freeze_epochs} epochs.")

    # ── Loss, optimizer, scheduler ────────────────────────────────────
    criterion = BalancedDepthLoss(
        lamb1=ft_cfg.get("lamb1", 0.5),
        lamb2=ft_cfg.get("lamb2", 1.0),
        lamb3=ft_cfg.get("lamb3", 1.0),
    ).to(device)

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=ft_cfg.lr, weight_decay=ft_cfg.weight_decay,
        betas=(0.9, 0.999))

    scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=ft_cfg.lr_step_size, gamma=ft_cfg.lr_gamma)

    # ── Data ──────────────────────────────────────────────────────────
    train_loader = get_depth_loader(cfg, device=device, split="train")
    val_every = ft_cfg.get("val_every", 5)

    scaler = GradScaler(enabled=(device == "cuda"))
    history = {"total": [], "depth": [], "grad": [], "norm": [], "ssim": []}
    best_metrics = {}

    # ── WandB ─────────────────────────────────────────────────────────
    wandb_active = _init_wandb(cfg) if cfg.get("use_wandb", False) else False

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()

        # Unfreeze encoder after freeze period
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            for param in model.encoder.parameters():
                param.requires_grad = True
            # Re-create optimizer with all parameters
            opt = torch.optim.AdamW(
                model.parameters(), lr=ft_cfg.lr,
                weight_decay=ft_cfg.weight_decay, betas=(0.9, 0.999))
            scheduler = torch.optim.lr_scheduler.StepLR(
                opt, step_size=ft_cfg.lr_step_size, gamma=ft_cfg.lr_gamma)
            log.info(f"Encoder unfrozen at epoch {epoch}.")

        ep_losses = {k: 0.0 for k in history}
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)

        for rgb, depth_gt in pbar:
            rgb = rgb.to(device, non_blocking=True)
            depth_gt = depth_gt.to(device, non_blocking=True)

            with autocast(device, dtype=torch.bfloat16):
                depth_pred = model(rgb)
                loss, components = criterion(depth_pred, depth_gt)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            for k in ep_losses:
                ep_losses[k] += components[k]
            pbar.set_postfix(loss=f"{components['total']:.4f}")

            if wandb_active:
                import wandb
                wandb.log({f"train/{k}": v for k, v in components.items()})

        # ── Epoch summary ─────────────────────────────────────────────
        scheduler.step()
        n = len(train_loader)
        for k in history:
            history[k].append(ep_losses[k] / n)

        log.info(f"Epoch {epoch:>3} | loss={history['total'][-1]:.4f} "
                 f"depth={history['depth'][-1]:.4f} "
                 f"grad={history['grad'][-1]:.4f} "
                 f"lr={scheduler.get_last_lr()[0]:.6f}")

        # ── Checkpoint ────────────────────────────────────────────────
        ckpt_every = cfg.get("ckpt_every", 10)
        if epoch % ckpt_every == 0:
            ckpt_dir = Path("checkpoints")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"meter_{variant}_epoch{epoch}.pth"
            torch.save(model.state_dict(), ckpt_path)
            log.info(f"Checkpoint saved: {ckpt_path}")

            # Depth prediction visualization
            from src.visualize import visualize_depth_inline
            vis_path = visualize_depth_inline(
                model, variant, epoch, out_dir="plots", device=device)
            log.info(f"Depth visualization saved: {vis_path}")
            model.train()

        # ── Validation ────────────────────────────────────────────────
        if epoch % val_every == 0 or epoch == epochs:
            metrics = _validate_depth(model, cfg, device)
            log.info(f"  Val | d1={metrics['delta1']:.4f} "
                     f"RMSE={metrics['rmse']:.4f} "
                     f"REL={metrics['rel']:.4f}")
            best_metrics = metrics

            if wandb_active:
                import wandb
                wandb.log({f"val/{k}": v for k, v in metrics.items()})

    # ── Save final model ──────────────────────────────────────────────
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_path = ckpt_dir / f"meter_{variant}_final.pth"
    torch.save(model.state_dict(), final_path)
    log.info(f"Final model saved: {final_path}")

    # Final depth visualization
    from src.visualize import visualize_depth_inline
    vis_path = visualize_depth_inline(
        model, variant, epochs, out_dir="plots", device=device)
    log.info(f"Final depth visualization: {vis_path}")

    if wandb_active:
        import wandb
        wandb.finish()

    # ── Plot loss curves ──────────────────────────────────────────────
    if history["total"]:
        _plot_depth_loss_curves(history, variant)

    return {"history": history, "best_metrics": best_metrics}


@torch.no_grad()
def _validate_depth(model, cfg, device) -> dict:
    """Run validation and compute depth metrics."""
    from src.data import get_depth_loader
    from src.loss import compute_depth_metrics

    model.eval()
    val_loader = get_depth_loader(cfg, device=device, split="val")

    all_metrics = []
    for rgb, depth_gt in val_loader:
        rgb = rgb.to(device, non_blocking=True)
        depth_gt = depth_gt.to(device, non_blocking=True)
        depth_pred = model(rgb)
        metrics = compute_depth_metrics(depth_pred, depth_gt)
        all_metrics.append(metrics)

    # Average metrics across batches
    avg = {}
    for key in all_metrics[0]:
        avg[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
    return avg


def _plot_depth_loss_curves(history: dict, variant: str):
    """Save depth fine-tuning loss curves."""
    import matplotlib.pyplot as plt

    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs_range = range(1, len(history["total"]) + 1)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    axes[0].plot(epochs_range, history["total"], "b-", linewidth=2)
    axes[0].set_title("Total BLF Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, history["depth"], "r-", linewidth=2)
    axes[1].set_title("L1 Depth Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs_range, history["grad"], "g-", linewidth=2)
    axes[2].set_title("Gradient Loss")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(epochs_range, history["ssim"], "m-", linewidth=2)
    axes[3].set_title("SSIM Loss")
    axes[3].set_xlabel("Epoch")
    axes[3].grid(True, alpha=0.3)

    plt.suptitle(f"METER Depth Fine-tuning — MobileViT-{variant.upper()}", fontsize=13)
    plt.tight_layout()
    out_path = plots_dir / f"depth_loss_curves_{variant}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    log.info(f"Depth loss curves saved: {out_path}")
    plt.close(fig)
