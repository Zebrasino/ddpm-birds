# train.py
# Line-by-line commented training loop for DDPM on CUB-200-2011.
from __future__ import annotations  # Future annotations

import os  # Filesystem operations
from argparse import ArgumentParser  # CLI argument parsing
from pathlib import Path  # Path handling

import torch  # Deep learning core
import torch.nn as nn  # For loss/optimizers
from torch.optim import AdamW  # Optimizer with decoupled weight decay
from torch.cuda.amp import GradScaler, autocast  # Mixed precision for speed
from tqdm import tqdm  # Progress bars

from utils import set_seed, TrainConfig, ensure_dir, save_grid, EMA  # Project utilities
from data import make_loader  # Data loading helper
from unet import UNet  # Model backbone
from diffusion import Diffusion  # Diffusion process

def parse_args() -> TrainConfig:
    """Parse command-line arguments into a TrainConfig dataclass."""
    ap = ArgumentParser()
    ap.add_argument('--data_root', type=str, default='./CUB_200_2011')
    ap.add_argument('--img_size', type=int, default=64)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--ema_decay', type=float, default=0.999)
    ap.add_argument('--num_steps', type=int, default=1000)
    ap.add_argument('--cond_mode', type=str, default='class', choices=['none', 'class'])
    ap.add_argument('--num_classes', type=int, default=200)
    ap.add_argument('--guidance_scale', type=float, default=1.0)
    ap.add_argument('--p_uncond', type=float, default=0.1)
    ap.add_argument('--schedule', type=str, default='cosine', choices=['linear', 'cosine'])
    ap.add_argument('--outdir', type=str, default='runs')
    args = ap.parse_args()

    # Populate TrainConfig from parsed args
    cfg = TrainConfig(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        ema_decay=args.ema_decay,
        num_steps=args.num_steps,
        cond_mode=args.cond_mode,
        num_classes=args.num_classes,
        guidance_scale=args.guidance_scale,
        p_uncond=args.p_uncond,
        schedule=args.schedule,
        outdir=args.outdir,
    )
    return cfg

def main() -> None:
    """Main training entrypoint."""
    cfg = parse_args()  # Read CLI args
    set_seed(42)  # Use a fixed seed for reproducibility

    device = cfg.device  # Choose CUDA if available
    ensure_dir(cfg.outdir)  # Create output directory for checkpoints and samples

    # Build dataloader
    loader = make_loader(cfg.data_root, cfg.img_size, cfg.batch_size, split="train")

    # Instantiate model and diffusion process
    model = UNet(img_channels=3, base=128, ch_mults=(1,2,2,4),
                 attn_res=(16,), num_classes=cfg.num_classes, cond_mode=cfg.cond_mode, t_dim=256).to(device)
    diffusion = Diffusion(T=cfg.num_steps, schedule=cfg.schedule, device=device)

    # Set up optimizer, EMA, and mixed precision scaler
    optim = AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    ema = EMA(model, decay=cfg.ema_decay)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # Training loop across epochs
    global_step = 0  # Counter for logging/checkpoints
    for epoch in range(cfg.epochs):
        model.train()  # Set model to training mode
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")  # Progress bar over batches
        for x, y in pbar:
            x = x.to(device)  # Move images to device
            y = y.to(device) if cfg.cond_mode == 'class' else None  # Labels if class-conditional
            # Sample a random timestep for each item in the batch
            t = torch.randint(0, cfg.num_steps, (x.size(0),), device=device).long()

            optim.zero_grad(set_to_none=True)  # Reset gradients
            with autocast(enabled=torch.cuda.is_available()):  # Mixed precision context
                # Compute DDPM loss (MSE between predicted and true noise)
                loss = diffusion.p_losses(model, x, t, y=y, p_uncond=cfg.p_uncond)

            # Backpropagate with gradient scaling to avoid underflow
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            # Update EMA weights after each step
            ema.update()

            global_step += 1  # Increment step counter
            pbar.set_postfix(loss=float(loss))  # Show current loss

        # At the end of each epoch, generate a small sample grid for qualitative monitoring
        model.eval()  # Switch to eval for sampling
        ema.store()  # Backup current weights
        ema.copy_to()  # Load EMA weights into the model for cleaner samples
        with torch.no_grad():
            # Sample 64 images with optional classifier-free guidance
            y_sample = None
            if cfg.cond_mode == 'class':
                y_sample = torch.randint(0, cfg.num_classes, (64,), device=device)
            samples = diffusion.sample(model, (64, 3, cfg.img_size, cfg.img_size),
                                       y=y_sample, guidance_scale=cfg.guidance_scale)
        save_grid(samples, os.path.join(cfg.outdir, f"samples_epoch_{epoch+1:03d}.png"), nrow=8)
        ema.restore()  # Restore original (non-EMA) weights

        # Save checkpoint after each epoch
        ckpt = {
            'model': model.state_dict(),
            'ema': ema.shadow,
            'optim': optim.state_dict(),
            'cfg': cfg.__dict__,
            'epoch': epoch + 1,
            'global_step': global_step,
        }
        torch.save(ckpt, os.path.join(cfg.outdir, 'last.ckpt'))

if __name__ == "__main__":
    main()
