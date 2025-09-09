# train.py
# Line-by-line commented training loop for DDPM on CUB-200-2011.
from __future__ import annotations  # Future annotations

import os  # Filesystem operations
from argparse import ArgumentParser  # CLI argument parsing

import torch  # Deep learning core
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
    # Optional: keep small workers on Colab
    ap.add_argument('--num_workers', type=int, default=2)
    args = ap.parse_args()

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
    # Attach non-dataclass args if needed
    cfg.num_workers = args.num_workers  # type: ignore[attr-defined]
    return cfg

def main() -> None:
    """Main training entrypoint."""
    cfg = parse_args()  # Read CLI args
    set_seed(42)  # Fixed seed for reproducibility

    device = cfg.device  # Choose CUDA if available
    ensure_dir(cfg.outdir)  # Output directory

    # Build dataloader (use few workers on Colab to avoid warnings/freezes)
    loader = make_loader(cfg.data_root, cfg.img_size, cfg.batch_size, split="train", num_workers=getattr(cfg, "num_workers", 2))

    # Instantiate model and diffusion process
    model = UNet(img_channels=3, base=128, ch_mults=(1,2,2,4),
                 attn_res=(16,), num_classes=cfg.num_classes, cond_mode=cfg.cond_mode, t_dim=256).to(device)
    diffusion = Diffusion(T=cfg.num_steps, schedule=cfg.schedule, device=device)

    # Optimizer, EMA, and mixed precision scaler
    optim = AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    ema = EMA(model, decay=cfg.ema_decay)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    global_step = 0  # For logging/checkpoints
    for epoch in range(cfg.epochs):  # Epoch loop
        model.train()  # Train mode
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")  # Progress bar
        for x, y in pbar:  # Batch loop
            x = x.to(device)  # Move images
            y = y.to(device) if cfg.cond_mode == 'class' else None  # Labels if conditional
            t = torch.randint(0, cfg.num_steps, (x.size(0),), device=device).long()  # Random timesteps

            optim.zero_grad(set_to_none=True)  # Reset gradients
            with autocast(enabled=torch.cuda.is_available()):  # Mixed precision
                loss = diffusion.p_losses(model, x, t, y=y, p_uncond=cfg.p_uncond)  # DDPM loss

            scaler.scale(loss).backward()  # Backprop (scaled)
            scaler.step(optim)  # Optimizer step
            scaler.update()  # Update scaler
            ema.update()  # EMA step

            global_step += 1  # Update step counter
            pbar.set_postfix(loss=float(loss))  # Log loss

        # At end of epoch: sample a grid for qualitative monitoring
        model.eval()  # Eval mode
        ema.store(); ema.copy_to()  # Swap to EMA weights
        with torch.no_grad():
            y_sample = None
            if cfg.cond_mode == 'class':
                y_sample = torch.randint(0, cfg.num_classes, (64,), device=device)  # Random labels
            samples = diffusion.sample(model, (64, 3, cfg.img_size, cfg.img_size),
                                       y=y_sample, guidance_scale=cfg.guidance_scale)  # Generate
        save_grid(samples, os.path.join(cfg.outdir, f"samples_epoch_{epoch+1:03d}.png"), nrow=8)  # Save grid
        ema.restore()  # Restore non-EMA weights

        # Save checkpoint
        ckpt = {
            'model': model.state_dict(),
            'ema': ema.shadow,
            'optim': optim.state_dict(),
            'cfg': cfg.__dict__,
            'epoch': epoch + 1,
            'global_step': global_step,
        }
        torch.save(ckpt, os.path.join(cfg.outdir, 'last.ckpt'))  # Write to disk

if __name__ == "__main__":
    main()
