from __future__ import annotations                       # future annotations
import os                                                # filesystem
import argparse                                          # CLI parsing
from typing import Optional                              # typing

import torch                                             # PyTorch
from torch.utils.data import DataLoader                  # data loader
from torchvision.utils import save_image                 # image saving

from data import make_cub_bbox_dataset                   # dataset factory
from unet import UNet                                    # model
from diffusion import Diffusion                          # DDPM core
from utils import EMA, seed_everything, cosine_warmup_lr, save_ckpt, load_ckpt  # utils

# AMP (use the cuda variant for widest compatibility on Colab)
from torch.cuda.amp import autocast, GradScaler          # AMP autocast + scaler

# -----------------------------
# CLI
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    """Define CLI arguments."""
    p = argparse.ArgumentParser()                                    # parser
    p.add_argument("--data_root", type=str, required=True, help="Path to CUB_200_2011 (folder with images/ and txts).")
    p.add_argument("--use_bbox", action="store_true", help="Crop around bounding boxes.")
    p.add_argument("--bbox_expand", type=float, default=1.0, help="Expansion factor for bbox crop (>=1).")
    p.add_argument("--class_limit", type=int, default=None, help="Limit number of classes (<=200).")
    p.add_argument("--subset", type=int, default=None, help="Limit number of images.")
    p.add_argument("--img_size", type=int, default=64, help="Square resolution.")
    p.add_argument("--outdir", type=str, required=True, help="Output directory for checkpoints and previews.")
    p.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    p.add_argument("--epochs", type=int, default=50, help="Max epochs (used only if max_steps is None).")
    p.add_argument("--max_steps", type=int, default=None, help="Stop after this many steps (preferred on Colab).")
    p.add_argument("--lr", type=float, default=2e-4, help="Base learning rate.")
    p.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    p.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value.")
    p.add_argument("--num_steps", type=int, default=1000, help="Diffusion total timesteps T.")
    p.add_argument("--schedule", type=str, choices=["cosine", "linear"], default="cosine", help="Beta schedule.")
    p.add_argument("--cond_mode", type=str, choices=["none", "class"], default="none", help="Conditioning type.")
    p.add_argument("--p_uncond", type=float, default=0.1, help="Classifier-free guidance drop prob during train.")
    p.add_argument("--guidance_scale", type=float, default=0.0, help="CFG guidance scale for previews.")
    p.add_argument("--base", type=int, default=64, help="UNet base width.")
    p.add_argument("--ema_mu", type=float, default=0.9995, help="EMA decay.")
    p.add_argument("--fg_weight", type=float, default=1.0, help="Foreground pixel weight multiplier in loss.")
    p.add_argument("--log_every", type=int, default=200, help="Log frequency (steps).")
    p.add_argument("--ckpt_every", type=int, default=2000, help="Checkpoint frequency (steps).")
    p.add_argument("--preview_every", type=int, default=0, help="Preview sampling frequency (steps, 0=off).")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    return p                                                          # return parser

# -----------------------------
# Preview helper
# -----------------------------
@torch.no_grad()
def preview_samples(
    model: UNet, diff: Diffusion, outdir: str, img_size: int, step: int,
    cond_mode: str, num_classes: Optional[int], guidance_scale: float
):
    """Save a small grid of samples (DDIM 35 steps, eta=0) for quick visual feedback."""
    model.eval()                                                      # eval mode
    os.makedirs(outdir, exist_ok=True)                                # ensure dir
    B = 16                                                            # grid size
    shape = (B, 3, img_size, img_size)                                # tensor shape

    # Choose labels for conditional model (random across seen classes)
    y = None                                                          # default None
    if cond_mode == "class" and num_classes is not None:              # conditional
        y = torch.randint(0, num_classes, (B,), device=next(model.parameters()).device)  # random labels

    # Use DDIM for fast previews
    x = diff.sample_ddim(
        model, shape, steps=35, eta=0.0, y=y,
        guidance_scale=(guidance_scale if y is not None else 0.0),    # CFG only if conditional
        skip_first=0
    )
    x = (x.clamp(-1, 1) + 1) / 2                                      # to [0,1]
    save_image(x, os.path.join(outdir, f"preview_step_{step:06d}.png"), nrow=4)  # save grid
    model.train()                                                     # back to train

# -----------------------------
# Main training
# -----------------------------
def main():
    args = build_parser().parse_args()                               # parse CLI

    seed_everything(args.seed)                                       # reproducibility

    device = "cuda" if torch.cuda.is_available() else "cpu"          # pick device
    torch.set_float32_matmul_precision("high")                       # speed hint

    os.makedirs(args.outdir, exist_ok=True)                          # ensure outdir

    # Build dataset (train split only)
    ds = make_cub_bbox_dataset(
        root=args.data_root,
        img_size=args.img_size,
        use_bbox=args.use_bbox,
        bbox_expand=args.bbox_expand,
        class_limit=args.class_limit,
        subset=args.subset,
        train_only=True,
    )                                                                 # CUB dataset

    # Safety: ensure non-empty dataset
    if len(ds) == 0:
        raise RuntimeError(f"No training samples found. Check --data_root, split, class_limit, subset.")

    # DataLoader
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )                                                                 # data loader

    # Build model (conditional if requested)
    num_classes = ds.num_classes if (args.cond_mode == "class") else None  # number of classes or None
    model = UNet(base=args.base, num_classes=num_classes).to(device)  # UNet

    # Diffusion buffers
    diffusion = Diffusion(T=args.num_steps, schedule=args.schedule, device=device)  # DDPM core

    # Optimizer / EMA / AMP scaler
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # AdamW
    ema = EMA(model, mu=args.ema_mu)                                 # EMA tracker
    scaler = GradScaler()                                             # AMP scaler

    # Optionally resume from checkpoint
    start_step = 0                                                    # default
    if args.resume is not None and os.path.isfile(args.resume):       # resume flag
        ck = load_ckpt(args.resume, map_location=device)              # load dict
        model.load_state_dict(ck["model"], strict=False)              # restore model
        if ck.get("ema") is not None:                                 # restore EMA
            ema.shadow.update(ck["ema"])
        start_step = int(ck.get("step", 0))                           # continue step

    # Compute total training steps
    if args.max_steps is not None:
        total_steps = args.max_steps                                  # explicit stop
    else:
        steps_per_epoch = len(dl)                                     # batches per epoch
        total_steps = steps_per_epoch * args.epochs                   # derive total

    step = start_step                                                 # init counter
    model.train()                                                     # train mode

    while step < total_steps:                                         # training loop
        for x, y, fg in dl:                                           # fetch batch (x: (B,3,H,W), y: int, fg: (B,1,H,W))
            if step >= total_steps:                                   # safety stop
                break

            x = x.to(device, non_blocking=True)                       # move image
            fg = fg.to(device, non_blocking=True)                     # move mask

            # For unconditional training, ignore labels; for class-conditional,
            # apply classifier-free guidance by dropping labels with prob p_uncond.
            y_in = None                                               # default no labels
            if args.cond_mode == "class":                             # conditional case
                y = y.to(device, non_blocking=True)                   # move labels
                if args.p_uncond > 0.0:                               # do CFG drop
                    drop = torch.rand_like(y.float()) < args.p_uncond # mask of drops
                    y_cf = y.clone()                                  # copy labels
                    y_cf[drop] = -1                                   # -1 signals NULL to UNet
                    y_in = y_cf                                       # use possibly-dropped labels
                else:
                    y_in = y                                          # use labels as-is

            # AMP autocast for forward/backward
            with autocast(enabled=(device == "cuda")):                # mixed precision
                loss = diffusion.p_losses(                             # compute loss
                    model, x0=x, y=y_in, fg_mask=fg, fg_weight=args.fg_weight
                )

            opt.zero_grad(set_to_none=True)                           # clear grads
            scaler.scale(loss).backward()                             # backprop in AMP
            if args.grad_clip is not None:                            # optional clip
                scaler.unscale_(opt)                                  # unscale first
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # clip
            scaler.step(opt)                                          # optimizer step
            scaler.update()                                           # update scaler

            # EMA update
            ema.update(model)                                         # maintain shadow weights

            # Cosine LR with warmup
            for g in opt.param_groups:                                # set LR per group
                g["lr"] = cosine_warmup_lr(step, total_steps, args.lr, warmup=min(1000, total_steps//20))

            # Logging each log_every
            if (step % args.log_every) == 0:                          # time to log
                print(f"step {step} | loss {loss.item():.4f} | lr {opt.param_groups[0]['lr']:.2e}")

            # Periodic preview (fast DDIM samples)
            if args.preview_every and (step % args.preview_every == 0) and (step > 0):
                # Use EMA weights for prettier previews
                shadow_model = UNet(base=args.base, num_classes=num_classes).to(device)  # temp model
                shadow_model.load_state_dict(ema.shadow, strict=False)                   # load EMA
                preview_samples(
                    shadow_model, diffusion, args.outdir, args.img_size, step,
                    args.cond_mode, num_classes, args.guidance_scale
                )

            # Periodic checkpointing
            if (step % args.ckpt_every) == 0 and (step > 0):          # save every N steps
                save_ckpt(os.path.join(args.outdir, "last.ckpt"), model, ema, vars(args), step)

            step += 1                                                 # increment step

            if step >= total_steps:                                   # stop if reached
                break

    # Save final checkpoint
    save_ckpt(os.path.join(args.outdir, "last.ckpt"), model, ema, vars(args), step)  # final save
    print(f"Saved checkpoint -> {os.path.join(args.outdir, 'last.ckpt')}")           # notify


if __name__ == "__main__":                                           # CLI entry
    main()                                                           # run main
