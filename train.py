# =============================== train.py =====================================
# Every line commented in English for clarity.

from __future__ import annotations                 # use postponed annotations (py>=3.7)

import os                                          # filesystem utilities
import argparse                                    # command-line interface
from typing import Optional                        # typing for optional fields

import torch                                       # main PyTorch
from torch.utils.data import DataLoader            # batching & workers
from torchvision.utils import save_image           # image grid writer

# Project modules (you already have these files)
from data import make_cub_bbox_dataset             # CUB dataset factory (bbox-aware)
from unet import UNet                              # U-Net backbone
from diffusion import Diffusion                    # DDPM core (schedule, samplers)
from utils import (                                # small helpers from your utils.py
    EMA,
    seed_everything,
    cosine_warmup_lr,
    save_ckpt,
    load_ckpt,
)

# AMP: use the CUDA variant (works as no-op on CPU)
from torch.cuda.amp import autocast, GradScaler    # autocast + gradient scaler


# -----------------------------
# CLI (argument parser)
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    """Define all CLI arguments used by this script."""
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--data_root", type=str, required=True,
                   help="Path to CUB_200_2011 (folder with images/ and txts).")
    p.add_argument("--use_bbox", action="store_true",
                   help="Crop around bounding boxes.")
    p.add_argument("--bbox_expand", type=float, default=1.0,
                   help="BBox expansion factor (>=1.0).")
    p.add_argument("--class_limit", type=int, default=None,
                   help="Limit number of classes to load (<=200).")
    p.add_argument("--subset", type=int, default=None,
                   help="Limit number of images to load (per run).")
    p.add_argument("--img_size", type=int, default=64,
                   help="Square image size (H=W).")

    # Output
    p.add_argument("--outdir", type=str, required=True,
                   help="Output directory (checkpoints, previews).")

    # Training loop
    p.add_argument("--batch_size", type=int, default=16,
                   help="Mini-batch size.")
    p.add_argument("--epochs", type=int, default=50,
                   help="Max epochs (used only if --max_steps is not set).")
    p.add_argument("--max_steps", type=int, default=None,
                   help="Stop after this many optimizer steps (preferred on Colab).")

    # Optimization
    p.add_argument("--lr", type=float, default=2e-4,
                   help="Base learning rate.")
    p.add_argument("--weight_decay", type=float, default=0.0,
                   help="AdamW weight decay.")
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Gradient norm clipping value (None/0 to disable).")

    # Diffusion
    p.add_argument("--num_steps", type=int, default=1000,
                   help="Total diffusion steps T.")
    p.add_argument("--schedule", type=str, choices=["cosine", "linear"], default="cosine",
                   help="Beta schedule type.")

    # Conditioning / CFG
    p.add_argument("--cond_mode", type=str, choices=["none", "class"], default="none",
                   help="Conditioning type: uncond or class-conditional.")
    p.add_argument("--p_uncond", type=float, default=0.1,
                   help="Classifier-free guidance: probability to drop labels at train.")
    p.add_argument("--guidance_scale", type=float, default=0.0,
                   help="CFG guidance scale used for *previews* during training.")

    # Model
    p.add_argument("--base", type=int, default=64,
                   help="U-Net base channel width.")
    p.add_argument("--ema_mu", type=float, default=0.9995,
                   help="EMA decay factor (closer to 1 = slower).")

    # Loss weighting (foreground mask)
    p.add_argument("--fg_weight", type=float, default=1.0,
                   help="Foreground pixel weight multiplier inside the loss.")

    # Logging / checkpoints / previews
    p.add_argument("--log_every", type=int, default=200,
                   help="Print loss and LR every N steps.")
    p.add_argument("--ckpt_every", type=int, default=2000,
                   help="Save a checkpoint every N steps.")
    p.add_argument("--preview_every", type=int, default=0,
                   help="Save an image preview every N steps (0 = off).")

    # Repro / resume
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed.")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from checkpoint path (.ckpt).")

    return p


# -----------------------------
# Quick DDIM preview helper
# -----------------------------
@torch.no_grad()
def preview_samples(
    model,                                   # UNet (you can pass EMA-loaded weights)
    diff,                                    # Diffusion object (exposes sample_ddim)
    outdir: str,                             # folder where to save the grid
    img_size: int,                           # resolution H=W
    step: int,                               # current global step (for filename)
    cond_mode: str,                          # 'none' or 'class'
    num_classes: Optional[int],              # number of classes if conditional
    guidance_scale: float                    # CFG scale to use at sampling time
):
    """Save a 4x4 grid preview using DDIM (fast & deterministic when eta=0.0)."""
    was_training = model.training            # remember current train/eval mode
    model.eval()                             # switch to eval for sampling
    os.makedirs(outdir, exist_ok=True)       # ensure output dir exists

    B = 16                                   # 16 images -> 4x4 grid
    shape = (B, 3, img_size, img_size)       # (B,C,H,W)
    device = next(model.parameters()).device # infer device from model

    # If class-conditional, sample random labels to cover the label space in preview.
    y = None                                 # default: unconditional sampling
    if cond_mode == "class" and (num_classes is not None) and (num_classes > 0):
        y = torch.randint(0, num_classes, (B,), device=device)  # random classes

    # Run DDIM with a small number of steps for a quick visual check.
    x = diff.sample_ddim(
        model=model,                         # UNet (EMA is recommended for previews)
        shape=shape,                         # output shape
        steps=35,                            # few steps -> fast
        eta=0.0,                             # deterministic
        y=y,                                 # None or labels
        guidance_scale=(guidance_scale if y is not None else 0.0),
        skip_first=0                         # do not skip early steps
    )

    # Convert from [-1,1] to [0,1] for saving and write a 4x4 grid.
    x = (x.clamp(-1, 1) + 1) / 2
    save_image(x, os.path.join(outdir, f"preview_step_{step:06d}.png"), nrow=4)

    model.train(was_training)                # restore original training/eval mode


# -----------------------------
# Main training entry
# -----------------------------
def main():
    # Parse CLI
    args = build_parser().parse_args()                   # read all CLI flags

    # Seeding for reproducibility
    seed_everything(args.seed)                           # set torch/python/numpy seeds

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # pick GPU if available
    torch.set_float32_matmul_precision("high")           # small speed hint on Ampere+

    # Output directory
    os.makedirs(args.outdir, exist_ok=True)              # make sure out folder exists

    # -------------------------
    # Dataset & DataLoader
    # -------------------------
    ds = make_cub_bbox_dataset(                          # build CUB dataset (train split)
        root=args.data_root,
        img_size=args.img_size,
        use_bbox=args.use_bbox,
        bbox_expand=args.bbox_expand,
        class_limit=args.class_limit,
        subset=args.subset,
        train_only=True,                                 # only training split
    )

    # Safety: no empty dataset
    if len(ds) == 0:
        raise RuntimeError(
            f"No training samples found at {args.data_root}. "
            f"Check --data_root / split / class_limit / subset."
        )

    # DataLoader with pinned memory on CUDA for speed
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # Decide conditionality (class-conditional = number of classes, else None)
    num_classes = ds.num_classes if (args.cond_mode == "class") else None

    # -------------------------
    # Model / Diffusion / Opt
    # -------------------------
    model = UNet(base=args.base, num_classes=num_classes).to(device)  # build UNet
    diffusion = Diffusion(T=args.num_steps, schedule=args.schedule, device=device)  # DDPM core

    # Optimizer (AdamW) and EMA
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = EMA(model, mu=args.ema_mu)                                 # keep EMA of weights

    # AMP scaler (safe no-op on CPU)
    scaler = GradScaler()

    # -------------------------
    # Resume (optional)
    # -------------------------
    start_step = 0                                                   # default start
    if args.resume is not None and os.path.isfile(args.resume):      # if resume path provided
        ck = load_ckpt(args.resume, map_location=device)             # load checkpoint dict
        # Restore model weights
        model.load_state_dict(ck["model"], strict=False)
        # Restore EMA weights (if present)
        if ck.get("ema") is not None:
            try:
                ema.shadow.load_state_dict(ck["ema"])                # if EMA stored as state_dict
            except Exception:
                # in some utils, ema.shadow may be a plain dict of tensors
                ema.shadow = ck["ema"]
        # Restore step counter
        start_step = int(ck.get("step", 0))
        # (Optionally) restore optimizer / RNG if your load_ckpt provides them.

    # -------------------------
    # Compute total steps
    # -------------------------
    if args.max_steps is not None:
        total_steps = int(args.max_steps)                            # prefer explicit stop
    else:
        steps_per_epoch = len(dl)                                    # batches per epoch
        total_steps = steps_per_epoch * int(args.epochs)             # derive from epochs

    # -------------------------
    # Training loop
    # -------------------------
    step = start_step                                                # global step counter
    model.train()                                                    # set train mode

    # Helper to materialize an EMA model for sampling previews
    def build_ema_model_for_preview() -> UNet:
        """Create a temp UNet, load EMA weights into it, and return it."""
        m = UNet(base=args.base, num_classes=num_classes).to(device) # same arch
        try:
            m.load_state_dict(ema.shadow, strict=False)              # load EMA shadow
        except Exception:
            # If ema.shadow is a dict of tensors keyed by param name, copy each tensor
            sd = m.state_dict()
            for k in sd:
                if k in ema.shadow:
                    sd[k].copy_(ema.shadow[k])
            m.load_state_dict(sd, strict=False)
        m.eval()                                                     # eval for sampling
        return m

    # Main optimize loop
    while step < total_steps:
        for batch in dl:                                             # iterate batches
            if step >= total_steps:                                  # stop exactly at total_steps
                break

            # ---------------------------------------------------------
            # Parse batch (supports (x,y,fg) or (x,y) or single x only)
            # ---------------------------------------------------------
            x0, y_in, fg = None, None, None                          # defaults
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    x0, y_in, fg = batch                             # images, labels, fg mask
                elif len(batch) == 2:
                    x0, y_in = batch                                 # images, labels
                else:
                    x0 = batch[0]                                    # images only
            else:
                x0 = batch                                           # images only

            # Move tensors to device
            x0 = x0.to(device, non_blocking=True)                    # (B,3,H,W) in [-1,1]
            if y_in is not None:
                y_in = y_in.to(device, non_blocking=True).long()     # class ids if provided
            if fg is not None:
                fg = fg.to(device, non_blocking=True).float()        # foreground mask (optional)

            # Classifier-Free Guidance label dropping at train (only if class-conditional)
            y_cf = None                                              # default: unconditional
            if num_classes is not None:
                if args.p_uncond > 0.0:
                    # sample a Bernoulli mask to drop labels with prob p_uncond
                    drop = torch.rand(x0.size(0), device=device) < args.p_uncond
                    y_cf = y_in.clone()
                    # set -1 for dropped items (UNet should map -1 -> NULL class)
                    y_cf[drop] = -1
                else:
                    y_cf = y_in

            # -------------------------
            # Forward + loss (AMP)
            # -------------------------
            optimizer.zero_grad(set_to_none=True)                    # clear gradients (fast)

            with autocast(enabled=(device.type == "cuda")):          # mixed precision on GPU
                # Call your diffusion training loss. The function itself
                # should sample timesteps/noise internally.
                loss = diffusion.p_losses(
                    model,
                    x0=x0,                                           # clean images in [-1,1]
                    y=y_cf,                                          # labels (or None / -1 for NULL)
                    fg_mask=fg,                                      # optional foreground mask
                    fg_weight=float(args.fg_weight),                 # foreground loss multiplier
                )

            # Backprop with gradient scaling
            scaler.scale(loss).backward()                            # scaled backward
            if args.grad_clip and args.grad_clip > 0:                # optional grad clip
                scaler.unscale_(optimizer)                           # unscale first
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)                                   # optimizer update
            scaler.update()                                          # scaler update

            # EMA update (track moving average of weights)
            ema.update(model)

            # Cosine LR with warmup (set LR each step)
            warmup = min(1000, max(1, total_steps // 20))            # e.g. 5% or 1000 steps
            lr_now = cosine_warmup_lr(step, total_steps, args.lr, warmup=warmup)
            for g in optimizer.param_groups:
                g["lr"] = lr_now

            # Increase global step counter
            step += 1

            # Logging
            if (step % args.log_every) == 0 or step == 1:
                print(f"step {step} | loss {loss.item():.4f} | lr {lr_now:.2e}")

            # Preview (DDIM) — if enabled
            if args.preview_every and args.preview_every > 0 and (step % args.preview_every) == 0:
                # Use EMA weights for cleaner previews
                ema_model = build_ema_model_for_preview()
                preview_samples(
                    model=ema_model,                                  # EMA snapshot
                    diff=diffusion,                                   # diffusion object
                    outdir=args.outdir,                               # save in outdir
                    img_size=args.img_size,                           # same resolution as train
                    step=step,                                        # file name uses step
                    cond_mode=args.cond_mode,                         # 'none' or 'class'
                    num_classes=num_classes,                          # used if conditional
                    guidance_scale=args.guidance_scale,               # typical 2.0-5.0
                )

            # Checkpointing — save both periodic and at the very end
            if args.ckpt_every and args.ckpt_every > 0 and (step % args.ckpt_every) == 0:
                save_ckpt(os.path.join(args.outdir, f"step_{step:06d}.ckpt"),
                          model, ema, vars(args), step)

            # Stop cleanly if we reached the limit
            if step >= total_steps:
                break

    # Final save (last.ckpt)
    save_ckpt(os.path.join(args.outdir, "last.ckpt"), model, ema, vars(args), step)
    print(f"Saved checkpoint -> {os.path.join(args.outdir, 'last.ckpt')}")


# Standard Python entry point
if __name__ == "__main__":
    main()
# ==============================================================================
                                         # run main
