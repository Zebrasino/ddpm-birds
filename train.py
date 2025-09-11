# train.py
# Training script for (class-conditional or unconditional) DDPM on CUB.
# Key features:
#  - Stable dataloader defaults for Colab (num_workers=0)
#  - AMP with compatible fallback
#  - EMA
#  - Cosine LR schedule
#  - P2 loss + optional foreground (bbox) weighting
#  - In-training DDIM preview (fast) with minimal overhead
# Every logical line is commented.

import os
import math
import argparse
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from diffusion import Diffusion
from data import make_cub_bbox_dataset
from unet import UNet


# ---------- small EMA helper ----------
class EMA:
    """Exponential Moving Average for model weights."""
    def __init__(self, model: nn.Module, mu: float = 0.999):
        self.mu = mu                                      # decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.mu).add_(v.detach(), alpha=1.0 - self.mu)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=False)


# ---------- AMP glue that works on PyTorch 1.13–2.x ----------
def make_amp(device: str):
    """Return (autocast_ctx_fn, GradScaler_or_None) appropriate for this runtime."""
    scaler = None
    if device == "cuda":
        # prefer torch.amp if present
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            def ctx():
                return torch.amp.autocast(device_type="cuda")
            scaler = torch.amp.GradScaler("cuda") if hasattr(torch.amp, "GradScaler") else None
            return ctx, scaler
        else:
            from torch.cuda.amp import autocast, GradScaler
            def ctx():
                return autocast()
            scaler = GradScaler()
            return ctx, scaler
    # CPU path: nullcontext
    from contextlib import nullcontext
    return (lambda: nullcontext()), None


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)          # CUB_200_2011 root
    ap.add_argument("--use_bbox", action="store_true")               # crop around bbox
    ap.add_argument("--bbox_expand", type=float, default=1.0)        # bbox expansion
    ap.add_argument("--subset", type=int, default=None)              # first K images
    ap.add_argument("--class_limit", type=int, default=None)         # first N classes

    ap.add_argument("--outdir", type=str, required=True)             # output folder
    ap.add_argument("--img_size", type=int, default=48)              # resolution
    ap.add_argument("--batch_size", type=int, default=16)            # batch size
    ap.add_argument("--epochs", type=int, default=999)               # dummy cap (we stop by steps)
    ap.add_argument("--max_steps", type=int, default=12000)          # hard stop by steps

    ap.add_argument("--lr", type=float, default=2e-4)                # learning rate
    ap.add_argument("--weight_decay", type=float, default=0.0)       # wd if you want

    ap.add_argument("--num_steps", type=int, default=100)            # diffusion steps T
    ap.add_argument("--schedule", type=str, default="cosine", choices=["cosine", "linear"])  # β schedule

    ap.add_argument("--cond_mode", type=str, default="none", choices=["none", "class"])  # conditioning
    ap.add_argument("--p_uncond", type=float, default=0.1)           # CFG dropout (training)
    ap.add_argument("--base", type=int, default=96)                  # UNet base channels
    ap.add_argument("--ema_mu", type=float, default=0.999)           # EMA decay

    ap.add_argument("--seed", type=int, default=0)                   # reproducibility
    ap.add_argument("--log_every", type=int, default=200)            # print/sample interval

    # foreground loss weighting (1.0 = off)
    ap.add_argument("--fg_weight", type=float, default=3.0)          # weight inside bbox
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint .ckpt")


    return ap.parse_args()


def set_seed(seed: int):
    """Deterministic-ish runs for debugging."""
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def cosine_lr(step: int, max_steps: int, base_lr: float) -> float:
    """Simple cosine schedule from step 0..max_steps."""
    if max_steps <= 0:
        return base_lr
    cos = 0.5 * (1 + math.cos(math.pi * min(step, max_steps) / max_steps))
    return base_lr * cos


def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"          # pick device
    amp_ctx, scaler = make_amp(device)                               # AMP ctx and scaler

    # ----- Data -----
    ds = make_cub_bbox_dataset(
        root=args.data_root,
        img_size=args.img_size,
        use_bbox=args.use_bbox,
        bbox_expand=args.bbox_expand,
        class_limit=args.class_limit,
        subset=args.subset,
    )
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, persistent_workers=False, pin_memory=False, drop_last=True
    )

    # ----- Model -----
    num_classes = None if args.cond_mode == "none" else (args.class_limit or 200)
    model = UNet(base=args.base, num_classes=num_classes).to(device)  # ε-predictor

    # ----- Optimizer / EMA / Diffusion -----
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = EMA(model, mu=args.ema_mu)
    diffusion = Diffusion(T=args.num_steps, schedule=args.schedule, device=torch.device(device))

    os.makedirs(args.outdir, exist_ok=True)                           # out dir

    # ----- Training loop -----
    step = 0
    model.train()
    while step < args.max_steps:
        for batch in dl:
            if step >= args.max_steps:
                break

            # batch can be (x, y, mask)
            x, y, mask = batch
            x = x.to(device)                                         # (B,3,H,W) in [-1,1]
            y = None if (y is None) else torch.as_tensor(y, device=device, dtype=torch.long)
            mask = mask.to(device)                                   # (B,1,H,W)

            # update LR with cosine schedule
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, args.max_steps, args.lr)

            # forward + loss (with AMP if available)
            with amp_ctx():
                loss = diffusion.p_losses(
                    model, x, y=y, p_uncond=args.p_uncond,
                    p2_gamma=0.5, p2_k=1.0,
                    fg_mask=mask, fg_weight=args.fg_weight
                )

            # backward (scaled if AMP)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)                                  # so we can clip if desired
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            opt.zero_grad(set_to_none=True)                           # zero grads fast
            ema.update(model)                                         # update EMA
            step += 1

            # ----- Logging / quick preview -----
            if step % args.log_every == 0 or step == 1:
                print(f"step {step} | loss {loss.item():.4f} | lr {opt.param_groups[0]['lr']:.2e}")
                # quick 4x4 grid using DDIM (fast & robust)
                model.eval()
                with torch.no_grad():
                    x_gen = diffusion.sample_ddim(
                        model, shape=(16, 3, args.img_size, args.img_size),
                        y=None, steps=40, eta=0.1, guidance_scale=0.0, skip_first=10
                    )
                    x_vis = (x_gen.clamp(-1, 1) + 1) / 2
                    save_image(x_vis, os.path.join(args.outdir, f"samples_epoch_{step//args.log_every:03d}.png"), nrow=4)
                model.train()

    # ----- Save checkpoint (raw + EMA) -----
    ckpt = {
        "args": vars(args),                           # store hyper-params for eval
        "model": model.state_dict(),                  # last weights
        "ema": ema.shadow,                            # EMA weights
        "step": step,
    }
    torch.save(ckpt, os.path.join(args.outdir, "last.ckpt"))
    print("Saved checkpoint ->", os.path.join(args.outdir, "last.ckpt"))


if __name__ == "__main__":
    main()

