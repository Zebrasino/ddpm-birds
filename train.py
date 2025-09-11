from __future__ import annotations                              # postponed hints
import os                                                       # filesystem
import argparse                                                 # CLI parsing
from typing import Optional                                     # typing

import torch                                                    # PyTorch
from torch.utils.data import DataLoader                         # DataLoader
from torchvision.utils import save_image                        # image save

from data import make_cub_bbox_dataset                          # dataset factory
from unet import UNet                                           # model
from diffusion import Diffusion                                 # DDPM core
from utils import EMA, seed_everything, cosine_warmup_lr, save_ckpt, load_ckpt  # utils

from torch.cuda.amp import autocast, GradScaler                 # AMP autocast+scaler


def build_parser() -> argparse.ArgumentParser:
    """Define CLI arguments."""
    p = argparse.ArgumentParser()                               # create parser
    # ---- data ----
    p.add_argument("--data_root", type=str, required=True, help="Path to CUB_200_2011")
    p.add_argument("--use_bbox", action="store_true", help="Crop around bounding boxes")
    p.add_argument("--bbox_expand", type=float, default=1.0, help="BBox expansion (>=1.0)")
    p.add_argument("--class_limit", type=int, default=None, help="Limit number of classes (<=200)")
    p.add_argument("--subset", type=int, default=None, help="Limit total images")
    p.add_argument("--img_size", type=int, default=64, help="Square resolution")
    p.add_argument("--outdir", type=str, required=True, help="Output directory")
    # ---- optimization ----
    p.add_argument("--batch_size", type=int, default=16, help="Batch size")
    p.add_argument("--epochs", type=int, default=50, help="Epochs if max_steps is None")
    p.add_argument("--max_steps", type=int, default=None, help="Stop after this many steps")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    p.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    p.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    # ---- diffusion ----
    p.add_argument("--num_steps", type=int, default=200, help="Diffusion steps T")
    p.add_argument("--schedule", type=str, choices=["cosine", "linear"], default="cosine", help="Beta schedule")
    # ---- conditioning / guidance ----
    p.add_argument("--cond_mode", type=str, choices=["none", "class"], default="none", help="Conditioning type")
    p.add_argument("--p_uncond", type=float, default=0.1, help="CFG drop prob during training")
    p.add_argument("--guidance_scale", type=float, default=0.0, help="CFG scale for previews")
    # ---- model / EMA ----
    p.add_argument("--base", type=int, default=64, help="UNet base width")
    p.add_argument("--ema_mu", type=float, default=0.9995, help="EMA decay")
    # ---- loss weighting ----
    p.add_argument("--fg_weight", type=float, default=1.0, help="Foreground mask weight multiplier")
    # ---- logging / save ----
    p.add_argument("--log_every", type=int, default=200, help="Log every N steps")
    p.add_argument("--ckpt_every", type=int, default=2000, help="Save checkpoint every N steps")
    p.add_argument("--preview_every", type=int, default=1000, help="Save preview every N steps (0=off)")
    # ---- misc ----
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--resume", type=str, default=None, help="Path to resume checkpoint")
    return p                                                     # return parser


@torch.no_grad()
def preview_samples(
    model: UNet,                                     # UNet (EMA weights recommended)
    diff: Diffusion,                                 # diffusion helper
    outdir: str,                                     # where to save
    img_size: int,                                   # resolution H=W
    step: int,                                       # global step for filename
    cond_mode: str,                                  # 'none' or 'class'
    num_classes: Optional[int],                      # number of classes if conditional
    guidance_scale: float,                           # CFG scale for sampling
):
    """Save a 4x4 DDIM preview (fast & stable)."""
    was_training = model.training                    # remember mode
    model.eval()                                     # eval for sampling
    os.makedirs(outdir, exist_ok=True)               # ensure folder

    B = 16                                           # grid size (4x4)
    shape = (B, 3, img_size, img_size)               # output tensor shape
    device = next(model.parameters()).device         # infer device

    y = None                                         # default: unconditional
    if cond_mode == "class" and (num_classes is not None) and (num_classes > 0):
        y = torch.randint(0, num_classes, (B,), device=device)     # random labels

    x = diff.sample_ddim(                            # run sampler
        model=model, shape=shape, steps=35, eta=0.0, y=y,
        guidance_scale=(guidance_scale if y is not None else 0.0),
        skip_first=0
    )
    x = (x.clamp(-1, 1) + 1) / 2                     # map to [0,1]
    save_image(x, os.path.join(outdir, f"preview_step_{step:06d}.png"), nrow=4)  # save grid
    model.train(was_training)                        # restore mode


def main():
    args = build_parser().parse_args()               # parse CLI
    seed_everything(args.seed)                       # reproducibility

    device = "cuda" if torch.cuda.is_available() else "cpu"     # pick device
    torch.set_float32_matmul_precision("high")       # small perf boost

    os.makedirs(args.outdir, exist_ok=True)          # ensure outdir exists

    # ---- dataset / loader ----
    ds = make_cub_bbox_dataset(                      # build dataset (train split)
        root=args.data_root,
        img_size=args.img_size,
        use_bbox=args.use_bbox,
        bbox_expand=args.bbox_expand,
        class_limit=args.class_limit,
        subset=args.subset,
        train_only=True,
    )
    if len(ds) == 0:                                 # sanity check
        raise RuntimeError("Empty dataset — check paths/split/class_limit/subset.")

    dl = DataLoader(                                 # DataLoader
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=(device == "cuda"), drop_last=True
    )

    # ---- model / diffusion / optimizer / ema / amp ----
    num_classes = ds.num_classes if (args.cond_mode == "class") else None  # conditional?
    model = UNet(base=args.base, num_classes=num_classes).to(device)       # UNet
    diff = Diffusion(T=args.num_steps, schedule=args.schedule, device=torch.device(device))  # DDPM core
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # optimizer
    ema = EMA(model, mu=args.ema_mu)                       # EMA tracker
    scaler = GradScaler()                                   # AMP scaler

    # ---- resume (optional) ----
    step = 0                                               # global step
    if args.resume is not None and os.path.isfile(args.resume):
        ck = load_ckpt(args.resume, map_location=device)   # load dict
        model.load_state_dict(ck["model"], strict=False)   # restore weights
        if ck.get("ema") is not None:                      # restore EMA if present
            ema.shadow.update(ck["ema"])
        step = int(ck.get("step", 0))                      # continue from here

    # ---- training length ----
    total_steps = args.max_steps if args.max_steps is not None else len(dl) * args.epochs

    # ---- train loop ----
    model.train()                                          # train mode
    while step < total_steps:                              # stop at total_steps
        for x, y, fg in dl:                                # fetch one batch
            if step >= total_steps:                        # hard stop
                break
            x = x.to(device, non_blocking=True)            # move image
            fg = fg.to(device, non_blocking=True)          # move fg mask

            # Optional class conditioning with classifier-free guidance drop
            y_in = None                                    # default: uncond
            if args.cond_mode == "class":
                y = y.to(device, non_blocking=True)        # move labels
                if args.p_uncond > 0.0:                    # apply label dropout
                    drop = torch.rand_like(y.float()) < args.p_uncond
                    y_cf = y.clone()
                    y_cf[drop] = -1                        # -1 → NULL class in UNet
                    y_in = y_cf
                else:
                    y_in = y

            # Forward + loss (AMP)
            with autocast(enabled=(device == "cuda")):
                loss = diff.p_losses(
                    model, x0=x, y=y_in, p_uncond=args.p_uncond,
                    fg_mask=fg, fg_weight=args.fg_weight
                )

            opt.zero_grad(set_to_none=True)                # clear grads
            scaler.scale(loss).backward()                  # backward (AMP)
            if args.grad_clip is not None:                 # optional grad clip
                scaler.unscale_(opt)                       # unscale before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)                               # optim step
            scaler.update()                                # update scaler

            ema.update(model)                              # EMA shadow update

            # Cosine LR with warmup
            for g in opt.param_groups:
                g["lr"] = cosine_warmup_lr(step, total_steps, args.lr, warmup=min(1000, total_steps // 20))

            # Logging
            if (step % args.log_every) == 0:
                print(f"step {step} | loss {loss.item():.4f} | lr {opt.param_groups[0]['lr']:.2e}")

            # Preview images (fast DDIM 35 steps)
            if args.preview_every and ((step == 1) or (step % args.preview_every == 0)):
                # copy EMA weights into a temp model for clean previews
                tmp = UNet(base=args.base, num_classes=num_classes).to(device)
                tmp.load_state_dict(model.state_dict(), strict=False)
                ema.copy_to(tmp)
                preview_samples(
                    model=tmp, diff=diff, outdir=args.outdir, img_size=args.img_size,
                    step=step, cond_mode=args.cond_mode, num_classes=num_classes,
                    guidance_scale=args.guidance_scale
                )
                del tmp

            # Periodic checkpoint
            if (step % args.ckpt_every) == 0 and (step > 0):
                save_ckpt(os.path.join(args.outdir, "last.ckpt"), model, ema, vars(args), step)

            step += 1                                        # next step
            if step >= total_steps:                          # hard stop
                break

    # Final save
    save_ckpt(os.path.join(args.outdir, "last.ckpt"), model, ema, vars(args), step)
    print(f"Saved checkpoint -> {os.path.join(args.outdir, 'last.ckpt')}")


if __name__ == "__main__":                                   # script entry
    main()                                                   # run training
