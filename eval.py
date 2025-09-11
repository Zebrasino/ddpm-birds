# eval.py
# Sampling/Eval script with:
#  - DDPM (ancestral/deterministic) and DDIM
#  - EMA option
#  - CFG (for class-conditional models)
#  - skip_first/eta controls for DDIM
# Every line is commented.

import argparse
import os
import torch
from torchvision.utils import save_image

from unet import UNet
from diffusion import Diffusion


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="sample", choices=["sample"])  # only sampling supported here
    ap.add_argument("--checkpoint", type=str, required=True)                   # path to .ckpt
    ap.add_argument("--img_size", type=int, default=None)                      # override resolution
    ap.add_argument("--outdir", type=str, required=True)                       # output dir
    ap.add_argument("--num_samples", type=int, default=16)                     # number of images
    ap.add_argument("--use_ema", action="store_true")                          # use EMA weights
    ap.add_argument("--guidance_scale", type=float, default=0.0)               # CFG scale

    # class-conditional label (if model was trained with classes)
    ap.add_argument("--label", type=int, default=None)

    # sampler options
    ap.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddpm_det", "ddim"])
    ap.add_argument("--ddim_steps", type=int, default=50)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--skip_first", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- load checkpoint & config ----
    ck = torch.load(args.checkpoint, map_location=device)
    cfg = ck["args"]
    img_size = int(args.img_size or cfg.get("img_size", 48))
    base = int(cfg.get("base", 96))
    T = int(cfg.get("num_steps", 100))
    schedule = cfg.get("schedule", "cosine")
    cond_mode = cfg.get("cond_mode", "none")

    # ---- build model ----
    num_classes = None if cond_mode == "none" else (cfg.get("class_limit") or 200)
    model = UNet(base=base, num_classes=num_classes).to(device).eval()

    # ---- load weights (EMA if requested and available) ----
    state = ck.get("ema") if (args.use_ema and ck.get("ema") is not None) else ck["model"]
    msd = model.state_dict()
    for k in msd:
        if k in state:
            msd[k].copy_(state[k])

    # ---- diffusion with same training schedule ----
    diff = Diffusion(T=T, schedule=schedule, device=torch.device(device))

    # ---- labels (optional) ----
    y = None
    if num_classes is not None:
        if args.label is not None:
            y = torch.full((args.num_samples,), int(args.label), device=device, dtype=torch.long)
        else:
            y = torch.randint(0, num_classes, (args.num_samples,), device=device, dtype=torch.long)

    # ---- sample ----
    B = args.num_samples
    H = img_size
    os.makedirs(args.outdir, exist_ok=True)

    if args.sampler == "ddpm":
        x = diff.sample_ddpm(model, (B, 3, H, H), y=y, guidance_scale=args.guidance_scale, deterministic=False)
    elif args.sampler == "ddpm_det":
        x = diff.sample_ddpm(model, (B, 3, H, H), y=y, guidance_scale=args.guidance_scale, deterministic=True)
    else:
        x = diff.sample_ddim(
            model, (B, 3, H, H), y=y,
            steps=args.ddim_steps, eta=args.eta,
            guidance_scale=args.guidance_scale,
            skip_first=args.skip_first
        )

    # ---- save grid ----
    x = (x.clamp(-1, 1) + 1) / 2
    save_image(x, os.path.join(args.outdir, "grid.png"), nrow=int(B ** 0.5) or 4)
    print("Saved ->", os.path.join(args.outdir, "grid.png"))


if __name__ == "__main__":
    main()
