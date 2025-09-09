# eval.py
# Simple sampler that loads a checkpoint and writes a grid of generated images.

import os
import argparse
import torch
from torchvision.utils import save_image

from unet import UNet
from diffusion import Diffusion
from utils import ensure_dir


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, choices=["sample"], required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--num_samples", type=int, default=36)
    p.add_argument("--class_id", type=int, default=None, help="optional class id for conditional model")
    return p


def load_checkpoint(path: str, device):
    ckpt = torch.load(path, map_location=device)
    model_cfg = ckpt["model_cfg"]
    model = UNet(**model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    # prefer EMA if available
    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"], strict=False)
    args = ckpt.get("args", {})
    return model, args


@torch.no_grad()
def task_sample(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.outdir)

    # load model and (re)build diffusion with the same T/schedule as in training
    model, train_args = load_checkpoint(args.checkpoint, device)
    T = int(train_args.get("num_steps", 400))
    schedule = train_args.get("schedule", "cosine")

    diffusion = Diffusion(T=T, schedule=schedule, device=device)
    model.eval()

    # labels: only if conditional
    y = None
    if getattr(model, "num_classes", None) is not None:
        n = args.num_samples
        if args.class_id is None:
            y = torch.randint(0, model.num_classes, (n,), device=device)
        else:
            y = torch.full((n,), int(args.class_id), device=device, dtype=torch.long)

    # sample
    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
        x = diffusion.sample(
            model,
            img_size=args.img_size,
            n=args.num_samples,
            y=y,
            guidance_scale=args.guidance_scale if y is not None else 1.0,
        )

    # save grid
    grid = (x + 1) / 2.0
    save_path = os.path.join(args.outdir, "samples.png")
    save_image(grid, save_path, nrow=6)
    print(f"Saved {save_path}")


def main():
    args = build_parser().parse_args()
    if args.task == "sample":
        task_sample(args)


if __name__ == "__main__":
    main()

