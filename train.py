# train.py
# Training loop for DDPM with UNet + EMA, AMP, and periodic sampling.
# Designed to run on Colab T4 with moderate VRAM.

import os
import argparse
from typing import Optional

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from unet import UNet
from diffusion import Diffusion
from utils import EMA, ensure_dir, set_seed, save_grid


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True, help="ImageFolder root (subdirs=classes)")
    p.add_argument("--outdir", type=str, required=True, help="Where to save checkpoints and samples")
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--num_steps", type=int, default=400)
    p.add_argument("--schedule", type=str, choices=["linear", "cosine"], default="cosine")
    p.add_argument("--cond_mode", type=str, choices=["none", "class"], default="class")
    p.add_argument("--num_classes", type=int, default=None)
    p.add_argument("--p_uncond", type=float, default=0.1)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--base", type=int, default=64, help="UNet base channels (keep 64 on T4)")
    p.add_argument("--save_every", type=int, default=5, help="epochs between checkpoints")
    p.add_argument("--sample_every", type=int, default=5, help="epochs between sample grids")
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.outdir)

    # ---------- dataset ----------
    # ImageFolder expects: data_root/class_x/xxx.png
    tfm = transforms.Compose([
        transforms.Resize(args.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),  # maps to [-1,1]
    ])
    dset = datasets.ImageFolder(args.data_root, transform=tfm)
    if args.cond_mode == "class" and args.num_classes is None:
        args.num_classes = len(dset.classes)

    loader = DataLoader(dset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # ---------- model ----------
    model = UNet(
        img_channels=3,
        base=args.base,                     # keep 64 for T4
        ch_mult=(1, 2, 2, 4),               # 64, 128, 128, 256
        attn_resolutions=(16,),             # attention at 16x16 only
        num_res_blocks=2,                   # light but expressive
        time_emb_dim=256,
        num_classes=(args.num_classes if args.cond_mode == "class" else None),
    ).to(device)

    # EMA shadow model
    ema = EMA(model, decay=args.ema_decay)

    # optimizer
    optim_ = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999))

    # diffusion
    diffusion = Diffusion(T=args.num_steps, schedule=args.schedule, device=device)

    # AMP scaler (new API)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    # save config for reproducibility
    torch.save({"train_args": vars(args)}, os.path.join(args.outdir, "train_args.pt"))

    # training
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True) if (args.cond_mode == "class") else None

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                loss = diffusion.p_losses(model, x, y=y, p_uncond=args.p_uncond)

            scaler.scale(loss).backward()
            scaler.step(optim_)
            scaler.update()
            optim_.zero_grad(set_to_none=True)
            ema.update()

            running += float(loss.detach())
            global_step += 1

        avg_loss = running / max(1, len(loader))
        print(f"Epoch {epoch}/{args.epochs} - loss: {avg_loss:.4f}")

        # sampling (with EMA weights for better quality)
        if (epoch % args.sample_every) == 0:
            model.eval()
            ema.copy_to(model)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                n = 36
                # choose labels if conditional
                if args.cond_mode == "class":
                    y_sample = torch.randint(0, args.num_classes, (n,), device=device)
                    x_gen = diffusion.sample(model, args.img_size, n, y=y_sample, guidance_scale=args.guidance_scale)
                else:
                    x_gen = diffusion.sample(model, args.img_size, n, y=None, guidance_scale=1.0)
            save_grid(x_gen, os.path.join(args.outdir, f"samples_epoch_{epoch:03d}.png"), nrow=6)

        # checkpoint
        if (epoch % args.save_every) == 0:
            ckpt = {
                "model": model.state_dict(),
                "ema": ema.shadow,
                "model_cfg": model.model_cfg,
                "args": vars(args),
            }
            torch.save(ckpt, os.path.join(args.outdir, f"epoch_{epoch:03d}.ckpt"))
            torch.save(ckpt, os.path.join(args.outdir, f"last.ckpt"))  # convenience

    # final checkpoint
    ckpt = {
        "model": model.state_dict(),
        "ema": ema.shadow,
        "model_cfg": model.model_cfg,
        "args": vars(args),
    }
    torch.save(ckpt, os.path.join(args.outdir, f"last.ckpt"))


if __name__ == "__main__":
    main()
