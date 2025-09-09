
from __future__ import annotations
import os
from argparse import ArgumentParser
import torch
from torch.optim import AdamW
from torch import amp
from tqdm import tqdm

from utils import set_seed, TrainConfig, ensure_dir, save_grid, EMA
from data import make_loader
from unet import UNet
from diffusion import Diffusion

def parse_args() -> TrainConfig:
    ap = ArgumentParser()
    ap.add_argument('--data_root', type=str, default='./CUB_200_2011')
    ap.add_argument('--img_size', type=int, default=64)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batches_per_epoch', type=int, default=10)   # NEW: short epoch
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--ema_decay', type=float, default=0.999)
    ap.add_argument('--num_steps', type=int, default=1000)
    ap.add_argument('--cond_mode', type=str, default='class', choices=['none','class'])
    ap.add_argument('--num_classes', type=int, default=200)
    ap.add_argument('--guidance_scale', type=float, default=1.0)
    ap.add_argument('--p_uncond', type=float, default=0.1)
    ap.add_argument('--schedule', type=str, default='cosine', choices=['linear','cosine'])
    ap.add_argument('--outdir', type=str, default='runs')
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--base', type=int, default=96)                # keep existing
    ap.add_argument('--no_attn', action='store_true')              # NEW: disable attention
    ap.add_argument('--sample_grid', type=int, default=9)          # NEW: few previews
    ap.add_argument('--sample_every', type=int, default=25)        # NEW: preview every K epochs
    args = ap.parse_args()

    cfg = TrainConfig(
        data_root=args.data_root, img_size=args.img_size, batch_size=args.batch_size,
        epochs=args.epochs, lr=args.lr, ema_decay=args.ema_decay, num_steps=args.num_steps,
        cond_mode=args.cond_mode, num_classes=args.num_classes, guidance_scale=args.guidance_scale,
        p_uncond=args.p_uncond, schedule=args.schedule, outdir=args.outdir
    )
    # attach extra fields (TrainConfig may not declare them)
    cfg.num_workers = args.num_workers  # type: ignore[attr-defined]
    cfg.base = args.base                # type: ignore[attr-defined]
    cfg.no_attn = args.no_attn          # type: ignore[attr-defined]
    cfg.batches_per_epoch = args.batches_per_epoch  # type: ignore[attr-defined]
    cfg.sample_grid = args.sample_grid  # type: ignore[attr-defined]
    cfg.sample_every = args.sample_every  # type: ignore[attr-defined]
    return cfg

def main() -> None:
    cfg = parse_args()
    set_seed(42)

    # speed-ups
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = cfg.device
    ensure_dir(cfg.outdir)

    loader = make_loader(
        cfg.data_root, cfg.img_size, cfg.batch_size, split="train",
        num_workers=getattr(cfg, "num_workers", 2)
    )

    attn_res = () if getattr(cfg, "no_attn", False) else (16,)
    model = UNet(
        img_channels=3, base=getattr(cfg, 'base', 96), ch_mults=(1, 2, 2, 4),
        attn_res=attn_res, num_classes=cfg.num_classes,
        cond_mode=cfg.cond_mode, t_dim=256
    ).to(device)
    model = model.to(memory_format=torch.channels_last)

    diffusion = Diffusion(T=cfg.num_steps, schedule=cfg.schedule, device=device)

    optim = AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    ema = EMA(model, decay=cfg.ema_decay)
    scaler = amp.GradScaler(enabled=torch.cuda.is_available())

    global_step = 0
    for epoch in range(cfg.epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.epochs}", mininterval=1.0)
        for i, (x, y) in enumerate(pbar):
            if i >= int(getattr(cfg, "batches_per_epoch", 10)):  # short epoch
                break
            x = x.to(device, memory_format=torch.channels_last, non_blocking=True)
            y = y.to(device) if cfg.cond_mode == 'class' else None
            t = torch.randint(0, cfg.num_steps, (x.size(0),), device=device).long()

            optim.zero_grad(set_to_none=True)
            with amp.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                loss = diffusion.p_losses(model, x, t, y=y, p_uncond=cfg.p_uncond)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            ema.update()

            global_step += 1
            pbar.set_postfix(loss=loss.detach().item())

        # preview samples occasionally
        if (epoch + 1) % int(getattr(cfg, "sample_every", 25)) == 0:
            model.eval(); ema.store(); ema.copy_to()
            with torch.no_grad():
                N = int(getattr(cfg, "sample_grid", 9))
                y_sample = None
                if cfg.cond_mode == 'class':
                    y_sample = torch.randint(0, cfg.num_classes, (N,), device=device)
                samples = diffusion.sample(
                    model, (N, 3, cfg.img_size, cfg.img_size),
                    y=y_sample, guidance_scale=cfg.guidance_scale
                )
            save_grid(samples, os.path.join(cfg.outdir, f"samples_epoch_{epoch+1:03d}.png"),
                      nrow=max(1, int(N ** 0.5)))
            ema.restore()

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
