
# eval.py
# Evaluation utilities: sampling, FID/PR (torch-fidelity), and a two-sample test.
from __future__ import annotations
import os
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm

from utils import ensure_dir, save_grid
from unet import UNet
from diffusion import Diffusion

# Optional FID/PR
try:
    from torch_fidelity import calculate_metrics
    HAS_TORCH_FIDELITY = True
except Exception:
    HAS_TORCH_FIDELITY = False

def make_real_loader(real_dir: str, img_size: int, batch_size: int = 64):
    tfm = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])
    ds = datasets.ImageFolder(real_dir, transform=tfm)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

class SmallDiscriminator(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base*2, 3, stride=2, padding=1), nn.BatchNorm2d(base*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base*2, base*4, 3, stride=2, padding=1), nn.BatchNorm2d(base*4), nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base*4, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)

def load_checkpoint(path: str, device: str = None):
    """Robusto: ricostruisce la UNet con gli stessi iperparametri del training.
    - Usa cfg salvata se presente
    - Altrimenti deduce 'base' dallo state_dict (in_conv out_channels)
    - Ritenta senza attention se necessario, poi strict=False come ultima spiaggia.
    """
    ckpt = torch.load(path, map_location='cpu')
    cfg = ckpt.get('cfg', {})
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    sd = ckpt['model']

    # 1) parametri principali
    base_from_sd = int(sd['in_conv.weight'].shape[0])  # out_channels of first conv = base
    base = int(cfg.get('base', base_from_sd))
    num_classes = int(cfg.get('num_classes', 200))
    cond_mode = cfg.get('cond_mode', 'class')
    t_dim = int(cfg.get('t_dim', 256))
    ch_mults = tuple(cfg.get('ch_mults', (1,2,2,4)))
    attn_res = tuple(cfg.get('attn_res', (16,)))  # se al training l’hai tolta, ritenteremo con attn_res=()

    def build(attn):
        m = UNet(img_channels=3, base=base, ch_mults=ch_mults, attn_res=attn,
                 num_classes=num_classes, cond_mode=cond_mode, t_dim=t_dim).to(device).eval()
        return m

    # 2) prova caricamento (attenzione come da cfg → senza attention → strict=False)
    model = build(attn_res)
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        try:
            model = build(())
            model.load_state_dict(sd, strict=True)
        except RuntimeError:
            model.load_state_dict(sd, strict=False)

    # 3) diffusion con schedule/steps del training (se salvati)
    diffusion = Diffusion(T=int(cfg.get('num_steps', 1000)),
                          schedule=cfg.get('schedule','cosine'),
                          device=device)

    # 4) carica EMA se presente
    if 'ema' in ckpt and isinstance(ckpt['ema'], dict):
        for name, param in model.named_parameters():
            if name in ckpt['ema']:
                param.data.copy_(ckpt['ema'][name])

    return model, diffusion, cfg, device

def task_sample(args):
    model, diffusion, cfg, device = load_checkpoint(args.checkpoint)
    N = args.num_samples
    y = None
    if cfg.get('cond_mode','class') == 'class':
        y = torch.randint(0, int(cfg.get('num_classes',200)), (N,), device=device)
    with torch.no_grad():
        x = diffusion.sample(model, (N, 3, args.img_size, args.img_size), y=y, guidance_scale=args.guidance_scale)
    ensure_dir(args.outdir)
    save_grid(x, os.path.join(args.outdir, f"samples_{N}.png"), nrow=int(N**0.5))

def task_fid(args):
    if not HAS_TORCH_FIDELITY:
        print("torch-fidelity non installato; esegui: pip install torch-fidelity")
        return
    model, diffusion, cfg, device = load_checkpoint(args.checkpoint)
    gen_dir = Path(args.outdir) / "gen_tmp"
    ensure_dir(str(gen_dir))
    target = args.num_gen if args.num_gen > 0 else 5000
    bs = 64
    written = 0
    with torch.no_grad():
        pbar = tqdm(range(0, target, bs), desc="Generating for FID")
        for _ in pbar:
            y = None
            if cfg.get('cond_mode','class') == 'class':
                y = torch.randint(0, int(cfg.get('num_classes',200)), (bs,), device=device)
            x = diffusion.sample(model, (bs, 3, args.img_size, args.img_size), y=y, guidance_scale=args.guidance_scale)
            x_vis = (x.clamp(-1,1) + 1) / 2
            for i in range(bs):
                save_grid(x_vis[i].unsqueeze(0), str(gen_dir / f"gen_{written+i:06d}.png"), nrow=1, normalize=False)
            written += bs
            if written >= target:
                break
    metrics = calculate_metrics(input1=str(gen_dir), input2=args.real_dir,
                                cuda=torch.cuda.is_available(),
                                isc=False, fid=True, kid=False, prc=True, verbose=True)
    print(metrics)

def task_two_sample(args):
    model, diffusion, cfg, device = load_checkpoint(args.checkpoint)
    real_loader = make_real_loader(args.real_dir, args.img_size, batch_size=64)
    D = SmallDiscriminator(in_ch=3, base=64).to(device)
    opt = torch.optim.Adam(D.parameters(), lr=2e-4)
    bce = nn.BCEWithLogitsLoss()
    for epoch in range(args.disc_epochs):
        pbar = tqdm(real_loader, desc=f"Disc epoch {epoch+1}/{args.disc_epochs}")
        for x_real, _ in pbar:
            x_real = x_real.to(device)
            with torch.no_grad():
                bs = x_real.size(0)
                y = None
                if cfg.get('cond_mode','class') == 'class':
                    y = torch.randint(0, int(cfg.get('num_classes',200)), (bs,), device=device)
                x_fake = diffusion.sample(model, (bs, 3, args.img_size, args.img_size), y=y, guidance_scale=args.guidance_scale)
            y_real = torch.ones(bs, device=device)
            y_fake = torch.zeros(bs, device=device)
            logits_real = D(x_real)
            logits_fake = D(x_fake.detach())
            loss = 0.5 * (bce(logits_real, y_real) + bce(logits_fake, y_fake))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    # Valutazione
    D.eval(); tot, correct = 0, 0
    with torch.no_grad():
        for x_real, _ in tqdm(real_loader, desc="Evaluating D"):
            x_real = x_real.to(device); bs = x_real.size(0)
            y_real = torch.ones(bs, device=device); y_fake = torch.zeros(bs, device=device)
            y = None
            if cfg.get('cond_mode','class') == 'class':
                y = torch.randint(0, int(cfg.get('num_classes',200)), (bs,), device=device)
            x_fake = diffusion.sample(model, (bs, 3, args.img_size, args.img_size), y=y, guidance_scale=args.guidance_scale)
            X = torch.cat([x_real, x_fake], dim=0); Y = torch.cat([y_real, y_fake], dim=0)
            logits = D(X); pred = (torch.sigmoid(logits) > 0.5).float()
            correct += (pred == Y).sum().item(); tot += Y.numel()
    acc = correct / tot
    print(f"Two-sample discriminator accuracy: {acc:.3f} (ideal ≈ 0.5)")

def parse_args():
    ap = ArgumentParser()
    ap.add_argument('--task', type=str, required=True, choices=['sample','fid','two_sample'])
    ap.add_argument('--checkpoint', type=str, required=True)
    ap.add_argument('--img_size', type=int, default=64)
    ap.add_argument('--outdir', type=str, default='runs')
    ap.add_argument('--guidance_scale', type=float, default=1.0)
    ap.add_argument('--real_dir', type=str, default='')
    ap.add_argument('--num_gen', type=int, default=5000)
    ap.add_argument('--disc_epochs', type=int, default=2)
    ap.add_argument('--num_samples', type=int, default=64)
    return ap.parse_args()

def main():
    args = parse_args(); ensure_dir(args.outdir)
    if args.task == 'sample':
        task_sample(args)
    elif args.task == 'fid':
        if not args.real_dir:
            raise ValueError("--real_dir è richiesto per FID/PR")
        task_fid(args)
    elif args.task == 'two_sample':
        if not args.real_dir:
            raise ValueError("--real_dir è richiesto per il two-sample test")
        task_two_sample(args)

if __name__ == "__main__":
    main()
