# eval.py
# Line-by-line commented evaluation utilities: FID/PR (if available) and two-sample test; also sampling.
from __future__ import annotations  # Future annotations

import os  # Filesystem paths
from argparse import ArgumentParser  # CLI arguments
from pathlib import Path  # Path utilities

import torch  # Tensors and device
import torch.nn as nn  # For the small discriminator
from torchvision import datasets, transforms  # To build a 'real' dataset directory loader
from tqdm import tqdm  # Progress bars

from utils import ensure_dir, save_grid  # I/O helpers
from unet import UNet  # Model for sampling
from diffusion import Diffusion  # Reverse process for generation

# Optional import: torch-fidelity for FID/PR if installed
try:
    from torch_fidelity import calculate_metrics  # Provides FID, Precision/Recall, etc.
    HAS_TORCH_FIDELITY = True
except Exception:
    HAS_TORCH_FIDELITY = False

def make_real_loader(real_dir: str, img_size: int, batch_size: int = 64):
    """Create a DataLoader from a directory of real images organized in class subfolders (ImageFolder)."""
    tfm = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])
    ds = datasets.ImageFolder(real_dir, transform=tfm)  # Folder structure: class folders under real_dir
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return dl

class SmallDiscriminator(nn.Module):
    """A tiny CNN used for the two-sample test (real vs generated)."""
    def __init__(self, in_ch: int = 3, base: int = 64):
        super().__init__()  # Initialize nn.Module
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base*2, 3, stride=2, padding=1), nn.BatchNorm2d(base*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base*2, base*4, 3, stride=2, padding=1), nn.BatchNorm2d(base*4), nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base*4, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)  # Output logits of shape (N,)

def load_checkpoint(path: str, device: str = None):
    """Load model and diffusion config from checkpoint path."""
    ckpt = torch.load(path, map_location='cpu')  # Load on CPU, then move
    cfg = ckpt['cfg']  # Retrieve stored config dict
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(img_channels=3, base=128, ch_mults=(1,2,2,4),
                 attn_res=(16,), num_classes=cfg['num_classes'], cond_mode=cfg['cond_mode'], t_dim=256)
    model.load_state_dict(ckpt['model'])  # Load weights
    model.to(device).eval()  # Move to device and eval mode
    diffusion = Diffusion(T=cfg['num_steps'], schedule=cfg['schedule'], device=device)
    # Optionally load EMA weights if present for cleaner sampling
    if 'ema' in ckpt and isinstance(ckpt['ema'], dict):
        for name, param in model.named_parameters():
            if name in ckpt['ema']:
                param.data.copy_(ckpt['ema'][name])
    return model, diffusion, cfg, device

def task_sample(args):
    """Generate a grid of samples using the trained (EMA) model."""
    model, diffusion, cfg, device = load_checkpoint(args.checkpoint)
    N = args.num_samples  # Number of images to sample
    y = None
    if cfg['cond_mode'] == 'class':
        # If conditional, randomly choose labels to diversify the grid
        y = torch.randint(0, cfg['num_classes'], (N,), device=device)
    # Sample images in [-1,1]
    with torch.no_grad():
        x = diffusion.sample(model, (N, 3, args.img_size, args.img_size),
                             y=y, guidance_scale=args.guidance_scale)
    # Save grid
    ensure_dir(args.outdir)
    save_grid(x, os.path.join(args.outdir, f"samples_{N}.png"), nrow=int(N**0.5))

def task_fid(args):
    """Compute FID (and optionally Precision/Recall) if torch-fidelity is available."""
    if not HAS_TORCH_FIDELITY:
        print("torch-fidelity not installed; please `pip install torch-fidelity` to compute FID.")
        return
    model, diffusion, cfg, device = load_checkpoint(args.checkpoint)
    # Prepare a temporary directory to write generated images for evaluation
    gen_dir = Path(args.outdir) / "gen_tmp"
    ensure_dir(str(gen_dir))
    # Generate approximately the same number of images as in the real set (or a fixed budget)
    target = args.num_gen if args.num_gen > 0 else 5000
    bs = 64
    written = 0
    with torch.no_grad():
        pbar = tqdm(range(0, target, bs), desc="Generating for FID")
        for _ in pbar:
            y = None
            if cfg['cond_mode'] == 'class':
                y = torch.randint(0, cfg['num_classes'], (bs,), device=device)
            x = diffusion.sample(model, (bs, 3, args.img_size, args.img_size),
                                 y=y, guidance_scale=args.guidance_scale)
            # Map from [-1,1] to [0,1] for saving to disk
            x_vis = (x.clamp(-1,1) + 1) / 2
            for i in range(bs):
                path = gen_dir / f"gen_{written+i:06d}.png"
                save_grid(x_vis[i].unsqueeze(0), str(path), nrow=1, normalize=False)
            written += bs
            if written >= target:
                break
    # Use torch-fidelity to compute metrics against real_dir
    metrics = calculate_metrics(input1=str(gen_dir), input2=args.real_dir,
                                cuda=torch.cuda.is_available(),
                                isc=False, fid=True, kid=False, prc=True, verbose=True)
    print(metrics)  # Print a dict with FID and PRC metrics

def task_two_sample(args):
    """Train a small discriminator to distinguish real vs generated; report accuracy."""
    model, diffusion, cfg, device = load_checkpoint(args.checkpoint)
    real_loader = make_real_loader(args.real_dir, args.img_size, batch_size=64)

    # Prepare discriminator, loss, and optimizer
    D = SmallDiscriminator(in_ch=3, base=64).to(device)
    opt = torch.optim.Adam(D.parameters(), lr=2e-4)
    bce = nn.BCEWithLogitsLoss()

    # Train for a few epochs (short, since it's a test, not a full GAN)
    for epoch in range(args.disc_epochs):
        pbar = tqdm(real_loader, desc=f"Disc epoch {epoch+1}/{args.disc_epochs}")
        for x_real, _ in pbar:
            x_real = x_real.to(device)
            # Generate a batch of fake samples
            with torch.no_grad():
                y = None
                if cfg['cond_mode'] == 'class':
                    y = torch.randint(0, cfg['num_classes'], (x_real.size(0),), device=device)
                x_fake = diffusion.sample(model, (x_real.size(0), 3, args.img_size, args.img_size),
                                          y=y, guidance_scale=args.guidance_scale)
            # Labels: real=1, fake=0
            y_real = torch.ones(x_real.size(0), device=device)
            y_fake = torch.zeros(x_real.size(0), device=device)

            # Update with real batch
            logits_real = D(x_real)
            loss_real = bce(logits_real, y_real)

            # Update with fake batch
            logits_fake = D(x_fake.detach())
            loss_fake = bce(logits_fake, y_fake)

            # Combine and step
            loss = (loss_real + loss_fake) * 0.5
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=float(loss))

    # Evaluate discriminator accuracy on a held-out set of mixed real/fake
    D.eval()
    tot, correct = 0, 0
    with torch.no_grad():
        for x_real, _ in tqdm(real_loader, desc="Evaluating D"):
            x_real = x_real.to(device)
            bs = x_real.size(0)
            # Half real, half fake
            y_real = torch.ones(bs, device=device)
            y_fake = torch.zeros(bs, device=device)
            # Fake batch
            y = None
            if cfg['cond_mode'] == 'class':
                y = torch.randint(0, cfg['num_classes'], (bs,), device=device)
            x_fake = diffusion.sample(model, (bs, 3, args.img_size, args.img_size),
                                      y=y, guidance_scale=args.guidance_scale)

            # Concatenate and predict
            X = torch.cat([x_real, x_fake], dim=0)
            Y = torch.cat([y_real, y_fake], dim=0)
            logits = D(X)
            pred = (torch.sigmoid(logits) > 0.5).float()
            correct += (pred == Y).sum().item()
            tot += Y.numel()
    acc = correct / tot
    print(f"Two-sample discriminator accuracy: {acc:.3f} (ideal ≈ 0.5 if distributions match)" )

def parse_args():
    """Parse CLI for evaluation tasks."""
    ap = ArgumentParser()
    ap.add_argument('--task', type=str, required=True, choices=['sample','fid','two_sample'])
    ap.add_argument('--checkpoint', type=str, required=True)
    ap.add_argument('--img_size', type=int, default=64)
    ap.add_argument('--outdir', type=str, default='runs')
    ap.add_argument('--guidance_scale', type=float, default=1.0)

    # For FID
    ap.add_argument('--real_dir', type=str, default='')
    ap.add_argument('--num_gen', type=int, default=5000)

    # For two-sample
    ap.add_argument('--disc_epochs', type=int, default=2)

    # For sampling
    ap.add_argument('--num_samples', type=int, default=64)
    return ap.parse_args()

def main():
    args = parse_args()
    ensure_dir(args.outdir)
    if args.task == 'sample':
        task_sample(args)
    elif args.task == 'fid':
        if not args.real_dir:
            raise ValueError("--real_dir is required for FID/PR computation")
        task_fid(args)
    elif args.task == 'two_sample':
        if not args.real_dir:
            raise ValueError("--real_dir is required for the two-sample test")
        task_two_sample(args)

if __name__ == "__main__":
    main()
