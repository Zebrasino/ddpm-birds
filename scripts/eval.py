#!/usr/bin/env python3  # Use the system's Python interpreter when run as a script
"""
DDPM evaluation / sampling helper.

- Loads a checkpoint (optionally the EMA weights).
- Draws samples using DDPM / deterministic DDPM / DDIM.
- Saves either: only a grid image, only individual tiles, or both.

This file is intentionally small and dependency-light so it can run in Colab.
"""

import argparse, os, sys, math, glob                 # Standard libs for CLI, paths, math, file listing
from pathlib import Path                              # Convenient path handling
import torch                                          # PyTorch core
from torchvision.utils import save_image, make_grid   # Helpers to build/save image grids

# Make sure we can import the local package without installing it.
# We append the repo's /src directory to sys.path so "from ddpm_birds import ..." works.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from ddpm_birds import UNet, Diffusion                # Model and diffusion wrapper from the project

def next_index(outdir: str) -> int:
    """Return the next integer index for 000123.png in outdir (to avoid overwrites)."""
    names = [Path(p).stem for p in glob.glob(os.path.join(outdir, "*.png"))]  # Collect existing PNG basenames
    nums = [int(s) for s in names if s.isdigit()]                              # Keep only pure numeric stems
    return (max(nums) + 1) if nums else 1                                      # Next index or 1 if none

@torch.no_grad()                                        # Disable autograd to save memory and speed up I/O
def save_grid_and_tiles(x_01: torch.Tensor, outdir: str, nrow: int = 10, padding: int = 2):
    """Save grid.png AND each individual sample as 000001.png, 000002.png, ..."""
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)                 # Ensure output directory exists
    grid = make_grid(x_01, nrow=nrow, padding=padding)                         # Create a grid from the batch
    save_image(grid, out / "grid.png")                                         # Save the grid image
    idx = next_index(outdir)                                                   # Determine next tile index
    for i, xi in enumerate(x_01, start=idx):                                   # Iterate samples with numbering
        save_image(xi, out / f"{i:06d}.png")                                   # Save each sample as a PNG

@torch.no_grad()                                        # Disable autograd for pure saving
def save_tiles_only(x_01: torch.Tensor, outdir: str):
    """Save ONLY the individual PNG tiles (no grid)."""
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)                 # Ensure output directory exists
    idx = next_index(outdir)                                                   # Determine next tile index
    for i, xi in enumerate(x_01, start=idx):                                   # Iterate samples with numbering
        save_image(xi, out / f"{i:06d}.png")                                   # Save each sample as a PNG

def parse_args():
    """Parse command-line arguments for sampling."""
    ap = argparse.ArgumentParser()                                             # CLI parser
    ap.add_argument("--task", type=str, default="sample", choices=["sample"])  # Only "sample" is supported
    ap.add_argument("--checkpoint", type=str, required=True)                   # Path to .ckpt file
    ap.add_argument("--img_size", type=int, default=None)                      # Optional override for image size
    ap.add_argument("--outdir", type=str, required=True)                       # Where to save outputs
    ap.add_argument("--num_samples", type=int, default=16)                     # How many images per call
    ap.add_argument("--use_ema", action="store_true")                          # Use EMA weights if present
    ap.add_argument("--guidance_scale", type=float, default=0.0)               # Classifier-free guidance scale
    ap.add_argument("--label", type=int, default=None)                         # Optional fixed class label

    ap.add_argument("--sampler", type=str, default="ddim",
                    choices=["ddpm","ddpm_det","ddim"])                        # Which sampler to use
    ap.add_argument("--ddim_steps", type=int, default=50)                      # Steps for DDIM
    ap.add_argument("--eta", type=float, default=0.0)                          # DDIM stochasticity (0 is deterministic)
    ap.add_argument("--skip_first", type=int, default=0)                       # Skip initial DDIM steps (speed trick)

    ap.add_argument("--nrow", type=int, default=None)                          # Grid columns (auto if None)
    ap.add_argument("--grid_only", action="store_true")                        # Save only grid.png
    ap.add_argument("--tiles_only", action="store_true")                       # Save only individual tiles
    return ap.parse_args()                                                     # Return parsed args

def main():
    """Entry point: load checkpoint, sample images, save outputs."""
    args = parse_args()                                                        # Parse CLI options
    if args.grid_only and args.tiles_only:                                     # Sanity: avoid conflicting flags
        raise SystemExit("Use at most one of --grid_only or --tiles_only.")

    device = "cuda" if torch.cuda.is_available() else "cpu"                    # Prefer GPU if available

    ck = torch.load(args.checkpoint, map_location=device)                      # Load checkpoint dict
    cfg = ck["args"]                                                            # Training args saved in ckpt
    img_size = int(args.img_size or cfg.get("img_size", 64))                   # Image size: override or from ckpt
    base = int(cfg.get("base", 96))                                            # UNet base channels from training
    T = int(cfg.get("num_steps", 400))                                         # Diffusion training steps
    schedule = cfg.get("schedule", "cosine")                                   # Noise schedule used in training
    cond_mode = cfg.get("cond_mode", "none")                                   # Conditioning mode ("none"/"class")

    num_classes = None if cond_mode == "none" else (cfg.get("class_limit") or 200)  # Num classes if class-cond
    model = UNet(base=base, num_classes=num_classes).to(device).eval()         # Build UNet and put into eval mode
    state = ck.get("ema") if (args.use_ema and ck.get("ema") is not None) else ck["model"]  # Choose weights
    msd = model.state_dict()                                                   # Model state dict (destination)
    for k in msd:                                                              # Copy only matching keys
        if k in state:
            msd[k].copy_(state[k])

    diff = Diffusion(T=T, schedule=schedule, device=torch.device(device))      # Instantiate diffusion wrapper

    y = None                                                                   # Default: unconditional
    if num_classes is not None:                                                # If model is class-conditional
        if args.label is not None:                                             # If label is provided, fix it
            y = torch.full((args.num_samples,), int(args.label), device=device, dtype=torch.long)
        else:                                                                  # Otherwise sample random labels
            y = torch.randint(0, num_classes, (args.num_samples,), device=device, dtype=torch.long)

    B, H = int(args.num_samples), int(img_size)                                # Batch size and spatial size
    os.makedirs(args.outdir, exist_ok=True)                                    # Ensure output directory exists

    with torch.no_grad():                                                      # No gradients needed for sampling
        if args.sampler == "ddpm":                                             # Plain DDPM (stochastic)
            x = diff.sample_ddpm(model, (B,3,H,H), y=y,
                                  guidance_scale=args.guidance_scale,
                                  deterministic=False)
        elif args.sampler == "ddpm_det":                                       # Deterministic DDPM (DDIM-like)
            x = diff.sample_ddpm(model, (B,3,H,H), y=y,
                                  guidance_scale=args.guidance_scale,
                                  deterministic=True)
        else:                                                                  # DDIM sampler
            x = diff.sample_ddim(model, (B,3,H,H), y=y,
                                 steps=args.ddim_steps, eta=args.eta,
                                 guidance_scale=args.guidance_scale,
                                 skip_first=args.skip_first)

    x = (x.clamp(-1,1) + 1) / 2                                                # Convert from [-1,1] to [0,1] range
    if args.tiles_only:                                                        # Save only tiles
        save_tiles_only(x, args.outdir)
        print(f"Saved tiles -> {args.outdir}")
    elif args.grid_only:                                                       # Save only a grid image
        nrow = int(args.nrow) if args.nrow is not None else max(1, int(math.isqrt(B)))
        grid = make_grid(x, nrow=nrow, padding=2)
        save_image(grid, os.path.join(args.outdir, "grid.png"))
        print(f"Saved grid -> {os.path.join(args.outdir, 'grid.png')}")
    else:                                                                      # Save both grid + tiles
        nrow = int(args.nrow) if args.nrow is not None else max(1, int(math.isqrt(B)))
        save_grid_and_tiles(x, args.outdir, nrow=nrow, padding=2)
        print(f"Saved grid + tiles -> {args.outdir}")

if __name__ == "__main__":                                                     # Script entry point
    main()                                                                     # Run main



