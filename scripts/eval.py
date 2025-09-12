import argparse, os, sys, math, glob
from pathlib import Path

import torch
from torchvision.utils import save_image, make_grid  # saving utilities from torchvision

# ---- allow `import ddpm_birds` without installing as a package ----
# We append ../src to sys.path so "from ddpm_birds import ..." works.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ddpm_birds import UNet, Diffusion  # our model and diffusion wrapper


# ---------- small helpers ----------

def next_index(outdir: str) -> int:
    """
    Find the next integer index for filenames like 000123.png in `outdir`.
    This lets multiple runs append without overwriting existing tiles.
    """
    # Grab basenames without extension for all PNGs in outdir.
    names = [Path(p).stem for p in glob.glob(os.path.join(outdir, "*.png"))]
    # Keep only those that are all-digits (e.g., "000123").
    nums = [int(s) for s in names if s.isdigit()]
    # If none found, start from 1; else continue from the max+1.
    return (max(nums) + 1) if nums else 1


@torch.no_grad()
def save_grid_and_tiles(x_01: torch.Tensor, outdir: str, nrow: int = 10, padding: int = 2, grid_only: bool = False):
    """
    Save one `grid.png` and (optionally) individual tiles as 000001.png, 000002.png, ...

    Args:
        x_01: Tensor in [0,1] of shape (N, 3, H, H) ready to be saved.
        outdir: Destination directory.
        nrow: Number of columns in the grid (rows are computed automatically).
        padding: Pixel padding used by torchvision's make_grid (for visual spacing).
        grid_only: If True, only save the grid and skip individual PNG tiles.
    """
    # Ensure the output directory exists.
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)

    # Create a grid tensor in [0,1] for visualization and save it.
    grid = make_grid(x_01, nrow=nrow, padding=padding)
    save_image(grid, out / "grid.png")

    # If only the grid is requested, we stop here.
    if grid_only:
        return

    # Determine where to start numbering PNG tiles so we don't overwrite prior runs.
    idx = next_index(outdir)

    # Save each sample individually as an RGB PNG (implicit by save_image).
    for i, xi in enumerate(x_01, start=idx):
        save_image(xi, out / f"{i:06d}.png")


# ---------- CLI args ----------

def parse_args():
    ap = argparse.ArgumentParser()
    # We only support sampling tasks in this script.
    ap.add_argument("--task", type=str, default="sample", choices=["sample"], help="Only sampling is supported.")
    # Path to a .ckpt produced by our training script.
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.ckpt).")
    # Optional override of training resolution; if omitted, we read it from the checkpoint args.
    ap.add_argument("--img_size", type=int, default=None, help="Override image size (default: from checkpoint).")
    # Output directory where grid/tiles will be written.
    ap.add_argument("--outdir", type=str, required=True, help="Folder to save samples.")
    # How many images to generate in a single run.
    ap.add_argument("--num_samples", type=int, default=16, help="Number of images to sample.")
    # Whether to use EMA weights stored in the checkpoint (recommended for eval).
    ap.add_argument("--use_ema", action="store_true", help="Use EMA shadow weights if available.")
    # Classifier-free guidance scale (0.0 = unconditional sampling).
    ap.add_argument("--guidance_scale", type=float, default=0.0, help="Classifier-free guidance scale.")

    # Class-conditional label: if provided, all samples use this class id; otherwise random classes.
    ap.add_argument("--label", type=int, default=None, help="Class id for class-conditional sampling (optional).")

    # Sampler selection: stochastic DDPM, deterministic DDPM, or DDIM.
    ap.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddpm_det", "ddim"],
                    help="Sampling method.")
    # DDIM parameters (ignored by DDPM).
    ap.add_argument("--ddim_steps", type=int, default=50, help="DDIM steps (if --sampler=ddim).")
    ap.add_argument("--eta", type=float, default=0.0, help="DDIM eta (noise amount).")
    ap.add_argument("--skip_first", type=int, default=0, help="Skip first K DDIM steps (speed trick).")

    # Grid/tiles saving controls.
    ap.add_argument("--nrow", type=int, default=None, help="Grid columns; default is a near-square layout.")
    ap.add_argument("--grid_only", action="store_true", help="Only save grid.png (no individual tiles).")
    return ap.parse_args()


# ---------- main ----------

def main():
    args = parse_args()

    # Pick device (CUDA if available).
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Load checkpoint dict on the chosen device (CPU works too) ----
    ck = torch.load(args.checkpoint, map_location=device)

    # The training script stored the argument namespace in ck["args"].
    cfg = ck["args"]

    # Read relevant hyper-parameters from the checkpoint (with fallbacks).
    img_size = int(args.img_size or cfg.get("img_size", 64))        # target resolution
    base = int(cfg.get("base", 96))                                 # UNet base channels
    T = int(cfg.get("num_steps", 400))                              # diffusion chain length
    schedule = cfg.get("schedule", "cosine")                        # beta schedule
    cond_mode = cfg.get("cond_mode", "none")                        # 'none' or 'class'

    # ---- Build the model skeleton and move to device ----
    num_classes = None if cond_mode == "none" else (cfg.get("class_limit") or 200)
    model = UNet(base=base, num_classes=num_classes).to(device).eval()

    # ---- Load weights: prefer EMA if requested and present ----
    state = ck.get("ema") if (args.use_ema and ck.get("ema") is not None) else ck["model"]
    # We copy only the matching keys to be robust to minor code changes.
    msd = model.state_dict()
    for k in msd:
        if k in state:
            msd[k].copy_(state[k])

    # ---- Instantiate diffusion with the training schedule ----
    diff = Diffusion(T=T, schedule=schedule, device=torch.device(device))

    # ---- Prepare (optional) class labels ----
    y = None
    if num_classes is not None:
        if args.label is not None:
            # If user provided a class id, repeat it for all samples.
            y = torch.full((args.num_samples,), int(args.label), device=device, dtype=torch.long)
        else:
            # Otherwise draw random class ids for variety.
            y = torch.randint(0, num_classes, (args.num_samples,), device=device, dtype=torch.long)

    # ---- Sampling shape (B, 3, H, H) ----
    B = int(args.num_samples)
    H = int(img_size)
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # ---- Generate samples under no grad; optional AMP to reduce VRAM ----
    model.eval()
    with torch.no_grad():
        # Use autocast on CUDA to reduce memory / speed up; on CPU it is a no-op.
        autocast_ctx = (torch.amp.autocast(device_type="cuda") if device == "cuda"
                        else torch.autocast("cpu") if hasattr(torch, "autocast") else torch.no_grad())
        with autocast_ctx:
            if args.sampler == "ddpm":
                # Stochastic DDPM (ancestral); note: skip_first is ignored here.
                x = diff.sample_ddpm(model, (B, 3, H, H), y=y,
                                     guidance_scale=args.guidance_scale, deterministic=False)
            elif args.sampler == "ddpm_det":
                # Deterministic DDPM (a.k.a. DDPM with fixed noise path).
                x = diff.sample_ddpm(model, (B, 3, H, H), y=y,
                                     guidance_scale=args.guidance_scale, deterministic=True)
            else:
                # DDIM sampler with optional step skipping and eta control.
                x = diff.sample_ddim(model, (B, 3, H, H), y=y,
                                     steps=args.ddim_steps, eta=args.eta,
                                     guidance_scale=args.guidance_scale,
                                     skip_first=args.skip_first)

    # ---- Convert from [-1,1] to [0,1] for saving ----
    x = (x.clamp(-1, 1) + 1) / 2

    # ---- Decide grid layout: default is near-square (e.g., 100 -> 10 columns) ----
    nrow = int(args.nrow) if args.nrow is not None else max(1, int(math.isqrt(B)))

    # ---- Save both grid and individual tiles ----
    save_grid_and_tiles(x, outdir, nrow=nrow, padding=2, grid_only=args.grid_only)

    # ---- Print a friendly message pointing to the files ----
    print(f"Saved -> {os.path.join(outdir, 'grid.png')}")
    if not args.grid_only:
        print(f"...and individual PNG tiles in {outdir}")


if __name__ == "__main__":
    main()

