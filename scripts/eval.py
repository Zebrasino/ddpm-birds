import argparse, os, sys, math, glob
from pathlib import Path
import torch
from torchvision.utils import save_image, make_grid

# Allow "from ddpm_birds import ..." without installing the package.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from ddpm_birds import UNet, Diffusion


def next_index(outdir: str) -> int:
    """Return the next integer index for 000123.png in outdir (to avoid overwrites)."""
    names = [Path(p).stem for p in glob.glob(os.path.join(outdir, "*.png"))]
    nums = [int(s) for s in names if s.isdigit()]
    return (max(nums) + 1) if nums else 1


@torch.no_grad()
def save_grid_and_tiles(x_01: torch.Tensor, outdir: str, nrow: int = 10, padding: int = 2):
    """Save grid.png and each sample as 000001.png, 000002.png, ..."""
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    grid = make_grid(x_01, nrow=nrow, padding=padding)
    save_image(grid, out / "grid.png")
    idx = next_index(outdir)
    for i, xi in enumerate(x_01, start=idx):
        save_image(xi, out / f"{i:06d}.png")


@torch.no_grad()
def save_tiles_only(x_01: torch.Tensor, outdir: str):
    """Save only the individual PNG tiles (no grid)."""
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    idx = next_index(outdir)
    for i, xi in enumerate(x_01, start=idx):
        save_image(xi, out / f"{i:06d}.png")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="sample", choices=["sample"], help="Only sampling is supported.")
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.ckpt).")
    ap.add_argument("--img_size", type=int, default=None, help="Override image size; default from checkpoint.")
    ap.add_argument("--outdir", type=str, required=True, help="Folder to save samples.")
    ap.add_argument("--num_samples", type=int, default=16, help="Number of images to sample.")
    ap.add_argument("--use_ema", action="store_true", help="Use EMA weights if available.")
    ap.add_argument("--guidance_scale", type=float, default=0.0, help="Classifier-free guidance scale.")
    ap.add_argument("--label", type=int, default=None, help="Class id for class-conditional sampling (optional).")

    # sampler options
    ap.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddpm_det", "ddim"], help="Sampler.")
    ap.add_argument("--ddim_steps", type=int, default=50)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--skip_first", type=int, default=0)

    # saving options
    ap.add_argument("--nrow", type=int, default=None, help="Grid columns; default: near-square.")
    ap.add_argument("--grid_only", action="store_true", help="Save only grid.png (no individual tiles).")
    ap.add_argument("--tiles_only", action="store_true", help="Save only individual PNG tiles (no grid).")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.grid_only and args.tiles_only:
        raise SystemExit("Use at most one of --grid_only or --tiles_only.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint & training config
    ck = torch.load(args.checkpoint, map_location=device)
    cfg = ck["args"]
    img_size = int(args.img_size or cfg.get("img_size", 64))
    base = int(cfg.get("base", 96))
    T = int(cfg.get("num_steps", 400))
    schedule = cfg.get("schedule", "cosine")
    cond_mode = cfg.get("cond_mode", "none")

    # Build model
    num_classes = None if cond_mode == "none" else (cfg.get("class_limit") or 200)
    model = UNet(base=base, num_classes=num_classes).to(device).eval()

    # Load weights (EMA if requested/available)
    state = ck.get("ema") if (args.use_ema and ck.get("ema") is not None) else ck["model"]
    msd = model.state_dict()
    for k in msd:
        if k in state:
            msd[k].copy_(state[k])

    # Diffusion wrapper with training schedule
    diff = Diffusion(T=T, schedule=schedule, device=torch.device(device))

    # Labels (optional)
    y = None
    if num_classes is not None:
        if args.label is not None:
            y = torch.full((args.num_samples,), int(args.label), device=device, dtype=torch.long)
        else:
            y = torch.randint(0, num_classes, (args.num_samples,), device=device, dtype=torch.long)

    # Sampling
    B, H = int(args.num_samples), int(img_size)
    os.makedirs(args.outdir, exist_ok=True)
    with torch.no_grad():
        if args.sampler == "ddpm":
            x = diff.sample_ddpm(model, (B, 3, H, H), y=y, guidance_scale=args.guidance_scale, deterministic=False)
        elif args.sampler == "ddpm_det":
            x = diff.sample_ddpm(model, (B, 3, H, H), y=y, guidance_scale=args.guidance_scale, deterministic=True)
        else:
            x = diff.sample_ddim(model, (B, 3, H, H), y=y,
                                 steps=args.ddim_steps, eta=args.eta,
                                 guidance_scale=args.guidance_scale,
                                 skip_first=args.skip_first)

    # Save
    x = (x.clamp(-1, 1) + 1) / 2
    if args.tiles_only:
        save_tiles_only(x, args.outdir)
        print(f"Saved individual PNG tiles in {args.outdir}")
    elif args.grid_only:
        nrow = int(args.nrow) if args.nrow is not None else max(1, int(math.isqrt(B)))
        grid = make_grid(x, nrow=nrow, padding=2)
        save_image(grid, os.path.join(args.outdir, "grid.png"))
        print(f"Saved -> {os.path.join(args.outdir, 'grid.png')}")
    else:
        nrow = int(args.nrow) if args.nrow is not None else max(1, int(math.isqrt(B)))
        save_grid_and_tiles(x, args.outdir, nrow=nrow, padding=2)
        print(f"Saved -> {os.path.join(args.outdir, 'grid.png')}")
        print(f"...and individual PNG tiles in {args.outdir}")


if __name__ == "__main__":
    main()



