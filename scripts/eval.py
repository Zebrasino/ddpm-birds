import argparse, os, sys, math, glob            # stdlib: CLI parsing, paths, math, file lookup
from pathlib import Path                        # path utils with nicer API
import torch                                    # PyTorch core
from torchvision.utils import save_image, make_grid  # helpers to write images and make grids

# Allow "from ddpm_birds import ..." without installing the package.
# We append ../src to sys.path so imports resolve when running from the repo.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ddpm_birds import UNet, Diffusion          # import your UNet and Diffusion wrapper


def next_index(outdir: str) -> int:
    """
    Return the next integer index for files named 000123.png in `outdir`.
    This avoids overwriting when you sample multiple times into the same folder.
    """
    names = [Path(p).stem for p in glob.glob(os.path.join(outdir, "*.png"))]  # all png basenames
    nums = [int(s) for s in names if s.isdigit()]                              # keep pure numeric names
    return (max(nums) + 1) if nums else 1                                      # next index or start at 1


@torch.no_grad()
def save_grid_and_tiles(x_01: torch.Tensor, outdir: str, nrow: int = 10, padding: int = 2):
    """
    Save both a grid.png and all images as numbered tiles (000001.png, ...).
    x_01 is expected in [0,1] range with shape (B,3,H,W).
    """
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)                 # ensure folder exists
    grid = make_grid(x_01, nrow=nrow, padding=padding)                         # build a grid tensor
    save_image(grid, out / "grid.png")                                         # write grid.png
    idx = next_index(outdir)                                                   # first numeric filename
    for i, xi in enumerate(x_01, start=idx):                                   # iterate batch
        save_image(xi, out / f"{i:06d}.png")                                   # write 000001.png etc.


@torch.no_grad()
def save_tiles_only(x_01: torch.Tensor, outdir: str):
    """
    Save only the individual PNG tiles (no grid).
    x_01 is expected in [0,1] range with shape (B,3,H,W).
    """
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)                 # ensure folder exists
    idx = next_index(outdir)                                                   # first numeric filename
    for i, xi in enumerate(x_01, start=idx):                                   # iterate batch
        save_image(xi, out / f"{i:06d}.png")                                   # write numbered tiles


def parse_args():
    """Define and parse command-line arguments."""
    ap = argparse.ArgumentParser()                                             # create CLI parser

    # Task selector (kept for future extension)
    ap.add_argument("--task", type=str, default="sample", choices=["sample"],
                    help="Only sampling is supported.")                        # only sampling for now

    # Checkpoint + IO
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path to checkpoint (.ckpt).")                        # model weights checkpoint
    ap.add_argument("--img_size", type=int, default=None,
                    help="Override image size; default is the training size stored in checkpoint.")
    ap.add_argument("--outdir", type=str, required=True,
                    help="Folder where samples are saved.")                     # output folder
    ap.add_argument("--num_samples", type=int, default=16,
                    help="Number of images to sample in this call.")           # batch size for sampling

    # Guidance / conditioning
    ap.add_argument("--use_ema", action="store_true",
                    help="Use EMA weights if available in the checkpoint.")    # EMA improves quality
    ap.add_argument("--guidance_scale", type=float, default=0.0,
                    help="Classifier-free guidance scale (0 = no guidance).")  # CFG strength
    ap.add_argument("--label", type=int, default=None,
                    help="Class id for class-conditional sampling (optional).")# fixed class if given

    # Sampler options
    ap.add_argument("--sampler", type=str, default="ddim",
                    choices=["ddpm", "ddpm_det", "ddim"],
                    help="Sampling algorithm to use.")                          # choose sampler
    ap.add_argument("--ddim_steps", type=int, default=50,
                    help="Number of DDIM steps (if --sampler=ddim).")          # DDIM steps
    ap.add_argument("--eta", type=float, default=0.0,
                    help="DDIM eta (0 = deterministic DDIM).")                 # DDIM stochasticity
    ap.add_argument("--skip_first", type=int, default=0,
                    help="Skip first K DDIM steps for speed (quality tradeoff).")  # fast skip trick

    # Saving options
    ap.add_argument("--nrow", type=int, default=None,
                    help="Columns in the grid; default is near-square.")       # grid layout
    ap.add_argument("--grid_only", action="store_true",
                    help="Save only grid.png (no individual tiles).")          # grid only mode
    ap.add_argument("--tiles_only", action="store_true",
                    help="Save only individual PNG tiles (no grid).")          # tiles only mode
    return ap.parse_args()                                                      # parse CLI args


def main():
    """Entry point for sampling."""
    args = parse_args()                                                        # read CLI args

    # Safety check: user cannot request both mutually exclusive modes.
    if args.grid_only and args.tiles_only:
        raise SystemExit("Use at most one of --grid_only or --tiles_only.")    # avoid ambiguity

    device = "cuda" if torch.cuda.is_available() else "cpu"                    # pick device

    # ---- Load checkpoint and training config ----
    ck = torch.load(args.checkpoint, map_location=device)                      # load torch checkpoint
    cfg = ck["args"]                                                           # training args dict
    img_size = int(args.img_size or cfg.get("img_size", 64))                   # final resolution
    base = int(cfg.get("base", 96))                                            # UNet base channels
    T = int(cfg.get("num_steps", 400))                                         # diffusion steps used at train
    schedule = cfg.get("schedule", "cosine")                                   # beta schedule type
    cond_mode = cfg.get("cond_mode", "none")                                   # conditioning mode at train

    # ---- Build the UNet (match training setup) ----
    num_classes = None if cond_mode == "none" else (cfg.get("class_limit") or 200)  # class count if conditional
    model = UNet(base=base, num_classes=num_classes).to(device).eval()         # instantiate UNet on device

    # ---- Load weights (prefer EMA if requested and present) ----
    state = ck.get("ema") if (args.use_ema and ck.get("ema") is not None) else ck["model"]  # choose state dict
    msd = model.state_dict()                                                   # model's current state dict
    for k in msd:                                                              # copy intersection keys
        if k in state:
            msd[k].copy_(state[k])                                             # in-place tensor copy

    # ---- Create diffusion wrapper with the same schedule ----
    diff = Diffusion(T=T, schedule=schedule, device=torch.device(device))      # sampling backend

    # ---- Prepare labels (optional) ----
    y = None                                                                   # default: unconditional
    if num_classes is not None:                                                # if model is class-conditional
        if args.label is not None:                                             # user provided a class id
            y = torch.full((args.num_samples,), int(args.label),
                           device=device, dtype=torch.long)                    # fixed class vector
        else:
            y = torch.randint(0, num_classes, (args.num_samples,),
                               device=device, dtype=torch.long)                # random classes

    # ---- Run the sampler ----
    B, H = int(args.num_samples), int(img_size)                                # batch size and size
    os.makedirs(args.outdir, exist_ok=True)                                    # ensure output folder
    with torch.no_grad():                                                      # no grad for inference
        if args.sampler == "ddpm":                                             # stochastic DDPM
            x = diff.sample_ddpm(model, (B, 3, H, H), y=y,
                                 guidance_scale=args.guidance_scale,
                                 deterministic=False)
        elif args.sampler == "ddpm_det":                                       # deterministic DDPM
            x = diff.sample_ddpm(model, (B, 3, H, H), y=y,
                                 guidance_scale=args.guidance_scale,
                                 deterministic=True)
        else:                                                                  # DDIM sampler
            x = diff.sample_ddim(model, (B, 3, H, H), y=y,
                                 steps=args.ddim_steps, eta=args.eta,
                                 guidance_scale=args.guidance_scale,
                                 skip_first=args.skip_first)

    # ---- Save results to disk ----
    x = (x.clamp(-1, 1) + 1) / 2                                               # map from [-1,1] to [0,1]

    if args.tiles_only:                                                        # only tiles mode
        save_tiles_only(x, args.outdir)                                        # write numbered PNGs
        print(f"Saved individual PNG tiles in {args.outdir}")                  # status message

    elif args.grid_only:                                                       # only grid mode
        nrow = int(args.nrow) if args.nrow is not None else max(1, int(math.isqrt(B)))  # grid width
        grid = make_grid(x, nrow=nrow, padding=2)                              # build grid
        save_image(grid, os.path.join(args.outdir, "grid.png"))                # write grid.png
        print(f"Saved -> {os.path.join(args.outdir, 'grid.png')}")             # status message

    else:                                                                      # default: grid + tiles
        nrow = int(args.nrow) if args.nrow is not None else max(1, int(math.isqrt(B)))  # grid width
        save_grid_and_tiles(x, args.outdir, nrow=nrow, padding=2)              # write grid and tiles
        print(f"Saved -> {os.path.join(args.outdir, 'grid.png')}")             # status message
        print(f"...and individual PNG tiles in {args.outdir}")                 # status message


if __name__ == "__main__":
    main()                                                                     # run the CLI entrypoint

