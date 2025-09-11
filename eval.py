from __future__ import annotations                              # postponed hints
import os                                                       # filesystem
import argparse                                                 # CLI
import torch                                                    # tensors
from torchvision.utils import save_image                        # save grids

from unet import UNet                                           # model
from diffusion import Diffusion                                 # samplers
from utils import load_ckpt                                     # checkpoint IO


def build_parser() -> argparse.ArgumentParser:
    """CLI for sampling/evaluation."""
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="sample", help="Only 'sample' is implemented here")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt")
    p.add_argument("--img_size", type=int, default=None, help="Override img_size (else use CKPT args)")
    p.add_argument("--outdir", type=str, required=True, help="Output folder")
    p.add_argument("--num_samples", type=int, default=16, help="Number of images")
    p.add_argument("--sampler", type=str, choices=["ddpm", "ddim"], default="ddim", help="Sampling algorithm")
    p.add_argument("--ddim_steps", type=int, default=50, help="DDIM steps")
    p.add_argument("--eta", type=float, default=0.0, help="DDIM eta (0=deterministic)")
    p.add_argument("--skip_first", type=int, default=0, help="Skip earliest steps (DDIM only)")
    p.add_argument("--use_ema", action="store_true", help="Use EMA weights if available")
    p.add_argument("--guidance_scale", type=float, default=0.0, help="CFG guidance scale for class-conditional")
    p.add_argument("--class_id", type=int, default=None, help="Force a single class id (0..C-1); omit for random")
    return p


@torch.no_grad()
def main():
    args = build_parser().parse_args()                           # parse CLI
    dev = "cuda" if torch.cuda.is_available() else "cpu"         # pick device
    ck = load_ckpt(args.checkpoint, map_location=dev)            # load ckpt
    A = ck.get("args", {})                                       # saved args

    # Restore training settings (with optional overrides from CLI)
    base = A.get("base", 64)                                     # UNet width
    num_steps = A.get("num_steps", 200)                          # diffusion steps
    schedule = A.get("schedule", "cosine")                       # schedule name
    cond_mode = A.get("cond_mode", "none")                       # 'none' or 'class'
    num_classes = None
    if cond_mode == "class":                                     # if class-conditional
        # Prefer value saved in args; as a fallback, try 'num_classes' field
        num_classes = A.get("num_classes", None)

    img_size = args.img_size or A.get("img_size", 64)            # resolution

    # Build model and load weights (EMA if requested)
    model = UNet(base=base, num_classes=num_classes).to(dev).eval()
    state = ck.get("ema" if args.use_ema and (ck.get("ema") is not None) else "model")
    model.load_state_dict(state, strict=False)                    # load weights

    # Diffusion helper
    diff = Diffusion(T=int(num_steps), schedule=schedule, device=torch.device(dev))

    # Prepare labels for conditional sampling (random by default)
    B = int(args.num_samples)                                     # batch size
    y = None
    if cond_mode == "class" and (num_classes is not None) and (num_classes > 0):
        if args.class_id is not None:                             # fixed class
            y = torch.full((B,), int(args.class_id), device=dev, dtype=torch.long)
        else:                                                     # random classes
            y = torch.randint(0, num_classes, (B,), device=dev)

    # Run the selected sampler
    shape = (B, 3, img_size, img_size)                            # output shape
    if args.sampler == "ddpm":                                    # DDPM
        x = diff.sample_ddpm(model, shape, y=y, guidance_scale=args.guidance_scale)
    else:                                                         # DDIM
        x = diff.sample_ddim(
            model, shape, y=y, steps=int(args.ddim_steps), eta=float(args.eta),
            guidance_scale=args.guidance_scale, skip_first=int(args.skip_first)
        )

    # Save grid
    os.makedirs(args.outdir, exist_ok=True)                       # ensure dir
    x = (x.clamp(-1, 1) + 1) / 2                                  # to [0,1]
    save_image(x, os.path.join(args.outdir, "grid.png"), nrow=int(max(1, B**0.5)))
    print("Saved ->", os.path.join(args.outdir, "grid.png"))      # log path


if __name__ == "__main__":                                       # entry
    main()                                                        # run
