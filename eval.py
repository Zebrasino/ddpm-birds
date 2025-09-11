from __future__ import annotations                           # future annotations
import os                                                    # filesystem
import argparse                                              # CLI
import torch                                                 # tensors
from torchvision.utils import save_image                     # image save
from torchvision import datasets, transforms                 # fallback datasets / transforms

from unet import UNet                                        # model
from diffusion import Diffusion                              # DDPM core
from utils import load_ckpt                                  # checkpoint I/O

# -----------------------------
# CLI
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()                                                 # parser
    p.add_argument("--task", type=str, choices=["sample", "fid", "two_sample"], required=True, help="What to run.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to last.ckpt.")
    p.add_argument("--img_size", type=int, default=64, help="Output image size.")
    p.add_argument("--outdir", type=str, default="samples", help="Where to save samples/plots.")
    p.add_argument("--num_samples", type=int, default=64, help="How many images to sample.")
    p.add_argument("--sampler", type=str, choices=["ddpm", "ddim"], default="ddim", help="Sampler type.")
    p.add_argument("--ddim_steps", type=int, default=50, help="DDIM steps.")
    p.add_argument("--eta", type=float, default=0.0, help="DDIM stochasticity (0=deterministic).")
    p.add_argument("--use_ema", action="store_true", help="Use EMA weights for sampling.")
    p.add_argument("--guidance_scale", type=float, default=0.0, help="CFG scale (class-conditional models).")
    p.add_argument("--label", type=int, default=None, help="If set, sample this class id for all images.")
    return p                                                                        # return parser

# -----------------------------
# Main
# -----------------------------
def main():
    args = build_parser().parse_args()                                            # parse
    dev = "cuda" if torch.cuda.is_available() else "cpu"                          # device

    ck = load_ckpt(args.checkpoint, map_location=dev)                             # load ckpt dict
    ck_args = ck["args"]                                                          # saved training args

    # Rebuild model (respect training conditioning)
    num_classes = None                                                            # default: unconditional
    if ck_args.get("cond_mode", "none") == "class":                               # conditional training
        num_classes = int(ck_args.get("class_limit", 200) or 200)                 # class count used in train

    model = UNet(base=int(ck_args.get("base", 64)), num_classes=num_classes).to(dev)  # model
    if args.use_ema and (ck.get("ema") is not None):                              # EMA sampling
        model.load_state_dict(ck["ema"], strict=False)                            # load EMA
    else:
        model.load_state_dict(ck["model"], strict=False)                          # load raw weights
    model.eval()                                                                  # eval mode

    # Diffusion object as in training
    diff = Diffusion(
        T=int(ck_args.get("num_steps", 1000)),                                    # same T
        schedule=str(ck_args.get("schedule", "cosine")),                          # same schedule
        device=dev
    )

    # ----- TASK: SAMPLE -----
    if args.task == "sample":
        os.makedirs(args.outdir, exist_ok=True)                                   # ensure dir
        B = args.num_samples                                                      # total samples
        C, H, W = 3, args.img_size, args.img_size                                 # shape
        # Labels: fixed class if provided; else random if conditional; else None
        y = None
        if num_classes is not None:                                               # conditional case
            if args.label is not None:                                            # fixed label
                y = torch.full((B,), int(args.label), device=dev, dtype=torch.long)  # all same
            else:
                y = torch.randint(0, num_classes, (B,), device=dev, dtype=torch.long)  # random labels

        # Pick sampler
        if args.sampler == "ddpm":                                                # ancestral
            x = diff.sample_ddpm(model, (B, C, H, W), y=y, guidance_scale=args.guidance_scale)
        else:                                                                      # DDIM
            x = diff.sample_ddim(model, (B, C, H, W), steps=args.ddim_steps, eta=args.eta,
                                 y=y, guidance_scale=args.guidance_scale, skip_first=0)

        # Save grid
        x = (x.clamp(-1, 1) + 1) / 2                                              # [0,1]
        save_image(x, os.path.join(args.outdir, "grid.png"), nrow=int(max(1, round(B ** 0.5))))  # grid

    # ----- TASK: FID / PR -----
    elif args.task == "fid":
        try:
            import torch_fidelity                                                # import metric lib
        except Exception:
            raise RuntimeError("Install torch-fidelity: pip install torch-fidelity")

        # Build temporary dirs with generated and real images, then call fidelity API.
        # (Implementation omitted for brevity in this didactic script.)
        raise NotImplementedError("FID/PR example omitted here to keep the script compact.")

    # ----- TASK: TWO-SAMPLE TEST -----
    elif args.task == "two_sample":
        # Train a small discriminator to distinguish real vs fake; report accuracy ~50% if good.
        # (Implementation omitted here to keep the script compact.)
        raise NotImplementedError("Two-sample test omitted here to keep the script compact.")

if __name__ == "__main__":                                                       # entry
    main()                                                                        # run
