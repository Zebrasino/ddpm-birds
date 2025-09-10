# eval.py
# Sampling / evaluation entry-point.
# Supports:
#   - DDPM (ancestral or deterministic) and DDIM sampling
#   - EMA or raw weights
#   - optional classifier-free guidance scale
#   - DDIM "skip_first" to truncate earliest noisy steps
# Every line is commented for clarity.

import argparse                              # CLI flags
import os                                    # filesystem
import torch                                  # PyTorch
from torchvision.utils import save_image      # image grid saving

from unet import UNet                         # ε-predictor backbone
from diffusion import Diffusion               # schedules + samplers


def parse_args():
    ap = argparse.ArgumentParser()                                    # create parser
    ap.add_argument("--task", type=str, default="sample", choices=["sample"])  # only sampling here
    ap.add_argument("--checkpoint", type=str, required=True)          # path to .ckpt
    ap.add_argument("--img_size", type=int, default=None)             # override size (optional)
    ap.add_argument("--outdir", type=str, required=True)              # output folder
    ap.add_argument("--num_samples", type=int, default=16)            # grid size (B)
    ap.add_argument("--use_ema", action="store_true")                 # use EMA weights if present
    ap.add_argument("--guidance_scale", type=float, default=0.0)      # CFG scale (0 = off)
    ap.add_argument("--label", type=int, default=None)                # class label (if conditional model)

    # sampler selection and options
    ap.add_argument("--sampler", type=str, default="ddim",
                    choices=["ddpm", "ddpm_det", "ddim"])             # choose algorithm
    ap.add_argument("--ddim_steps", type=int, default=50)             # DDIM steps
    ap.add_argument("--eta", type=float, default=0.0)                 # DDIM eta (0=deterministic)
    ap.add_argument("--skip_first", type=int, default=0)              # truncate earliest steps in DDIM

    return ap.parse_args()                                            # parsed args


def main():
    args = parse_args()                                               # read CLI
    device = "cuda" if torch.cuda.is_available() else "cpu"          # choose device

    # ---- load checkpoint & hyper-params ----
    ckpt = torch.load(args.checkpoint, map_location=device)          # load dict
    conf = ckpt["args"]                                              # training args saved
    img_size = args.img_size or int(conf.get("img_size", 48))        # image size to sample
    base = int(conf.get("base", 64))                                 # UNet base channels
    T = int(conf.get("num_steps", 200))                              # diffusion steps used in training
    schedule = conf.get("schedule", "cosine")                        # beta schedule type
    cond_mode = conf.get("cond_mode", "none")                        # "none" or "class"

    # ---- build model (conditional only if training was) ----
    num_classes = None                                               # default unconditional
    if cond_mode == "class":
        # best-effort: try to recover number of classes from training args
        # (if unavailable, you must pass labels within valid range manually)
        num_classes = int(conf.get("num_classes", conf.get("class_limit", 0) or 200))

    model = UNet(base=base, num_classes=num_classes).to(device).eval()  # instantiate & eval

    # ---- load weights: EMA preferred if requested and present ----
    target_state = None                                              # choose state dict to load
    if args.use_ema and ckpt.get("ema", None) is not None:
        target_state = ckpt["ema"]                                   # EMA weights
    else:
        target_state = ckpt["model"]                                 # raw training weights
    # copy the (possibly partial) state dict into model parameters
    msd = model.state_dict()
    for k in msd:
        if k in target_state:
            msd[k].copy_(target_state[k])

    # ---- build diffusion object with the SAME T and schedule ----
    diff = Diffusion(T=T, schedule=schedule, device=torch.device(device))

    # ---- prepare label tensor if class-conditional ----
    y = None                                                         # default unconditional
    if num_classes is not None:
        # If user passed a single label, repeat it; else sample random labels.
        if args.label is not None:
            y = torch.full((args.num_samples,), int(args.label), device=device, dtype=torch.long)
        else:
            y = torch.randint(0, num_classes, (args.num_samples,), device=device, dtype=torch.long)

    # ---- run the chosen sampler ----
    B = args.num_samples                                             # number of samples
    H = img_size                                                     # resolution
    os.makedirs(args.outdir, exist_ok=True)                          # ensure outdir exists

    if args.sampler == "ddpm":
        # ancestral DDPM (noisy) — more diversity, less sharp for small models
        x = diff.sample_ddpm(model, (B, 3, H, H), y=y,
                             guidance_scale=max(0.0, float(args.guidance_scale)),
                             deterministic=False)
    elif args.sampler == "ddpm_det":
        # deterministic DDPM (sigma=0) — crisper than ancestral, still many steps
        x = diff.sample_ddpm(model, (B, 3, H, H), y=y,
                             guidance_scale=max(0.0, float(args.guidance_scale)),
                             deterministic=True)
    else:
        # DDIM — fewer steps, deterministic if eta=0, robust for weak models
        x = diff.sample_ddim(model, (B, 3, H, H), y=y,
                             steps=int(args.ddim_steps),
                             eta=float(args.eta),
                             guidance_scale=max(0.0, float(args.guidance_scale)),
                             skip_first=max(0, int(args.skip_first)))

    # ---- save grid in [0,1] ----
    x = (x.clamp(-1, 1) + 1) / 2                                     # map to [0,1]
    save_image(x, os.path.join(args.outdir, "grid.png"), nrow=int(B ** 0.5) or 4)
    print("Saved ->", os.path.join(args.outdir, "grid.png"))         # notify path


if __name__ == "__main__":
    main()                                                           # entrypoint
