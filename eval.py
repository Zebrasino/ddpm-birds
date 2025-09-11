# eval.py — chunked sampler for DDIM/DDPM with CFG and EMA
import argparse, os, math, torch
from torchvision.utils import save_image
from diffusion import Diffusion
from unet import UNet

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="sample")          # kept for parity
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--num_samples", type=int, default=64)
    ap.add_argument("--chunk_size", type=int, default=64)          # NEW: per-GPU batch at sampling
    ap.add_argument("--sampler", type=str, default="ddim", choices=["ddim","ddpm","ddpm_det"])
    ap.add_argument("--ddim_steps", type=int, default=120)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--skip_first", type=int, default=60)
    ap.add_argument("--guidance_scale", type=float, default=2.3)
    ap.add_argument("--use_ema", action="store_true")
    ap.add_argument("--label", type=int, default=None)             # class-cond label or None
    ap.add_argument("--base", type=int, default=96)                # UNet base if shape check is needed
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- model/diffusion ----
    num_classes = 200 if args.label is not None else None          # simple heuristic
    model = UNet(base=args.base, num_classes=num_classes).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    # Load raw weights
    model.load_state_dict(ckpt["model"], strict=False)

    # If requested, load EMA weights into the model
    if args.use_ema and "ema" in ckpt:
        shadow = {k: v.to(device) for k, v in ckpt["ema"].items()}
        model.load_state_dict(shadow, strict=False)

    model.eval()
    diff = Diffusion(T=ckpt.get("args", {}).get("num_steps", 400),
                     schedule=ckpt.get("args", {}).get("schedule", "cosine"),
                     device=device)

    H = W = args.img_size
    total = args.num_samples
    bs = max(1, min(args.chunk_size, total))                       # per-chunk batch

    # ---- optional AMP to save VRAM ----
    amp_ctx = (lambda: torch.amp.autocast(device_type="cuda")) if device.type=="cuda" else (lambda: torch.autocast("cpu"))

    # ---- first small grid for sanity (does not OOM) ----
    B0 = min(64, total, bs)
    with amp_ctx():
        y = None if args.label is None else torch.full((B0,), int(args.label), device=device, dtype=torch.long)
        if args.sampler == "ddim":
            x = diff.sample_ddim(model, (B0,3,H,W), y=y, steps=args.ddim_steps, eta=args.eta,
                                  guidance_scale=args.guidance_scale, skip_first=args.skip_first)
        elif args.sampler == "ddpm_det":
            x = diff.sample_ddpm_deterministic(model, (B0,3,H,W), y=y, guidance_scale=args.guidance_scale)
        else:  # ddpm stochastic (ancestral)
            x = diff.sample_ddpm(model, (B0,3,H,W), y=y, guidance_scale=args.guidance_scale)
    save_image((x.clamp(-1,1)+1)/2, os.path.join(args.outdir, "grid.png"), nrow=int(math.sqrt(B0)) or 4)
    print("Saved grid ->", os.path.join(args.outdir, "grid.png"))

    # ---- generate the rest in chunks, saving individual PNGs ----
    produced = 0
    idx_global = 0
    while produced < total:
        B = min(bs, total - produced)
        with amp_ctx():
            y = None if args.label is None else torch.full((B,), int(args.label), device=device, dtype=torch.long)
            if args.sampler == "ddim":
                x = diff.sample_ddim(model, (B,3,H,W), y=y, steps=args.ddim_steps, eta=args.eta,
                                      guidance_scale=args.guidance_scale, skip_first=args.skip_first)
            elif args.sampler == "ddpm_det":
                x = diff.sample_ddpm_deterministic(model, (B,3,H,W), y=y, guidance_scale=args.guidance_scale)
            else:
                x = diff.sample_ddpm(model, (B,3,H,W), y=y, guidance_scale=args.guidance_scale)

        x = (x.clamp(-1,1)+1)/2
        # Save B individual images: 000001.png, 000002.png, ...
        for i in range(B):
            idx_global += 1
            save_image(x[i], os.path.join(args.outdir, f"{idx_global:06d}.png"))
        produced += B

        # free some VRAM between chunks
        del x; torch.cuda.empty_cache() if device.type=="cuda" else None
        print(f"Generated {produced}/{total}")

    print("Done.")

if __name__ == "__main__":
    main()
