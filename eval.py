# eval.py
# Sampling / FID-Precision-Recall / Two-sample test with a small discriminator.
# Every line commented.

import argparse                                   # CLI
import os                                         # paths
import torch                                      # torch
from torchvision.utils import save_image          # save grids
from torchvision import datasets, transforms      # real data for metrics (optional)

from unet import UNet                             # model
from diffusion import Diffusion                   # scheduler
from utils import load_ckpt                       # ckpt helper

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, required=True, choices=["sample","fid","two_sample"])  # mode
    ap.add_argument("--checkpoint", type=str, required=True)            # ckpt path
    ap.add_argument("--img_size", type=int, default=48)                 # size
    ap.add_argument("--outdir", type=str, required=True)                # output dir
    ap.add_argument("--num_samples", type=int, default=64)              # #samples for sample mode
    ap.add_argument("--use_ema", action="store_true")                   # use EMA weights
    ap.add_argument("--class_id", type=int, default=None)               # class to condition on (sample)
    ap.add_argument("--guidance_scale", type=float, default=0.0)        # CFG scale
    # FID/PR specific
    ap.add_argument("--num_fake", type=int, default=500)                # how many fakes to generate
    ap.add_argument("--use_bbox_real", action="store_true")             # use cropped reals (for CUB)
    ap.add_argument("--cub_root", type=str, default=None)               # path to CUB (for real stats)
    # Two-sample discriminator test
    ap.add_argument("--disc_steps", type=int, default=5000)             # discrim training steps
    return ap.parse_args()

def build_model_from_ckpt(ckpt_path, device="cuda"):
    ck = load_ckpt(ckpt_path, map_location=device)                      # load dict
    a = ck["args"]                                                      # saved args
    cond = (a.get("cond_mode","none") == "class")                       # whether cond
    num_classes = a.get("class_limit", 200) if cond else None           # #classes
    model = UNet(base=a.get("base",64), num_classes=num_classes).to(device)  # build same arch
    state = ck["ema"] if ("ema" in ck and ck["ema"] is not None) else ck["model"]  # pick weights
    model.load_state_dict(state, strict=False)                          # load
    model.eval()                                                        # eval mode
    diff = Diffusion(T=a.get("num_steps",200), schedule=a.get("schedule","cosine"),
                     device=torch.device(device))                       # scheduler
    return model, diff, a, cond, num_classes

def do_sample(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"             # device
    model, diff, a, cond, num_classes = build_model_from_ckpt(args.checkpoint, device)  # build
    os.makedirs(args.outdir, exist_ok=True)                             # create
    B, H = args.num_samples, args.img_size                              # batch,size
    # Build conditioning vector if requested
    y = None
    if cond and args.class_id is not None:
        y = torch.full((B,), int(args.class_id), device=device, dtype=torch.long)  # fixed class
    # Sample images
    with torch.no_grad():                                               # no grads
        x = diff.sample(model, (B,3,H,H), y=y, guidance_scale=(args.guidance_scale if cond else 0.0),
                        deterministic=False)                             # ancestral sampler
        x = (x.clamp(-1,1) + 1) / 2                                     # [0,1]
        save_image(x, os.path.join(args.outdir, "grid.png"), nrow=int(B**0.5))  # save
        print("Saved ->", os.path.join(args.outdir, "grid.png"))        # log

def do_fid(args):
    # Requires: pip install torch-fidelity
    from torch_fidelity import calculate_metrics                         # import here
    device = "cuda" if torch.cuda.is_available() else "cpu"             # device
    model, diff, a, cond, num_classes = build_model_from_ckpt(args.checkpoint, device)  # build
    os.makedirs(args.outdir, exist_ok=True)                             # outdir
    # Generate fake images into a temp folder
    fake_dir = os.path.join(args.outdir, "fakes"); os.makedirs(fake_dir, exist_ok=True)
    H = args.img_size
    n = args.num_fake
    bs = 64
    with torch.no_grad():
        wrote = 0
        while wrote < n:
            b = min(bs, n - wrote)
            y = None
            if cond:
                # uniform over classes available in training (approx)
                y = torch.randint(0, num_classes, (b,), device=device)
            x = diff.sample(model, (b,3,H,H), y=y, guidance_scale=(args.guidance_scale if cond else 0.0))
            x = (x.clamp(-1,1)+1)/2
            for i in range(b):
                save_image(x[i], os.path.join(fake_dir, f"{wrote+i:06d}.png"))
            wrote += b
            print(f"generated {wrote}/{n}")
    # Real images directory
    if args.use_bbox_real and args.cub_root is not None:
        # For quick evaluation we can just point to a folder of cropped reals you saved;
        # otherwise, use the original images dir (a bit unfair but quick).
        real_dir = os.path.join(args.cub_root, "images")
    else:
        real_dir = os.path.join(os.path.dirname(args.outdir), "reals")   # fallback (user-provided)
    # Compute metrics
    metrics = calculate_metrics(input1=fake_dir, input2=real_dir, cuda=(device=="cuda"),
                                isc=False, fid=True, kid=True, precision=True, recall=True)
    print(metrics)

def do_two_sample(args):
    # Trains a tiny discriminator to distinguish real vs fake (should be ~50% if perfect)
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, diff, a, cond, num_classes = build_model_from_ckpt(args.checkpoint, device)
    H = args.img_size

    # Very small conv discriminator
    D = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1), nn.LeakyReLU(0.2, True),
        nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True),
        nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(256, 1)
    ).to(device)

    opt = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    # Real images loader (user should point to a small dir of reals sized H)
    tfm = transforms.Compose([
        transforms.Resize(H), transforms.CenterCrop(H),
        transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])
    # Default: use CUB images folder
    real_dir = os.path.join(os.path.dirname(args.outdir), "reals")
    if not os.path.isdir(real_dir):
        real_dir = args.cub_root or "/content/CUB_200_2011/images"
    real_ds = datasets.ImageFolder(real_dir, transform=tfm)
    real_dl = torch.utils.data.DataLoader(real_ds, batch_size=32, shuffle=True, num_workers=2, drop_last=True)

    # Train discrim for a bit
    D.train()
    steps = 0
    while steps < args.disc_steps:
        for real, _ in real_dl:
            real = real.to(device)
            bs = real.size(0)
            # make fakes
            with torch.no_grad():
                y = None
                if cond:
                    y = torch.randint(0, num_classes, (bs,), device=device)
                fake = diff.sample(model, (bs,3,H,H), y=y, guidance_scale=(args.guidance_scale if cond else 0.0))
            # compute loss
            opt.zero_grad(set_to_none=True)
            pred_real = D(real); pred_fake = D(fake)
            loss = bce(pred_real, torch.ones_like(pred_real)) + bce(pred_fake, torch.zeros_like(pred_fake))
            loss.backward(); opt.step()
            steps += 1
            if steps % 200 == 0:
                print(f"D steps {steps}/{args.disc_steps} | loss {loss.item():.3f}")
            if steps >= args.disc_steps: break

    # Evaluate accuracy on a small mixed set
