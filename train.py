# train.py
# DDPM training script (class-conditional optional). Stable defaults for CUB.
# Every line commented.

import argparse                                   # CLI parser
import os                                         # paths
import torch                                      # torch
import torch.nn as nn                             # nn
from torch.cuda.amp import autocast, GradScaler   # mixed precision
from torchvision.utils import save_image          # save grids

from unet import UNet                             # our UNet
from diffusion import Diffusion                   # scheduler + loss + sample
from data import make_transforms, make_cub_bbox_dataset, make_loader  # data
from utils import EMA, seed_everything, cosine_warmup_lr, save_ckpt, load_ckpt  # utils

def parse_args():
    # Define all CLI flags with sane defaults for Colab/T4
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)                # CUB root
    ap.add_argument("--use_bbox", action="store_true")                     # use bbox crop
    ap.add_argument("--bbox_expand", type=float, default=1.0)              # expansion factor
    ap.add_argument("--subset", type=int, default=None)                    # max samples
    ap.add_argument("--class_limit", type=int, default=None)               # limit classes
    ap.add_argument("--outdir", type=str, required=True)                   # where to save

    ap.add_argument("--img_size", type=int, default=48)                    # resolution
    ap.add_argument("--batch_size", type=int, default=16)                  # batch
    ap.add_argument("--epochs", type=int, default=999)                     # epochs (unused if max_steps)
    ap.add_argument("--max_steps", type=int, default=None)                 # hard stop on steps
    ap.add_argument("--lr", type=float, default=2e-4)                      # base LR
    ap.add_argument("--weight_decay", type=float, default=0.0)             # wd
    ap.add_argument("--num_steps", type=int, default=200)                  # diffusion T
    ap.add_argument("--schedule", type=str, default="cosine", choices=["cosine","linear"])  # schedule

    ap.add_argument("--cond_mode", type=str, default="none", choices=["none","class"])      # conditioning
    ap.add_argument("--p_uncond", type=float, default=0.1)                 # CFG drop prob (train)
    ap.add_argument("--guidance_scale", type=float, default=0.0)           # ONLY used for preview sampling

    ap.add_argument("--base", type=int, default=64)                        # UNet width
    ap.add_argument("--ema_mu", type=float, default=0.9995)                # EMA decay
    ap.add_argument("--seed", type=int, default=0)                         # seed
    ap.add_argument("--resume", type=str, default=None)                    # path to ckpt to resume
    ap.add_argument("--log_every", type=int, default=200)                  # preview/saving frequency
    return ap.parse_args()

def main():
    args = parse_args()                                                    # parse CLI
    device = "cuda" if torch.cuda.is_available() else "cpu"               # pick device
    seed_everything(args.seed)                                             # reproducibility

    # ---------------- Data ----------------
    tfm = make_transforms(args.img_size)                                   # resize->centercrop->norm
    expand = args.bbox_expand if args.use_bbox else 1.0                    # expansion factor
    ds, num_classes = make_cub_bbox_dataset(args.data_root, tfm, expand,
                                            class_limit=args.class_limit,
                                            subset=args.subset)            # dataset
    dl = make_loader(ds, args.batch_size, shuffle=True, num_workers=2)     # dataloader

    # ------------- Model & Opt -------------
    cond = (args.cond_mode == "class")                                     # whether conditional
    model = UNet(base=args.base, num_classes=(num_classes if cond else None)).to(device)  # UNet
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # AdamW
    scaler = GradScaler()                                                  # AMP scaler
    ema = EMA(model, mu=args.ema_mu)                                       # EMA tracker
    diffusion = Diffusion(T=args.num_steps, schedule=args.schedule, device=torch.device(device))  # scheduler

    # Resume (must match architecture flags; do not change base/T!)
    step0 = 0                                                               # initial step
    if args.resume is not None and os.path.isfile(args.resume):             # resume if path exists
        ckpt = load_ckpt(args.resume, map_location=device)                  # load ckpt
        model.load_state_dict(ckpt["model"], strict=False)                  # restore weights
        if ckpt.get("ema", None) is not None:                               # restore EMA if present
            ema.shadow.update(ckpt["ema"]) if hasattr(ema.shadow, "update") else ema.shadow.update(ckpt["ema"])
            ema.shadow = ckpt["ema"]                                        # assign shadow
        step0 = int(ckpt.get("step", 0))                                    # restore step
        print(f"[resume] loaded {args.resume} at step {step0}")             # log resume

    # ------------- Train Loop -------------
    os.makedirs(args.outdir, exist_ok=True)                                 # create outdir
    model.train()                                                           # training mode
    global_step = step0                                                     # track steps

    for epoch in range(args.epochs):                                        # epochs loop
        for x, y in dl:                                                     # iterate minibatches
            x = x.to(device, non_blocking=True)                             # move images
            y = y.to(device, non_blocking=True) if cond else None           # move labels if any

            # LR schedule per step
            lr = cosine_warmup_lr(global_step, args.max_steps or 1000000, args.lr, warmup=200) # compute lr
            for pg in opt.param_groups: pg["lr"] = lr                       # set lr

            # Forward + loss (AMP)
            opt.zero_grad(set_to_none=True)                                 # clear grads
            with autocast(device_type="cuda", enabled=(device=="cuda")):    # mixed precision
                loss = diffusion.p_losses(model, x, y=y, p_uncond=(args.p_uncond if cond else 0.0)) # noise loss
            scaler.scale(loss).backward()                                    # backward scaled
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)         # grad clip for stability
            scaler.step(opt)                                                 # optimizer step
            scaler.update()                                                  # update scaler
            ema.update(model)                                                # update EMA

            # Logging
            if global_step % 100 == 0:                                       # print a bit more often
                print(f"step {global_step} | loss {loss.item():.4f} | lr {lr:.2e}")

            # Periodic sampling preview (cheap: small grid)
            if (global_step > 0) and (global_step % args.log_every == 0):
                model.eval()                                                 # eval mode
                # Use EMA weights for sampling preview
                ema_model = UNet(base=args.base, num_classes=(num_classes if cond else None)).to(device) # mirror
                ema.copy_to(ema_model)                                       # load EMA
                with torch.no_grad():                                        # no grads
                    B = 16                                                   # small grid
                    y_samp = None
                    if cond:                                                 # build a small y vector if conditional
                        # Sample labels uniformly over kept classes (0..num_classes-1)
                        y_samp = torch.randint(0, num_classes, (B,), device=device)
                    x_gen = diffusion.sample(ema_model, (B,3,args.img_size,args.img_size),
                                               y=y_samp, guidance_scale=(args.guidance_scale if cond else 0.0),
                                               deterministic=False)          # ancestral
                    grid = (x_gen.clamp(-1,1) + 1) / 2                       # to [0,1]
                    save_image(grid, os.path.join(args.outdir, f"samples_epoch_{epoch:03d}.png"), nrow=4)  # save grid
                model.train()                                                 # back to train

            global_step += 1                                                  # increment steps
            if args.max_steps is not None and global_step >= args.max_steps: # stopping condition
                print("Reached max_steps, stopping training.")                # info
                # Final checkpoint
                save_ckpt(os.path.join(args.outdir, "last.ckpt"), model, ema, vars(args), global_step)
                return                                                        # exit

        # End of epoch: save ckpt
        save_ckpt(os.path.join(args.outdir, "last.ckpt"), model, ema, vars(args), global_step)  # save

if __name__ == "__main__":                                                   # entrypoint
    main()                                                                   # run main


