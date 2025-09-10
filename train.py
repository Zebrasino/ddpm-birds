# train.py
# DDPM training script (class-conditional optional).
# Uses the new torch.amp autocast/GradScaler API (no deprecation warnings).
# Every line is commented for clarity.

import argparse                                  # parse CLI flags
import os                                        # filesystem utilities
import contextlib                                # nullcontext for CPU autocast
import torch                                     # PyTorch core
import torch.nn as nn                            # neural network helpers
from torch import amp                            # NEW API: torch.amp for autocast/scaler
from torchvision.utils import save_image         # image grid saving

from unet import UNet                            # our U-Net ε-predictor
from diffusion import Diffusion                  # DDPM scheduler + loss + sampler
from data import (                               # dataset builders / loader
    make_transforms,
    make_cub_bbox_dataset,
    make_loader,
)
from utils import (                              # misc training utilities
    EMA,
    seed_everything,
    cosine_warmup_lr,
    save_ckpt,
    load_ckpt,
)

# --------------------------
# CLI arguments definition
# --------------------------
def parse_args():
    ap = argparse.ArgumentParser()                                        # create parser

    # Data / I/O
    ap.add_argument("--data_root", type=str, required=True)               # path to CUB_200_2011
    ap.add_argument("--use_bbox", action="store_true")                    # crop to provided bbox
    ap.add_argument("--bbox_expand", type=float, default=1.0)             # bbox expansion factor
    ap.add_argument("--subset", type=int, default=None)                   # use only first N items
    ap.add_argument("--class_limit", type=int, default=None)              # keep only first K classes
    ap.add_argument("--outdir", type=str, required=True)                  # output directory

    # Training hyper-parameters
    ap.add_argument("--img_size", type=int, default=48)                   # training resolution
    ap.add_argument("--batch_size", type=int, default=16)                 # batch size
    ap.add_argument("--epochs", type=int, default=999)                    # max epochs (guard)
    ap.add_argument("--max_steps", type=int, default=None)                # hard stop after N steps
    ap.add_argument("--lr", type=float, default=2e-4)                     # base learning rate
    ap.add_argument("--weight_decay", type=float, default=0.0)            # weight decay
    ap.add_argument("--num_steps", type=int, default=200)                 # diffusion steps T
    ap.add_argument("--schedule", type=str, default="cosine",             # beta schedule
                    choices=["cosine", "linear"])

    # Conditioning / guidance
    ap.add_argument("--cond_mode", type=str, default="none",              # enable class conditioning
                    choices=["none", "class"])
    ap.add_argument("--p_uncond", type=float, default=0.1)                # CFG dropout prob at train
    ap.add_argument("--guidance_scale", type=float, default=0.0)          # preview guidance at sample

    # Model / EMA / misc
    ap.add_argument("--base", type=int, default=64)                       # UNet base channels
    ap.add_argument("--ema_mu", type=float, default=0.9995)               # EMA decay
    ap.add_argument("--seed", type=int, default=0)                        # RNG seed
    ap.add_argument("--resume", type=str, default=None)                   # checkpoint to resume
    ap.add_argument("--log_every", type=int, default=200)                 # preview/sample frequency

    return ap.parse_args()                                                # return populated args


# --------------------------
# Main training entrypoint
# --------------------------
def main():
    args = parse_args()                                                   # parse flags
    device = "cuda" if torch.cuda.is_available() else "cpu"              # choose device
    seed_everything(args.seed)                                            # reproducibility knobs

    # ---------- Build data ----------
    tfm = make_transforms(args.img_size)                                  # Resize->CenterCrop->ToTensor->Norm
    expand = args.bbox_expand if args.use_bbox else 1.0                   # expansion factor (1.0 = tight bbox)
    ds, num_classes = make_cub_bbox_dataset(                              # dataset + #classes actually used
        args.data_root, tfm, expand,
        class_limit=args.class_limit,
        subset=args.subset,
    )
    dl = make_loader(ds, args.batch_size, shuffle=True, num_workers=2)    # small num_workers for Colab stability

    # ---------- Build model / opt / EMA / scheduler ----------
    cond = (args.cond_mode == "class")                                    # whether class conditioning is active
    model = UNet(                                                         # instantiate UNet
        base=args.base,
        num_classes=(num_classes if cond else None),
    ).to(device)
    opt = torch.optim.AdamW(                                              # AdamW optimizer
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = amp.GradScaler('cuda') if device == 'cuda' else None         # NEW API GradScaler (None on CPU)
    ema = EMA(model, mu=args.ema_mu)                                      # exponential moving average
    diffusion = Diffusion(                                                # DDPM scheduler and loss
        T=args.num_steps, schedule=args.schedule, device=torch.device(device)
    )

    # ---------- (Optional) resume ----------
    global_step = 0                                                       # initialize step counter
    if args.resume is not None and os.path.isfile(args.resume):           # if resume path exists
        ckpt = load_ckpt(args.resume, map_location=device)                # load checkpoint dict
        model.load_state_dict(ckpt["model"], strict=False)                # restore model weights (strict=False tolerant)
        if ckpt.get("ema", None) is not None:                             # restore EMA shadow if present
            ema.shadow = ckpt["ema"]                                      # assign shadow weights dictionary
        global_step = int(ckpt.get("step", 0))                            # continue step count
        print(f"[resume] loaded '{args.resume}' at step {global_step}")   # log resume info

    # ---------- Make sure outdir exists ----------
    os.makedirs(args.outdir, exist_ok=True)                               # create output directory

    # ---------- Training loop ----------
    model.train()                                                         # set training mode
    for epoch in range(args.epochs):                                      # epoch guard (often we stop via max_steps)
        for x, y in dl:                                                   # iterate mini-batches
            x = x.to(device, non_blocking=True)                           # move images to device
            y = y.to(device, non_blocking=True) if cond else None         # move labels if conditional

            # Per-step cosine LR with warmup (smooth & stable on small batches)
            lr = cosine_warmup_lr(                                        # compute LR for this step
                global_step, args.max_steps or 1_000_000, args.lr, warmup=200
            )
            for pg in opt.param_groups:                                   # apply LR to all param groups
                pg["lr"] = lr

            # Choose autocast context depending on device (null on CPU)
            amp_ctx = amp.autocast('cuda', dtype=torch.float16) if device == 'cuda' else contextlib.nullcontext()

            opt.zero_grad(set_to_none=True)                               # clear previous gradients
            with amp_ctx:                                                 # mixed precision region
                loss = diffusion.p_losses(                                # DDPM ε-MSE loss
                    model, x, y=y, p_uncond=(args.p_uncond if cond else 0.0)
                )

            # Backpropagation + optimizer step (with or without GradScaler)
            if scaler is not None:                                        # CUDA: use GradScaler
                scaler.scale(loss).backward()                             # scaled backward
                scaler.unscale_(opt)                                      # unscale before clipping (best practice)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # gradient clipping for stability
                scaler.step(opt)                                          # optimizer step (scaled)
                scaler.update()                                           # update scaler
            else:                                                         # CPU path: plain FP32
                loss.backward()                                           # backprop
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # gradient clipping
                opt.step()                                                # optimizer step

            ema.update(model)                                             # update EMA shadow weights

            # Light logging to stdout
            if global_step % 100 == 0:                                    # print every 100 steps
                print(f"step {global_step} | loss {loss.item():.4f} | lr {lr:.2e}")

            # Periodic sampling preview (small grid, EMA weights)
            if (global_step > 0) and (global_step % args.log_every == 0): # time to preview
                model.eval()                                              # switch to eval
                # Build a mirror model and load EMA to keep training model untouched
                ema_model = UNet(base=args.base,                          # same architecture as training
                                  num_classes=(num_classes if cond else None)).to(device)
                ema.copy_to(ema_model)                                    # copy EMA weights into mirror
                with torch.no_grad():                                     # no grad during sampling
                    B = 16                                                # grid size (4×4)
                    H = args.img_size                                     # resolution
                    y_samp = None                                         # default: unconditional
                    if cond:                                              # if model is conditional
                        # uniform random labels in the kept class set (0..num_classes-1)
                        y_samp = torch.randint(0, num_classes, (B,), device=device, dtype=torch.long)
                    # Sample via DDPM ancestral sampler (CFG only if cond)
                    x_gen = diffusion.sample(ema_model, (B, 3, H, H),
                                             y=y_samp,
                                             guidance_scale=(args.guidance_scale if cond else 0.0),
                                             deterministic=False)
                    grid = (x_gen.clamp(-1, 1) + 1) / 2                   # map to [0,1] for saving
                    save_path = os.path.join(args.outdir, f"samples_epoch_{epoch:03d}.png")
                    save_image(grid, save_path, nrow=4)                   # write preview grid
                model.train()                                             # back to training mode

            global_step += 1                                              # advance global step
            # Hard stop condition by steps (typical on Colab to limit time)
            if args.max_steps is not None and global_step >= args.max_steps:
                print("Reached max_steps, stopping training.")            # notify user
                save_ckpt(os.path.join(args.outdir, "last.ckpt"),         # save final checkpoint
                          model, ema, vars(args), global_step)
                return                                                    # leave main()

        # End of epoch: always save a checkpoint (allows resume later)
        save_ckpt(os.path.join(args.outdir, "last.ckpt"),                 # save periodic checkpoint
                  model, ema, vars(args), global_step)

# Standard Python entrypoint: run training when executed as a script
if __name__ == "__main__":
    main()


