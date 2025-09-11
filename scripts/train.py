import os, math, argparse, random
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Allow running without installation: add ../src to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

# Import from the package
from ddpm_birds import Diffusion, make_cub_bbox_dataset, UNet
from ddpm_birds import EMA, cosine_warmup_lr  # if you want to reuse these instead of local copies



# ---------- small EMA helper ----------
class EMA:
    """Exponential Moving Average for model weights."""
    def __init__(self, model: nn.Module, mu: float = 0.999):
        self.mu = mu                                 # EMA decay (closer to 1.0 = slower update)
        # Take an initial snapshot of model weights (detached, cloned tensors)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        # Shadow <- mu * Shadow + (1 - mu) * Weight
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.mu).add_(v.detach(), alpha=1.0 - self.mu)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        # Load EMA weights into the model (use strict=False to survive shape diffs)
        model.load_state_dict(self.shadow, strict=False)


# ---------- AMP glue that works on PyTorch 1.13–2.x ----------
def make_amp(device: str):
    """Return (autocast_ctx_fn, GradScaler_or_None) appropriate for this runtime."""
    scaler = None                                    # default: no scaler on CPU / older runtimes
    if device == "cuda":                             # CUDA-only AMP
        # Prefer torch.amp (new API) if available
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            def ctx():
                return torch.amp.autocast(device_type="cuda")  # autocast context manager
            scaler = torch.amp.GradScaler("cuda") if hasattr(torch.amp, "GradScaler") else None
            return ctx, scaler
        else:
            # Fallback to torch.cuda.amp (older API)
            from torch.cuda.amp import autocast, GradScaler
            def ctx():
                return autocast()
            scaler = GradScaler()
            return ctx, scaler
    # CPU path: use a null context (no autocast, no scaler)
    from contextlib import nullcontext
    return (lambda: nullcontext()), None


def parse_args():
    """Define and parse all command-line arguments."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)          # CUB_200_2011 root
    ap.add_argument("--use_bbox", action="store_true")               # crop/weight around bbox
    ap.add_argument("--bbox_expand", type=float, default=1.0)        # enlarge bbox by this factor
    ap.add_argument("--subset", type=int, default=None)              # use only first K images
    ap.add_argument("--class_limit", type=int, default=None)         # keep only first N classes

    ap.add_argument("--outdir", type=str, required=True)             # output directory
    ap.add_argument("--img_size", type=int, default=64)              # training resolution
    ap.add_argument("--batch_size", type=int, default=16)            # batch size
    ap.add_argument("--epochs", type=int, default=999)               # dummy cap, we stop by steps
    ap.add_argument("--max_steps", type=int, default=25000)          # stop after this many steps

    ap.add_argument("--lr", type=float, default=2e-4)                # base learning rate
    ap.add_argument("--weight_decay", type=float, default=0.0)       # weight decay (optional)

    ap.add_argument("--num_steps", type=int, default=400)            # diffusion steps T
    ap.add_argument("--schedule", type=str, default="cosine",
                    choices=["cosine", "linear"])                    # beta schedule type

    ap.add_argument("--cond_mode", type=str, default="class",
                    choices=["none", "class"])                       # conditioning mode
    ap.add_argument("--p_uncond", type=float, default=0.3)           # classifier-free dropout prob
    ap.add_argument("--base", type=int, default=96)                  # UNet base channels
    ap.add_argument("--ema_mu", type=float, default=0.999)           # EMA decay factor

    ap.add_argument("--seed", type=int, default=0)                   # RNG seed
    ap.add_argument("--log_every", type=int, default=250)            # log/sample every N steps

    ap.add_argument("--fg_weight", type=float, default=5.0)          # loss weight inside bbox

    # ---- NEW: resume + warmup ----
    ap.add_argument("--resume", type=str, default=None,              # path to a .ckpt to resume
                    help="Path to checkpoint (.ckpt) to resume from")
    ap.add_argument("--warmup", type=int, default=1000,              # warmup steps (0 = off)
                    help="Number of linear warmup steps for the LR (0 disables)")

    return ap.parse_args()                                           # return parsed arguments


def set_seed(seed: int):
    """Deterministic-ish runs for debugging (keeps CuDNN fast)."""
    random.seed(seed)                                               # Python RNG
    torch.manual_seed(seed)                                         # CPU RNG
    torch.cuda.manual_seed_all(seed)                                # all GPU RNGs
    # Keep CuDNN fast and non-deterministic (deterministic=True often slows down a lot)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def cosine_lr(step: int, max_steps: int, base_lr: float) -> float:
    """Simple cosine schedule from step 0..max_steps (no warmup)."""
    if max_steps <= 0:                                              # guard for degenerate case
        return base_lr
    cos = 0.5 * (1 + math.cos(math.pi * min(step, max_steps) / max_steps))
    return base_lr * cos                                            # decay from base_lr -> 0


def cosine_warmup_lr(step: int, max_steps: int, base_lr: float, warmup: int = 0) -> float:
    """Cosine decay with an initial linear warmup."""
    if warmup and step < warmup:                                    # linear warmup 0 -> base_lr
        return base_lr * float(step) / float(max(1, warmup))
    s = max(0, step - warmup)                                       # shift to start cosine at 0
    m = max(1, max_steps - warmup)                                  # length of cosine phase
    cos = 0.5 * (1 + math.cos(math.pi * min(s, m) / m))             # cosine factor
    return base_lr * cos                                            # cosine-decayed LR


def main():
    args = parse_args()                                             # parse CLI
    set_seed(args.seed)                                             # set seeds

    device = "cuda" if torch.cuda.is_available() else "cpu"         # pick device string
    amp_ctx, scaler = make_amp(device)                              # AMP context + scaler
    torch_device = torch.device(device)                             # torch.device handle

    # ----- Data -----
    ds = make_cub_bbox_dataset(                                     # create dataset (with masks)
        root=args.data_root,
        img_size=args.img_size,
        use_bbox=args.use_bbox,
        bbox_expand=args.bbox_expand,
        class_limit=args.class_limit,
        subset=args.subset,
    )
    dl = DataLoader(                                                # simple, stable loader for Colab
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, persistent_workers=False, pin_memory=False, drop_last=True
    )

    # ----- Model -----
    num_classes = None if args.cond_mode == "none" else (args.class_limit or 200)
    model = UNet(base=args.base, num_classes=num_classes).to(torch_device)  # ε-predictor UNet

    # ----- Optimizer / EMA / Diffusion -----
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # AdamW opt
    ema = EMA(model, mu=args.ema_mu)                                   # EMA wrapper
    diffusion = Diffusion(T=args.num_steps, schedule=args.schedule, device=torch_device)  # DDPM core

    os.makedirs(args.outdir, exist_ok=True)                            # ensure output dir exists

    # ----- Optional resume from a checkpoint -----
    start_step = 0                                                     # default: start from scratch
    if args.resume is not None:                                        # resume path provided
        ckpt = torch.load(args.resume, map_location="cpu")             # load on CPU
        if "model" in ckpt:                                            # raw model weights
            model.load_state_dict(ckpt["model"], strict=False)
        if "ema" in ckpt:                                              # EMA weights for eval
            model_device = next(model.parameters()).device
            # move EMA tensors to current device
            ema.shadow = {k: v.to(model_device) for k, v in ckpt["ema"].items()}
            # If you prefer training *from* EMA weights, uncomment:
            # ema.copy_to(model)
        start_step = int(ckpt.get("step", 0))                          # resume step count
        print(f"Resumed from {args.resume} @ step {start_step}")       # log resume info

    # ----- Training loop -----
    step = start_step                                                  # continue global step
    model.train()                                                      # set train mode
    while step < args.max_steps:                                       # stop by global steps
        for batch in dl:                                               # iterate over batches
            if step >= args.max_steps:                                 # safety break
                break

            # Unpack batch: image tensor, class labels, bbox mask
            x, y, mask = batch                                         # x in [-1,1], mask in {0,1}
            x = x.to(torch_device, non_blocking=False)                 # move images to device
            y = None if (y is None) else torch.as_tensor(              # move labels (if any)
                y, device=torch_device, dtype=torch.long
            )
            mask = mask.to(torch_device, non_blocking=False)           # move mask to device

            # Update LR (cosine + optional warmup)
            lr_now = cosine_warmup_lr(step, args.max_steps, args.lr, warmup=args.warmup)
            for g in opt.param_groups:                                 # set LR for all groups
                g["lr"] = lr_now

            # Forward pass + loss (with AMP if available)
            with amp_ctx():                                            # autocast (if CUDA AMP)
                loss = diffusion.p_losses(                             # DDPM denoising loss
                    model, x, y=y,                                     # model + conditioning
                    p_uncond=args.p_uncond,                            # classifier-free dropout
                    p2_gamma=0.5, p2_k=1.0,                            # P2 loss weighting
                    fg_mask=mask, fg_weight=args.fg_weight             # foreground weighting
                )

            # Backward pass (scaled if AMP) + optimization step
            if scaler is not None:                                     # AMP path
                scaler.scale(loss).backward()                          # scaled grads
                scaler.unscale_(opt)                                   # unscale for clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
                scaler.step(opt)                                       # optimizer step
                scaler.update()                                        # update scaler
            else:                                                      # non-AMP path
                loss.backward()                                        # regular backward
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
                opt.step()                                             # optimizer step

            opt.zero_grad(set_to_none=True)                            # faster grad reset
            ema.update(model)                                          # update EMA weights
            step += 1                                                  # increment global step

            # ----- Logging / quick preview -----
            if step % args.log_every == 0 or step == 1:                # periodic logging
                print(f"step {step} | loss {loss.item():.4f} | lr {opt.param_groups[0]['lr']:.2e}")
                # Quick 4x4 grid using DDIM (fast, deterministic-ish, no CFG)
                model.eval()                                           # switch to eval for sampling
                with torch.no_grad():                                  # no-grad sampling
                    x_gen = diffusion.sample_ddim(                     # 40-step DDIM preview
                        model, shape=(16, 3, args.img_size, args.img_size),
                        y=None, steps=40, eta=0.1,                     # light stochasticity
                        guidance_scale=0.0,                            # no CFG in preview
                        skip_first=10                                  # skip only very early noise
                    )
                    x_vis = (x_gen.clamp(-1, 1) + 1) / 2               # back to [0,1] for saving
                    save_path = os.path.join(                          # build preview filename
                        args.outdir, f"samples_epoch_{step // args.log_every:03d}.png"
                    )
                    save_image(x_vis, save_path, nrow=4)               # save 4x4 grid
                model.train()                                          # back to train mode

    # ----- Save checkpoint (raw + EMA) -----
    ckpt = {
        "args": vars(args),                                            # store hyper-params for eval
        "model": model.state_dict(),                                   # last raw weights
        "ema": ema.shadow,                                             # EMA weights for clean eval
        "step": step,                                                  # final global step
    }
    ckpt_path = os.path.join(args.outdir, "last.ckpt")                 # checkpoint path
    torch.save(ckpt, ckpt_path)                                        # serialize to disk
    print("Saved checkpoint ->", ckpt_path)                            # log save path


if __name__ == "__main__":                                             # script entrypoint
    main()                                                             # run training
