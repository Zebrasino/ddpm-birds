import os                                # filesystem utilities
import math                              # math helpers (ceil, etc.)
import json                              # to store args alongside checkpoints
import random                            # Python RNG
from argparse import ArgumentParser      # CLI parsing

import torch                             # core PyTorch
import torch.nn as nn                    # neural network layers
import torch.optim as optim              # optimizers
from torch.utils.data import DataLoader  # data loader
from torchvision.utils import save_image # to save sample grids

# Local modules (assumed present in your repo)
from data import make_cub_bbox_dataset   # dataset factory (images, labels[, mask])
from diffusion import Diffusion          # DDPM core (q_sample, p_losses, sample_ddim)
from unet import UNet                    # U-Net backbone (conditional or not)

# -----------------------------------------------------------------------------
# Small EMA helper (kept here so train.py is standalone)
# -----------------------------------------------------------------------------
class EMA:
    """Exponential Moving Average wrapper for a model."""
    def __init__(self, model: nn.Module, mu: float = 0.9995):
        # Create a shadow copy of the model (same device/dtype)
        self.ema_model = UNet(base=model.base, num_classes=model.num_classes).to(next(model.parameters()).device)
        self.ema_model.load_state_dict(model.state_dict(), strict=True)  # start equal to online weights
        self.mu = mu                                                    # decay factor

        # EMA model is only used for evaluation/sampling -> disable gradients
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, online: nn.Module):
        """EMA <- mu * EMA + (1-mu) * online"""
        for p_ema, p in zip(self.ema_model.parameters(), online.parameters()):
            p_ema.data.mul_(self.mu).add_(p.data, alpha=(1.0 - self.mu))

    def state_dict(self):
        """Return the EMA model weights (for saving)."""
        return self.ema_model.state_dict()

    def load_state_dict(self, sd):
        """Load weights into the EMA model (for resuming)."""
        self.ema_model.load_state_dict(sd, strict=True)


# -----------------------------------------------------------------------------
# Utility functions: seeding, checkpoint IO, tiny debug sampler
# -----------------------------------------------------------------------------
def seed_everything(seed: int):
    """Make results more repeatable."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN deterministic (slower but predictable). Comment if you prefer speed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(path, model, ema, optimizer, args, step: int):
    """Save a training checkpoint."""
    pkg = {
        "model": model.state_dict(),                     # online model weights
        "ema": (ema.state_dict() if ema is not None else None),  # EMA weights (or None)
        "optimizer": optimizer.state_dict(),             # optimizer state
        "args": vars(args),                              # full CLI args as dict
        "step": step,                                    # global step for resume
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)    # ensure folder exists
    torch.save(pkg, path)                                # write .ckpt file


def load_checkpoint(path, model, ema, optimizer, device):
    """Load a training checkpoint. Returns restored global step."""
    ckpt = torch.load(path, map_location=device)         # load to the right device

    # Restore model weights
    model.load_state_dict(ckpt["model"], strict=True)

    # Restore EMA weights (if present and EMA is enabled)
    if ema is not None and ckpt.get("ema", None) is not None:
        ema.load_state_dict(ckpt["ema"])

    # Restore optimizer (if provided)
    if optimizer is not None and ckpt.get("optimizer", None) is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    # Return global step (default 0 if missing)
    return int(ckpt.get("step", 0))


@torch.no_grad()
def tiny_debug_sample(diffusion: Diffusion,
                      model_for_eval: nn.Module,
                      outdir: str,
                      img_size: int,
                      cond_mode: str,
                      num_classes: int | None,
                      step_tag: str,
                      device: torch.device):
    """
    Generate a tiny DDIM sample grid during training to track progress.
    Very cheap: 8 images, 25 steps DDIM, guidance_scale=3.5 by default.
    """
    # Number of images in the grid
    n = 8

    # Build a random label batch if class-conditional
    if cond_mode == "class" and num_classes is not None:
        y = torch.randint(0, num_classes, (n,), device=device)
        guidance_scale = 3.5
    else:
        y = None
        guidance_scale = 0.0

    # Shape: (N, C, H, W)
    shape = (n, 3, img_size, img_size)

    # Generate samples with a small DDIM schedule
    xg = diffusion.sample_ddim(
        model_for_eval, shape,
        steps=25,            # few steps -> fast
        eta=0.0,             # deterministic DDIM
        skip_first=10,       # skip the very-noisy first part
        y=y,                 # labels or None
        guidance_scale=guidance_scale
    )

    # Map from [-1, 1] to [0, 1]
    xg = (xg.clamp(-1, 1) + 1) / 2

    # Save grid
    os.makedirs(outdir, exist_ok=True)
    save_image(xg, os.path.join(outdir, f"dbg_{step_tag}.png"), nrow=4)


# -----------------------------------------------------------------------------
# Argument parser
# -----------------------------------------------------------------------------
def build_parser():
    p = ArgumentParser(description="Train a DDPM on CUB-200-2011")

    # Data & IO
    p.add_argument("--data_root", type=str, required=True, help="Root of CUB_200_2011")
    p.add_argument("--outdir", type=str, required=True, help="Where to write logs/checkpoints/samples")
    p.add_argument("--img_size", type=int, default=64, help="Training resolution")
    p.add_argument("--batch_size", type=int, default=16, help="Batch size per step")
    p.add_argument("--epochs", type=int, default=9999, help="Max epochs (unused when max_steps reached)")
    p.add_argument("--max_steps", type=int, default=60000, help="Stop after this many optimizer steps")
    p.add_argument("--log_every", type=int, default=500, help="Print loss every N steps")
    p.add_argument("--ckpt_every", type=int, default=0, help="Save checkpoint every N steps (0=off)")
    p.add_argument("--sample_every", type=int, default=0, help="Save tiny DDIM samples every N steps (0=off)")
    p.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from (if empty, try last.ckpt)")

    # Dataset options
    p.add_argument("--use_bbox", action="store_true", help="Crop images by bounding boxes")
    p.add_argument("--bbox_expand", type=float, default=1.1, help="BBox expansion factor")
    p.add_argument("--subset", type=int, default=0, help="Limit images per class (0 = use all)")
    p.add_argument("--class_limit", type=int, default=200, help="Limit number of classes (<= 200)")
    p.add_argument("--aug_color", action="store_true", help="Enable light color jitter aug")

    # Model & diffusion
    p.add_argument("--base", type=int, default=64, help="UNet base channel multiplier (64/96)")
    p.add_argument("--cond_mode", type=str, choices=["none", "class"], default="class", help="Conditioning type")
    p.add_argument("--num_steps", type=int, default=300, help="Diffusion training steps T")
    p.add_argument("--schedule", type=str, choices=["cosine", "linear"], default="cosine", help="Beta schedule")
    p.add_argument("--fg_weight", type=float, default=2.0, help="Foreground pixel weight (if mask is available)")

    # Optim / EMA / CFG
    p.add_argument("--lr", type=float, default=2e-4, help="Adam learning rate")
    p.add_argument("--weight_decay", type=float, default=0.0, help="AdamW weight decay")
    p.add_argument("--ema_mu", type=float, default=0.9995, help="EMA decay")
    p.add_argument("--p_uncond", type=float, default=0.1, help="Classifier-free guidance unconditional prob (training)")

    # System
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")

    return p


# -----------------------------------------------------------------------------
# Main training function
# -----------------------------------------------------------------------------
def main():
    # Parse CLI args
    args = build_parser().parse_args()

    # Fix randomness for reproducibility
    seed_everything(args.seed)

    # Choose device (prefer CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output folder
    os.makedirs(args.outdir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Build dataset & loader
    # - make_cub_bbox_dataset should return (dataset, num_classes)
    # - Each sample ideally yields: (image_tensor[-1..1], class_idx, fg_mask[0/1])
    #   If fg_mask not available, we’ll handle it below.
    # -------------------------------------------------------------------------
    # Build dataset (only with supported arguments)
    ds, num_classes = make_cub_bbox_dataset(
        root=args.data_root,
        img_size=args.img_size,
        use_bbox=args.use_bbox,
        bbox_expand=args.bbox_expand,   # works, present in your data.py
        class_limit=args.class_limit,
        subset=args.subset,
    )



    # Tell the model how many classes (None if unconditional)
    model_num_classes = (num_classes if args.cond_mode == "class" else None)

    # DataLoader (pin memory on CUDA; persistent workers speed things up)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0)
    )

    # -------------------------------------------------------------------------
    # Build model + diffusion + EMA
    # -------------------------------------------------------------------------
    model = UNet(base=args.base, num_classes=model_num_classes).to(device)
    diffusion = Diffusion(T=args.num_steps, schedule=args.schedule, device=device)

    # Optimizer (AdamW; weight_decay defaults to 0.0)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # EMA wrapper
    ema = EMA(model, mu=args.ema_mu)

    # Mixed precision scaler (new API)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # -------------------------------------------------------------------------
    # Optional resume: load from args.resume or from outdir/last.ckpt if exists
    # -------------------------------------------------------------------------
    global_step = 0
    resume_path = args.resume if args.resume else os.path.join(args.outdir, "last.ckpt")
    if os.path.isfile(resume_path):
        print(f"[resume] Loading checkpoint: {resume_path}")
        global_step = load_checkpoint(resume_path, model, ema, opt, device)
        print(f"[resume] Restored global_step={global_step}")

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    model.train()                                # set train mode
    # Pre-compute label count for sampling routine
    cond_is_class = (args.cond_mode == "class" and model_num_classes is not None)

    for epoch in range(args.epochs):             # epoch counter (mostly cosmetic)
        for batch in dl:                         # iterate mini-batches
            # Stop if we hit max_steps (hard limit)
            if global_step >= args.max_steps:
                break

            # Unpack batch flexibly:
            # - (x, y, mask) or (x, y) or (x,) depending on dataset implementation
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    x, y, fg_mask = batch
                elif len(batch) == 2:
                    x, y = batch
                    fg_mask = None
                else:
                    # Only images
                    x = batch[0]
                    y, fg_mask = None, None
            else:
                x, y, fg_mask = batch, None, None

            # Move tensors to GPU/CPU as appropriate
            x = x.to(device, non_blocking=True)
            y = (y.to(device, non_blocking=True) if (y is not None) else None)
            fg_mask = (fg_mask.to(device, non_blocking=True) if (fg_mask is not None) else None)

            # Classifier-free guidance during training:
            # with probability p_uncond, drop the label (set to -1) so the model learns the null condition.
            if cond_is_class and (y is not None):
                drop = (torch.rand_like(y.float()) < args.p_uncond)
                y = torch.where(drop, torch.full_like(y, -1), y)

            # Zero optimizer gradients
            opt.zero_grad(set_to_none=True)

            # Forward pass inside autocast for speed/VRAM (CUDA only)
            if device.type == "cuda":
                autocast_ctx = torch.amp.autocast("cuda")
            else:
                # No autocast on CPU
                class _NullCtx:
                    def __enter__(self): pass
                    def __exit__(self, *exc): pass
                autocast_ctx = _NullCtx()

            with autocast_ctx:
                # Compute DDPM loss; pass foreground mask/weight if available
                loss = diffusion.p_losses(
                    model,
                    x,
                    y_in=y,                         # class labels or None
                    fg_mask=fg_mask,                # optional [B,1,H,W] mask
                    fg_weight=args.fg_weight        # scalar weight for foreground pixels
                )

            # Backprop with AMP or full precision
            if scaler is not None:
                scaler.scale(loss).backward()
                # Clip to avoid exploding gradients on small batches
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

            # EMA update (after optimizer)
            ema.update(model)

            # Increase global step
            global_step += 1

            # Logging
            if (global_step % args.log_every) == 0 or global_step == 1:
                # Compute a cosine LR “by hand” from optimizer if you use a scheduler elsewhere
                # Here we just print the current optimizer LR (single param group assumed)
                lr_now = opt.param_groups[0]["lr"]
                print(f"step {global_step} | loss {loss.item():.4f} | lr {lr_now:.2e}")

            # Frequent checkpointing (optional)
            if args.ckpt_every and (global_step % args.ckpt_every == 0):
                ck_path = os.path.join(args.outdir, f"step_{global_step:06d}.ckpt")
                save_checkpoint(ck_path, model, ema, opt, args, step=global_step)
                print(f"[ckpt] Saved -> {ck_path}")

            # Tiny debug sampling with DDIM (optional, very cheap)
            if args.sample_every and (global_step % args.sample_every == 0):
                model.eval()
                tiny_debug_sample(
                    diffusion=diffusion,
                    model_for_eval=ema.ema_model,    # sample from EMA weights
                    outdir=args.outdir,
                    img_size=args.img_size,
                    cond_mode=args.cond_mode,
                    num_classes=model_num_classes,
                    step_tag=f"{global_step:06d}",
                    device=device
                )
                model.train()

            # Always write/update "last.ckpt" so Colab preemptions hurt less
            if (global_step % max(2500, args.log_every)) == 0:
                last = os.path.join(args.outdir, "last.ckpt")
                save_checkpoint(last, model, ema, opt, args, step=global_step)

        # Hard exit when max_steps reached (break out of epoch loop too)
        if global_step >= args.max_steps:
            break

    # Final save at the end of training
    last = os.path.join(args.outdir, "last.ckpt")
    save_checkpoint(last, model, ema, opt, args, step=global_step)
    print(f"[done] Reached max_steps or epochs. Saved final -> {last}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()


