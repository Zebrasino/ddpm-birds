# train.py — fully commented, line-by-line, with foreground-weighted loss support

import os  # filesystem utilities
import math  # for math helpers
import argparse  # CLI argument parsing
import random  # RNG for Python
from pathlib import Path  # convenient path handling
import time  # measure elapsed time

import torch  # PyTorch main package
import torch.nn as nn  # neural network building blocks
import torch.optim as optim  # optimizers
from torch.utils.data import DataLoader  # data loader
from torch.cuda.amp import GradScaler, autocast  # mixed-precision tools

from data import make_cub_bbox_dataset  # your dataset factory
from unet import UNet  # the model (U-Net)
from diffusion import Diffusion  # diffusion schedule + losses + samplers
from utils import EMA  # exponential moving average helper (your utility)

# ---------------------------
# Small helpers
# ---------------------------
def set_seed(seed: int):
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)  # Python RNG
    torch.manual_seed(seed)  # CPU RNG
    torch.cuda.manual_seed_all(seed)  # GPU RNG
    torch.backends.cudnn.deterministic = True  # deterministic convolution
    torch.backends.cudnn.benchmark = False  # disable shape auto-tuning for determinism

def center_gaussian_mask(B, H, W, device, sigma=0.5):
    """
    Create a centered 2D Gaussian mask in [0,1] of shape (B,1,H,W).
    sigma is expressed as a fraction of half-size; smaller sigma = sharper peak.
    """
    # Coordinate grid normalized to [-1, 1]
    ys = torch.linspace(-1, 1, steps=H, device=device)  # y coordinates
    xs = torch.linspace(-1, 1, steps=W, device=device)  # x coordinates
    Y, X = torch.meshgrid(ys, xs, indexing="ij")  # 2D grid
    # Gaussian radial profile
    R2 = X**2 + Y**2  # squared radius
    # Convert sigma (fraction) to variance in this normalized grid
    var = (sigma**2)  # variance
    mask2d = torch.exp(-0.5 * R2 / (var + 1e-8))  # Gaussian function
    mask2d = (mask2d - mask2d.min()) / (mask2d.max() - mask2d.min() + 1e-8)  # normalize to [0,1]
    return mask2d.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)  # (B,1,H,W)

# ---------------------------
# Arg parser
# ---------------------------
def build_argparser():
    """Create an argument parser with all training knobs."""
    p = argparse.ArgumentParser(description="Train DDPM on CUB with optional class conditioning.")

    # Data / I/O
    p.add_argument("--data_root", type=str, required=True, help="Path to CUB_200_2011 root folder")
    p.add_argument("--use_bbox", action="store_true", help="Use bounding boxes (crop around bird)")
    p.add_argument("--bbox_expand", type=float, default=1.0, help="BBox expansion factor (>=1.0)")
    p.add_argument("--class_limit", type=int, default=None, help="Limit number of classes (e.g., 200)")
    p.add_argument("--subset", type=int, default=None, help="Limit number of images per class")
    p.add_argument("--img_size", type=int, default=64, help="Training image size")

    p.add_argument("--outdir", type=str, required=True, help="Output directory for checkpoints and logs")

    # Optimization
    p.add_argument("--batch_size", type=int, default=16, help="Batch size")
    p.add_argument("--epochs", type=int, default=9999, help="Max epochs (used if max_steps not reached)")
    p.add_argument("--max_steps", type=int, default=None, help="Stop after this many optimization steps")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    p.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for AdamW")
    p.add_argument("--grad_clip", type=float, default=1.0, help="Gradient norm clipping threshold")

    # Diffusion schedule
    p.add_argument("--num_steps", type=int, default=300, help="Number of diffusion steps T")
    p.add_argument("--schedule", type=str, choices=["cosine", "linear"], default="cosine",
                   help="Beta schedule type")

    # Conditioning
    p.add_argument("--cond_mode", type=str, choices=["none", "class"], default="none",
                   help="Conditioning mode (none or class)")
    p.add_argument("--p_uncond", type=float, default=0.1,
                   help="Classifier-free guidance training prob: drop condition with this prob")
    p.add_argument("--guidance_scale", type=float, default=0.0,
                   help="CFG guidance scale used only for in-training previews/sampling")

    # Model
    p.add_argument("--base", type=int, default=64, help="UNet base channel multiplier")

    # EMA
    p.add_argument("--ema_mu", type=float, default=0.999, help="EMA decay for model parameters")

    # Logging / checkpoints
    p.add_argument("--log_every", type=int, default=500, help="Log every N steps")
    p.add_argument("--ckpt_every", type=int, default=2000, help="Save checkpoint every N steps")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from")
    p.add_argument("--preview_every", type=int, default=0,
                   help="If >0, run a quick sample preview every N steps")

    # Foreground weighting (this is the reintroduced flag)
    p.add_argument("--fg_weight", type=float, default=1.0,
                   help="Extra weight for foreground pixels in the loss (>=1).")
    p.add_argument("--fg_sigma", type=float, default=0.5,
                   help="Sigma for center-gaussian mask if no mask is provided by the dataset.")

    return p

# ---------------------------
# Main
# ---------------------------
def main():
    """Main training routine."""
    # Parse arguments
    args = build_argparser().parse_args()  # parse all CLI arguments

    # Prepare output directory
    outdir = Path(args.outdir)  # turn into Path object
    outdir.mkdir(parents=True, exist_ok=True)  # create folder if not exists

    # Set random seeds for reproducibility
    set_seed(args.seed)  # sync seeds

    # Pick device
    device = "cuda" if torch.cuda.is_available() else "cpu"  # choose GPU if available

    # Build dataset (returns a PyTorch Dataset)
    # NOTE: make_cub_bbox_dataset returns a Dataset (not (ds, num_classes))
    ds = make_cub_bbox_dataset(
        root=args.data_root,          # CUB root path
        img_size=args.img_size,       # training resolution
        use_bbox=args.use_bbox,       # crop around bbox if requested
        bbox_expand=args.bbox_expand, # expand factor
        class_limit=args.class_limit, # limit number of classes
        subset=args.subset            # limit images per class
    )

    # Infer number of classes if we are in class-conditional mode
    if args.cond_mode == "class":
        # Assumes the dataset exposes an attribute 'num_classes'
        assert hasattr(ds, "num_classes"), "Dataset must have 'num_classes' attribute for class conditioning."
        num_classes = ds.num_classes  # number of labels in the dataset
    else:
        num_classes = None  # unconditioned model

    # DataLoader for training
    dl = DataLoader(
        ds,                      # dataset
        batch_size=args.batch_size,  # batch size
        shuffle=True,                # shuffle data
        num_workers=2,               # small number of workers (Colab-friendly)
        pin_memory=True,             # faster host->device transfers
        drop_last=True               # enforce full batches
    )

    # Create model
    model = UNet(base=args.base, num_classes=num_classes).to(device)  # instantiate UNet

    # Exponential Moving Average wrapper
    ema = EMA(model, mu=args.ema_mu)  # create EMA tracker

    # Diffusion driver (schedule + forward training loss + samplers)
    diffusion = Diffusion(
        T=args.num_steps,            # number of diffusion steps
        schedule=args.schedule,      # cosine or linear
        device=torch.device(device)  # target device
    )

    # Optimizer: AdamW
    optim_params = [p for p in model.parameters() if p.requires_grad]  # trainable params
    opt = optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)  # optimizer

    # AMP GradScaler for mixed precision stability
    scaler = GradScaler()  # create scaler

    # Optionally resume from checkpoint
    global_step = 0  # count optimizer steps
    start_epoch = 0  # starting epoch index
    if args.resume is not None and os.path.isfile(args.resume):  # if a resume path exists
        ckpt = torch.load(args.resume, map_location=device)  # load checkpoint
        model.load_state_dict(ckpt["model"], strict=False)  # restore model weights (strict=False for safety)
        if "ema" in ckpt:  # if EMA states are present
            ema.shadow.load_state_dict(ckpt["ema"])  # restore EMA weights
        if "opt" in ckpt:  # if optimizer state present
            opt.load_state_dict(ckpt["opt"])  # restore optimizer
        global_step = ckpt.get("global_step", 0)  # restore step count
        start_epoch = ckpt.get("epoch", 0)  # restore epoch count
        print(f"[resume] Loaded checkpoint from {args.resume} (step={global_step}, epoch={start_epoch})")  # log resume

    # Training loop
    t0 = time.time()  # start timer
    for epoch in range(start_epoch, args.epochs):  # iterate over epochs
        for batch in dl:  # iterate over batches
            # Unpack batch; support (x,y) or (x,y,mask)
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                x, y, mask = batch  # dataset provided a mask tensor
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, y = batch  # only image and label
                mask = None   # no mask provided
            else:
                # If dataset returns only images (unconditional)
                x = batch  # image tensor
                y = None   # no labels
                mask = None  # no mask

            x = x.to(device, non_blocking=True)  # move images to device
            if y is not None:
                y = y.to(device, non_blocking=True)  # move labels to device

            # Build a center prior mask if no mask is present and fg_weight > 1
            if mask is None and args.fg_weight > 1.0:
                B, C, H, W = x.shape  # extract shapes
                mask = center_gaussian_mask(B, H, W, device, sigma=args.fg_sigma)  # build center mask
            elif mask is not None:
                # Make sure mask has shape (B,1,H,W) and is float in [0,1]
                if mask.dim() == 3:  # (B,H,W)
                    mask = mask.unsqueeze(1)  # add channel dim
                mask = mask.to(device, dtype=x.dtype)  # type & device
                mask = mask.clamp(0, 1)  # clamp to [0,1]

            # Forward + loss under AMP autocast
            opt.zero_grad(set_to_none=True)  # clear gradients
            with autocast():  # enable mixed precision on CUDA automatically
                # Compute DDPM loss; pass optional foreground mask + weight
                loss = diffusion.p_losses(
                    model,              # UNet
                    x0=x,               # clean image
                    y=y,                # labels (or None)
                    p_uncond=args.p_uncond if args.cond_mode == "class" else 0.0,  # CFG drop prob
                    mask=mask,          # optional foreground mask
                    fg_weight=args.fg_weight  # extra weight for mask==1 pixels
                )  # returns scalar loss

            # Backprop with AMP scaling
            scaler.scale(loss).backward()  # backward pass (scaled)
            if args.grad_clip is not None and args.grad_clip > 0:  # if gradient clipping enabled
                scaler.unscale_(opt)  # unscale grads for clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # clip by norm
            scaler.step(opt)  # optimizer step (scaled)
            scaler.update()  # update scaler

            # Update EMA after optimizer step
            ema.update(model)  # blend current weights into EMA shadow

            # Increase global step
            global_step += 1  # increment

            # Logging
            if (global_step % args.log_every) == 0:  # periodic logging
                elapsed = time.time() - t0  # seconds elapsed
                print(f"step {global_step} | loss {loss.item():.4f} | lr {opt.param_groups[0]['lr']:.2e} | {elapsed/60:.1f} min")  # log line

            # Optional preview sampling during training (lightweight DDIM)
            if args.preview_every > 0 and (global_step % args.preview_every) == 0:  # if preview is requested
                model_eval = ema.shadow  # use EMA for nicer samples
                model_eval.eval()  # eval mode
                with torch.no_grad():  # no grads
                    # quick sample (deterministic DDIM, few steps)
                    x_gen = diffusion.sample_ddim(
                        model_eval,                    # model for sampling
                        shape=(16, 3, args.img_size, args.img_size),  # sample grid
                        steps=min(50, args.num_steps), # number of DDIM steps
                        eta=0.0,                       # deterministic
                        y=None,                        # unconditional preview
                        guidance_scale=0.0             # no CFG here
                    )  # returns (B,3,H,W) in [-1,1]
                    # save preview grid
                    from torchvision.utils import save_image  # lazy import
                    grid = (x_gen.clamp(-1, 1) + 1) / 2.0  # to [0,1]
                    save_image(grid, str(outdir / f"preview_step_{global_step:06d}.png"), nrow=4)  # write file
                model_eval.train()  # back to train (for safety)

            # Periodic checkpointing
            if (global_step % args.ckpt_every) == 0:  # every N steps
                ck = {
                    "args": vars(args),        # store args
                    "model": model.state_dict(),  # model weights
                    "ema": ema.shadow.state_dict(),  # EMA weights
                    "opt": opt.state_dict(),   # optimizer state
                    "global_step": global_step,  # step count
                    "epoch": epoch             # epoch count
                }  # checkpoint dict
                torch.save(ck, str(outdir / "last.ckpt"))  # save file
                print(f"Saved checkpoint -> {outdir / 'last.ckpt'}")  # log path

            # Early stop on max_steps
            if args.max_steps is not None and global_step >= args.max_steps:  # reached target steps
                print("Reached max_steps, stopping training.")  # message
                # Save a final checkpoint at exit
                ck = {
                    "args": vars(args),
                    "model": model.state_dict(),
                    "ema": ema.shadow.state_dict(),
                    "opt": opt.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch
                }  # final checkpoint
                torch.save(ck, str(outdir / "last.ckpt"))  # write file
                return  # end training early

        # End of epoch: you could place epoch-based logic here (optional)

    # Finished all epochs: save checkpoint
    ck = {
        "args": vars(args),
        "model": model.state_dict(),
        "ema": ema.shadow.state_dict(),
        "opt": opt.state_dict(),
        "global_step": global_step,
        "epoch": args.epochs
    }  # final checkpoint (epochs completed)
    torch.save(ck, str(outdir / "last.ckpt"))  # write file
    print(f"Training completed. Saved checkpoint -> {outdir / 'last.ckpt'}")  # log final save

# Entry point guard
if __name__ == "__main__":  # if script is executed directly
    main()  # run main


