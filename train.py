# ============================
# train.py  — DDPM training
# Every line is commented.
# ============================

# ---- standard libs
import os  # filesystem utilities
import json  # to save args as json in the checkpoint
import math  # math helpers (may be used in scheduling)
import glob  # to search for folders when resolving data_root
import random  # for python-level seeding
from dataclasses import asdict  # for neat args serialization (optional)

# ---- numpy / torch
import numpy as np  # numerical utilities
import torch  # PyTorch main package
import torch.nn as nn  # neural network layers
import torch.optim as optim  # optimizers
from torch.utils.data import DataLoader  # batched data iteration
from torchvision import transforms  # image transforms
import torchvision.datasets as tvds  # ImageFolder fallback
from torchvision.utils import save_image  # to save sample grids

# ---- project modules
from diffusion import Diffusion  # DDPM schedules and losses
from unet import UNet  # U-Net denoiser

# ---- typing helpers
from typing import Optional, Tuple  # type hints

# ---- argparse
import argparse  # command-line parsing


# --------------------------
# Utility: set all random seeds for reproducibility
# --------------------------
def set_seed(seed: int) -> None:
    """Set Python/NumPy/Torch seeds for reproducibility."""
    random.seed(seed)  # seed python's RNG
    np.random.seed(seed)  # seed numpy RNG
    torch.manual_seed(seed)  # seed torch CPU RNG
    torch.cuda.manual_seed_all(seed)  # seed all CUDA devices
    torch.backends.cudnn.deterministic = True  # deterministic convs
    torch.backends.cudnn.benchmark = False  # disable autotuner for determinism


# --------------------------
# EMA helper: maintains an exponential moving average of model weights
# --------------------------
class EMA:
    """Exponential Moving Average for model parameters."""

    def __init__(self, model: nn.Module, mu: float = 0.999) -> None:
        self.mu = mu  # decay factor (closer to 1 = smoother)
        self.shadow = {}  # dict param_name -> averaged tensor (on same device)
        # Initialize shadow weights as a copy of model's current state
        for k, v in model.state_dict().items():  # iterate over all parameters/buffers
            self.shadow[k] = v.detach().clone()  # store a clone detached from graph

    @torch.no_grad()  # no gradients for EMA updates
    def update(self, model: nn.Module) -> None:
        """Update EMA weights from the current model parameters."""
        msd = model.state_dict()  # get current model state dict
        for k, v in msd.items():  # iterate parameters/buffers
            if k not in self.shadow:  # if missing (unlikely), initialize
                self.shadow[k] = v.detach().clone()
            else:
                # shadow = mu * shadow + (1 - mu) * new_value
                self.shadow[k].lerp_(v.detach(), 1.0 - self.mu)

    def state_dict(self) -> dict:
        """Return the EMA weights as a state_dict-like dict."""
        return self.shadow  # return the internal dict (already tensors)


# --------------------------
# Resolve data_root to the folder that directly contains "images/"
# --------------------------
def resolve_data_root(root: str) -> str:
    """Return a folder path that directly contains 'images/'."""
    if os.path.isdir(os.path.join(root, "images")):  # if root/images exists
        return root  # already correct
    cand = os.path.join(root, "CUB_200_2011")  # common nesting in Kaggle zips
    if os.path.isdir(os.path.join(cand, "images")):  # if cand/images exists
        return cand  # use nested folder
    # Otherwise, search 2 levels deep for any folder that has images/
    for d in glob.glob(os.path.join(root, "*")) + glob.glob(os.path.join(root, "*", "*")):
        if os.path.isdir(os.path.join(d, "images")):  # found a valid folder
            print(f"[auto] Using data_root -> {d}")  # inform user of auto-selection
            return d  # return discovered folder
    return root  # give back original (may error later, but prints help)


# --------------------------
# Try to build CUB dataset with BBoxes; fallback to ImageFolder if empty/fails
# --------------------------
def build_dataset_with_fallback(
    args: argparse.Namespace,
) -> Tuple[torch.utils.data.Dataset, int]:
    """
    Attempt to build the dataset via make_cub_bbox_dataset (from data.py).
    If it fails or returns 0 samples, fallback to torchvision ImageFolder on images/.
    Returns: (dataset, num_classes)
    """
    ds = None  # dataset placeholder
    num_classes = None  # number of classes placeholder

    # First, try the user's custom CUB loader that supports bounding boxes
    try:
        # Import locally to avoid hard import errors if signature differs
        from data import make_cub_bbox_dataset  # project-provided function

        # Call ONLY with kwargs we know exist (older data.py may differ)
        maybe = make_cub_bbox_dataset(
            root=args.data_root,  # root folder that contains 'images/'
            img_size=args.img_size,  # target square size
            use_bbox=args.use_bbox,  # whether to crop by GT bboxes
            bbox_expand=args.bbox_expand,  # expansion factor around bbox
            class_limit=args.class_limit,  # limit number of species
            subset=args.subset,  # limit number of total images
        )
        # Some versions may return just the dataset; others (dataset, num_classes)
        if isinstance(maybe, tuple) and len(maybe) == 2:  # two-return variant
            ds, num_classes = maybe  # unpack dataset and num_classes
        else:
            ds = maybe  # just dataset; we'll derive num_classes later
        n_items = len(ds)  # attempt to count items (requires __len__)
        print(f"[info] BBOX dataset built: {n_items} samples")  # print stats
    except TypeError as e:
        # Signature mismatch (e.g., unrecognized kw); warn and fallback
        print("[warn] BBOX builder signature mismatch:", e)  # show error
        ds, n_items = None, 0  # mark as unusable
    except Exception as e:
        # Any other runtime failure (e.g., missing files); warn and fallback
        print("[warn] BBOX builder failed:", repr(e))  # show error
        ds, n_items = None, 0  # mark as unusable

    # If failed OR empty, fallback to standard ImageFolder-based dataset
    if ds is None or n_items == 0:  # if no dataset or empty
        print("[fallback] Using ImageFolder (no bbox)")  # inform user
        # Standard diffusion-normalizing transform pipeline
        tfm = transforms.Compose(
            [
                transforms.Resize(args.img_size, interpolation=transforms.InterpolationMode.BICUBIC),  # resize
                transforms.CenterCrop(args.img_size),  # center crop to square
                transforms.ToTensor(),  # to tensor [0,1]
                transforms.Normalize((0.5,) * 3, (0.5,) * 3),  # to [-1,1]
            ]
        )
        img_root = os.path.join(args.data_root, "images")  # ImageFolder root
        ds = tvds.ImageFolder(img_root, transform=tfm)  # create dataset

        # If requested, restrict to the first K class folders (sorted)
        if args.class_limit:  # if user limits classes
            keep_classes = sorted(ds.classes)[: int(args.class_limit)]  # select first K
            keep_idx = [ds.class_to_idx[c] for c in keep_classes]  # map to indices
            # Filter samples that belong to the kept indices
            ds.samples = [s for s in ds.samples if s[1] in keep_idx]  # filter (path, label)
            ds.targets = [s[1] for s in ds.samples]  # update targets vector
            ds.classes = keep_classes  # update class names to the kept ones

        # If requested, limit to the first N images overall
        if args.subset:  # if user limits total images
            ds.samples = ds.samples[: int(args.subset)]  # slice first N images
            ds.targets = [s[1] for s in ds.samples]  # update labels vector

        # ImageFolder expects .imgs attribute as alias (some APIs use it)
        ds.imgs = ds.samples  # align alias
        num_classes = len(ds.classes)  # derive number of classes from ImageFolder
        n_items = len(ds)  # count items
        print(f"[info] ImageFolder dataset built: {n_items} samples, {num_classes} classes")  # print stats

    # If still empty, throw a helpful error to the user
    if n_items == 0:  # nothing to train on
        raise RuntimeError(
            f"Dataset is empty at '{args.data_root}'.\n"
            f"- Ensure this folder contains an 'images/' subfolder.\n"
            f"- If using --use_bbox, verify 'bounding_boxes.txt' exists and matches 'images.txt'.\n"
            f"- Check class_limit={args.class_limit} and subset={args.subset}."
        )

    # If num_classes still unknown (custom builder returned only dataset), deduce it
    if num_classes is None:  # if still None
        # Use the user's intended limit or default to 200 (CUB standard)
        num_classes = int(args.class_limit) if args.class_limit else 200  # reasonable default

    print(f"[info] Final dataset: {len(ds)} samples | num_classes={num_classes}")  # summary line
    return ds, num_classes  # return dataset and class count


# --------------------------
# Save a checkpoint dict to disk
# --------------------------
def save_checkpoint(
    path: str,  # target file path (.ckpt)
    model: nn.Module,  # current model
    ema: Optional[EMA],  # EMA wrapper (may be None)
    step: int,  # current training step
    args: argparse.Namespace,  # parsed CLI args
) -> None:
    """Save training checkpoint to 'path'."""
    os.makedirs(os.path.dirname(path), exist_ok=True)  # ensure folder exists
    ck = {  # build a dict with state
        "step": step,  # save current step
        "model": model.state_dict(),  # model weights
        "ema": (ema.state_dict() if ema is not None else None),  # EMA weights if any
        "args": vars(args),  # serialize args as plain dict (JSON-friendly)
    }
    torch.save(ck, path)  # write to disk
    print(f"[ckpt] Saved -> {path}")  # notify user


# --------------------------
# Optional: quick sampling preview during training (DDIM)
# --------------------------
@torch.no_grad()  # disable gradients for sampling
def preview_samples(
    model: nn.Module,  # trained denoiser
    diffusion: Diffusion,  # diffusion helper
    img_size: int,  # output size (H = W)
    outdir: str,  # folder to save the grid
    step: int,  # step id for filename
    use_ema: bool = False,  # if you pass an EMA-wrapped model outside
    num_classes: Optional[int] = None,  # number of classes for conditional sampling
    guidance_scale: float = 0.0,  # guidance scale (0 to disable)
    cond_mode: str = "none",  # 'none' or 'class'
) -> None:
    """Generate a small grid of samples to monitor training."""
    model.eval()  # set eval mode (disable dropout etc.)
    n = 16  # number of samples in the grid
    device = next(model.parameters()).device  # current device
    # Prepare class labels if conditional sampling is enabled
    y = None  # no labels by default
    if cond_mode == "class" and num_classes is not None:  # if class-conditional
        y = torch.randint(low=0, high=num_classes, size=(n,), device=device)  # random classes
    # Use DDIM sampler for fast and robust preview
    x = diffusion.sample_ddim(  # call DDIM sampler provided by diffusion.py
        model=model,  # denoiser
        shape=(n, 3, img_size, img_size),  # output shape
        steps=min(50, diffusion.T),  # take up to 50 steps for speed
        eta=0.0,  # deterministic DDIM
        y=y,  # labels (or None)
        guidance_scale=guidance_scale,  # CFG strength for sampling
    )
    x = (x.clamp(-1, 1) + 1) / 2  # map from [-1,1] to [0,1] for saving
    os.makedirs(outdir, exist_ok=True)  # ensure folder exists
    grid_path = os.path.join(outdir, f"samples_step_{step:06d}.png")  # file path
    save_image(x, grid_path, nrow=4)  # save 4x4 grid
    print(f"[sample] Wrote {grid_path}")  # notify user


# --------------------------
# Argument parser
# --------------------------
def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="DDPM training on CUB-200-2011")  # CLI help title

    # Data settings
    p.add_argument("--data_root", type=str, required=True, help="Folder that contains 'images/'")  # data root
    p.add_argument("--use_bbox", action="store_true", help="Use GT bounding boxes if available")  # bbox flag
    p.add_argument("--bbox_expand", type=float, default=1.0, help="BBox expansion factor")  # bbox expand
    p.add_argument("--class_limit", type=int, default=None, help="Limit number of classes")  # class limit
    p.add_argument("--subset", type=int, default=None, help="Limit number of images")  # subset limit
    p.add_argument("--img_size", type=int, default=64, help="Square image size")  # target size

    # Training hyperparams
    p.add_argument("--outdir", type=str, required=True, help="Output directory for logs/ckpts")  # outdir
    p.add_argument("--batch_size", type=int, default=16, help="Batch size")  # batch size
    p.add_argument("--epochs", type=int, default=9999, help="Upper bound on epochs (looped via steps)")  # epochs
    p.add_argument("--max_steps", type=int, default=60000, help="Stop after this many steps")  # early stop
    p.add_argument("--lr", type=float, default=2e-4, help="AdamW learning rate")  # learning rate
    p.add_argument("--weight_decay", type=float, default=0.0, help="AdamW weight decay")  # weight decay
    p.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping max-norm")  # grad clip

    # Diffusion process
    p.add_argument("--num_steps", type=int, default=300, help="Number of diffusion time steps T")  # T
    p.add_argument("--schedule", type=str, default="cosine", choices=["cosine", "linear"], help="Beta schedule")  # schedule

    # Conditioning & CFG
    p.add_argument("--cond_mode", type=str, default="none", choices=["none", "class"], help="Conditioning mode")  # cond
    p.add_argument("--p_uncond", type=float, default=0.1, help="Drop-label prob (CFG) during training")  # p_uncond
    p.add_argument("--guidance_scale", type=float, default=0.0, help="Guidance scale for on-the-fly preview")  # scale

    # Model size
    p.add_argument("--base", type=int, default=64, help="UNet base channels")  # base channels

    # EMA
    p.add_argument("--ema_mu", type=float, default=0.999, help="EMA decay")  # EMA decay

    # Logging / checkpointing
    p.add_argument("--log_every", type=int, default=1000, help="Print loss every N steps")  # logging freq
    p.add_argument("--ckpt_every", type=int, default=1000, help="Save checkpoint every N steps")  # ckpt freq

    # Misc
    p.add_argument("--seed", type=int, default=0, help="Random seed")  # seed
    p.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")  # resume path
    p.add_argument("--preview_every", type=int, default=0, help="If >0, run DDIM preview every N steps")  # preview

    return p.parse_args()  # parse and return


# --------------------------
# Main training function
# --------------------------
def main() -> None:
    """Main entry point to train the DDPM."""
    args = get_args()  # parse CLI flags
    set_seed(args.seed)  # set all RNG seeds for reproducibility

    device = "cuda" if torch.cuda.is_available() else "cpu"  # choose device

    args.data_root = resolve_data_root(args.data_root)  # normalize data_root
    print("[info] data_root resolved to:", args.data_root)  # print result

    ds, num_classes = build_dataset_with_fallback(args)  # dataset + class count

    # Build the model; if cond_mode is 'class', pass num_classes else None
    model = UNet(base=args.base, num_classes=(num_classes if args.cond_mode == "class" else None)).to(device)  # UNet
    model.train()  # enable train mode

    # Build diffusion helper (betas schedule, alphas, etc.)
    diffusion = Diffusion(T=args.num_steps, schedule=args.schedule, device=torch.device(device))  # DDPM object

    # Prepare optimizer (AdamW is a good default for diffusion models)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # optimizer

    # AMP autocast and GradScaler (new API removes deprecation warnings)
    autocast = torch.amp.autocast  # alias to torch.amp.autocast context manager
    scaler = torch.amp.GradScaler(device_type="cuda" if device == "cuda" else "cpu")  # AMP scaler

    # Optional EMA over parameters
    ema = EMA(model, mu=args.ema_mu)  # create EMA helper

    # DataLoader for training
    dl = DataLoader(
        ds,  # dataset built above
        batch_size=args.batch_size,  # batch size from args
        shuffle=True,  # shuffle each epoch
        num_workers=2,  # a couple of workers is enough on Colab
        pin_memory=(device == "cuda"),  # pin host memory if GPU
        drop_last=True,  # drop last incomplete batch for stable shapes
    )

    # Optionally resume from a previous checkpoint
    start_step = 0  # initialize step counter
    if args.resume is not None and os.path.isfile(args.resume):  # if valid resume path
        ckpt = torch.load(args.resume, map_location=device)  # load checkpoint dict
        model.load_state_dict(ckpt["model"], strict=False)  # restore model weights
        if ckpt.get("ema") is not None:  # if EMA weights exist
            # re-init EMA with current model then copy its shadow weights
            ema = EMA(model, mu=args.ema_mu)  # re-create EMA wrapper
            for k in ema.shadow:  # iterate EMA dict keys
                if k in ckpt["ema"]:  # only copy existing keys
                    ema.shadow[k].copy_(ckpt["ema"][k])  # copy tensor
        start_step = int(ckpt.get("step", 0))  # resume step counter
        print(f"[resume] Loaded {args.resume} at step {start_step}")  # log resume

    # Create outdir
    os.makedirs(args.outdir, exist_ok=True)  # ensure output directory exists

    # Training loop by steps (epochs just bound the outer loop)
    step = start_step  # current global step
    # Outer loop (epochs) just cycles over the dataloader
    for ep in range(args.epochs):  # potentially large number (we stop by max_steps)
        # Iterate over the dataloader
        for batch in dl:  # each batch from dataset
            if step >= args.max_steps:  # stop condition
                print("Reached max_steps, stopping training.")  # print and stop
                break  # break inner loop

            # Unpack batch: ImageFolder returns (img, label)
            if isinstance(batch, (tuple, list)) and len(batch) >= 1:  # tuple/list check
                x = batch[0].to(device, non_blocking=True)  # images tensor to device
                y = batch[1].to(device, non_blocking=True) if (len(batch) > 1) else None  # labels if present
            else:
                # If dataset returns only images, set y=None
                x = batch.to(device, non_blocking=True)  # images to device
                y = None  # no labels

            # If training unconditioned, ignore labels entirely
            if args.cond_mode != "class":  # not class-conditional
                y_in = None  # pass None to diffusion loss
            else:
                y_in = y  # class labels used for conditional training

            # Mixed precision forward + loss
            with autocast(device_type=("cuda" if device == "cuda" else "cpu")):  # AMP autocast
                # Compute epsilon-prediction MSE loss at random timesteps
                loss = diffusion.p_losses(  # DDPM loss function
                    model=model,  # denoiser model
                    x_start=x,  # clean images
                    y=y_in,  # labels or None
                    p_uncond=args.p_uncond,  # drop-label probability for CFG
                )

            # Backward pass with AMP scaling
            scaler.scale(loss).backward()  # compute scaled gradients

            # Gradient clipping (important for stability)
            if args.grad_clip is not None and args.grad_clip > 0:  # if enabled
                scaler.unscale_(opt)  # unscale before clipping
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)  # clip

            # Optimizer step with scaler
            scaler.step(opt)  # apply optimizer step
            scaler.update()  # update scaler state
            opt.zero_grad(set_to_none=True)  # reset gradients (set_to_none saves memory)

            # EMA update
            ema.update(model)  # update moving average of weights

            # Step counter increments
            step += 1  # increment step

            # Logging every N steps
            if (step % args.log_every) == 0:  # check frequency
                print(f"step {step} | loss {loss.item():.4f} | lr {args.lr:.2e}")  # print metrics

            # Optional sampling preview every N steps
            if args.preview_every and (step % args.preview_every) == 0:  # if enabled
                # Temporarily swap to EMA weights for sampling preview
                sd_backup = model.state_dict()  # backup current weights
                model.load_state_dict(ema.state_dict(), strict=False)  # load EMA weights
                try:
                    preview_samples(  # run quick DDIM sampling
                        model=model,  # denoiser
                        diffusion=diffusion,  # helper
                        img_size=args.img_size,  # output size
                        outdir=os.path.join(args.outdir, "previews"),  # preview folder
                        step=step,  # filename id
                        use_ema=True,  # we used EMA weights
                        num_classes=(num_classes if args.cond_mode == "class" else None),  # cond
                        guidance_scale=args.guidance_scale,  # CFG scale
                        cond_mode=args.cond_mode,  # cond mode
                    )
                finally:
                    model.load_state_dict(sd_backup, strict=False)  # restore training weights

            # Periodic checkpoint save
            if (step % args.ckpt_every) == 0:  # on frequency
                # Save both rolling 'last.ckpt' and step-tagged ckpt
                save_checkpoint(  # write step-tagged ckpt
                    path=os.path.join(args.outdir, f"ckpt_{step:06d}.ckpt"),  # path with step
                    model=model,  # current model
                    ema=ema,  # EMA wrapper
                    step=step,  # current step
                    args=args,  # args dict
                )
                save_checkpoint(  # write/overwrite last.ckpt
                    path=os.path.join(args.outdir, "last.ckpt"),  # rolling last
                    model=model,  # model
                    ema=ema,  # EMA
                    step=step,  # step
                    args=args,  # args
                )

        # If we hit max_steps in the inner loop, exit the outer loop as well
        if step >= args.max_steps:  # double-check after epoch boundary
            break  # stop training

    # Final checkpoint save when training finishes (ensure we have last state)
    save_checkpoint(  # save final 'last.ckpt'
        path=os.path.join(args.outdir, "last.ckpt"),  # final last file
        model=model,  # model
        ema=ema,  # EMA
        step=step,  # final step
        args=args,  # args
    )
    print("[done] Training finished.")  # training over


# --------------------------
# Entry point
# --------------------------
if __name__ == "__main__":  # python main guard
    main()  # launch training


