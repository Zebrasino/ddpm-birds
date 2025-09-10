import argparse  # command-line interface parsing
import os        # filesystem utilities
import math      # math helpers
import torch     # PyTorch core
from torch import optim  # optimizers (AdamW)
from torch.utils.data import DataLoader  # mini-batch loader
from torchvision import transforms       # image transforms

from diffusion import Diffusion          # DDPM process (forward/loss/sampler)
from unet import UNet                    # U-Net backbone (epsilon-predictor)
from utils import set_seed, save_grid, EMAHelper  # misc utils (seed/grid/EMA)
from data import make_cub_bbox_dataset, make_imagefolder_dataset  # datasets


def cosine_with_warmup(optimizer, total_steps: int, warmup_ratio: float = 0.03, min_lr_ratio: float = 0.1):
    """Cosine LR schedule with linear warmup; returns a LambdaLR scheduler."""
    warmup = max(1, int(total_steps * warmup_ratio))            # number of warmup steps
    def lr_lambda(step):                                        # step → scale factor
        if step < warmup:                                       # warmup region
            return (step + 1) / warmup                          # linear 0→1
        prog = (step - warmup) / max(1, total_steps - warmup)   # normalized progress in [0,1]
        cos = 0.5 * (1 + math.cos(math.pi * prog))              # cosine from 1→0
        return min_lr_ratio + (1 - min_lr_ratio) * cos          # clamp to [min_lr_ratio, 1]
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)  # hook into optimizer


def parse_args():
    """Define and parse all CLI arguments."""
    p = argparse.ArgumentParser()
    # data
    p.add_argument('--data_root', type=str, required=True)         # path to CUB root (contains images/, txt files) or to ImageFolder
    p.add_argument('--use_bbox', action='store_true')              # crop with CUB ground-truth bounding boxes
    p.add_argument('--bbox_expand', type=float, default=1.0)       # expansion factor around bbox (1.0 => exact box, no margin)
    p.add_argument('--subset', type=int, default=None)             # limit number of images (quick overfit)
    p.add_argument('--class_limit', type=int, default=None)        # limit number of classes
    # training
    p.add_argument('--outdir', type=str, required=True)            # output directory (checkpoints/samples)
    p.add_argument('--img_size', type=int, default=64)             # resolution used by the model
    p.add_argument('--batch_size', type=int, default=16)           # mini-batch size
    p.add_argument('--epochs', type=int, default=40)               # max epochs (also bounded by max_steps)
    p.add_argument('--lr', type=float, default=2e-4)               # learning rate
    p.add_argument('--weight_decay', type=float, default=0.0)      # weight decay (often 0 for diffusion)
    p.add_argument('--num_steps', type=int, default=400)           # diffusion chain length T (train+sample)
    p.add_argument('--schedule', type=str, default='cosine', choices=['cosine','linear'])  # beta schedule
    p.add_argument('--cond_mode', type=str, choices=['none','class'], default='class')     # conditioning type
    p.add_argument('--p_uncond', type=float, default=0.1)          # classifier-free dropout prob during training
    p.add_argument('--guidance_scale', type=float, default=1.0)    # CFG scale used for end-of-epoch preview
    p.add_argument('--base', type=int, default=64)                 # U-Net base channel width
    p.add_argument('--ema_mu', type=float, default=0.9995)         # EMA decay for target weights
    p.add_argument('--max_steps', type=int, default=None)          # stop after N optimizer steps (global)
    p.add_argument('--seed', type=int, default=42)                 # RNG seed for reproducibility
    p.add_argument('--resume', type=str, default=None)             # checkpoint path to resume from
    p.add_argument('--log_every', type=int, default=500)           # print loss/LR every N steps
    return p.parse_args()


def main():
    """Training entry point."""
    args = parse_args()                                           # parse CLI
    os.makedirs(args.outdir, exist_ok=True)                       # ensure output dir exists
    set_seed(args.seed)                                           # set all seeds (python/numpy/torch)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # choose device

    # --- transforms: DETERMINISTIC for reliable overfit/sanity ---
    tfm = transforms.Compose([
        transforms.Resize(args.img_size, interpolation=transforms.InterpolationMode.BICUBIC),  # resize (keeps aspect → then center-crop)
        transforms.CenterCrop(args.img_size),                                                  # center-crop to square
        transforms.ToTensor(),                                                                 # HWC uint8 → CHW float in [0,1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),                                # map to [-1,1]
    ])

    # --- dataset: CUB with bbox (no margin) OR plain ImageFolder ---
    if args.use_bbox:  # use ground-truth bird boxes (no extra expansion by default)
        ds, num_classes = make_cub_bbox_dataset(
            args.data_root, tfm, expand=args.bbox_expand, subset=args.subset, class_limit=args.class_limit
        )
    else:              # or a simple folder of class subfolders
        ds, num_classes = make_imagefolder_dataset(
            args.data_root, tfm, subset=args.subset, class_limit=args.class_limit
        )

    # standard data loader (pin memory speeds host→GPU copies; num_workers can be tuned)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # --- build model and diffusion process ---
    model = UNet(base=args.base, num_classes=(num_classes if args.cond_mode == 'class' else None)).to(device)  # U-Net
    diffusion = Diffusion(T=args.num_steps, schedule=args.schedule, device=device)                              # DDPM

    # optimizer, AMP scaler, EMA, and LR scheduler
    optim_ = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)   # AdamW
    scaler = torch.amp.GradScaler(device.type)                                                                  # AMP scaler ('cuda' or 'cpu')
    ema = EMAHelper(mu=args.ema_mu)                                                                             # EMA tracker
    ema.register(model)                                                                                         # initialize EMA weights

    # optional resume from checkpoint (also restores EMA)
    start_epoch = 0      # epoch index to resume from
    global_step = 0      # optimizer step counter
    if args.resume is not None and os.path.isfile(args.resume):                                                 # resume if file exists
        ckpt = torch.load(args.resume, map_location=device)                                                     # load checkpoint
        model.load_state_dict(ckpt['model'], strict=False)                                                      # restore model weights
        if 'ema' in ckpt:                                                                                       # restore EMA if present
            ema.shadow = ckpt['ema']
        start_epoch = int(ckpt.get('epoch', -1)) + 1                                                            # next epoch
        global_step = int(ckpt.get('global_step', 0))                                                           # steps so far
        print(f"Resumed from {args.resume} at epoch {start_epoch}, step {global_step}")                         # log

    # total LR schedule horizon (either max_steps or estimated from epochs × loader length)
    total_steps = args.max_steps if args.max_steps is not None else (len(dl) * max(1, args.epochs - start_epoch))
    scheduler = cosine_with_warmup(optim_, total_steps=total_steps, warmup_ratio=0.03, min_lr_ratio=0.1)        # cosine LR

    # --- training loop ---
    for epoch in range(start_epoch, args.epochs):                                                               # outer epoch loop
        model.train()                                                                                           # enable train mode (dropout, etc.)
        for batch in dl:                                                                                        # iterate mini-batches
            x = batch[0].to(device, non_blocking=True)                                                          # images tensor in [-1,1]
            y = batch[1].to(device, non_blocking=True) if (args.cond_mode == 'class') else None                # labels or None

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):                                 # mixed precision forward
                loss = diffusion.p_losses(model, x, y=y, p_uncond=args.p_uncond)                                # MSE on predicted noise

            optim_.zero_grad(set_to_none=True)                                                                  # clear grads (fast path)
            scaler.scale(loss).backward()                                                                       # backward with AMP scaling
            scaler.unscale_(optim_)                                                                             # unscale before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)                                             # grad clip for stability
            scaler.step(optim_)                                                                                 # optimizer step
            scaler.update()                                                                                     # update scaler (AMP)
            scheduler.step()                                                                                    # LR schedule step

            ema.update(model)                                                                                   # update EMA weights

            if (global_step % args.log_every) == 0:                                                             # periodic logging
                lr_now = optim_.param_groups[0]['lr']                                                           # current LR
                print(f"step {global_step} | loss {loss.item():.4f} | lr {lr_now:.2e}")                         # print metrics

            global_step += 1                                                                                    # count step
            if args.max_steps is not None and global_step >= args.max_steps:                                    # stop by budget
                break                                                                                           # exit batch loop

        # --- end-of-epoch preview using EMA (usually cleaner) ---
        model.eval()                                                                                            # eval mode
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}                                 # store current weights
        ema.copy_to(model)                                                                                      # load EMA weights for sampling
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):                    # no grad & AMP
            n_preview = 36                                                                                      # number of preview images
            if args.cond_mode == 'class':                                                                       # conditional preview
                y_s = torch.randint(0, model.num_classes, (n_preview,), device=device)                          # random classes
            else:
                y_s = None                                                                                      # unconditional preview
            x_gen = diffusion.sample(                                                                           
                model, n=n_preview, img_size=args.img_size, y=y_s,
                guidance_scale=(args.guidance_scale if y_s is not None else 0.0)                                # CFG only if conditional
            )
            x_gen = (x_gen + 1) / 2.0                                                                           # back to [0,1]
            save_grid(x_gen, os.path.join(args.outdir, f'samples_epoch_{epoch:03d}.png'), nrow=6)               # write preview grid
        model.load_state_dict(backup, strict=False)                                                              # restore train weights

        # --- checkpoint at end of epoch ---
        ckpt_args = vars(args).copy()                                                                            # copy CLI args
        ckpt_args['num_classes'] = (num_classes if args.cond_mode == 'class' else None)                          # persist class count
        ckpt = {                                                                                                 # serialize state
            'model': model.state_dict(),
            'ema': ema.shadow,
            'args': ckpt_args,
            'epoch': epoch,
            'global_step': global_step,
        }
        torch.save(ckpt, os.path.join(args.outdir, 'last.ckpt'))                                                 # save checkpoint

        if args.max_steps is not None and global_step >= args.max_steps:                                         # break epochs if done
            print("Reached max_steps, stopping training.")
            break


if __name__ == '__main__':  # script guard
    main()                   # run training


