import argparse  # CLI parsing
import os  # filesystem
import math  # math utilities
import torch  # torch core
from torch import optim  # optimizers
from torch.utils.data import DataLoader  # data loader
from torchvision import transforms  # image transforms

from diffusion import Diffusion  # DDPM core
from unet import UNet  # U-Net backbone
from utils import set_seed, save_grid, EMAHelper  # utils (seed, grid, EMA)
from data import make_cub_bbox_dataset, make_imagefolder_dataset  # datasets


def cosine_with_warmup(optimizer, total_steps: int, warmup_ratio: float = 0.03, min_lr_ratio: float = 0.1):  # lr scheduler factory
    warmup = max(1, int(total_steps * warmup_ratio))  # warmup steps
    def lr_lambda(step):  # step → multiplier
        if step < warmup:  # warmup phase
            return (step + 1) / warmup  # linear warmup
        prog = (step - warmup) / max(1, total_steps - warmup)  # progress 0..1
        cos = 0.5 * (1 + math.cos(math.pi * prog))  # cosine 1..0
        return min_lr_ratio + (1 - min_lr_ratio) * cos  # scale to [min,1]
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)  # scheduler


def parse_args():  # build CLI
    p = argparse.ArgumentParser()  # parser
    # data
    p.add_argument('--data_root', type=str, required=True)  # path to images/ or CUB root
    p.add_argument('--use_bbox', action='store_true')  # use CUB GT bounding boxes
    p.add_argument('--bbox_expand', type=float, default=1.2)  # bbox margin factor
    p.add_argument('--subset', type=int, default=None)  # limit number of images (quick overfit)
    p.add_argument('--class_limit', type=int, default=None)  # limit number of classes
    # training
    p.add_argument('--outdir', type=str, required=True)  # output dir
    p.add_argument('--img_size', type=int, default=64)  # resolution
    p.add_argument('--batch_size', type=int, default=16)  # batch size
    p.add_argument('--epochs', type=int, default=40)  # epochs
    p.add_argument('--lr', type=float, default=2e-4)  # learning rate
    p.add_argument('--weight_decay', type=float, default=0.0)  # weight decay (0 is typical for diffusion)
    p.add_argument('--num_steps', type=int, default=400)  # diffusion steps (T)
    p.add_argument('--schedule', type=str, default='cosine', choices=['cosine','linear'])  # beta schedule
    p.add_argument('--cond_mode', type=str, choices=['none','class'], default='class')  # conditioning
    p.add_argument('--p_uncond', type=float, default=0.1)  # CFG dropout prob
    p.add_argument('--guidance_scale', type=float, default=1.0)  # CFG scale for end-of-epoch samples
    p.add_argument('--base', type=int, default=64)  # UNet base channels
    p.add_argument('--ema_mu', type=float, default=0.999)  # EMA decay
    p.add_argument('--max_steps', type=int, default=None)  # stop after N steps globally
    p.add_argument('--seed', type=int, default=42)  # RNG seed
    p.add_argument('--resume', type=str, default=None)  # resume from ckpt path
    p.add_argument('--log_every', type=int, default=500)  # print every N steps
    return p.parse_args()  # parse


def main():  # main entry
    args = parse_args()  # parse args
    os.makedirs(args.outdir, exist_ok=True)  # ensure outdir
    set_seed(args.seed)  # seed

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # pick device

    # transforms: zoom into the bird; normalize to [-1,1]
    tfm = transforms.Compose([
    transforms.Resize(args.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(args.img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


    # dataset: use bbox (CUB root) or imagefolder (images/)
    if args.use_bbox:  # using GT boxes
        ds, num_classes = make_cub_bbox_dataset(args.data_root, tfm, expand=args.bbox_expand, subset=args.subset, class_limit=args.class_limit)
    else:  # plain folder
        ds, num_classes = make_imagefolder_dataset(args.data_root, tfm, subset=args.subset, class_limit=args.class_limit)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)  # data loader

    # build model & diffusion
    model = UNet(base=args.base, num_classes=(num_classes if args.cond_mode=='class' else None)).to(device)  # UNet
    diffusion = Diffusion(T=args.num_steps, schedule=args.schedule, device=device)  # DDPM

    # optimizer, scaler, ema, scheduler
    optim_ = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.999), weight_decay=args.weight_decay)  # AdamW
    scaler = torch.amp.GradScaler(device.type)  # AMP scaler for active device

    ema = EMAHelper(mu=args.ema_mu)  # EMA helper
    ema.register(model)  # initialize EMA

    # optional resume
    start_epoch = 0  # epoch counter
    global_step = 0  # step counter
    if args.resume is not None and os.path.isfile(args.resume):  # resume path given
        ckpt = torch.load(args.resume, map_location=device)  # load ckpt
        model.load_state_dict(ckpt['model'], strict=False)  # load weights
        if 'ema' in ckpt:  # restore ema if present
            ema.shadow = ckpt['ema']
        if 'epoch' in ckpt:  # resume epoch index
            start_epoch = int(ckpt['epoch']) + 1
        if 'global_step' in ckpt:  # resume step
            global_step = int(ckpt['global_step'])
        print(f"Resumed from {args.resume} at epoch {start_epoch}, step {global_step}")  # log resume

    # scheduler using total expected steps if max_steps not set
    total_steps = args.max_steps if args.max_steps is not None else (len(dl) * max(1, args.epochs - start_epoch))  # steps budget
    scheduler = cosine_with_warmup(optim_, total_steps=total_steps, warmup_ratio=0.03, min_lr_ratio=0.1)  # cosine schedule

    for epoch in range(start_epoch, args.epochs):  # training epochs
        model.train()  # train mode
        for batch in dl:  # iterate batches
            x = batch[0].to(device, non_blocking=True)  # images
            y = batch[1].to(device, non_blocking=True) if (args.cond_mode=='class') else None  # labels or None

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):  # mixed precision
                loss = diffusion.p_losses(model, x, y=y, p_uncond=args.p_uncond)  # ddpm loss

            optim_.zero_grad(set_to_none=True)  # zero grads
            scaler.scale(loss).backward()  # backward
            scaler.unscale_(optim_)  # unscale for clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip for stability
            scaler.step(optim_)  # optimizer step
            scaler.update()  # update scaler
            scheduler.step()  # lr scheduler step

            ema.update(model)  # update ema

            if (global_step % args.log_every) == 0:  # periodic log
                lr_now = optim_.param_groups[0]['lr']  # current lr
                print(f"step {global_step} | loss {loss.item():.4f} | lr {lr_now:.2e}")  # print

            global_step += 1  # increment step
            if args.max_steps is not None and global_step >= args.max_steps:  # stop if reached budget
                break  # exit batch loop

        # end-of-epoch sampling with EMA
        model.eval()  # eval mode
        _backup = {k: v.detach().clone() for k, v in model.state_dict().items()}  # backup weights
        ema.copy_to(model)  # load ema weights
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):  # no grad & AMP
            n = 36  # number of samples
            if args.cond_mode=='class':  # class-conditional grid
                y_s = torch.randint(0, model.num_classes, (n,), device=device)  # random classes
            else:  # unconditional grid
                y_s = None  # no labels
            x_gen = diffusion.sample(model, img_size=args.img_size, n=n, y=y_s, guidance_scale=(args.guidance_scale if y_s is not None else 0.0))  # sample
            x_gen = (x_gen + 1) / 2.0  # to [0,1]
            save_grid(x_gen, os.path.join(args.outdir, f'samples_epoch_{epoch:03d}.png'), nrow=6)  # save grid
        model.load_state_dict(_backup, strict=False)  # restore training weights

        # save checkpoint at end of epoch
        ckpt_args = vars(args).copy()  # copy args
        ckpt_args['num_classes'] = (num_classes if args.cond_mode=='class' else None)  # store class count
        ckpt = {  # ckpt dict
            'model': model.state_dict(),  # weights
            'ema': ema.shadow,  # ema state
            'args': ckpt_args,  # training args (with num_classes)
            'epoch': epoch,  # last epoch
            'global_step': global_step,  # steps so far
        }  # end dict
        torch.save(ckpt, os.path.join(args.outdir, 'last.ckpt'))  # write ckpt

        if args.max_steps is not None and global_step >= args.max_steps:  # if finished by steps budget
            print("Reached max_steps, stopping training.")  # log
            break  # exit epoch loop


if __name__ == '__main__':  # entry guard
    main()  # run main

