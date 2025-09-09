import argparse  # import modules
import os  # import modules
import torch  # import modules
from torch import optim  # import names from module
from torch.utils.data import DataLoader  # import names from module
from torchvision import transforms  # import names from module
from torchvision.utils import save_image  # import names from module

from diffusion import Diffusion  # import names from module
from unet import UNet  # import names from module
from utils import set_seed, save_grid, EMAHelper  # import names from module
from data import make_dataset  # import names from module


def parse_args():  # define function parse_args
    p = argparse.ArgumentParser()  # variable assignment
    p.add_argument('--data_root', type=str, required=True)  # statement
    p.add_argument('--outdir', type=str, required=True)  # statement
    p.add_argument('--img_size', type=int, default=64)  # statement
    p.add_argument('--batch_size', type=int, default=16)  # statement
    p.add_argument('--epochs', type=int, default=40)  # statement
    p.add_argument('--lr', type=float, default=2e-4)  # statement
    p.add_argument('--num_steps', type=int, default=400)  # statement
    p.add_argument('--schedule', type=str, default='cosine')  # statement
    p.add_argument('--cond_mode', type=str, choices=['none','class'], default='class')  # statement
    p.add_argument('--p_uncond', type=float, default=0.1)  # statement
    p.add_argument('--guidance_scale', type=float, default=3.0)  # statement
    p.add_argument('--base', type=int, default=64)  # statement
    p.add_argument('--seed', type=int, default=42)  # statement
    return p.parse_args()  # return value


def main():  # define function main
    args = parse_args()  # variable assignment
    os.makedirs(args.outdir, exist_ok=True)  # statement
    set_seed(args.seed)  # statement

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # variable assignment

    tfm = transforms.Compose([  # variable assignment
        transforms.Resize(args.img_size),  # statement
        transforms.CenterCrop(args.img_size),  # statement
        transforms.RandomHorizontalFlip(),  # statement
        transforms.ToTensor(),  # statement
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # maps to [-1,1]  # PyTorch operation
    ])

    ds, num_classes = make_dataset(args.data_root, tfm)  # variable assignment
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)  # variable assignment

    model = UNet(base=args.base, num_classes=(num_classes if args.cond_mode=='class' else None)).to(device)  # variable assignment
    diffusion = Diffusion(T=args.num_steps, schedule=args.schedule, device=device)  # variable assignment

    optim_ = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.999), weight_decay=0.01)  # variable assignment
    scaler = torch.cuda.amp.GradScaler()  # variable assignment
    ema = EMAHelper(mu=0.999)  # variable assignment
    ema.register(model)  # statement

    global_step = 0  # variable assignment
    for epoch in range(args.epochs):  # loop
        model.train()  # PyTorch operation
        for batch in dl:  # loop
            x = batch[0].to(device, non_blocking=True)  # PyTorch operation
            y = batch[1].to(device, non_blocking=True) if (args.cond_mode=='class') else None  # PyTorch operation

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):  # context manager
                loss = diffusion.p_losses(model, x, y=y, p_uncond=args.p_uncond)  # PyTorch operation

            optim_.zero_grad(set_to_none=True)  # PyTorch operation
            scaler.scale(loss).backward()  # PyTorch operation
            scaler.unscale_(optim_)  # PyTorch operation
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # PyTorch operation
            scaler.step(optim_)  # PyTorch operation
            scaler.update()  # PyTorch operation

            ema.update(model)  # statement

            if global_step % 500 == 0:  # control flow
                print(f"step {global_step} loss {loss.item():.4f}")  # debug/print
            global_step += 1  # variable assignment

        # sampling at epoch end with EMA  # comment  # statement
        model.eval()  # PyTorch operation
        _backup = {k: v.detach().clone() for k, v in model.state_dict().items()}  # variable assignment
        ema.copy_to(model)  # statement
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):  # context manager
            n = 36  # variable assignment
            if args.cond_mode=='class':  # control flow
                y_s = torch.randint(0, model.num_classes, (n,), device=device)  # PyTorch operation
            else:  # control flow
                y_s = None  # variable assignment
            x_gen = diffusion.sample(model, img_size=args.img_size, n=n, y=y_s, guidance_scale=(args.guidance_scale if y_s is not None else 0.0))  # PyTorch operation
            x_gen = (x_gen + 1) / 2.0  # PyTorch operation
            save_grid(x_gen, os.path.join(args.outdir, f'samples_epoch_{epoch:03d}.png'), nrow=6)  # statement
        model.load_state_dict(_backup, strict=False)  # PyTorch operation

        # checkpoint  # comment  # statement
        ckpt = {
            'model': model.state_dict(),  # variable assignment
            'ema': ema.shadow,  # variable assignment
            'args': vars(args),  # variable assignment
        }  # variable assignment
        torch.save(ckpt, os.path.join(args.outdir, 'last.ckpt'))  # PyTorch operation


if __name__ == '__main__':  # control flow
    main()  # statement
