import argparse  # import modules
import os  # import modules
import torch  # import modules
from torchvision.utils import save_image  # import names from module
from torchvision import datasets, transforms  # import names from module
import tempfile  # import modules
import torchvision  # import modules

from diffusion import Diffusion  # import names from module
from unet import UNet  # import names from module


def load_checkpoint(path: str, device: torch.device):  # define function load_checkpoint
    ckpt = torch.load(path, map_location=device)  # PyTorch operation
    args = ckpt.get('args', {})  # variable assignment
    num_classes = None if args.get('cond_mode','class')=='none' else None  # variable assignment
    # build model using training args; if classes unknown, infer from state dict  # comment  # statement
    state = ckpt['model']  # variable assignment
    if 't_pos.dim' in state or True:  # control flow
        m = UNet(base=args.get('base',64), num_classes=args.get('num_classes', None))  # variable assignment
    else:  # control flow
        m = UNet(base=64, num_classes=None)  # variable assignment
    m.load_state_dict(state, strict=False)  # PyTorch operation
    m.to(device)  # PyTorch operation
    return m, args  # return value


def _generate_folder_of_fakes(model, diffusion, out_dir, num_images, img_size, y=None, guidance_scale=3.5, n_per_chunk=64):  # define function _generate_folder_of_fakes
    os.makedirs(out_dir, exist_ok=True)  # statement
    saved = 0  # variable assignment
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):  # context manager
        while saved < num_images:  # loop
            n = min(n_per_chunk, num_images - saved)  # variable assignment
            y_batch = None  # variable assignment
            if getattr(model, 'num_classes', None) is not None:  # control flow
                if y is None:  # control flow
                    y_batch = torch.randint(0, model.num_classes, (n,), device=next(model.parameters()).device)  # PyTorch operation
                else:  # control flow
                    y_batch = torch.full((n,), int(y), device=next(model.parameters()).device, dtype=torch.long)  # PyTorch operation
            x = diffusion.sample(model, img_size=img_size, n=n, y=y_batch, guidance_scale=(guidance_scale if y_batch is not None else 1.0))  # PyTorch operation
            x = (x + 1) / 2.0  # PyTorch operation
            for i in range(n):  # loop
                save_image(x[i], os.path.join(out_dir, f"fake_{saved+i:06d}.png"))  # statement
            saved += n  # variable assignment
    return out_dir  # return value


def task_sample(args):  # define function task_sample
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # variable assignment
    model, train_args = load_checkpoint(args.checkpoint, device)  # variable assignment
    T = int(train_args.get('num_steps', 400))  # variable assignment
    schedule = train_args.get('schedule', 'cosine')  # variable assignment
    diffusion = Diffusion(T=T, schedule=schedule, device=device)  # variable assignment

    y = None  # variable assignment
    if args.class_id is not None:  # control flow
        y = torch.full((args.num_samples,), int(args.class_id), device=device, dtype=torch.long)  # PyTorch operation

    os.makedirs(args.outdir, exist_ok=True)  # statement
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):  # context manager
        x = diffusion.sample(model, n=args.num_samples, img_size=args.img_size, y=y, guidance_scale=args.guidance_scale)  # PyTorch operation
    x = (x + 1) / 2.0  # PyTorch operation
    save_image(x, os.path.join(args.outdir, 'grid.png'), nrow=int(args.num_samples**0.5))  # statement


def task_fid(args):  # define function task_fid
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # variable assignment
    if args.real_dir is None:  # control flow
        raise SystemExit('--real_dir is required for FID')  # raise exception
    model, train_args = load_checkpoint(args.checkpoint, device)  # variable assignment
    T = int(train_args.get('num_steps', 400))  # variable assignment
    schedule = train_args.get('schedule', 'cosine')  # variable assignment
    diffusion = Diffusion(T=T, schedule=schedule, device=device)  # variable assignment
    model.eval()  # PyTorch operation
    with tempfile.TemporaryDirectory() as tmpdir:  # context manager
        _generate_folder_of_fakes(model, diffusion, tmpdir, args.num_fake, args.img_size, y=args.class_id, guidance_scale=args.guidance_scale)  # statement
        try:  # try block
            import torch_fidelity  # import modules
        except ImportError:  # exception handler
            raise SystemExit('torch-fidelity not installed. Install with: pip install torch-fidelity')  # raise exception
        metrics = torch_fidelity.calculate_metrics(input1=args.real_dir, input2=tmpdir, cuda=torch.cuda.is_available(), isc=False, fid=True, kid=True, prc=True, verbose=True)  # PyTorch operation
        print('FID:', metrics.get('frechet_inception_distance'))  # debug/print
        print('KID:', metrics.get('kernel_inception_distance_mean'))  # debug/print
        print('Precision:', metrics.get('precision'))  # debug/print
        print('Recall:', metrics.get('recall'))  # debug/print


class SmallDisc(torch.nn.Module):  # define class SmallDisc
    def __init__(self, in_ch=3):  # define function __init__
        super().__init__()  # call parent constructor  # statement
        self.net = torch.nn.Sequential(  # variable assignment
            torch.nn.Conv2d(in_ch, 32, 4, 2, 1), torch.nn.LeakyReLU(0.2, inplace=True),  # PyTorch operation
            torch.nn.Conv2d(32, 64, 4, 2, 1), torch.nn.BatchNorm2d(64), torch.nn.LeakyReLU(0.2, inplace=True),  # PyTorch operation
            torch.nn.Conv2d(64, 128, 4, 2, 1), torch.nn.BatchNorm2d(128), torch.nn.LeakyReLU(0.2, inplace=True),  # PyTorch operation
            torch.nn.AdaptiveAvgPool2d(1),  # PyTorch operation
            torch.nn.Flatten(),  # PyTorch operation
            torch.nn.Linear(128, 1)  # PyTorch operation
        )
    def forward(self, x):  # define function forward
        return self.net(x).squeeze(1)  # return value


@torch.no_grad()  # decorator
def _load_real_samples(real_dir, img_size, num, device):  # define function _load_real_samples
    tfm = transforms.Compose([  # variable assignment
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),  # statement
        transforms.CenterCrop(img_size),  # statement
        transforms.ToTensor(),  # statement
        transforms.Normalize((0.5,)*3, (0.5,)*3),  # statement
    ])
    ds = datasets.ImageFolder(real_dir, transform=tfm)  # variable assignment
    if len(ds.samples) == 0:  # control flow
        raise SystemExit('real_dir must contain subfolders with images.')  # raise exception
    idxs = torch.randperm(len(ds.samples))[:num].tolist()  # PyTorch operation
    xs = []  # variable assignment
    for i in idxs:  # loop
        path, _ = ds.samples[i]  # variable assignment
        img = torchvision.io.read_image(path)[:3].float()/255.0  # PyTorch operation
        img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(img_size, img_size), mode='bicubic', align_corners=False).squeeze(0)  # PyTorch operation
        img = (img - 0.5)/0.5  # PyTorch operation
        xs.append(img)  # statement
    x = torch.stack(xs, dim=0).to(device)  # PyTorch operation
    return x  # return value


def task_two_sample(args):  # define function task_two_sample
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # variable assignment
    if args.real_dir is None:  # control flow
        raise SystemExit('--real_dir is required for two_sample')  # raise exception
    model, train_args = load_checkpoint(args.checkpoint, device)  # variable assignment
    T = int(train_args.get('num_steps', 400))  # variable assignment
    schedule = train_args.get('schedule', 'cosine')  # variable assignment
    diffusion = Diffusion(T=T, schedule=schedule, device=device)  # variable assignment
    model.eval()  # PyTorch operation

    with tempfile.TemporaryDirectory() as tmpdir:  # context manager
        fake_dir = _generate_folder_of_fakes(model, diffusion, tmpdir, args.num_fake, args.img_size, y=args.class_id, guidance_scale=args.guidance_scale)  # statement

        x_real = _load_real_samples(args.real_dir, args.img_size, min(args.num_real, args.num_fake), device)  # PyTorch operation
        fpaths = sorted([os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith('.png')])[:x_real.size(0)]  # variable assignment
        xs = []  # variable assignment
        for p in fpaths:  # loop
            img = torchvision.io.read_image(p)[:3].float()/255.0  # PyTorch operation
            img = (img - 0.5)/0.5  # PyTorch operation
            xs.append(img)  # statement
        x_fake = torch.stack(xs, dim=0).to(device)  # PyTorch operation

    y = torch.cat([torch.ones(x_real.size(0), device=device), torch.zeros(x_fake.size(0), device=device)], dim=0)  # PyTorch operation
    x = torch.cat([x_real, x_fake], dim=0)  # PyTorch operation
    perm = torch.randperm(x.size(0), device=device)  # PyTorch operation
    x, y = x[perm], y[perm]  # PyTorch operation

    n = x.size(0)  # variable assignment
    n_train = int(0.8*n)  # variable assignment
    x_tr, y_tr = x[:n_train], y[:n_train]  # variable assignment
    x_te, y_te = x[n_train:], y[n_train:]  # variable assignment

    disc = SmallDisc().to(device)  # variable assignment
    opt = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))  # variable assignment
    for epoch in range(5):  # loop
        for i in range(0, n_train, 64):  # loop
            xb = x_tr[i:i+64]  # variable assignment
            yb = y_tr[i:i+64]  # variable assignment
            logits = disc(xb)  # PyTorch operation
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb)  # PyTorch operation
            opt.zero_grad()  # PyTorch operation
            loss.backward()  # PyTorch operation
            opt.step()  # PyTorch operation
    with torch.no_grad():  # context manager
        logits = disc(x_te)  # PyTorch operation
        acc = ((logits > 0).float() == y_te).float().mean().item()  # PyTorch operation
    print(f"Two-sample discriminator accuracy: {acc*100:.2f}% (ideal ≈ 50%)")  # debug/print


def parse_args():  # define function parse_args
    p = argparse.ArgumentParser()  # variable assignment
    p.add_argument('--task', type=str, choices=['sample','fid','two_sample'], required=True)  # statement
    p.add_argument('--checkpoint', type=str, required=True)  # statement
    p.add_argument('--img_size', type=int, default=64)  # statement
    p.add_argument('--outdir', type=str, required=True)  # statement
    p.add_argument('--num_samples', type=int, default=36)  # statement
    p.add_argument('--class_id', type=int, default=None)  # statement
    p.add_argument('--guidance_scale', type=float, default=3.5)  # statement
    p.add_argument('--real_dir', type=str, default=None)  # statement
    p.add_argument('--num_real', type=int, default=500)  # statement
    p.add_argument('--num_fake', type=int, default=500)  # statement
    return p.parse_args()  # return value


def main():  # define function main
    args = parse_args()  # variable assignment
    if args.task == 'sample':  # control flow
        task_sample(args)  # statement
    elif args.task == 'fid':  # control flow
        task_fid(args)  # statement
    elif args.task == 'two_sample':  # control flow
        task_two_sample(args)  # statement
    else:  # control flow
        raise SystemExit(f'Unknown task {args.task}')  # raise exception


if __name__ == '__main__':  # control flow
    main()  # statement
