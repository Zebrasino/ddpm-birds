import argparse                       # CLI parsing
import os                             # filesystem ops
import tempfile                       # temp dirs for FID/PR
import torch                          # torch core
from torchvision.utils import save_image  # save grids
from torchvision import datasets, transforms  # datasets/transforms
import torchvision                    # image IO/ops
from PIL import Image as _PILImage    # PIL for bbox cropping in FID

from diffusion import Diffusion       # DDPM process
from unet import UNet                 # U-Net
from data import _read_images_txt, _read_bboxes_txt  # CUB readers (for real crops)


def load_checkpoint(path: str, device: torch.device, use_ema: bool = True):
    """Load model from our training checkpoint (optionally with EMA weights)."""
    ckpt = torch.load(path, map_location=device)                                    # load ckpt dict
    args = ckpt.get('args', {})                                                     # stored train args
    cond_mode = args.get('cond_mode', 'class')                                      # conditioning type
    n_cls = args.get('num_classes', None) if (cond_mode == 'class') else None       # num classes (if any)
    model = UNet(base=args.get('base', 64), num_classes=n_cls).to(device)           # rebuild model
    model.load_state_dict(ckpt['model'], strict=False)                               # load weights
    if use_ema and ('ema' in ckpt):                                                 # optionally swap to EMA
        state = model.state_dict()
        for k in state.keys():
            if k in ckpt['ema']:
                state[k].copy_(ckpt['ema'][k])
    model.eval()                                                                     # eval mode
    return model, args                                                               # return model + saved args


def _generate_folder_of_fakes(model, diffusion, out_dir, num_images, img_size, y=None, guidance_scale=3.5, n_per_chunk=64):
    """Generate num_images samples to out_dir (PNG files), possibly class-conditional."""
    os.makedirs(out_dir, exist_ok=True)                                             # ensure dir
    saved = 0                                                                       # counter
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):  # AMP sampling
        while saved < num_images:                                                   # until quota
            n = min(n_per_chunk, num_images - saved)                                # current batch size
            y_batch = None                                                          # default: unconditional
            if getattr(model, 'num_classes', None) is not None:                     # conditional model?
                if y is None:                                                       # random classes if none specified
                    y_batch = torch.randint(0, model.num_classes, (n,), device=next(model.parameters()).device)
                else:                                                               # fixed class for all samples
                    y_batch = torch.full((n,), int(y), device=next(model.parameters()).device, dtype=torch.long)
            x = diffusion.sample(model, img_size=img_size, n=n, y=y_batch, guidance_scale=(guidance_scale if y_batch is not None else 0.0))
            x = (x + 1) / 2.0                                                       # back to [0,1]
            for i in range(n):                                                      # save images
                save_image(x[i], os.path.join(out_dir, f"fake_{saved+i:06d}.png"))
            saved += n                                                               # increment counter
    return out_dir                                                                   # return folder path


def task_sample(args):
    """Draw a grid of generated samples (DDPM ancestral; CFG if conditional)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')           # device
    model, train_args = load_checkpoint(args.checkpoint, device, use_ema=args.use_ema)  # load
    T = int(train_args.get('num_steps', 400))                                       # diffusion steps
    schedule = train_args.get('schedule', 'cosine')                                 # schedule type
    diffusion = Diffusion(T=T, schedule=schedule, device=device)                    # build process

    y = None                                                                        # optional labels
    if args.class_id is not None:
        y = torch.full((args.num_samples,), int(args.class_id), device=device, dtype=torch.long)
    os.makedirs(args.outdir, exist_ok=True)                                         # ensure dir
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
        x = diffusion.sample(model, n=args.num_samples, img_size=args.img_size, y=y, guidance_scale=args.guidance_scale)
    x = (x + 1) / 2.0                                                               # back to [0,1]
    save_image(x, os.path.join(args.outdir, 'grid.png'), nrow=int(args.num_samples**0.5))  # grid save


def _prepare_real_dir_with_bboxes(cub_root: str, img_size: int, max_images: int) -> str:
    """Create a temp folder with real images cropped to CUB bbox (no expansion), resized to img_size."""
    imgs = _read_images_txt(cub_root)                                               # id→rel path
    bboxes = _read_bboxes_txt(cub_root)                                             # id→bbox
    tfm = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),   # resize
        transforms.CenterCrop(img_size),                                                   # center-crop
    ])
    tmpdir = tempfile.mkdtemp()                                                     # make temp dir
    count = 0                                                                       # counter
    for img_id, rel in imgs.items():                                                # iterate CUB list
        if count >= max_images:
            break
        path = os.path.join(cub_root, rel if rel.startswith("images/") else os.path.join("images", rel))  # robust join
        if not os.path.isfile(path):
            continue
        img = torchvision.io.read_image(path)[:3]                                   # RGB (C,H,W)
        pil = _PILImage.fromarray(img.permute(1, 2, 0).numpy().astype('uint8'))    # to PIL
        if img_id in bboxes:                                                        # crop bbox if available
            x, y, w, h = bboxes[img_id]
            pil = pil.crop((int(x), int(y), int(x + w), int(y + h)))
        pil = tfm(pil)                                                              # resize+crop
        save_path = os.path.join(tmpdir, f"real_{count:06d}.png")                   # filename
        pil.save(save_path)                                                         # save
        count += 1                                                                  # increment
    return tmpdir                                                                   # temp folder


def task_fid(args):
    """Compute FID / KID / Precision–Recall (requires torch-fidelity)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')           # device
    if args.real_dir is None and not (args.use_bbox_real and args.cub_root):        # guard
        raise SystemExit('--real_dir or (--use_bbox_real and --cub_root) is required for FID')
    model, train_args = load_checkpoint(args.checkpoint, device, use_ema=args.use_ema)    # model
    T = int(train_args.get('num_steps', 400))                                       # steps
    schedule = train_args.get('schedule', 'cosine')                                 # schedule
    diffusion = Diffusion(T=T, schedule=schedule, device=device)                    # ddpm

    # Pick reals (plain folder OR CUB bbox-cropped temp folder)
    if args.use_bbox_real:
        real_dir = _prepare_real_dir_with_bboxes(args.cub_root, args.img_size, args.num_real)
    else:
        real_dir = args.real_dir

    # Generate fakes and compute metrics
    with tempfile.TemporaryDirectory() as tmpdir:
        _generate_folder_of_fakes(model, diffusion, tmpdir, args.num_fake, args.img_size, y=args.class_id, guidance_scale=args.guidance_scale, n_per_chunk=args.n_per_chunk)
        try:
            import torch_fidelity                                                    # external metrics
        except ImportError:
            raise SystemExit('torch-fidelity not installed. Install with: pip install torch-fidelity')
        metrics = torch_fidelity.calculate_metrics(input1=real_dir, input2=tmpdir, cuda=torch.cuda.is_available(), isc=False, fid=True, kid=True, prc=True, verbose=True)
        print('FID:', metrics.get('frechet_inception_distance'))                    # print FID
        print('KID:', metrics.get('kernel_inception_distance_mean'))                # print KID
        print('Precision:', metrics.get('precision'))                               # print Precision
        print('Recall:', metrics.get('recall'))                                     # print Recall


class SmallDisc(torch.nn.Module):
    """Tiny CNN discriminator used for a simple two-sample test."""
    def __init__(self, in_ch=3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, 32, 4, 2, 1), torch.nn.LeakyReLU(0.2, inplace=True),         # 64→32
            torch.nn.Conv2d(32, 64, 4, 2, 1), torch.nn.BatchNorm2d(64), torch.nn.LeakyReLU(0.2),# 32→16
            torch.nn.Conv2d(64, 128, 4, 2, 1), torch.nn.BatchNorm2d(128), torch.nn.LeakyReLU(0.2),# 16→8
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(128, 1)          # → logit
        )
    def forward(self, x):
        return self.net(x).squeeze(1)                                                            # (B,)


@torch.no_grad()
def _load_real_samples_folder(real_dir, img_size, num, device):
    """Load a small batch of real images from a folder; returns a normalized tensor in [-1,1]."""
    tfm = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    ds = datasets.ImageFolder(real_dir, transform=tfm)                                           # folder dataset
    if len(ds.samples) == 0:
        raise SystemExit("real_dir must contain subfolders with images.")                        # guard
    idxs = torch.randperm(len(ds.samples))[:num].tolist()                                        # random subset
    xs = []
    for i in idxs:
        path, _ = ds.samples[i]
        img = torchvision.io.read_image(path)[:3].float()/255.0                                  # (C,H,W) in [0,1]
        img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(img_size, img_size), mode="bicubic", align_corners=False).squeeze(0)
        img = (img - 0.5)/0.5                                                                    # normalize
        xs.append(img)
    x = torch.stack(xs, dim=0).to(device)
    return x


def task_two_sample(args):
    """Train a tiny discriminator to distinguish real vs fake (ideal ≈ 50% acc if distributions match)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, train_args = load_checkpoint(args.checkpoint, device, use_ema=args.use_ema)
    T = int(train_args.get('num_steps', 400))
    schedule = train_args.get('schedule', 'cosine')
    diffusion = Diffusion(T=T, schedule=schedule, device=device)

    # generate fakes to disk first (same preprocessing of fakes/reals)
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_dir = _generate_folder_of_fakes(model, diffusion, tmpdir, args.num_fake, args.img_size, y=args.class_id, guidance_scale=args.guidance_scale, n_per_chunk=args.n_per_chunk)
        # load reals
        x_real = _load_real_samples_folder(args.real_dir, args.img_size, min(args.num_real, args.num_fake), device)
        # read fakes back
        fpaths = sorted([os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith('.png')])[:x_real.size(0)]
        xs = []
        for p in fpaths:
            img = torchvision.io.read_image(p)[:3].float()/255.0
            img = (img - 0.5)/0.5
            xs.append(img)
        x_fake = torch.stack(xs, dim=0).to(device)

    # labels and shuffle
    y = torch.cat([torch.ones(x_real.size(0), device=device), torch.zeros(x_fake.size(0), device=device)], dim=0)
    x = torch.cat([x_real, x_fake], dim=0)
    perm = torch.randperm(x.size(0), device=device)
    x, y = x[perm], y[perm]

    # train tiny discriminator
    n = x.size(0); n_train = int(0.8*n)
    x_tr, y_tr = x[:n_train], y[:n_train]
    x_te, y_te = x[n_train:], y[n_train:]
    disc = SmallDisc().to(device)
    opt = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))
    for epoch in range(5):
        for i in range(0, n_train, 64):
            xb, yb = x_tr[i:i+64], y_tr[i:i+64]
            logits = disc(xb)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        logits = disc(x_te)
        acc = ((logits > 0).float() == y_te).float().mean().item()
    print(f"Two-sample discriminator accuracy: {acc*100:.2f}% (ideal ≈ 50%)")


def parse_args():
    """CLI for eval tasks."""
    p = argparse.ArgumentParser()
    p.add_argument('--task', type=str, choices=['sample','fid','two_sample'], required=True)  # which evaluation
    p.add_argument('--checkpoint', type=str, required=True)            # path to .ckpt
    p.add_argument('--img_size', type=int, default=64)                 # resolution
    p.add_argument('--outdir', type=str, required=True)                # output folder
    p.add_argument('--num_samples', type=int, default=36)              # number for grid
    p.add_argument('--class_id', type=int, default=None)               # fixed class id (optional)
    p.add_argument('--guidance_scale', type=float, default=3.5)        # CFG scale (if conditional)
    p.add_argument('--use_ema', action='store_true')                   # use EMA weights
    p.add_argument('--n_per_chunk', type=int, default=64)              # batch gen size
    p.add_argument('--real_dir', type=str, default=None)               # real images folder (FID/two-sample)
    p.add_argument('--num_real', type=int, default=500)                # #reals for FID
    p.add_argument('--num_fake', type=int, default=500)                # #fakes for FID
    p.add_argument('--use_bbox_real', action='store_true')             # use CUB bbox for reals in FID
    p.add_argument('--cub_root', type=str, default=None)               # CUB root for bbox cropping
    return p.parse_args()


def main():
    """Dispatch by task name."""
    args = parse_args()
    if args.task == 'sample':
        task_sample(args)
    elif args.task == 'fid':
        task_fid(args)
    elif args.task == 'two_sample':
        task_two_sample(args)
    else:
        raise SystemExit(f'Unknown task {args.task}')


if __name__ == '__main__':
    main()
