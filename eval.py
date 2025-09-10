import argparse  # CLI parsing
import os  # filesystem
import tempfile  # temp dirs
import torch  # torch core
from torchvision.utils import save_image  # save grids
from torchvision import datasets, transforms  # datasets & transforms
import torchvision  # IO and ops
from PIL import Image as _PILImage  # PIL image

from diffusion import Diffusion  # DDPM core
from unet import UNet  # U-Net backbone
from data import _read_images_txt, _read_bboxes_txt  # CUB readers for bbox-cropping reals


def load_checkpoint(path: str, device: torch.device, use_ema: bool = True):  # load model from ckpt
    ckpt = torch.load(path, map_location=device)  # load ckpt
    args = ckpt.get('args', {})  # saved train args
    cond_mode = args.get('cond_mode', 'class')  # 'class' or 'none'
    n_cls = args.get('num_classes', None) if (cond_mode == 'class') else None  # number of classes (if known)
    model = UNet(base=args.get('base', 64), num_classes=n_cls).to(device)  # build model
    model.load_state_dict(ckpt['model'], strict=False)  # load weights
    if use_ema and ('ema' in ckpt):  # optionally load EMA weights
        state = model.state_dict()  # model state
        for k in state.keys():  # for each tensor
            if k in ckpt['ema']:  # if present in ema
                state[k].copy_(ckpt['ema'][k])  # copy ema weight
    model.eval()  # eval mode
    return model, args  # return


def _generate_folder_of_fakes(model, diffusion, out_dir, num_images, img_size, y=None, guidance_scale=3.5, n_per_chunk=64):  # generate images to folder
    os.makedirs(out_dir, exist_ok=True)  # ensure dir
    saved = 0  # counter
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):  # no grad & AMP
        while saved < num_images:  # until done
            n = min(n_per_chunk, num_images - saved)  # batch size
            y_batch = None  # default no labels
            if getattr(model, 'num_classes', None) is not None:  # if conditional
                if y is None:  # random classes
                    y_batch = torch.randint(0, model.num_classes, (n,), device=next(model.parameters()).device)
                else:  # fixed class
                    y_batch = torch.full((n,), int(y), device=next(model.parameters()).device, dtype=torch.long)
            x = diffusion.sample(model, img_size=img_size, n=n, y=y_batch, guidance_scale=(guidance_scale if y_batch is not None else 0.0))  # sample
            x = (x + 1) / 2.0  # to [0,1]
            for i in range(n):  # save loop
                save_image(x[i], os.path.join(out_dir, f"fake_{saved+i:06d}.png"))  # write file
            saved += n  # inc counter
    return out_dir  # path back


def task_sample(args):  # sample grid
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # device
    model, train_args = load_checkpoint(args.checkpoint, device, use_ema=args.use_ema)  # load model
    T = int(train_args.get('num_steps', 400))  # diffusion steps
    schedule = train_args.get('schedule', 'cosine')  # beta scheduler
    diffusion = Diffusion(T=T, schedule=schedule, device=device)  # DDPM

    y = None  # labels (optional)
    if args.class_id is not None:  # fixed class id
        y = torch.full((args.num_samples,), int(args.class_id), device=device, dtype=torch.long)  # labels batch

    os.makedirs(args.outdir, exist_ok=True)  # ensure dir
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):  # no grad & AMP
        x = diffusion.sample(model, n=args.num_samples, img_size=args.img_size, y=y, guidance_scale=args.guidance_scale)  # sample batch
    x = (x + 1) / 2.0  # to [0,1]
    save_image(x, os.path.join(args.outdir, 'grid.png'), nrow=int(args.num_samples**0.5))  # save grid


def _prepare_real_dir_with_bboxes(cub_root: str, img_size: int, max_images: int) -> str:  # build temp real dir cropped by bbox
    imgs = _read_images_txt(cub_root)  # id→path
    bboxes = _read_bboxes_txt(cub_root)  # id→bbox
    tfm = transforms.Compose([  # transform
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),  # resize
        transforms.CenterCrop(img_size),  # center crop
    ])  # end transform
    tmpdir = tempfile.mkdtemp()  # temp folder
    count = 0  # saved counter
    for img_id, rel in imgs.items():  # iterate all
        if count >= max_images:  # stop at quota
            break  # exit
        path = os.path.join(cub_root, rel)  # abs path
        if not os.path.isfile(path):  # skip missing
            continue  # next
        img = torchvision.io.read_image(path)[:3]  # read RGB (C,H,W)
        pil = _PILImage.fromarray(img.permute(1, 2, 0).numpy().astype('uint8'))  # to PIL
        if img_id in bboxes:  # if bbox exists
            x, y, w, h = bboxes[img_id]  # bbox
            pil = pil.crop((int(x), int(y), int(x + w), int(y + h)))  # crop bbox
        pil = tfm(pil)  # resize+crop
        save_path = os.path.join(tmpdir, f"real_{count:06d}.png")  # filename
        pil.save(save_path)  # save file
        count += 1  # inc
    return tmpdir  # return temp dir


def task_fid(args):  # compute FID/KID/PR
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # device
    if args.real_dir is None and not (args.use_bbox_real and args.cub_root):  # must have real source
        raise SystemExit('--real_dir or (--use_bbox_real and --cub_root) is required for FID')  # error
    model, train_args = load_checkpoint(args.checkpoint, device, use_ema=args.use_ema)  # load model
    T = int(train_args.get('num_steps', 400))  # steps
    schedule = train_args.get('schedule', 'cosine')  # schedule
    diffusion = Diffusion(T=T, schedule=schedule, device=device)  # ddpm

    # Prepare reals: either direct folder or temp bbox-cropped folder from CUB root
    if args.use_bbox_real:  # use bbox for fairness
        real_dir = _prepare_real_dir_with_bboxes(args.cub_root, args.img_size, args.num_real)  # temp reals
    else:  # use provided dir
        real_dir = args.real_dir  # plain reals

    # Generate fakes
    with tempfile.TemporaryDirectory() as tmpdir:  # temp folder for fakes
        _generate_folder_of_fakes(model, diffusion, tmpdir, args.num_fake, args.img_size, y=args.class_id, guidance_scale=args.guidance_scale, n_per_chunk=args.n_per_chunk)  # make fakes
        try:  # torch-fidelity import
            import torch_fidelity  # metrics
        except ImportError:  # missing package
            raise SystemExit('torch-fidelity not installed. Install with: pip install torch-fidelity')  # instruction
        metrics = torch_fidelity.calculate_metrics(  # compute metrics
            input1=real_dir, input2=tmpdir, cuda=torch.cuda.is_available(), isc=False, fid=True, kid=True, prc=True, verbose=True
        )  # run
        print('FID:', metrics.get('frechet_inception_distance'))  # log FID
        print('KID:', metrics.get('kernel_inception_distance_mean'))  # log KID
        print('Precision:', metrics.get('precision'))  # log precision
        print('Recall:', metrics.get('recall'))  # log recall


class SmallDisc(torch.nn.Module):  # tiny discriminator for two-sample test
    def __init__(self, in_ch=3):  # ctor
        super().__init__()  # init
        self.net = torch.nn.Sequential(  # conv stack
            torch.nn.Conv2d(in_ch, 32, 4, 2, 1), torch.nn.LeakyReLU(0.2, inplace=True),  # 64→32
            torch.nn.Conv2d(32, 64, 4, 2, 1), torch.nn.BatchNorm2d(64), torch.nn.LeakyReLU(0.2, inplace=True),  # 32→16
            torch.nn.Conv2d(64, 128, 4, 2, 1), torch.nn.BatchNorm2d(128), torch.nn.LeakyReLU(0.2, inplace=True),  # 16→8
            torch.nn.AdaptiveAvgPool2d(1),  # global avg
            torch.nn.Flatten(),  # flatten
            torch.nn.Linear(128, 1)  # logit
        )  # net
    def forward(self, x):  # forward
        return self.net(x).squeeze(1)  # (B,)


@torch.no_grad()  # no grad for loading reals
def _load_real_samples_folder(real_dir, img_size, num, device):  # load real images from folder
    tfm = transforms.Compose([  # transform
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),  # resize
        transforms.CenterCrop(img_size),  # center crop
        transforms.ToTensor(),  # to tensor
        transforms.Normalize((0.5,)*3, (0.5,)*3),  # to [-1,1]
    ])  # pipeline
    ds = datasets.ImageFolder(real_dir, transform=tfm)  # dataset
    if len(ds.samples) == 0:  # sanity check
        raise SystemExit("real_dir must contain subfolders with images.")  # error
    idxs = torch.randperm(len(ds.samples))[:num].tolist()  # random subset
    xs = []  # buffer
    for i in idxs:  # loop
        path, _ = ds.samples[i]  # pick path
        img = torchvision.io.read_image(path)[:3].float()/255.0  # read as float
        img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(img_size, img_size), mode="bicubic", align_corners=False).squeeze(0)  # resize
        img = (img - 0.5)/0.5  # [-1,1]
        xs.append(img)  # append
    x = torch.stack(xs, dim=0).to(device)  # stack batch
    return x  # tensor


def task_two_sample(args):  # discriminator-based two-sample test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # device
    model, train_args = load_checkpoint(args.checkpoint, device, use_ema=args.use_ema)  # model
    T = int(train_args.get('num_steps', 400))  # steps
    schedule = train_args.get('schedule', 'cosine')  # schedule
    diffusion = Diffusion(T=T, schedule=schedule, device=device)  # ddpm

    # Generate fakes
    with tempfile.TemporaryDirectory() as tmpdir:  # temp dir
        fake_dir = _generate_folder_of_fakes(model, diffusion, tmpdir, args.num_fake, args.img_size, y=args.class_id, guidance_scale=args.guidance_scale, n_per_chunk=args.n_per_chunk)  # gen fakes

        # Load reals from folder (simple path)
        x_real = _load_real_samples_folder(args.real_dir, args.img_size, min(args.num_real, args.num_fake), device)  # real batch
        # Load fakes back from disk
        fpaths = sorted([os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith('.png')])[:x_real.size(0)]  # fake paths
        xs = []  # buffer
        for p in fpaths:  # loop
            img = torchvision.io.read_image(p)[:3].float()/255.0  # read
            img = (img - 0.5)/0.5  # normalize
            xs.append(img)  # append
        x_fake = torch.stack(xs, dim=0).to(device)  # stack

    # labels: real=1, fake=0
    y = torch.cat([torch.ones(x_real.size(0), device=device), torch.zeros(x_fake.size(0), device=device)], dim=0)  # labels
    x = torch.cat([x_real, x_fake], dim=0)  # concat data
    perm = torch.randperm(x.size(0), device=device)  # shuffle
    x, y = x[perm], y[perm]  # permute

    # split train/test
    n = x.size(0)  # total
    n_train = int(0.8*n)  # 80% train
    x_tr, y_tr = x[:n_train], y[:n_train]  # train
    x_te, y_te = x[n_train:], y[n_train:]  # test

    # train tiny discriminator
    disc = SmallDisc().to(device)  # model
    opt = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))  # optimizer
    for epoch in range(5):  # short train
        for i in range(0, n_train, 64):  # batches
            xb = x_tr[i:i+64]  # batch x
            yb = y_tr[i:i+64]  # batch y
            logits = disc(xb)  # forward
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb)  # BCE loss
            opt.zero_grad()  # zero
            loss.backward()  # backward
            opt.step()  # step
    with torch.no_grad():  # eval
        logits = disc(x_te)  # forward
        acc = ((logits > 0).float() == y_te).float().mean().item()  # accuracy
    print(f"Two-sample discriminator accuracy: {acc*100:.2f}% (ideal ≈ 50%)")  # report


def parse_args():  # cli for eval
    p = argparse.ArgumentParser()  # parser
    p.add_argument('--task', type=str, choices=['sample','fid','two_sample'], required=True)  # which task
    p.add_argument('--checkpoint', type=str, required=True)  # ckpt path
    p.add_argument('--img_size', type=int, default=64)  # resolution
    p.add_argument('--outdir', type=str, required=True)  # output folder
    p.add_argument('--num_samples', type=int, default=36)  # grid size
    p.add_argument('--class_id', type=int, default=None)  # optional class id
    p.add_argument('--guidance_scale', type=float, default=3.5)  # CFG scale
    p.add_argument('--use_ema', action='store_true')  # use EMA weights
    p.add_argument('--n_per_chunk', type=int, default=64)  # batch size for gen
    p.add_argument('--real_dir', type=str, default=None)  # folder of real images
    p.add_argument('--num_real', type=int, default=500)  # number of real imgs
    p.add_argument('--num_fake', type=int, default=500)  # number of fake imgs
    p.add_argument('--use_bbox_real', action='store_true')  # crop reals by bbox (FID)
    p.add_argument('--cub_root', type=str, default=None)  # CUB root (images.txt etc.)
    return p.parse_args()  # parse


def main():  # dispatch
    args = parse_args()  # parse args
    if args.task == 'sample':  # sampling
        task_sample(args)  # run
    elif args.task == 'fid':  # fid/kid/pr
        task_fid(args)  # run
    elif args.task == 'two_sample':  # 2-sample
        task_two_sample(args)  # run
    else:  # unknown
        raise SystemExit(f'Unknown task {args.task}')  # error


if __name__ == '__main__':  # entry
    main()  # run

