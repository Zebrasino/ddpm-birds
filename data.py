import os  # filesystem
from typing import Tuple, List, Dict, Optional  # typing helpers
from PIL import Image  # image IO
from torchvision import datasets, transforms  # torchvision datasets & transforms

# ---------- CUB helper readers ----------

def _read_images_txt(root: str) -> Dict[int, str]:  # map: image_id -> relative path
    mp = {}  # initialize map
    with open(os.path.join(root, "images.txt"), "r") as f:  # open annotation
        for line in f:  # for each line
            i, p = line.strip().split(" ", 1)  # split id and path
            mp[int(i)] = p  # store mapping
    return mp  # return map


def _read_bboxes_txt(root: str) -> Dict[int, Tuple[float, float, float, float]]:  # map: image_id -> (x,y,w,h)
    mp = {}  # initialize map
    with open(os.path.join(root, "bounding_boxes.txt"), "r") as f:  # open annotation
        for line in f:  # for each line
            i, x, y, w, h = line.strip().split()  # parse 5 columns
            mp[int(i)] = (float(x) - 1.0, float(y) - 1.0, float(w), float(h))  # CUB 1-based → shift to 0-based
    return mp  # return map


class CUBBBoxDataset:  # dataset that crops by GT bbox with margin
    def __init__(self, root: str, transform, expand: float = 1.2, subset: Optional[int] = None, class_limit: Optional[int] = None):  # ctor
        self.root = root  # dataset root (folder with images.txt, images/, ...)
        self.transform = transform  # transforms pipeline
        self.expand = float(expand)  # bbox expansion factor
        self.images = _read_images_txt(root)  # id -> rel path
        self.bboxes = _read_bboxes_txt(root)  # id -> bbox
        classes = sorted({p.split("/")[1] for p in self.images.values()})  # unique class names
        if class_limit is not None:  # optionally limit classes
            classes = classes[:int(class_limit)]  # take first K classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}  # class → index
        self.samples: List[Tuple[str, int, int]] = []  # list of (abs_path, label, img_id)
        for img_id, rel in self.images.items():  # iterate all images
            cls = rel.split("/")[1]  # class name = folder
            if cls not in self.class_to_idx:  # skip if outside class_limit
                continue  # ignore
            label = self.class_to_idx[cls]  # label id
            self.samples.append((os.path.join(root, rel), label, img_id))  # store triplet
        if subset is not None:  # optional subset for quick overfit
            self.samples = self.samples[:int(subset)]  # cut list

    def __len__(self) -> int:  # dataset size
        return len(self.samples)  # number of samples

    def _crop_with_bbox(self, img: Image.Image, bbox) -> Image.Image:  # crop around bbox with margin
        W, H = img.size  # original size
        x, y, w, h = bbox  # bbox fields
        cx, cy = x + w / 2.0, y + h / 2.0  # center
        side = max(w * self.expand, h * self.expand)  # square around bbox (expanded)
        nx1 = max(0, int(cx - side / 2.0))  # left
        ny1 = max(0, int(cy - side / 2.0))  # top
        nx2 = min(W, int(cx + side / 2.0))  # right
        ny2 = min(H, int(cy + side / 2.0))  # bottom
        if (nx2 - nx1) < 8 or (ny2 - ny1) < 8:  # avoid degenerate crops
            return img  # fallback to full image
        return img.crop((nx1, ny1, nx2, ny2))  # return crop

    def __getitem__(self, idx: int):  # get sample
        path, label, img_id = self.samples[idx]  # unpack sample
        img = Image.open(path).convert("RGB")  # read rgb image
        bbox = self.bboxes.get(img_id, None)  # fetch bbox
        if bbox is not None:  # if exists
            img = self._crop_with_bbox(img, bbox)  # crop by bbox
        if self.transform is not None:  # apply transforms
            img = self.transform(img)  # transform
        return img, label  # return (tensor, class)


def make_cub_bbox_dataset(root: str, transform, expand: float = 1.2, subset: Optional[int] = None, class_limit: Optional[int] = None):  # CUB bbox builder
    ds = CUBBBoxDataset(root, transform, expand=expand, subset=subset, class_limit=class_limit)  # create dataset
    num_classes = len(ds.class_to_idx)  # count classes
    return ds, num_classes  # return dataset & num_classes


def make_imagefolder_dataset(root: str, transform, subset: Optional[int] = None, class_limit: Optional[int] = None):  # vanilla ImageFolder
    ds = datasets.ImageFolder(root=root, transform=transform)  # load folder
    if class_limit is not None:  # optionally limit classes
        keep_classes = sorted(ds.classes)[:int(class_limit)]  # top-K
        keep_idx = [ds.class_to_idx[c] for c in keep_classes]  # indices
        ds.samples = [s for s in ds.samples if s[1] in keep_idx]  # filter samples
        ds.targets = [s[1] for s in ds.samples]  # rebuild targets
        ds.classes = keep_classes  # shrink classes list
    if subset is not None:  # limit number of samples
        ds.samples = ds.samples[:int(subset)]  # cut
        ds.targets = [s[1] for s in ds.samples]  # rebuild targets
    num_classes = len(ds.classes)  # class count
    return ds, num_classes  # return dataset & num_classes

