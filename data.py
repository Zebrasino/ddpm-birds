# data.py
# CUB-200-2011 loaders with bbox cropping and simple transforms.
# Every line commented.

from typing import Tuple, Dict, List, Optional                        # typing
import os                                                             # filesystem
from PIL import Image                                                 # image IO
import torch                                                          # torch
from torch.utils.data import Dataset, DataLoader                      # dataset
from torchvision import transforms as T                               # aug/resize

# ----------------------------
# CUB metadata readers
# ----------------------------

def _read_images_txt(root: str) -> Dict[int, str]:
    # Parse images.txt -> {image_id: relative_path}
    p = os.path.join(root, "images.txt")                              # file path
    table = {}                                                        # id->path
    with open(p, "r") as f:                                           # open file
        for line in f:                                                # iterate lines
            idx, rel = line.strip().split()                           # "1 001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg"
            table[int(idx)] = rel                                     # store
    return table                                                      # mapping

def _read_bboxes_txt(root: str) -> Dict[int, Tuple[float, float, float, float]]:
    # Parse bounding_boxes.txt -> {image_id: (x,y,w,h)}
    p = os.path.join(root, "bounding_boxes.txt")                      # file path
    table = {}                                                        # id->bbox
    with open(p, "r") as f:                                           # open
        for line in f:                                                # iterate
            i, x, y, w, h = line.strip().split()                      # tokens
            table[int(i)] = (float(x), float(y), float(w), float(h))  # store
    return table                                                      # mapping

def _read_image_class_labels(root: str) -> Dict[int, int]:
    # Parse image_class_labels.txt -> {image_id: class_id}
    p = os.path.join(root, "image_class_labels.txt")                  # file path
    table = {}                                                        # mapping
    with open(p, "r") as f:                                           # open
        for line in f:                                                # iterate
            i, c = line.strip().split()                               # tokens
            table[int(i)] = int(c) - 1                                # 0-based class
    return table                                                      # mapping

# ----------------------------
# Dataset with bbox crop
# ----------------------------

class CUBBBoxDataset(Dataset):
    # Reads CUB images, crops to bbox (optional expand), applies transforms.
    def __init__(self, root: str, transform, expand: float = 1.0,
                 class_limit: Optional[int] = None, subset: Optional[int] = None):
        super().__init__()                                            # init
        self.root = root                                              # dataset root
        self.transform = transform                                    # torchvision transform
        self.expand = expand                                          # bbox expansion factor
        self.images = _read_images_txt(root)                          # id->relative path
        self.bboxes = _read_bboxes_txt(root)                          # id->bbox
        self.labels = _read_image_class_labels(root)                  # id->class (0..199)

        # Optionally filter classes (take first N by id)
        if class_limit is not None:                                   # limit classes
            keep = set(range(class_limit))                            # {0..class_limit-1}
            self.items = [i for i in sorted(self.images.keys())
                          if self.labels[i] in keep]                  # filter by class
        else:
            self.items = sorted(self.images.keys())                   # all ids

        # Optionally reduce to a fixed subset size
        if subset is not None:                                        # if subset requested
            self.items = self.items[:subset]                          # take first K

    def __len__(self):
        return len(self.items)                                        # dataset length

    def __getitem__(self, idx: int):
        img_id = self.items[idx]                                      # integer id
        rel = self.images[img_id]                                     # relative path
        path = os.path.join(self.root, "images", rel)                 # full path
        img = Image.open(path).convert("RGB")                         # load rgb

        # Get bbox and crop (expand around center)
        x, y, w, h = self.bboxes[img_id]                              # bbox floats
        cx, cy = x + w/2, y + h/2                                     # center
        w2, h2 = w*self.expand, h*self.expand                         # expanded size
        # compute integer box ensuring image bounds
        left   = max(0, int(cx - w2/2))
        top    = max(0, int(cy - h2/2))
        right  = min(img.width, int(cx + w2/2))
        bottom = min(img.height, int(cy + h2/2))
        img = img.crop((left, top, right, bottom))                    # crop to box

        x = self.transform(img)                                       # apply transforms
        y = self.labels[img_id]                                       # class label int
        return x, y                                                   # sample

# ----------------------------
# Builders
# ----------------------------

def make_transforms(img_size: int):
    # Deterministic transforms: resize → center-crop → normalize to [-1,1].
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize((0.5,)*3, (0.5,)*3),
    ])

def make_cub_bbox_dataset(root: str, transform, expand: float,
                          class_limit: Optional[int], subset: Optional[int]):
    # Helper to build dataset + number of classes kept.
    ds = CUBBBoxDataset(root, transform, expand=expand,
                        class_limit=class_limit, subset=subset)
    if class_limit is None:
        num_classes = 200                                            # full CUB
    else:
        num_classes = class_limit                                    # restricted
    return ds, num_classes

def make_loader(ds: Dataset, batch_size: int, shuffle: bool = True, num_workers: int = 2):
    # Default small num_workers works on Colab/T4 without dataloader crashes.
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=True)

