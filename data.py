from __future__ import annotations                # future annotations
import os                                         # filesystem ops
from typing import List, Tuple, Optional          # typing helpers
import math                                       # math utilities
import numpy as np                                # arrays
from PIL import Image                             # image I/O
import torch                                      # tensors
from torch.utils.data import Dataset              # PyTorch dataset
import torchvision.transforms as T                # transforms
import torchvision.transforms.functional as TF    # functional transforms

# -----------------------------
# Small helpers to read CUB txt
# -----------------------------
def _read_lines(path: str) -> List[str]:
    """Read a text file and return non-empty stripped lines."""
    with open(path, "r") as f:                    # open file
        lines = [ln.strip() for ln in f.readlines()]  # strip newline
    return [ln for ln in lines if ln]             # drop empties

def _load_cub_index(root: str):
    """
    Parse CUB-200-2011 annotation files.
    Returns dicts indexed by image_id (int starting at 1).
    """
    # Paths to official txt files shipped with the dataset
    p_images   = os.path.join(root, "images.txt")                # id -> relative path
    p_labels   = os.path.join(root, "image_class_labels.txt")    # id -> class id
    p_bbox     = os.path.join(root, "bounding_boxes.txt")        # id -> bbox (x,y,w,h)
    p_split    = os.path.join(root, "train_test_split.txt")      # id -> train(1)/test(0)

    # Read all lines
    lines_images = _read_lines(p_images)                         # e.g. "1 001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg"
    lines_labels = _read_lines(p_labels)                         # e.g. "1 1"
    lines_bbox   = _read_lines(p_bbox)                           # e.g. "1 101 67 221 167"
    lines_split  = _read_lines(p_split)                          # e.g. "1 1"

    # Parse to dicts keyed by image id
    id2path = {}                                                 # image id -> relative path under images/
    id2label = {}                                                # image id -> class index (1..200)
    id2bbox = {}                                                 # image id -> (x,y,w,h) in original image coords
    id2train = {}                                                # image id -> is_train (bool)

    for ln in lines_images:                                      # iterate entries
        idx, rel = ln.split(" ", 1)                              # split id/path
        id2path[int(idx)] = rel                                  # store mapping

    for ln in lines_labels:                                      # iterate labels
        idx, lab = ln.split(" ", 1)                              # id / class id (1..200)
        id2label[int(idx)] = int(lab) - 1                        # convert to 0-based

    for ln in lines_bbox:                                        # iterate bboxes
        a = ln.split()                                           # id x y w h
        idx = int(a[0])                                          # image id
        x, y, w, h = map(float, a[1:])                           # bbox floats
        id2bbox[idx] = (x, y, w, h)                              # store

    for ln in lines_split:                                       # iterate split
        idx, is_tr = ln.split(" ", 1)                            # id / 0|1
        id2train[int(idx)] = (int(is_tr) == 1)                   # bool

    return id2path, id2label, id2bbox, id2train                  # return dicts

# -------------------------------------
# Foreground mask from a bbox rectangle
# -------------------------------------
def _bbox_mask(h: int, w: int, x0: int, y0: int, x1: int, y1: int) -> torch.Tensor:
    """
    Build a binary mask 1 inside the (x0,y0,x1,y1) rectangle, 0 outside, shape (1,H,W).
    Coordinates are clamped to [0,w]x[0,h].
    """
    # Clamp bounds to image
    x0 = max(0, min(w, x0))                                      # clamp x0
    y0 = max(0, min(h, y0))                                      # clamp y0
    x1 = max(0, min(w, x1))                                      # clamp x1
    y1 = max(0, min(h, y1))                                      # clamp y1

    # Create zeros then fill the rectangle with ones
    m = torch.zeros((1, h, w), dtype=torch.float32)              # (1,H,W) zeros
    m[:, y0:y1, x0:x1] = 1.0                                     # set rectangle = 1
    return m                                                     # binary mask

# ---------------------------
# The CUB dataset class
# ---------------------------
class CUBBBoxDataset(Dataset):
    """
    CUB dataset with optional bbox crop and mask.
    Yields tuples: (image_tensor, class_label, fg_mask_tensor).
    """
    def __init__(
        self,
        root: str,                         # CUB_200_2011 directory (contains images/ + *.txt)
        img_size: int = 64,                # final square size after transforms
        use_bbox: bool = True,             # whether to crop around bbox
        bbox_expand: float = 1.0,          # bbox expansion factor (>=1.0)
        class_limit: Optional[int] = None, # limit number of classes (min 1)
        subset: Optional[int] = None,      # limit number of images overall
        train_only: bool = True,           # use only training split
    ):
        super().__init__()                                              # init dataset
        self.root = root                                                # store root
        self.img_size = img_size                                        # target size
        self.use_bbox = use_bbox                                        # store flag
        self.bbox_expand = bbox_expand                                  # expansion factor

        # Load mapping dicts from txts
        id2path, id2label, id2bbox, id2train = _load_cub_index(root)    # parse metadata

        # Build candidate image ids respecting train/test split
        ids = [i for i in id2path.keys() if (id2train[i] if train_only else True)]  # keep train if requested

        # If class_limit is set, keep only those images whose label < class_limit
        if class_limit is not None:                                      # if limiting classes
            ids = [i for i in ids if id2label[i] < class_limit]          # filter ids

        # Sort ids for reproducibility
        ids.sort()                                                       # stable order

        # If subset is set, keep only the first subset images
        if subset is not None:                                           # if limiting images
            ids = ids[:int(subset)]                                      # slice list

        # Store resolved lists for fast indexing
        self.ids = ids                                                   # final id list
        self.id2path = id2path                                           # mappings
        self.id2label = id2label
        self.id2bbox = id2bbox

        # Number of classes visible to the dataset (for class-conditional training)
        self.num_classes = (class_limit if class_limit is not None else 200)  # default full 200

        # Define base transforms (normalize to [-1,1])
        self.to_tensor = T.Compose([                                     # compose transforms
            T.ToTensor(),                                                # (H,W,C)[0..1] -> (C,H,W)
            T.Normalize((0.5,)*3, (0.5,)*3),                             # to [-1,1]
        ])

    def __len__(self) -> int:
        """Return number of available images."""
        return len(self.ids)                                             # dataset length

    def __getitem__(self, idx: int):
        """
        Read one sample and return:
        - x:     image tensor (3,H,W) normalized to [-1,1]
        - y:     class label (int)
        - fg:    foreground mask (1,H,W) in {0,1} (after resize to img_size)
        """
        # Resolve image id
        img_id = self.ids[idx]                                           # image id
        rel = self.id2path[img_id]                                       # relative path under images/
        y = self.id2label[img_id]                                        # 0-based class index

        # Load RGB image from disk
        p_img = os.path.join(self.root, "images", rel)                   # absolute path
        with Image.open(p_img) as im:                                    # open file
            im = im.convert("RGB")                                       # ensure RGB

            # Optional bbox-based crop (expand around center)
            if self.use_bbox:                                            # if cropping enabled
                x, y0, w, h = self.id2bbox[img_id]                       # original bbox floats
                cx = x + w/2.0                                           # bbox center x
                cy = y0 + h/2.0                                          # bbox center y
                s  = max(w, h) * self.bbox_expand                        # square side after expansion
                # Build square crop corners
                left   = int(round(cx - s/2.0))                          # left x
                top    = int(round(cy - s/2.0))                          # top y
                right  = int(round(cx + s/2.0))                          # right x
                bottom = int(round(cy + s/2.0))                          # bottom y
                # Pad the image if the crop goes out-of-bounds
                pad_l = max(0, -left)                                    # left pad
                pad_t = max(0, -top)                                     # top pad
                pad_r = max(0, right  - im.width)                        # right pad
                pad_b = max(0, bottom - im.height)                       # bottom pad
                if pad_l or pad_t or pad_r or pad_b:                     # if any pad needed
                    im = TF.pad(im, (pad_l, pad_t, pad_r, pad_b), fill=0)# zero-pad
                    left  += pad_l                                       # shift crop
                    top   += pad_t
                    right += pad_l
                    bottom+= pad_t
                # Finally crop to the square
                im = im.crop((left, top, right, bottom))                 # perform crop

                # Foreground mask = all ones after bbox crop (whole patch is bird-focused)
                fg = torch.ones(1, im.height, im.width, dtype=torch.float32)  # (1,H,W) ones
            else:
                # If we do not crop, we can build a rectangular mask aligned to the original image
                x, y0, w, h = self.id2bbox[img_id]                       # bbox coords
                # Convert to integer box corners
                x0 = int(round(x))
                y1 = int(round(y0 + h))
                x1 = int(round(x + w))
                y0 = int(round(y0))
                fg = _bbox_mask(im.height, im.width, x0, y0, x1, y1)     # rectangle mask

            # Resize image and mask to target squared size (bicubic for image, nearest for mask)
            im = im.resize((self.img_size, self.img_size), Image.BICUBIC)     # image resize
            fg = TF.resize(fg, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.NEAREST)  # mask resize

            # Convert to normalized tensor
            x = self.to_tensor(im)                                       # [-1,1] tensor
            fg = fg.clamp(0, 1)                                          # safety clamp

        return x, y, fg                                                  # return triplet


# ---------------------------
# Factory function
# ---------------------------
def make_cub_bbox_dataset(
    root: str,
    img_size: int,
    use_bbox: bool = True,
    bbox_expand: float = 1.0,
    class_limit: Optional[int] = None,
    subset: Optional[int] = None,
    train_only: bool = True,
) -> CUBBBoxDataset:
    """
    Build and return a CUBBBoxDataset instance configured with the given options.
    """
    return CUBBBoxDataset(
        root=root,
        img_size=img_size,
        use_bbox=use_bbox,
        bbox_expand=bbox_expand,
        class_limit=class_limit,
        subset=subset,
        train_only=train_only,
    )
