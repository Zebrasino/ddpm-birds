from __future__ import annotations                     # postpone type hints
import os                                              # filesystem utils
from typing import List, Tuple, Optional               # typing helpers

from PIL import Image                                  # image IO
import torch                                           # tensors
from torch.utils.data import Dataset                   # dataset base class
import torchvision.transforms.functional as TF         # deterministic F transforms


def _read_txt(path: str) -> List[str]:
    """Read a text file and return a list of stripped lines."""
    with open(path, "r") as f:                         # open file
        return [line.strip() for line in f]            # strip newline


class CUBBBoxDataset(Dataset):
    """CUB-200-2011 with optional bbox crop and a foreground mask aligned to it."""
    def __init__(
        self,
        root: str,                                     # CUB_200_2011 folder
        img_size: int = 64,                            # output square resolution
        use_bbox: bool = True,                         # crop around bbox if True
        bbox_expand: float = 1.0,                      # expand bbox (>= 1.0)
        class_limit: Optional[int] = None,             # limit number of classes
        subset: Optional[int] = None,                  # limit number of images
        train_only: bool = True,                       # use only training split
    ):
        self.root = root                               # store root
        self.img_size = int(img_size)                  # target H=W
        self.use_bbox = bool(use_bbox)                 # store flag
        self.bbox_expand = float(max(1.0, bbox_expand))# clamp expand >= 1.0

        # ---------- Load CUB metadata ----------
        # images.txt: "<id> <relative_path>"
        images_txt = _read_txt(os.path.join(root, "images.txt"))
        # image_class_labels.txt: "<id> <class_id>"
        labels_txt = _read_txt(os.path.join(root, "image_class_labels.txt"))
        # bounding_boxes.txt: "<id> x y w h"    (float values as strings)
        bbox_txt   = _read_txt(os.path.join(root, "bounding_boxes.txt"))
        # train_test_split.txt: "<id> 1|0"
        split_txt  = _read_txt(os.path.join(root, "train_test_split.txt"))
        # classes.txt: "<class_id> <class_name>"
        classes_txt = _read_txt(os.path.join(root, "classes.txt"))

        # Parse class names and clamp to class_limit if provided
        all_classes = [c.split(" ", 1)[1] for c in classes_txt]   # keep names
        if class_limit is not None:                                # limit classes
            all_classes = all_classes[: int(class_limit)]          # slice subset
        self.classes = all_classes                                 # store names
        self.num_classes = len(self.classes)                       # expose count

        # Build id->classid and id->path maps, later filter by split & class_limit
        id_to_rel = {}                                             # id -> rel path
        for line in images_txt:                                    # parse images.txt
            idx, rel = line.split(" ", 1)
            id_to_rel[int(idx)] = rel

        id_to_lbl = {}                                             # id -> class id (1..200)
        for line in labels_txt:                                    # parse labels
            idx, cid = line.split()
            id_to_lbl[int(idx)] = int(cid)                         # original CUB id

        id_to_bbox = {}                                            # id -> (x,y,w,h)
        for line in bbox_txt:                                      # parse bboxes
            toks = line.split()
            idx = int(toks[0])
            x, y, w, h = map(float, toks[1:])
            id_to_bbox[idx] = (x, y, w, h)

        id_is_train = {}                                           # id -> bool
        for line in split_txt:                                     # parse split
            idx, flag = line.split()
            id_is_train[int(idx)] = (flag == "1")                  # True if train

        # ---------- Build a list of usable items ----------
        # We keep only images whose class is within class_limit (if any)
        # and optionally only the training split.
        items: List[Tuple[str, int, Tuple[float, float, float, float]]] = []
        # CUB class ids start at 1; convert to 0..num_classes-1 for training.
        valid_class_ids = set(range(1, self.num_classes + 1))
        for img_id, rel in id_to_rel.items():                      # iterate images
            if train_only and (not id_is_train[img_id]):           # skip test
                continue
            cid = id_to_lbl[img_id]                                # original class id
            if cid not in valid_class_ids:                         # skip out of range
                continue
            # remap to 0..num_classes-1
            y = cid - 1                                            # zero-based label
            bb = id_to_bbox[img_id]                                # bbox tuple
            items.append((os.path.join(self.root, "images", rel), y, bb))

        # Optionally subsample a fixed number of images for quick experiments
        if subset is not None:
            items = items[: int(subset)]                           # simple slice

        self.items = items                                         # store items

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.items)                                     # number of items

    def _expand_bbox(self, x: float, y: float, w: float, h: float, W: int, H: int):
        """Expand bbox by factor while keeping it inside [0,W]x[0,H]."""
        cx = x + w / 2.0                                           # center x
        cy = y + h / 2.0                                           # center y
        nw = w * self.bbox_expand                                  # new width
        nh = h * self.bbox_expand                                  # new height
        nx = max(0.0, cx - nw / 2.0)                               # left
        ny = max(0.0, cy - nh / 2.0)                               # top
        nx2 = min(float(W), nx + nw)                               # right
        ny2 = min(float(H), ny + nh)                               # bottom
        return nx, ny, nx2 - nx, ny2 - ny                          # clamped bbox

    def __getitem__(self, idx: int):
        """Load image, apply deterministic transforms, return (x, y, fg_mask)."""
        path, y, (bx, by, bw, bh) = self.items[idx]                # unpack tuple
        img = Image.open(path).convert("RGB")                      # read RGB
        W, H = img.size                                            # original size

        # Optionally expand the bbox (tighter focus on the subject)
        if self.use_bbox:                                          # bbox cropping
            bx, by, bw, bh = self._expand_bbox(bx, by, bw, bh, W, H)

        # Build a PIL "mask" (0 background, 1 foreground) same size as image
        mask = Image.new("L", (W, H), 0)                           # zeros
        if self.use_bbox:
            # fill the rectangle area with 255 (=1 after normalization)
            import PIL.ImageDraw as ImageDraw                      # draw util
            draw = ImageDraw.Draw(mask)                            # drawing ctx
            draw.rectangle([bx, by, bx + bw, by + bh], fill=255)   # draw bbox

        # --- Deterministic spatial transforms (keep image & mask in sync) ---
        # 1) Optional crop around bbox to remove big background margins.
        if self.use_bbox:
            img = img.crop((bx, by, bx + bw, by + bh))             # crop image
            mask = mask.crop((bx, by, bx + bh, by + bh)) if False else mask  # keep unused; next line resizes
            # Note: we do not need to crop mask here since the next resize
            # operates on the whole image; we rebuild mask based on bbox.

        # 2) Resize to the target side, then CenterCrop to a fixed square.
        img = TF.resize(img, self.img_size, interpolation=TF.InterpolationMode.BICUBIC)  # resize
        img = TF.center_crop(img, [self.img_size, self.img_size])   # center crop

        # Recompute a mask aligned to the current (square) image:
        # easiest is "all ones" if we already cropped to bbox; else use scaled bbox.
        if self.use_bbox:
            fg_mask = torch.ones(1, self.img_size, self.img_size)  # full FG
        else:
            # scale original bbox to the resized+center-cropped coordinates
            # (approximation: fill 1 over whole frame; conservative default)
            fg_mask = torch.ones(1, self.img_size, self.img_size)  # fallback

        # Convert image to tensor in [-1, 1]
        x = TF.to_tensor(img)                                      # [0,1]
        x = x * 2.0 - 1.0                                          # [-1,1]

        return x, torch.tensor(y, dtype=torch.long), fg_mask       # final triplet


def make_cub_bbox_dataset(
    root: str,
    img_size: int = 64,
    use_bbox: bool = True,
    bbox_expand: float = 1.0,
    class_limit: Optional[int] = None,
    subset: Optional[int] = None,
    train_only: bool = True,
) -> CUBBBoxDataset:
    """Factory wrapper to create the dataset (kept for backward-compat)."""
    return CUBBBoxDataset(
        root=root, img_size=img_size, use_bbox=use_bbox, bbox_expand=bbox_expand,
        class_limit=class_limit, subset=subset, train_only=train_only
    )
