# data.py
# CUB-200-2011 loader with bounding-box crops and a foreground mask.
# Returns (image_tensor, label or None, fg_mask) so the loss can upweight the bird.
# Every line is commented for clarity.

import os
from typing import Optional, Tuple, List

import torch
from torch.utils.data import Dataset
from PIL import Image


def _read_indexed_txt(path: str) -> List[Tuple[int, str]]:
    """Read files like images.txt: 'idx path' per line → list of (idx, value)."""
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx_str, val = line.split(" ", 1)
            out.append((int(idx_str), val))
    return out


def _load_cub_lists(root: str):
    """Load CUB protocol files needed for bbox and class ids."""
    images = _read_indexed_txt(os.path.join(root, "images.txt"))           # [(id, relpath)]
    bboxes = _read_indexed_txt(os.path.join(root, "bounding_boxes.txt"))   # [(id, 'x y w h')]
    labels = _read_indexed_txt(os.path.join(root, "image_class_labels.txt"))  # [(id, cls)]

    # build id→path, id→(x,y,w,h), id→class
    id_to_path = {i: os.path.join(root, "images", p) for i, p in images}
    id_to_bbox = {}
    for i, s in bboxes:
        x, y, w, h = [float(v) for v in s.split()]
        id_to_bbox[i] = (x, y, w, h)
    id_to_cls = {i: int(c) - 1 for i, c in labels}  # to 0-based

    return id_to_path, id_to_bbox, id_to_cls


class CUBBBoxDataset(Dataset):
    """CUB dataset that crops around bbox and returns a foreground mask.

    root points to the CUB_200_2011 folder containing:
      - images/...
      - images.txt, bounding_boxes.txt, image_class_labels.txt
    """

    def __init__(
        self,
        root: str,                            # /content/CUB_200_2011
        img_size: int = 48,                  # output resolution
        use_bbox: bool = True,               # crop around bbox if True
        bbox_expand: float = 1.0,            # expand/shrink bbox (1.0 = exact)
        class_limit: Optional[int] = None,   # keep only first N classes (optional)
        subset: Optional[int] = None,        # keep only first K images (optional)
        normalize: bool = True,              # map to [-1,1]
    ):
        super().__init__()
        self.root = root
        self.img_size = int(img_size)
        self.use_bbox = bool(use_bbox)
        self.bbox_expand = float(bbox_expand)
        self.normalize = bool(normalize)

        id_to_path, id_to_bbox, id_to_cls = _load_cub_lists(root)   # protocol dicts

        # build list of ids obeying filters
        ids = sorted(id_to_path.keys())
        if class_limit is not None:
            # keep ids with class in [0..class_limit-1]
            ids = [i for i in ids if id_to_cls[i] < int(class_limit)]
        if subset is not None:
            ids = ids[: int(subset)]

        self.items = [(i, id_to_path[i], id_to_bbox.get(i, None), id_to_cls[i]) for i in ids]

    def __len__(self) -> int:
        return len(self.items)

    def _crop_bbox(self, img: Image.Image, bbox_xywh):
        """Crop around an expanded bbox, clamped to image bounds."""
        if bbox_xywh is None:
            return img                                   # no bbox → return whole image
        W, H = img.size
        x, y, w, h = bbox_xywh
        # center + expanded box
        cx, cy = x + w / 2.0, y + h / 2.0
        scale = self.bbox_expand
        w2, h2 = w * scale / 2.0, h * scale / 2.0
        x0, y0 = max(0, int(cx - w2)), max(0, int(cy - h2))
        x1, y1 = min(W, int(cx + w2)), min(H, int(cy + h2))
        if x1 <= x0 or y1 <= y0:                         # degenerate → full image
            return img
        return img.crop((x0, y0, x1, y1))

    def __getitem__(self, idx: int):
        img_id, path, bbox, cls = self.items[idx]        # unpack index entry

        # load PIL (RGB)
        img = Image.open(path).convert("RGB")            # image at original size

        # (1) crop around bbox (if enabled)
        if self.use_bbox:
            img_c = self._crop_bbox(img, bbox)
        else:
            img_c = img

        # we also create a foreground mask (1 inside bbox, 0 outside) AFTER resize
        # build mask in original coords, then apply same crop+resize
        if self.use_bbox and bbox is not None:
            W, H = img.size
            x, y, w, h = bbox
            # expanded bbox rectangle (same as crop)
            cx, cy = x + w / 2.0, y + h / 2.0
            scale = self.bbox_expand
            w2, h2 = w * scale / 2.0, h * scale / 2.0
            x0, y0 = max(0, int(cx - w2)), max(0, int(cy - h2))
            x1, y1 = min(W, int(cx + w2)), min(H, int(cy + h2))

            # make a binary mask on the cropped window
            import numpy as np
            mask = Image.new("L", (img_c.size[0], img_c.size[1]), 0)
            # bbox in the cropped frame (shift by x0,y0)
            bx0, by0 = max(0, int(x - x0)), max(0, int(y - y0))
            bx1, by1 = min(img_c.size[0], int(x + w - x0)), min(img_c.size[1], int(y + h - y0))
            if bx1 > bx0 and by1 > by0:
                from PIL import ImageDraw
                draw = ImageDraw.Draw(mask)
                draw.rectangle([bx0, by0, bx1, by1], fill=255)
            # resize image and mask to model size
            img_c = img_c.resize((self.img_size, self.img_size), resample=Image.BICUBIC)
            mask = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)
            # to tensors
            x = torch.from_numpy((torch.ByteTensor(torch.ByteStorage.from_buffer(img_c.tobytes()))
                                  .view(self.img_size, self.img_size, 3)
                                  .numpy().astype("float32") / 255.0)).permute(2, 0, 1)
            m = torch.from_numpy(
                (torch.ByteTensor(torch.ByteStorage.from_buffer(mask.tobytes()))
                 .view(self.img_size, self.img_size)
                 .numpy().astype("float32") / 255.0)
            ).unsqueeze(0)
        else:
            # simple resize (no bbox)
            img_c = img_c.resize((self.img_size, self.img_size), resample=Image.BICUBIC)
            import numpy as np
            x = torch.from_numpy((torch.ByteTensor(torch.ByteStorage.from_buffer(img_c.tobytes()))
                                  .view(self.img_size, self.img_size, 3)
                                  .numpy().astype("float32") / 255.0)).permute(2, 0, 1)
            m = torch.ones(1, self.img_size, self.img_size, dtype=torch.float32)

        # normalize to [-1, 1]
        if self.normalize:
            x = x * 2.0 - 1.0

        return x, cls, m                                   # image, label (int), mask


def make_cub_bbox_dataset(
    root: str,
    img_size: int,
    use_bbox: bool,
    bbox_expand: float,
    class_limit: Optional[int],
    subset: Optional[int],
):
    """Factory that returns the dataset configured for training/eval."""
    ds = CUBBBoxDataset(
        root=root,
        img_size=img_size,
        use_bbox=use_bbox,
        bbox_expand=bbox_expand,
        class_limit=class_limit,
        subset=subset,
        normalize=True,
    )
    return ds


