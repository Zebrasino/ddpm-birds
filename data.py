
import os
from typing import Tuple, List, Dict, Optional
from PIL import Image
from torchvision import datasets, transforms

# ---------- CUB helper readers ----------

def _read_images_txt(root: str) -> Dict[int, str]:
    # CUB images.txt lines look like:
    #   "<img_id> <rel_path_inside_images/>"
    # Example:
    #   1 001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg
    # NOTE: this path is RELATIVE to the "images/" directory (no "images/" prefix here).
    mp = {}
    with open(os.path.join(root, "images.txt"), "r") as f:
        for line in f:
            i, p = line.strip().split(" ", 1)
            mp[int(i)] = p
    return mp

def _read_bboxes_txt(root: str) -> Dict[int, Tuple[float, float, float, float]]:
    # lines: "<img_id> <x> <y> <w> <h>"  (1-based coords in CUB)
    mp = {}
    with open(os.path.join(root, "bounding_boxes.txt"), "r") as f:
        for line in f:
            i, x, y, w, h = line.strip().split()
            mp[int(i)] = (float(x) - 1.0, float(y) - 1.0, float(w), float(h))
    return mp

class CUBBBoxDataset:
    """
    Use CUB GT bounding boxes to crop images before transforms.
    root = dataset root containing: images/, images.txt, bounding_boxes.txt
    """
    def __init__(
        self,
        root: str,
        transform,
        expand: float = 1.2,
        subset: Optional[int] = None,
        class_limit: Optional[int] = None
    ):
        self.root = root
        self.transform = transform
        self.expand = float(expand)
        self.images = _read_images_txt(root)     # id -> rel path INSIDE images/
        self.bboxes = _read_bboxes_txt(root)     # id -> (x,y,w,h)

        # Class names are the FIRST path component (no "images/" here)
        # e.g. "001.Black_footed_Albatross/Black_Footed_....jpg" -> class = "001.Black_footed_Albatross"
        classes = sorted({p.split("/")[0] for p in self.images.values()})
        if class_limit is not None:
            classes = classes[:int(class_limit)]
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        self.samples: List[Tuple[str, int, int]] = []  # (abs_path, label, img_id)
        for img_id, rel in self.images.items():
            cls = rel.split("/")[0]
            if cls not in self.class_to_idx:
                continue
            label = self.class_to_idx[cls]
            # IMPORTANT: join root + "images" + relative path from images.txt
            abs_path = os.path.join(root, "images", rel)
            self.samples.append((abs_path, label, img_id))

        if subset is not None:
            self.samples = self.samples[:int(subset)]

    def __len__(self) -> int:
        return len(self.samples)

    def _crop_with_bbox(self, img: Image.Image, bbox) -> Image.Image:
        W, H = img.size
        x, y, w, h = bbox
        cx, cy = x + w / 2.0, y + h / 2.0
        side = max(w * self.expand, h * self.expand)
        nx1 = max(0, int(cx - side / 2.0))
        ny1 = max(0, int(cy - side / 2.0))
        nx2 = min(W, int(cx + side / 2.0))
        ny2 = min(H, int(cy + side / 2.0))
        if (nx2 - nx1) < 8 or (ny2 - ny1) < 8:
            return img
        return img.crop((nx1, ny1, nx2, ny2))

    def __getitem__(self, idx: int):
        path, label, img_id = self.samples[idx]
        img = Image.open(path).convert("RGB")
        bbox = self.bboxes.get(img_id, None)
        if bbox is not None:
            img = self._crop_with_bbox(img, bbox)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

def make_cub_bbox_dataset(root: str, transform, expand: float = 1.2, subset: Optional[int] = None, class_limit: Optional[int] = None):
    ds = CUBBBoxDataset(root, transform, expand=expand, subset=subset, class_limit=class_limit)
    num_classes = len(ds.class_to_idx)
    return ds, num_classes

# --- Optional: plain ImageFolder for non-CUB folders (root must be .../images) ---
def make_imagefolder_dataset(root: str, transform, subset: Optional[int] = None, class_limit: Optional[int] = None):
    ds = datasets.ImageFolder(root=root, transform=transform)
    if class_limit is not None:
        keep_classes = sorted(ds.classes)[:int(class_limit)]
        keep_idx = [ds.class_to_idx[c] for c in keep_classes]
        ds.samples = [s for s in ds.samples if s[1] in keep_idx]
        ds.targets = [s[1] for s in ds.samples]
        ds.classes = keep_classes
    if subset is not None:
        ds.samples = ds.samples[:int(subset)]
        ds.targets = [s[1] for s in ds.samples]
    ds.imgs = ds.samples  # keep imgs alias in sync with samples
    num_classes = len(ds.classes)
    return ds, num_classes



