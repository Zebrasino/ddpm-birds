import os                                      # filesystem
from typing import Tuple, List, Dict, Optional # typing
from PIL import Image                          # image IO
from torchvision import datasets, transforms   # torchvision datasets/transforms

# ---------- CUB helper readers ----------

def _read_images_txt(root: str) -> Dict[int, str]:
    """Parse CUB images.txt → dict {img_id: relative_path_inside_images/}."""
    mp = {}                                                                          # accumulator
    with open(os.path.join(root, "images.txt"), "r") as f:                           # open file
        for line in f:                                                               # iterate lines
            i, p = line.strip().split(" ", 1)                                        # split id + path (no 'images/' prefix)
            mp[int(i)] = p                                                           # store mapping
    return mp                                                                        # return dict

def _read_bboxes_txt(root: str) -> Dict[int, Tuple[float, float, float, float]]:
    """Parse CUB bounding_boxes.txt → dict {img_id: (x, y, w, h)}; convert to 0-based."""
    mp = {}                                                                          # accumulator
    with open(os.path.join(root, "bounding_boxes.txt"), "r") as f:                   # open file
        for line in f:                                                               # iterate lines
            i, x, y, w, h = line.strip().split()                                     # tokens
            mp[int(i)] = (float(x) - 1.0, float(y) - 1.0, float(w), float(h))        # 0-based coords
    return mp                                                                        # return dict

class CUBBBoxDataset:
    """
    Dataset that crops each CUB image to its **ground-truth bounding box** (no extra expansion by default).
    The dataset root must contain: images/ , images.txt , bounding_boxes.txt
    """
    def __init__(
        self,
        root: str,
        transform,                               # torchvision transform pipeline
        expand: float = 1.0,                     # **1.0 = exact bbox, no margin (per request)**
        subset: Optional[int] = None,            # take only first N images
        class_limit: Optional[int] = None        # keep only first K classes
    ):
        self.root = root                         # dataset root
        self.transform = transform               # transform pipeline
        self.expand = float(expand)              # expansion factor around bbox
        self.images = _read_images_txt(root)     # id -> relative path (inside images/)
        self.bboxes = _read_bboxes_txt(root)     # id -> bbox tuple

        # Build class list from first path component (e.g., "001.Black_footed_Albatross/...")
        classes = sorted({p.split("/")[0] for p in self.images.values()})            # unique class names
        if class_limit is not None:                                                  # optionally limit classes
            classes = classes[:int(class_limit)]
        self.class_to_idx = {c: i for i, c in enumerate(classes)}                    # mapping name → label idx

        # Build sample list: (absolute_path, label_idx, img_id)
        self.samples: List[Tuple[str, int, int]] = []
        for img_id, rel in self.images.items():                                      # iterate all images
            cls = rel.split("/")[0]                                                  # class name
            if cls not in self.class_to_idx:                                         # skip if not selected
                continue
            label = self.class_to_idx[cls]                                           # label id
            abs_path = os.path.join(root, "images", rel)                             # **join root + "images" + rel**
            self.samples.append((abs_path, label, img_id))                           # store sample tuple

        if subset is not None:                                                       # limit dataset length
            self.samples = self.samples[:int(subset)]

    def __len__(self) -> int:
        return len(self.samples)                                                     # dataset size

    def _crop_with_bbox(self, img: Image.Image, bbox) -> Image.Image:
        """Crop PIL image to bbox, optionally expanding by self.expand (1.0 → exact box)."""
        W, H = img.size                                                              # image size
        x, y, w, h = bbox                                                            # bbox fields
        cx, cy = x + w / 2.0, y + h / 2.0                                            # bbox center
        # square side after expansion (max of w,h) — with expand=1.0 this equals the bbox size
        side = max(w * self.expand, h * self.expand)
        nx1 = max(0, int(cx - side / 2.0))                                           # left
        ny1 = max(0, int(cy - side / 2.0))                                           # top
        nx2 = min(W, int(cx + side / 2.0))                                           # right
        ny2 = min(H, int(cy + side / 2.0))                                           # bottom
        if (nx2 - nx1) < 8 or (ny2 - ny1) < 8:                                       # avoid degenerate crops
            return img
        return img.crop((nx1, ny1, nx2, ny2))                                        # crop and return

    def __getitem__(self, idx: int):
        path, label, img_id = self.samples[idx]                                      # sample tuple
        img = Image.open(path).convert("RGB")                                        # load RGB
        bbox = self.bboxes.get(img_id, None)                                         # lookup bbox
        if bbox is not None:                                                         # crop if available
            img = self._crop_with_bbox(img, bbox)
        if self.transform is not None:                                               # apply transforms
            img = self.transform(img)
        return img, label                                                            # return (tensor, label)


def make_cub_bbox_dataset(root: str, transform, expand: float = 1.0, subset: Optional[int] = None, class_limit: Optional[int] = None):
    """Factory: create CUB dataset with bbox cropping; returns (dataset, num_classes)."""
    ds = CUBBBoxDataset(root, transform, expand=expand, subset=subset, class_limit=class_limit)
    num_classes = len(ds.class_to_idx)
    return ds, num_classes


# --- Optional: plain ImageFolder for non-CUB folders (root must point to class subfolders) ---
def make_imagefolder_dataset(root: str, transform, subset: Optional[int] = None, class_limit: Optional[int] = None):
    """Factory: torchvision ImageFolder with optional class/size subset."""
    ds = datasets.ImageFolder(root=root, transform=transform)                         # standard folder dataset
    if class_limit is not None:                                                       # keep only first K classes
        keep_classes = sorted(ds.classes)[:int(class_limit)]
        keep_idx = [ds.class_to_idx[c] for c in keep_classes]
        ds.samples = [s for s in ds.samples if s[1] in keep_idx]                      # filter samples
        ds.targets = [s[1] for s in ds.samples]                                       # update targets
        ds.classes = keep_classes                                                     # update class names
    if subset is not None:                                                            # truncate dataset
        ds.samples = ds.samples[:int(subset)]
        ds.targets = [s[1] for s in ds.samples]
    ds.imgs = ds.samples                                                              # keep alias consistent
    num_classes = len(ds.classes)
    return ds, num_classes                                                            # dataset + number of classes




