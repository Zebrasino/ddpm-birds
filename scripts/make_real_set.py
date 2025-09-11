import argparse, os, math, random
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from PIL import Image

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cub-root", type=str, required=True,
                    help="Path to CUB_200_2011 (contains images/, images.txt, etc.)")
    ap.add_argument("--outdir", type=str, required=True,
                    help="Where to write the flat folder of PNGs")
    ap.add_argument("--img-size", type=int, default=64,
                    help="Output square size (e.g., 64 or 128)")
    ap.add_argument("--split", type=str, default="train", choices=["train", "test", "all"],
                    help="Which subset to export")
    ap.add_argument("--use-bbox", action="store_true",
                    help="Crop around bounding box before resizing")
    ap.add_argument("--bbox-expand", type=float, default=1.5,
                    help="Scale factor to expand bbox (1.0 = no expand)")
    ap.add_argument("--num", type=int, default=None,
                    help="Total number of images to export (random sample). If unset, export all in the split.")
    ap.add_argument("--balanced", action="store_true",
                    help="Sample the same number per class (requires --per-class)")
    ap.add_argument("--per-class", type=int, default=None,
                    help="How many images per class when --balanced is set")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed")
    return ap.parse_args()

# ---- Helpers to read CUB metadata -------------------------------------------------------------

def read_index_map(txt_path: Path) -> Dict[int, str]:
    """Read 'images.txt' => {image_id: relative_path}."""
    mp: Dict[int, str] = {}
    with open(txt_path, "r") as f:
        for line in f:
            i, p = line.strip().split()
            mp[int(i)] = p
    return mp

def read_split(txt_path: Path) -> Dict[int, int]:
    """Read 'train_test_split.txt' => {image_id: 1(train)|0(test)}."""
    mp: Dict[int, int] = {}
    with open(txt_path, "r") as f:
        for line in f:
            i, s = line.strip().split()
            mp[int(i)] = int(s)
    return mp

def read_labels(txt_path: Path) -> Dict[int, int]:
    """Read 'image_class_labels.txt' => {image_id: class_id (1..200)}."""
    mp: Dict[int, int] = {}
    with open(txt_path, "r") as f:
        for line in f:
            i, y = line.strip().split()
            mp[int(i)] = int(y)
    return mp

def read_bboxes(txt_path: Path) -> Dict[int, Tuple[float, float, float, float]]:
    """Read 'bounding_boxes.txt' => {image_id: (x, y, w, h)} in pixel coords."""
    mp: Dict[int, Tuple[float, float, float, float]] = {}
    with open(txt_path, "r") as f:
        for line in f:
            i, x, y, w, h = line.strip().split()
            mp[int(i)] = (float(x), float(y), float(w), float(h))
    return mp

# ---- Image ops -------------------------------------------------------------------------------

def expand_and_clip_bbox(x: float, y: float, w: float, h: float,
                         W: int, H: int, scale: float) -> Tuple[int, int, int, int]:
    """Expand bbox by 'scale' around its center and clip to image boundaries."""
    cx = x + w / 2.0
    cy = y + h / 2.0
    w2 = w * scale
    h2 = h * scale
    x0 = int(round(max(0, cx - w2 / 2.0)))
    y0 = int(round(max(0, cy - h2 / 2.0)))
    x1 = int(round(min(W, cx + w2 / 2.0)))
    y1 = int(round(min(H, cy + h2 / 2.0)))
    # Make sure we have at least 1px area
    if x1 <= x0: x1 = min(W, x0 + 1)
    if y1 <= y0: y1 = min(H, y0 + 1)
    return x0, y0, x1, y1

def load_and_prepare(img_path: Path,
                     use_bbox: bool,
                     bbox_expand: float,
                     bbox_map: Optional[Dict[int, Tuple[float,float,float,float]]],
                     img_id: int,
                     out_size: int) -> Image.Image:
    """Load an image, optionally crop around expanded bbox, then resize to (out_size,out_size)."""
    im = Image.open(img_path).convert("RGB")
    if use_bbox and bbox_map is not None and img_id in bbox_map:
        x, y, w, h = bbox_map[img_id]
        W, H = im.size
        x0, y0, x1, y1 = expand_and_clip_bbox(x, y, w, h, W, H, bbox_expand)
        im = im.crop((x0, y0, x1, y1))
    # Resize to square with bicubic (same family used in your pipeline)
    im = im.resize((out_size, out_size), Image.BICUBIC)
    return im

# ---- Sampling logic --------------------------------------------------------------------------

def sample_ids(ids: List[int],
               labels: Dict[int,int],
               balanced: bool,
               per_class: Optional[int],
               total_num: Optional[int],
               seed: int) -> List[int]:
    """Return a list of image ids to export (random or balanced per-class)."""
    random.seed(seed)
    if balanced:
        assert per_class is not None, "When --balanced, please specify --per-class."
        # Group ids by class (labels are 1..200)
        by_class: Dict[int, List[int]] = {}
        for i in ids:
            y = labels[i]
            by_class.setdefault(y, []).append(i)
        out: List[int] = []
        for y, lst in by_class.items():
            random.shuffle(lst)
            take = min(per_class, len(lst))
            out.extend(lst[:take])
        random.shuffle(out)
        return out
    else:
        if total_num is None or total_num >= len(ids):
            random.shuffle(ids)
            return ids
        else:
            return random.sample(ids, total_num)

# ---- Main ------------------------------------------------------------------------------------

def main():
    args = parse_args()
    root = Path(args.cub_root)
    out  = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # Read metadata
    idx_map   = read_index_map(root/"images.txt")
    split_map = read_split(root/"train_test_split.txt")
    label_map = read_labels(root/"image_class_labels.txt")
    bbox_map  = read_bboxes(root/"bounding_boxes.txt") if args.use_bbox else None

    # Build the list of ids for the requested split
    if args.split == "train":
        ids = [i for i in idx_map if split_map[i] == 1]
    elif args.split == "test":
        ids = [i for i in idx_map if split_map[i] == 0]
    else:
        ids = sorted(idx_map.keys())

    # Sample image ids (random or balanced)
    keep_ids = sample_ids(ids, label_map, args.balanced, args.per_class, args.num, args.seed)

    # Export loop
    count_ok = 0
    for k, img_id in enumerate(keep_ids, 1):
        rel = idx_map[img_id]
        src = root/"images"/rel
        try:
            im = load_and_prepare(src, args.use_bbox, args.bbox_expand, bbox_map, img_id, args.img_size)
            # Save with 6-digit zero padding for stable sorting
            im.save(out/f"{k:06d}.png")
            count_ok += 1
        except Exception as e:
            print(f"[warn] skipping {src}: {e}")

    # Small report
    print(f"Saved {count_ok} images to {out}")
    print("Done.")

if __name__ == "__main__":
    main()
