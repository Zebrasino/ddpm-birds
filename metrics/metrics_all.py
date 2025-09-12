import argparse, os, glob              # CLI parsing, filesystem, and globbing
from pathlib import Path               # convenient path handling

import numpy as np                     # numerical arrays
import torch                           # tensor library (GPU/CPU)
import torchvision as tv               # models (InceptionV3)
from torchvision import transforms as T  # preprocessing transforms
from torchvision.models.feature_extraction import create_feature_extractor  # feature hooks
from torch.utils.data import Dataset, DataLoader   # data pipeline
from PIL import Image                  # image loading

from sklearn.neighbors import NearestNeighbors     # kNN for PRDC

from torch_fidelity import calculate_metrics as tf_calculate_metrics  # FID/KID backend


# ----------------------------- IO & datasets -----------------------------

IMG_EXT = (".png", ".jpg", ".jpeg")    # supported image extensions


class FlatImageFolder(Dataset):
    """Flat (non-recursive) folder of images."""
    def __init__(self, folder: str, transform):
        # Collect files matching supported extensions (case-insensitive)
        self.paths = sorted([p for p in glob.glob(os.path.join(folder, "*"))
                             if p.lower().endswith(IMG_EXT)])
        self.transform = transform     # preprocessing pipeline

    def __len__(self):
        return len(self.paths)         # number of images

    def __getitem__(self, idx):
        p = self.paths[idx]            # path at index
        img = Image.open(p).convert("RGB")  # load and ensure 3 channels
        return self.transform(img)     # return preprocessed tensor


def build_inception_and_transform(device: torch.device):
    """
    Build an InceptionV3 model and its preprocessing transform.

    - Prefer the official weights API (newer torchvision).
    - Fallback to pretrained=True for older versions.
    - IMPORTANT: aux_logits=True is required by recent torchvision.
    - Return a feature extractor that outputs the 'avgpool' activations (pre-logits).
    """
    try:
        # Newer API: use weights enum + ready-made transforms
        weights = tv.models.Inception_V3_Weights.IMAGENET1K_V1
        model = tv.models.inception_v3(weights=weights, aux_logits=True).to(device).eval()
        transform = weights.transforms()
    except Exception:
        # Older API: use pretrained=True and build transforms manually
        model = tv.models.inception_v3(pretrained=True, aux_logits=True).to(device).eval()
        transform = T.Compose([
            T.Resize(299, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(299),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    # Map the 'avgpool' node to 'feat' so forward(model(x)) returns {'feat': ...}
    feat_extractor = create_feature_extractor(model, return_nodes={'avgpool': 'feat'})
    return feat_extractor, transform


@torch.no_grad()
def extract_inception_features(folder: str, batch: int, device: torch.device) -> np.ndarray:
    """
    Extract 2048-D Inception features for all images in `folder`.
    Returns a numpy array of shape (N, 2048) in float32.
    """
    model, transform = build_inception_and_transform(device)  # model + preprocessing
    ds = FlatImageFolder(folder, transform)                   # dataset view of the folder
    if len(ds) == 0:                                          # edge case: empty folder
        return np.zeros((0, 2048), dtype=np.float32)
    # DataLoader with small number of workers; pin_memory helps on CUDA
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2,
                    pin_memory=(device.type == "cuda"))
    feats = []                                                # list of (B, 2048) tensors
    for x in dl:
        x = x.to(device, non_blocking=True)                   # move batch to device
        y = model(x)['feat'].squeeze(-1).squeeze(-1)         # (B,2048,1,1)->(B,2048)
        feats.append(y.cpu().numpy().astype(np.float32))      # collect on CPU as float32
    return np.concatenate(feats, axis=0)                      # stack into (N, 2048)


# ----------------------------- PRDC -----------------------------

def compute_prdc(real: np.ndarray, fake: np.ndarray, k: int = 5):
    """
    Compute Precision/Recall/Density/Coverage (Kynkäänniemi et al., 2019)
    on top of feature embeddings.

    Notes:
    - For Density, sklearn's radius_neighbors accepts a scalar radius; we loop
      over fake samples and query with the matched real radius individually.
    """
    # Fit kNN on real features (to get per-sample radii from k-th neighbor)
    nn_real = NearestNeighbors(n_neighbors=k).fit(real)
    rr_dists, _ = nn_real.kneighbors(real)   # distances among real points
    radii = rr_dists[:, -1]                  # radius = distance to k-th neighbor (per real point)

    # For each fake, find nearest real (distance + index)
    rf_dists, rf_idx = nn_real.kneighbors(fake, n_neighbors=1)
    rf_dists = rf_dists.ravel()
    rf_idx   = rf_idx.ravel()

    # Precision: fraction of fake within the radius of their nearest real
    precision = float(np.mean(rf_dists <= radii[rf_idx]))

    # Recall: fraction of real that have at least one fake within their radius
    nn_fake = NearestNeighbors(n_neighbors=1).fit(fake)        # kNN over fake
    fr_dists, _ = nn_fake.kneighbors(real, n_neighbors=1)      # nearest fake to each real
    fr_dists = fr_dists.ravel()
    recall = float(np.mean(fr_dists <= radii))

    # Density: average # of real neighbors within the matched radius of each fake (normalized by k)
    counts = []
    eps = 1e-12
    for j in range(fake.shape[0]):                              # loop per fake (OK for ~1k)
        r = float(max(radii[rf_idx[j]], eps))                   # scalar radius for this fake
        neighs = nn_real.radius_neighbors(fake[j:j+1], radius=r, return_distance=False)
        counts.append(len(neighs[0]))                           # #reals within radius r
    density = float(np.mean(np.asarray(counts, dtype=np.float32) / float(k)))

    # Coverage: fraction of real covered by at least one fake within each real's radius
    coverage = float(np.mean(fr_dists <= radii))

    return {
        "precision": precision,
        "recall":    recall,
        "density":   density,
        "coverage":  coverage,
    }


# ----------------------------- KID helpers -----------------------------

def read_kid_fields(d: dict):
    """
    Robustly read KID value and uncertainty from torch-fidelity dict across versions.
    Returns (value, uncertainty, tag) where tag is one of {'var', 'std', ''}.
    """
    val = d.get('kernel_inception_distance') \
          or d.get('kernel_inception_distance_mean') \
          or d.get('kid')                                # fallbacks across versions
    var = d.get('kernel_inception_distance_variance')    # variance (newer)
    std = d.get('kernel_inception_distance_std') or d.get('kid_std')  # std (older)
    if var is not None:
        return val, var, 'var'
    if std is not None:
        return val, std, 'std'
    return val, None, ''


# ----------------------------- CLI & main -----------------------------

def parse_args():
    """Parse command-line arguments for folders and options."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake_dir", type=str, required=True)        # generated images folder
    ap.add_argument("--real_dir", type=str, required=True)        # reference images folder
    ap.add_argument("--batch", type=int, default=64)              # batch size for feature extraction
    ap.add_argument("--kid-subsets", dest="kid_subsets", type=int, default=50)     # KID subsets
    ap.add_argument("--kid-subset-size", dest="kid_subset_size", type=int, default=1000)  # subset size
    ap.add_argument("--seed", type=int, default=None)             # (unused here but reserved)
    return ap.parse_args()


def main():
    args = parse_args()                                           # read CLI args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # pick device

    # Count files to avoid silent failures
    fake_n = len([p for p in glob.glob(os.path.join(args.fake_dir, "*")) if p.lower().endswith(IMG_EXT)])
    real_n = len([p for p in glob.glob(os.path.join(args.real_dir, "*")) if p.lower().endswith(IMG_EXT)])
    print(f"[check] fake={fake_n} png | real={real_n} png")
    assert fake_n >= 100 and real_n >= 100, "Need enough images (>=100); aim for 1000 vs 1000."

    # -------------------- 1) FID / KID --------------------
    print("\n[1/2] Computing FID / KID ...\n")
    fidkid = tf_calculate_metrics(
        input1=args.fake_dir,                     # generated images
        input2=args.real_dir,                     # reference images
        cuda=torch.cuda.is_available(),           # use GPU if available
        isc=False,                                # no Inception Score here
        fid=True,                                 # compute FID
        kid=True,                                 # compute KID
        kid_subset_size=args.kid_subset_size,     # KID subset size (e.g., 1000)
        kid_subsets=args.kid_subsets,             # number of subsets (e.g., 50)
        verbose=False,                            # quiet torch-fidelity logging
    )
    fid = fidkid.get('frechet_inception_distance')              # FID value
    kid_val, kid_unc, kid_tag = read_kid_fields(fidkid)         # robust KID parsing

    print("================= METRICS REPORT =================")
    if fid is not None:
        print(f"FID: {fid:.4f}")                                # print FID
    if kid_val is not None:                                     # print KID + uncertainty if present
        line = f"KID: {kid_val:.8f}"
        if kid_unc is not None:
            line += f" ({kid_tag}: {kid_unc:.8f})"              # '(var: ...)' or '(std: ...)'
        print(line)

    # -------------------- 2) PRDC --------------------
    print("\n[2/2] Extracting Inception features for PRDC ...")
    fake_feat = extract_inception_features(args.fake_dir, args.batch, device)  # (M, 2048)
    real_feat = extract_inception_features(args.real_dir, args.batch, device)  # (N, 2048)
    print(f"Feature shapes: fake={fake_feat.shape}, real={real_feat.shape}")

    print("\n[2/2] Computing PRDC (k=5) ...")
    print(f"Num real: {len(real_feat)} Num fake: {len(fake_feat)}")
    prdc = compute_prdc(real_feat, fake_feat, k=5)                            # compute metrics
    print(f"Precision: {prdc['precision']:.4f}")
    print(f"Recall:    {prdc['recall']:.4f}")
    print(f"Density:   {prdc['density']:.4f}")
    print(f"Coverage:  {prdc['coverage']:.4f}")


if __name__ == "__main__":
    main()  # entry point when executed as a script



