# metrics/metrics_all.py
#!/usr/bin/env python3
"""
Unified metrics script:
- FID / KID via torch-fidelity (robust across versions).
- PRDC (precision/recall/density/coverage) from Inception-2048 features.
- C2ST (Classifier Two-Sample Test) via logistic regression + permutation p-value.

Usage:
  python metrics_all.py --fake_dir <FAKE> --real_dir <REAL> --batch 64 --kid-subsets 50 --kid-subset-size 1000

This file is patched to:
- Use torchvision inception_v3 with aux_logits=True (required by recent torchvision).
- Tolerate different KID dict keys from torch-fidelity (mean/std/variance variants).
"""

import argparse, os, math, glob
from pathlib import Path

import numpy as np
import torch
import torchvision as tv
from torchvision import transforms as T
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle as sk_shuffle

# FID/KID backend
from torch_fidelity import calculate_metrics as tf_calculate_metrics


# ----------------------------- IO & datasets -----------------------------

IMG_EXT = (".png", ".jpg", ".jpeg")


class FlatImageFolder(Dataset):
    """Flat folder of images."""
    def __init__(self, folder: str, transform):
        self.paths = sorted([p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith(IMG_EXT)])
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img)


def build_inception_and_transform(device: torch.device):
    """
    Build an InceptionV3 and its preprocessing transform.
    Uses torchvision weights API when available; falls back to pretrained=True otherwise.
    IMPORTANT: aux_logits=True for recent torchvision.
    Returns (model.eval().to(device), transform)
    """
    try:
        weights = tv.models.Inception_V3_Weights.IMAGENET1K_V1
        model = tv.models.inception_v3(weights=weights, aux_logits=True).to(device).eval()
        transform = weights.transforms()
    except Exception:
        model = tv.models.inception_v3(pretrained=True, aux_logits=True).to(device).eval()
        transform = T.Compose([
            T.Resize(299, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(299),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    # Extract 2048-dim features from 'avgpool'
    feat_extractor = create_feature_extractor(model, return_nodes={'avgpool': 'feat'})
    return feat_extractor, transform


@torch.no_grad()
def extract_inception_features(folder: str, batch: int, device: torch.device) -> np.ndarray:
    """Return Nx2048 features for all images in `folder`."""
    model, transform = build_inception_and_transform(device)
    ds = FlatImageFolder(folder, transform)
    if len(ds) == 0:
        return np.zeros((0, 2048), dtype=np.float32)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))
    feats = []
    for x in dl:
        x = x.to(device, non_blocking=True)
        y = model(x)['feat'].squeeze(-1).squeeze(-1)  # (B, 2048, 1, 1) -> (B, 2048)
        feats.append(y.cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0)


# ----------------------------- PRDC -----------------------------

def compute_prdc(real: np.ndarray, fake: np.ndarray, k: int = 5):
    """
    Precision/Recall/Density/Coverage as in Kynkäänniemi et al. (2019).
    real, fake: (N, D) arrays of features.
    """
    # Fit NN on real for kNN radius
    nn_real = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(real)
    # k-th NN distances (radius) in real-real space
    rr_dists, _ = nn_real.kneighbors(real)   # (N, k)
    radii = rr_dists[:, -1]                  # (N,)
    # For fast membership tests, we need nearest real neighbor for each fake
    rf_dists, rf_idx = nn_real.kneighbors(fake, n_neighbors=1)  # (M,1)
    rf_dists = rf_dists.squeeze(1)         # (M,)
    rf_idx = rf_idx.squeeze(1)             # (M,)

    # Precision: fraction of fake within the radius of their nearest real
    precision = float(np.mean(rf_dists <= radii[rf_idx]))

    # Recall: fraction of real that have at least one fake within their radius
    nn_fake = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(fake)
    fr_dists, _ = nn_fake.kneighbors(real, n_neighbors=1)  # (N,1)
    fr_dists = fr_dists.squeeze(1)                         # (N,)
    recall = float(np.mean(fr_dists <= radii))

    # Density: average number of real neighbors within radius (normalized by k)
    # Count how many real points are within each fake's radius (of its matched real).
    # Approximate via distance ratio exp(-dist/radius) surrogate:
    # A simpler practical proxy: precision * k (as in some open-source impls).
    # We'll compute a more faithful proxy via brute-force local neighborhood per fake:
    # Use the same nn_real structure to query how many neighbors are within radius for each fake.
    # sklearn kneighbors doesn't support variable radius directly; use radius_neighbors.
    # If radius is zero for some real point, fall back to small epsilon.
    eps = 1e-12
    radii_fake = np.maximum(radii[rf_idx], eps)  # radius of matched real
    # Count neighbors within each radius using radius_neighbors (returns list of arrays)
    # To avoid OOM, process in chunks
    counts = []
    step = 200
    for i in range(0, fake.shape[0], step):
        neighs = nn_real.radius_neighbors(fake[i:i+step], radii_fake[i:i+step], return_distance=False)
        counts.extend([len(n) for n in neighs])
    density = float(np.mean(np.array(counts, dtype=np.float32) / float(k)))

    # Coverage: fraction of real covered by at least one fake (within that real's radius)
    # Use same fr_dists <= radii criterion
    coverage = float(np.mean(fr_dists <= radii))

    return {
        "precision": precision,
        "recall": recall,
        "density": density,
        "coverage": coverage,
    }


# ----------------------------- C2ST -----------------------------

def c2st_logreg(real: np.ndarray, fake: np.ndarray, permutations: int = 200, seed: int | None = None):
    """
    Classifier two-sample test with logistic regression.
    Returns dict with AUC, null distribution mean/std, p-value.
    """
    rng = np.random.RandomState(seed)
    X = np.vstack([real, fake]).astype(np.float32)
    y = np.hstack([np.zeros(len(real), dtype=np.int64), np.ones(len(fake), dtype=np.int64)])

    # Train/Val split (stratified)
    idx = np.arange(len(X))
    idx = sk_shuffle(idx, random_state=rng)
    split = int(0.8 * len(idx))
    tr, va = idx[:split], idx[split:]

    clf = LogisticRegression(max_iter=1000, n_jobs=None)
    clf.fit(X[tr], y[tr])
    prob = clf.predict_proba(X[va])[:, 1]
    auc = roc_auc_score(y[va], prob)

    # Permutation test: shuffle labels, refit, compute AUC each time
    null_aucs = []
    for _ in range(permutations):
        y_perm = y.copy()
        rng.shuffle(y_perm)
        clf_p = LogisticRegression(max_iter=300, n_jobs=None)
        clf_p.fit(X[tr], y_perm[tr])
        prob_p = clf_p.predict_proba(X[va])[:, 1]
        null_aucs.append(roc_auc_score(y[va], prob_p))
    null_aucs = np.asarray(null_aucs, dtype=np.float32)
    # p-value: fraction of null >= observed (right-tailed)
    pval = float((null_aucs >= auc).mean())

    return {
        "auc": float(auc),
        "null_mean": float(null_aucs.mean()),
        "null_std": float(null_aucs.std(ddof=1) if len(null_aucs) > 1 else 0.0),
        "p_value": pval,
    }


# ----------------------------- KID helpers -----------------------------

def read_kid_fields(d: dict):
    """
    Robustly read KID value and uncertainty from torch-fidelity dict across versions.
    Returns (value, uncertainty, tag) where tag is 'var' or 'std' or ''.
    """
    val = d.get('kernel_inception_distance') \
          or d.get('kernel_inception_distance_mean') \
          or d.get('kid')
    var = d.get('kernel_inception_distance_variance')
    std = d.get('kernel_inception_distance_std') or d.get('kid_std')
    if var is not None:
        return val, var, 'var'
    if std is not None:
        return val, std, 'std'
    return val, None, ''


# ----------------------------- Main -----------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake_dir", type=str, required=True)
    ap.add_argument("--real_dir", type=str, required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--kid-subsets", type=int, default=50)
    ap.add_argument("--kid-subset-size", type=int, default=1000)
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--seed", type=int, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Count files
    fake_n = len([p for p in glob.glob(os.path.join(args.fake_dir, "*")) if p.lower().endswith(IMG_EXT)])
    real_n = len([p for p in glob.glob(os.path.join(args.real_dir, "*")) if p.lower().endswith(IMG_EXT)])
    print(f"[check] fake={fake_n} png | real={real_n} png")
    assert fake_n >= 100 and real_n >= 100, "Need enough images (>=100); aim for 1000 vs 1000."

    # 1) FID / KID (torch-fidelity)
    print("\n[1/3] Computing FID / KID ...\n")
    fidkid = tf_calculate_metrics(
        input1=args.fake_dir,
        input2=args.real_dir,
        cuda=torch.cuda.is_available(),
        isc=False,
        fid=True,
        kid=True,
        kid_subset_size=args.kid_subset_size,
        kid_subsets=args.kid_subsets,
        verbose=False,
    )
    fid = fidkid.get('frechet_inception_distance')
    kid_val, kid_unc, kid_tag = read_kid_fields(fidkid)
    print("================= METRICS REPORT =================")
    if fid is not None:
        print(f"FID: {fid:.4f}")
    if kid_val is not None:
        line = f"KID: {kid_val:.8f}"
        if kid_unc is not None:
            line += f" ({kid_tag}: {kid_unc:.8f})"
        print(line)

    # 2) PRDC (from 2048-dim features)
    print("\n[2/3] Extracting Inception features for PRDC/C2ST ...")
    fake_feat = extract_inception_features(args.fake_dir, args.batch, device)  # (M,2048)
    real_feat = extract_inception_features(args.real_dir, args.batch, device)  # (N,2048)
    print(f"Feature shapes: fake={fake_feat.shape}, real={real_feat.shape}")

    print("\n[2/3] Computing PRDC (k=5) ...")
    print(f"Num real: {len(real_feat)} Num fake: {len(fake_feat)}")
    prdc = compute_prdc(real_feat, fake_feat, k=5)
    print(f"Precision: {prdc['precision']:.4f}")
    print(f"Recall:    {prdc['recall']:.4f}")
    print(f"Density:   {prdc['density']:.4f}")
    print(f"Coverage:  {prdc['coverage']:.4f}")

    # 3) C2ST (logistic regression + permutation)
    print("\n[3/3] Running C2ST (logistic regression, permutation test) ...")
    c2 = c2st_logreg(real_feat, fake_feat, permutations=args.permutations, seed=args.seed)
    print(f"AUC:       {c2['auc']:.4f}")
    print(f"Null μ:    {c2['null_mean']:.4f}  σ: {c2['null_std']:.4f}")
    print(f"p-value:   {c2['p_value']:.6f}")


if __name__ == "__main__":
    main()
