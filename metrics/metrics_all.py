#!/usr/bin/env python3
"""
Unified metrics script:
- FID / KID via torch-fidelity.
- PRDC (precision/recall/density/coverage) da feature Inception-2048.
- C2ST (Classifier Two-Sample Test) con logistic regression + permutation test.

Compat:
- torchvision InceptionV3 con aux_logits=True (API nuove/vecchie).
- KID dict fields robusti (mean/std/variance).
"""

import argparse, os, glob
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

from torch_fidelity import calculate_metrics as tf_calculate_metrics


# ----------------------------- IO & datasets -----------------------------

IMG_EXT = (".png", ".jpg", ".jpeg")


class FlatImageFolder(Dataset):
    """Flat folder of images."""
    def __init__(self, folder: str, transform):
        self.paths = sorted([p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith(IMG_EXT)])
        self.transform = transform

    def __len__(self): 
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img)


def build_inception_and_transform(device: torch.device):
    """
    Build an InceptionV3 and its preprocessing transform.
    Preferisce la weights API; fallback a pretrained=True.
    NB: aux_logits=True richiesto dalle ultime torchvision.
    Ritorna (feature_extractor, transform).
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
        y = model(x)['feat'].squeeze(-1).squeeze(-1)  # (B,2048,1,1)->(B,2048)
        feats.append(y.cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0)


# ----------------------------- PRDC -----------------------------

def compute_prdc(real: np.ndarray, fake: np.ndarray, k: int = 5):
    """
    Precision/Recall/Density/Coverage (Kynkäänniemi et al., 2019).
    Per la Density usiamo radius_neighbors in loop (un raggio scalare per fake).
    """
    nn_real = NearestNeighbors(n_neighbors=k).fit(real)
    rr_dists, _ = nn_real.kneighbors(real)       # (N,k)
    radii = rr_dists[:, -1]                      # (N,)

    rf_dists, rf_idx = nn_real.kneighbors(fake, n_neighbors=1)
    rf_dists = rf_dists.ravel()
    rf_idx   = rf_idx.ravel()

    # Precision: frazione di fake dentro il raggio del real più vicino
    precision = float(np.mean(rf_dists <= radii[rf_idx]))

    # Recall: frazione di real con almeno un fake entro il proprio raggio
    nn_fake = NearestNeighbors(n_neighbors=1).fit(fake)
    fr_dists, _ = nn_fake.kneighbors(real, n_neighbors=1)
    fr_dists = fr_dists.ravel()
    recall = float(np.mean(fr_dists <= radii))

    # Density: numero medio di real entro il raggio r_j del fake j (normalizzato per k)
    counts = []
    eps = 1e-12
    for j in range(fake.shape[0]):
        r = float(max(radii[rf_idx[j]], eps))  # raggio scalare
        neighs = nn_real.radius_neighbors(fake[j:j+1], radius=r, return_distance=False)
        counts.append(len(neighs[0]))
    density = float(np.mean(np.asarray(counts, dtype=np.float32) / float(k)))

    # Coverage: frazione di real coperti da almeno un fake entro il loro raggio
    coverage = float(np.mean(fr_dists <= radii))

    return {
        "precision": precision,
        "recall":    recall,
        "density":   density,
        "coverage":  coverage,
    }


# ----------------------------- C2ST -----------------------------

def c2st_logreg(real: np.ndarray, fake: np.ndarray, permutations: int = 200, seed: int | None = None):
    """
    Classifier two-sample test con Logistic Regression + permutation p-value.
    Ritorna dict con AUC, media/std della null e p-value.
    """
    rng = np.random.RandomState(seed)
    X = np.vstack([real, fake]).astype(np.float32)
    y = np.hstack([np.zeros(len(real), dtype=np.int64), np.ones(len(fake), dtype=np.int64)])

    idx = np.arange(len(X))
    idx = sk_shuffle(idx, random_state=rng)
    split = int(0.8 * len(idx))
    tr, va = idx[:split], idx[split:]

    clf = LogisticRegression(max_iter=2000, n_jobs=None)
    clf.fit(X[tr], y[tr])
    prob = clf.predict_proba(X[va])[:, 1]
    auc = roc_auc_score(y[va], prob)

    null_aucs = []
    for _ in range(permutations):
        y_perm = y.copy()
        rng.shuffle(y_perm)
        clf_p = LogisticRegression(max_iter=600, n_jobs=None)
        clf_p.fit(X[tr], y_perm[tr])
        prob_p = clf_p.predict_proba(X[va])[:, 1]
        null_aucs.append(roc_auc_score(y[va], prob_p))
    null_aucs = np.asarray(null_aucs, dtype=np.float32)
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
    Estrae KID e incertezza da torch-fidelity, gestendo chiavi diverse.
    Ritorna (value, uncertainty, tag) dove tag in {var,std,''}.
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
    ap.add_argument("--kid-subsets", dest="kid_subsets", type=int, default=50)
    ap.add_argument("--kid-subset-size", dest="kid_subset_size", type=int, default=1000)
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

    # 1) FID / KID
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

    # 2) PRDC (usando feature Inception 2048)
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

    # 3) C2ST
    print("\n[3/3] Running C2ST (logistic regression, permutation test) ...")
    c2 = c2st_logreg(real_feat, fake_feat, permutations=args.permutations, seed=args.seed)
    print(f"AUC:       {c2['auc']:.4f}")
    print(f"Null μ:    {c2['null_mean']:.4f}  σ: {c2['null_std']:.4f}")
    print(f"p-value:   {c2['p_value']:.6f}")


if __name__ == "__main__":
    main()


