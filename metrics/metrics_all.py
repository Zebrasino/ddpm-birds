# metrics/metrics_all.py
# Compute FID, KID (torch-fidelity), PRDC (precision/recall/density/coverage),
# and C2ST (classifier two-sample test) on Inception features.
# Usage:
#   python metrics/metrics_all.py \
#       --fake_dir "/content/.../ddim_1k" \
#       --real_dir "/content/cub64_real_train_1k" \
#       --batch 64 --kid-subsets 50 --kid-subset-size 1000 --permutations 200

import argparse, os, math, warnings
from pathlib import Path
from PIL import Image
import numpy as np
import torch, torchvision as tv
from torchvision import transforms as T

# ---- Optional: silence PIL/Torch warnings to keep logs clean
warnings.filterwarnings("ignore", category=UserWarning)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake_dir", type=str, required=True, help="Folder with generated PNGs")
    ap.add_argument("--real_dir", type=str, required=True, help="Folder with real PNGs (same resolution)")
    ap.add_argument("--batch", type=int, default=64, help="Batch size for feature extraction")
    ap.add_argument("--kid-subsets", type=int, default=50, help="Number of subsets for KID")
    ap.add_argument("--kid-subset-size", type=int, default=1000, help="Subset size for each KID estimate")
    ap.add_argument("--permutations", type=int, default=200, help="Permutations for C2ST p-value")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    return ap.parse_args()

# ---------- Inception feature extractor (2048-D pre-logits) ----------
def build_inception_and_transform(device):
    """Return (inception_model, transform) for 2048-D pre-logit features."""
    try:
        weights = tv.models.Inception_V3_Weights.IMAGENET1K_V1
        model = tv.models.inception_v3(weights=weights, aux_logits=True).to(device).eval()
        transform = weights.transforms()
    except Exception:
        # Fallback for older torchvision
        model = tv.models.inception_v3(pretrained=True, aux_logits=True).to(device).eval()
        transform = T.Compose([
            T.Resize(299, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(299),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    return model, transform

@torch.no_grad()
def extract_inception_features(dirpath, model, transform, batch=64, device="cuda"):
    """Extract 2048-D features from all PNG files in dirpath."""
    paths = sorted([p for p in Path(dirpath).glob("*.png")])
    if len(paths) == 0:
        raise FileNotFoundError(f"No PNG files found in {dirpath}. Ensure a flat folder of images.")
    features, hook_out = [], []
    # Hook the avgpool to grab pre-logit features (B,2048,1,1)
    def _hook(m, i, o): hook_out.append(o.squeeze(-1).squeeze(-1))
    h = model.avgpool.register_forward_hook(_hook)
    for i in range(0, len(paths), batch):
        imgs = [transform(Image.open(p).convert("RGB")) for p in paths[i:i+batch]]
        x = torch.stack(imgs, 0).to(device)
        hook_out.clear()
        _ = model(x)
        features.append(hook_out[0].detach().cpu())
    h.remove()
    return torch.cat(features, 0).numpy()  # (N, 2048)

# ---------- FID/KID via torch-fidelity (Python API) ----------
def compute_fid_kid(fake_dir, real_dir, kid_subset_size=1000, kid_subsets=50, batch_size=64):
    from torch_fidelity import calculate_metrics
    kid_subset_size = int(min(kid_subset_size,
                              len(list(Path(fake_dir).glob("*.png"))),
                              len(list(Path(real_dir).glob("*.png")))))
    metrics = calculate_metrics(
        input1=fake_dir, input2=real_dir,
        fid=True, kid=True, isc=False,               # IS optional; often noisy on small N
        kid_subset_size=kid_subset_size, kid_subsets=kid_subsets,
        batch_size=batch_size,
    )
    return metrics  # dict with 'frechet_inception_distance', 'kernel_inception_distance_mean', '..._var'

# ---------- PRDC (precision, recall, density, coverage) ----------
def compute_prdc(fake_feats, real_feats, k=5):
    from prdc import compute_prdc
    res = compute_prdc(real_features=real_feats, fake_features=fake_feats, nearest_k=k)
    # Convert to Python floats for clean printing
    return {k: float(v) for k, v in res.items()}

# ---------- C2ST (classifier two-sample test) with Logistic Regression ----------
def c2st_auc_pvalue(fake_feats, real_feats, seed=0, permutations=200):
    """Train logistic regression to discriminate fake vs real on features;
       return (roc_auc, permutation_p_value)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    X = np.concatenate([fake_feats, real_feats], axis=0)
    y = np.concatenate([np.zeros(len(fake_feats)), np.ones(len(real_feats))], axis=0)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    clf = LogisticRegression(max_iter=1000, n_jobs=1)
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    auc_obs = float(roc_auc_score(yte, proba))

    # Permutation test on the test split
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(permutations):
        yperm = rng.permutation(yte)
        auc_p = roc_auc_score(yperm, proba)
        if auc_p >= auc_obs:
            cnt += 1
    pval = (cnt + 1) / (permutations + 1)           # add-one smoothing
    return auc_obs, pval

def main():
    args = parse_args()
    # Seed everything lightly
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # Quick sanity: count files
    fake_n = len(list(Path(args.fake_dir).glob("*.png")))
    real_n = len(list(Path(args.real_dir).glob("*.png")))
    print(f"[check] fake={fake_n} png | real={real_n} png")
    assert fake_n >= 100 and real_n >= 100, "Need enough images (>=100); aim for 1000 vs 1000."

    # Device for feature extraction
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inception, transform = build_inception_and_transform(device)

    # ---- 1) FID / KID
    print("\n[1/3] Computing FID / KID ...")
    fidkid = compute_fid_kid(args.fake_dir, args.real_dir,
                             kid_subset_size=args.kid_subset_size,
                             kid_subsets=args.kid_subsets,
                             batch_size=args.batch)
    # ---- 2) PRDC
    print("\n[2/3] Extracting Inception features for PRDC/C2ST ...")
    fake_feats = extract_inception_features(args.fake_dir, inception, transform,
                                            batch=args.batch, device=device)
    real_feats = extract_inception_features(args.real_dir, inception, transform,
                                            batch=args.batch, device=device)
    print(f"Feature shapes: fake={fake_feats.shape}, real={real_feats.shape}")

    print("\n[2/3] Computing PRDC (k=5) ...")
    prdc = compute_prdc(fake_feats, real_feats, k=5)

    # ---- 3) C2ST (discriminator-based two-sample test)
    print("\n[3/3] Running C2ST (logistic regression, permutation test) ...")
    auc, pval = c2st_auc_pvalue(fake_feats, real_feats,
                                seed=args.seed, permutations=args.permutations)

    # ---- Report
    print("\n================= METRICS REPORT =================")
    print(f"FID: {fidkid['frechet_inception_distance']:.4f}")
    print(f"KID mean: {fidkid['kernel_inception_distance_mean']:.6f} "
          f"(var: {fidkid['kernel_inception_distance_variance']:.6f})")
    print(f"PRDC: precision={prdc['precision']:.4f}, recall={prdc['recall']:.4f}, "
          f"density={prdc['density']:.4f}, coverage={prdc['coverage']:.4f}")
    print(f"C2ST: ROC-AUC={auc:.4f}, p-value={pval:.4f}")
    print("==================================================")

if __name__ == "__main__":
    main()

