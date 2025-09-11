import argparse, json, math, warnings
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

import torch
import torchvision as tv
from torchvision import transforms as T

# Silence noisy warnings from PIL/torch to keep logs readable
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------ CLI ---------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Two-sample tests (C2ST, MMD) on Inception features.")
    ap.add_argument("--fake_dir", type=str, required=True, help="Folder with generated images (flat).")
    ap.add_argument("--real_dir", type=str, required=True, help="Folder with real images (flat).")
    ap.add_argument("--batch", type=int, default=64, help="Batch size for feature extraction.")
    ap.add_argument("--permutations", type=int, default=200, help="Permutations for p-value estimation.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    ap.add_argument("--save_json", type=str, default=None, help="Optional path to save a JSON report.")
    ap.add_argument("--test", type=str, default="both", choices=["c2st", "mmd", "both"],
                    help="Which test to run.")
    return ap.parse_args()

# --------------- InceptionV3 feature extractor (2048-D) -------------
def build_inception_and_transform(device: torch.device):
    """
    Build an InceptionV3 model and the corresponding preprocessing transform.
    We use pre-logit (avgpool) features (shape: Bx2048).
    """
    try:
        # Newer torchvision: official weights and ready-made transforms
        weights = tv.models.Inception_V3_Weights.IMAGENET1K_V1
        model = tv.models.inception_v3(weights=weights, aux_logits=False).to(device).eval()
        transform = weights.transforms()
    except Exception:
        # Fallback for older versions
        model = tv.models.inception_v3(pretrained=True, aux_logits=False).to(device).eval()
        transform = T.Compose([
            T.Resize(299, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(299),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    return model, transform

@torch.no_grad()
def extract_features(dirpath: str, model, transform, batch: int, device: torch.device) -> np.ndarray:
    """
    Walk the folder, load images, push through Inception, and collect 2048-D features.
    Returns a numpy array of shape (N, 2048).
    """
    # Collect all PNG/JPG files (flat folder expected)
    paths = sorted([p for p in Path(dirpath).glob("*.png")] +
                   [p for p in Path(dirpath).glob("*.jpg")]  +
                   [p for p in Path(dirpath).glob("*.jpeg")])
    if len(paths) == 0:
        raise FileNotFoundError(f"No images found in {dirpath}. Provide a flat folder of PNG/JPG.")

    feats_list = []             # where we accumulate batch features
    hook_out = []               # temporary storage for the forward hook

    # We hook avgpool to capture pre-logit activations: (B, 2048, 1, 1)
    def _hook(m, i, o):
        hook_out.append(o.squeeze(-1).squeeze(-1))

    handle = model.avgpool.register_forward_hook(_hook)

    # Iterate in batches to avoid OOM
    for i in range(0, len(paths), batch):
        imgs = [transform(Image.open(p).convert("RGB")) for p in paths[i:i+batch]]
        x = torch.stack(imgs, 0).to(device)
        hook_out.clear()
        _ = model(x)            # run forward; hook fills hook_out
        feats_list.append(hook_out[0].detach().cpu())

    handle.remove()
    feats = torch.cat(feats_list, 0).numpy().astype(np.float32)  # (N, 2048) float32
    return feats

# ----------------------------- C2ST ---------------------------------
def c2st_auc_pvalue(fake: np.ndarray, real: np.ndarray, seed: int, n_perm: int) -> Tuple[float, float]:
    """
    Classifier Two-Sample Test:
      - Split features into train/test (stratified)
      - Train Logistic Regression (L2)
      - Report ROC-AUC on test
      - Permute test labels to obtain a p-value (how often permuted AUC >= observed)
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    # Build dataset: y=0 (fake), y=1 (real)
    X = np.concatenate([fake, real], axis=0)
    y = np.concatenate([np.zeros(len(fake)), np.ones(len(real))], axis=0)

    # Stratified split (70/30)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)

    # Pipeline: Standardize -> Logistic Regression (handles scale and is fast/robust)
    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(max_iter=1000, n_jobs=1)
    )
    clf.fit(Xtr, ytr)

    # Probabilities on test to compute ROC-AUC
    proba = clf.predict_proba(Xte)[:, 1]
    auc_obs = float(roc_auc_score(yte, proba))

    # Permutation test on the *test* labels only (condition on trained classifier)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        yperm = rng.permutation(yte)
        auc_p = roc_auc_score(yperm, proba)
        if auc_p >= auc_obs:
            cnt += 1
    pval = (cnt + 1) / (n_perm + 1)  # add-one smoothing for stability

    return auc_obs, pval

# ----------------------------- MMD ----------------------------------
def _pdist2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Pairwise squared Euclidean distances between rows of a (n×d) and b (m×d).
    Returns an (n×m) matrix. Computed on CPU to keep GPU free.
    """
    a2 = (a * a).sum(1, keepdim=True)          # (n, 1)
    b2 = (b * b).sum(1, keepdim=True).T        # (1, m)
    return (a2 + b2 - 2.0 * (a @ b.T)).clamp_min_(0.0)

def mmd2_rbf_unbiased(fake: np.ndarray, real: np.ndarray, sigma: float | None = None) -> Tuple[float, float]:
    """
    Unbiased estimate of MMD^2 with an RBF kernel.
    - sigma: bandwidth. If None, uses the median heuristic computed on joint distances.
    Returns (mmd2, sigma).
    """
    # Move to torch CPU tensors (float64 for numerical stability)
    X = torch.from_numpy(fake).double()
    Y = torch.from_numpy(real).double()

    # Compute pairwise squared distances
    Kxx_d2 = _pdist2(X, X)
    Kyy_d2 = _pdist2(Y, Y)
    Kxy_d2 = _pdist2(X, Y)

    # Median heuristic for sigma if not provided (avoid using diagonal zeros)
    if sigma is None:
        # Concatenate upper-triangular (without diag) of Kxx and Kyy plus full Kxy
        # For speed and simplicity here we take the median of all strictly positive entries.
        all_d2 = torch.cat([Kxx_d2.flatten(), Kyy_d2.flatten(), Kxy_d2.flatten()], 0)
        all_d2 = all_d2[all_d2 > 0]
        med = torch.median(all_d2)
        sigma = math.sqrt(0.5 * float(med.item())) if med.numel() > 0 else 1.0
        sigma = max(sigma, 1e-6)

    gamma = 1.0 / (2.0 * sigma * sigma)

    # Kernel matrices (no diagonal contributions in Kxx/Kyy for unbiased estimator)
    Kxx = torch.exp(-gamma * Kxx_d2)
    Kyy = torch.exp(-gamma * Kyy_d2)
    Kxy = torch.exp(-gamma * Kxy_d2)

    n = X.size(0); m = Y.size(0)
    # Remove diagonal terms from intra-set kernels
    mmd2 = (Kxx.sum() - Kxx.diag().sum()) / (n * (n - 1)) \
         + (Kyy.sum() - Kyy.diag().sum()) / (m * (m - 1)) \
         - 2.0 * Kxy.mean()

    return float(mmd2.item()), float(sigma)

def mmd_permutation_pvalue(fake: np.ndarray, real: np.ndarray, mmd_obs: float, sigma: float,
                           n_perm: int, seed: int) -> float:
    """
    Permutation p-value for MMD^2: shuffle labels across the joint sample and recompute MMD^2.
    """
    Z = np.concatenate([fake, real], axis=0)
    n = fake.shape[0]
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        idx = rng.permutation(Z.shape[0])
        Xp = Z[idx[:n]]
        Yp = Z[idx[n:]]
        mmd_p, _ = mmd2_rbf_unbiased(Xp, Yp, sigma=sigma)
        if mmd_p >= mmd_obs:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)  # add-one smoothing

# ------------------------------- MAIN -------------------------------
def main():
    args = parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    # Device for Inception: prefer GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = build_inception_and_transform(device)

    # Extract features (Nx2048) for both sets
    print("[features] extracting Inception features ...")
    F = extract_features(args.fake_dir, model, transform, args.batch, device)
    R = extract_features(args.real_dir, model, transform, args.batch, device)
    print(f"fake: {F.shape}, real: {R.shape}")

    report = {}

    # ---- C2ST ----
    if args.test in ("c2st", "both"):
        print("[C2ST] training logistic regression and running permutation test ...")
        auc, p = c2st_auc_pvalue(F, R, seed=args.seed, n_perm=args.permutations)
        report["c2st_auc"] = auc
        report["c2st_p_value"] = p
        print(f"C2ST ROC-AUC: {auc:.4f} | p-value: {p:.4f}")

    # ---- MMD ----
    if args.test in ("mmd", "both"):
        print("[MMD] computing unbiased MMD^2 (RBF) and permutation test ...")
        mmd2, sigma = mmd2_rbf_unbiased(F, R, sigma=None)
        pval = mmd_permutation_pvalue(F, R, mmd_obs=mmd2, sigma=sigma,
                                      n_perm=args.permutations, seed=args.seed)
        report["mmd2_rbf"] = mmd2
        report["mmd_sigma"] = sigma
        report["mmd_p_value"] = pval
        print(f"MMD^2 (RBF, sigma={sigma:.4g}): {mmd2:.6f} | p-value: {pval:.4f}")

    # Optionally dump a JSON report
    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[report] saved -> {args.save_json}")

if __name__ == "__main__":
    main()
