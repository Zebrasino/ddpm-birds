import argparse, json, math, warnings          # CLI parsing, JSON output, math utils, and silencing warnings
from pathlib import Path                        # convenient, cross-platform paths
from typing import Tuple                        # type annotations for function returns

import numpy as np                              # numerical arrays, vectorized ops
from PIL import Image                           # image loading

import torch                                    # tensors, GPU/CPU acceleration
import torchvision as tv                        # models (InceptionV3)
from torchvision import transforms as T         # imagenet preprocessing transforms

# Silence noisy warnings from PIL/torch to keep logs readable
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------ CLI ---------------------------------
def parse_args():
    """Parse command-line arguments for input folders, batch size, test type, etc."""
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
    We capture pre-logit (avgpool) features of shape (B, 2048).
    Using aux_logits=True for torchvision compatibility (newer releases require it).
    """
    try:
        # Newer torchvision API: use official weights object + ready-made transforms
        weights = tv.models.Inception_V3_Weights.IMAGENET1K_V1     # pretrained weights enum
        model = tv.models.inception_v3(weights=weights, aux_logits=True).to(device).eval()  # eval mode
        transform = weights.transforms()                            # proper resize/normalize pipeline
    except Exception:
        # Fallback for older torchvision where "pretrained=True" is the standard flag
        model = tv.models.inception_v3(pretrained=True, aux_logits=True).to(device).eval()
        transform = T.Compose([                                     # manual ImageNet preprocessing
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
    Returns a numpy array of shape (N, 2048) in float32.
    """
    # Collect all PNG/JPG files from a *flat* directory (no recursion)
    paths = sorted([p for p in Path(dirpath).glob("*.png")] +
                   [p for p in Path(dirpath).glob("*.jpg")]  +
                   [p for p in Path(dirpath).glob("*.jpeg")])
    if len(paths) == 0:
        raise FileNotFoundError(f"No images found in {dirpath}. Provide a flat folder of PNG/JPG.")

    feats_list = []                 # accumulates per-batch feature tensors
    hook_out = []                   # stores the output captured by the forward hook

    # Define a forward hook on avgpool to capture pre-logit activations: (B, 2048, 1, 1)
    def _hook(m, i, o):
        hook_out.append(o.squeeze(-1).squeeze(-1))   # squeeze spatial dims -> (B, 2048)

    handle = model.avgpool.register_forward_hook(_hook)  # attach the hook

    # Iterate in mini-batches to control memory usage
    for i in range(0, len(paths), batch):
        imgs = [transform(Image.open(p).convert("RGB")) for p in paths[i:i+batch]]  # load+preprocess
        x = torch.stack(imgs, 0).to(device)                                         # (B, 3, 299, 299)
        hook_out.clear()                                                            # reset hook buffer
        _ = model(x)                                                                # forward pass; hook fills hook_out
        feats_list.append(hook_out[0].detach().cpu())                               # collect (B, 2048)

    handle.remove()                                                                 # detach the hook
    feats = torch.cat(feats_list, 0).numpy().astype(np.float32)                     # (N, 2048) float32
    return feats

# ----------------------------- C2ST ---------------------------------
def c2st_auc_pvalue(fake: np.ndarray, real: np.ndarray, seed: int, n_perm: int) -> Tuple[float, float]:
    """
    Classifier Two-Sample Test:
      - Stratified train/test split
      - Train Logistic Regression (with standardization)
      - ROC-AUC on test as the statistic
      - Permute only the *test* labels to obtain a p-value (how often permuted AUC >= observed)
    Returns (auc_observed, p_value).
    """
    from sklearn.model_selection import train_test_split      # stratified split util
    from sklearn.preprocessing import StandardScaler          # feature standardization
    from sklearn.pipeline import make_pipeline                # compose scaler + classifier
    from sklearn.linear_model import LogisticRegression       # logistic regression classifier
    from sklearn.metrics import roc_auc_score                 # ROC-AUC metric

    # Build the dataset: label 0 = fake, label 1 = real
    X = np.concatenate([fake, real], axis=0)                  # features stacked
    y = np.concatenate([np.zeros(len(fake)), np.ones(len(real))], axis=0)  # binary labels

    # Stratified split (70% train / 30% test) for balanced evaluation
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)

    # Standardize -> Logistic Regression (robust and fast)
    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),        # zero-mean/unit-variance scaling
        LogisticRegression(max_iter=1000, n_jobs=1)           # classifier; n_jobs used by some solvers
    )
    clf.fit(Xtr, ytr)                                         # train on the training split

    # Predicted probabilities on the test split for ROC-AUC
    proba = clf.predict_proba(Xte)[:, 1]                      # probability of class "real"
    auc_obs = float(roc_auc_score(yte, proba))                # observed AUC

    # Permutation test on the test labels (conditioned on the trained classifier)
    rng = np.random.default_rng(seed)                         # reproducible RNG
    cnt = 0                                                   # count how many permuted AUC >= observed
    for _ in range(n_perm):
        yperm = rng.permutation(yte)                          # shuffle labels
        auc_p = roc_auc_score(yperm, proba)                   # AUC under permuted labels
        if auc_p >= auc_obs:
            cnt += 1
    pval = (cnt + 1) / (n_perm + 1)                           # add-one smoothing for stability

    return auc_obs, pval                                       # return statistic and p-value

# ----------------------------- MMD ----------------------------------
def _pdist2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Pairwise squared Euclidean distances between rows of a (n×d) and b (m×d).
    Returns an (n×m) matrix. Computed on CPU to keep GPU free.
    """
    a2 = (a * a).sum(1, keepdim=True)          # (n, 1)   sum of squares per row of a
    b2 = (b * b).sum(1, keepdim=True).T        # (1, m)   sum of squares per row of b, transposed
    return (a2 + b2 - 2.0 * (a @ b.T)).clamp_min_(0.0)  # ||a-b||^2 with numerical safety

def mmd2_rbf_unbiased(fake: np.ndarray, real: np.ndarray, sigma: float | None = None) -> Tuple[float, float]:
    """
    Unbiased estimate of MMD^2 with an RBF kernel.
    - sigma: bandwidth. If None, uses the median heuristic computed on joint distances.
    Returns (mmd2, sigma_used).
    """
    # Move to torch CPU tensors (float64 improves numerical stability)
    X = torch.from_numpy(fake).double()        # (n, d)
    Y = torch.from_numpy(real).double()        # (m, d)

    # Compute pairwise squared distances for Kxx, Kyy, Kxy
    Kxx_d2 = _pdist2(X, X)                     # (n, n)
    Kyy_d2 = _pdist2(Y, Y)                     # (m, m)
    Kxy_d2 = _pdist2(X, Y)                     # (n, m)

    # Median heuristic for sigma if not provided (exclude zeros on diagonals)
    if sigma is None:
        all_d2 = torch.cat([Kxx_d2.flatten(), Kyy_d2.flatten(), Kxy_d2.flatten()], 0)  # pool all distances
        all_d2 = all_d2[all_d2 > 0]                 # remove zeros (self-distances)
        med = torch.median(all_d2)                  # median of positive distances
        sigma = math.sqrt(0.5 * float(med.item())) if med.numel() > 0 else 1.0  # heuristic bandwidth
        sigma = max(sigma, 1e-6)                    # avoid degenerate bandwidth

    gamma = 1.0 / (2.0 * sigma * sigma)             # RBF gamma parameter

    # Kernel matrices (exclude diagonal terms for unbiased MMD^2)
    Kxx = torch.exp(-gamma * Kxx_d2)                # (n, n)
    Kyy = torch.exp(-gamma * Kyy_d2)                # (m, m)
    Kxy = torch.exp(-gamma * Kxy_d2)                # (n, m)

    n = X.size(0); m = Y.size(0)                    # sample sizes
    # Unbiased MMD^2 estimate (Gretton et al., remove self-similarities)
    mmd2 = (Kxx.sum() - Kxx.diag().sum()) / (n * (n - 1)) \
         + (Kyy.sum() - Kyy.diag().sum()) / (m * (m - 1)) \
         - 2.0 * Kxy.mean()

    return float(mmd2.item()), float(sigma)         # return scalar MMD^2 and the bandwidth used

def mmd_permutation_pvalue(fake: np.ndarray, real: np.ndarray, mmd_obs: float, sigma: float,
                           n_perm: int, seed: int) -> float:
    """
    Permutation p-value for MMD^2: shuffle labels across the joint sample and recompute MMD^2.
    Returns the (right-tailed) p-value.
    """
    Z = np.concatenate([fake, real], axis=0)        # joint pool of features
    n = fake.shape[0]                               # size of the first group
    rng = np.random.default_rng(seed)               # RNG for permutations
    cnt = 0                                         # count how many permuted >= observed
    for _ in range(n_perm):
        idx = rng.permutation(Z.shape[0])           # permute indices
        Xp = Z[idx[:n]]                             # permuted group A
        Yp = Z[idx[n:]]                             # permuted group B
        mmd_p, _ = mmd2_rbf_unbiased(Xp, Yp, sigma=sigma)  # recompute MMD^2 using fixed sigma
        if mmd_p >= mmd_obs:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)                 # add-one smoothing

# ------------------------------- MAIN -------------------------------
def main():
    args = parse_args()                              # read CLI args
    np.random.seed(args.seed); torch.manual_seed(args.seed)  # set seeds for reproducibility

    # Device for Inception: prefer GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = build_inception_and_transform(device)  # model + preprocessing

    # Extract features (N×2048) for both sets
    print("[features] extracting Inception features ...")
    F = extract_features(args.fake_dir, model, transform, args.batch, device)  # fake features
    R = extract_features(args.real_dir, model, transform, args.batch, device)  # real features
    print(f"fake: {F.shape}, real: {R.shape}")

    report = {}                                      # dictionary to optionally serialize as JSON

    # ---- C2ST ----
    if args.test in ("c2st", "both"):
        print("[C2ST] training logistic regression and running permutation test ...")
        auc, p = c2st_auc_pvalue(F, R, seed=args.seed, n_perm=args.permutations)  # statistic + p-value
        report["c2st_auc"] = auc
        report["c2st_p_value"] = p
        print(f"C2ST ROC-AUC: {auc:.4f} | p-value: {p:.4f}")

    # ---- MMD ----
    if args.test in ("mmd", "both"):
        print("[MMD] computing unbiased MMD^2 (RBF) and permutation test ...")
        mmd2, sigma = mmd2_rbf_unbiased(F, R, sigma=None)                         # compute statistic
        pval = mmd_permutation_pvalue(F, R, mmd_obs=mmd2, sigma=sigma,
                                      n_perm=args.permutations, seed=args.seed)   # permutation p-value
        report["mmd2_rbf"] = mmd2
        report["mmd_sigma"] = sigma
        report["mmd_p_value"] = pval
        print(f"MMD^2 (RBF, sigma={sigma:.4g}): {mmd2:.6f} | p-value: {pval:.4f}")

    # Optionally dump a JSON report
    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)            # ensure folder exists
        with open(args.save_json, "w") as f:
            json.dump(report, f, indent=2)                                        # write results
        print(f"[report] saved -> {args.save_json}")

if __name__ == "__main__":
    main()  # hand off to main() when run as a script

