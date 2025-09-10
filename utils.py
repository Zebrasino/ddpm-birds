# utils.py
# Training utilities: EMA, seed, checkpoint save/load, cosine LR with warmup.
# Every line commented.

import os                                  # filesystem
import math                                # math ops
from typing import Dict                    # typing
import torch                               # torch

class EMA:
    # Exponential moving average of parameters.
    def __init__(self, model, mu: float = 0.9995):
        self.mu = mu                        # decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}  # copy

    def update(self, model):
        # Update shadow weights in-place.
        for k, v in model.state_dict().items():   # iterate params/buffers
            if v.dtype.is_floating_point:         # only float tensors
                self.shadow[k].mul_(self.mu).add_(v.detach(), alpha=1.0 - self.mu)

    def copy_to(self, model):
        # Load shadow (EMA) into a model (for eval/sampling)
        model.load_state_dict(self.shadow, strict=False)

def seed_everything(seed: int = 0):
    # Make runs deterministic-ish (GPU nondeterminism may remain).
    import random                           # python RNG
    import numpy as np                      # numpy RNG
    random.seed(seed)                       # seed py
    np.random.seed(seed)                    # seed np
    torch.manual_seed(seed)                 # seed torch CPU
    torch.cuda.manual_seed_all(seed)        # seed torch GPU
    torch.backends.cudnn.benchmark = False  # deterministic kernels
    torch.backends.cudnn.deterministic = True

def cosine_warmup_lr(step: int, total: int, base_lr: float, warmup: int = 100):
    # Cosine LR with linear warmup for first 'warmup' steps.
    if step < warmup:                        # warmup phase
        return base_lr * (step + 1) / warmup
    # cosine decay from warmup..total
    t = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * t))

def save_ckpt(path: str, model, ema: EMA, args: Dict, step: int):
    # Save current and EMA weights plus args/step.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "ema": ema.shadow if ema is not None else None,
        "args": args,
        "step": step,
    }, path)

def load_ckpt(path: str, map_location="cpu"):
    # Load checkpoint dict.
    return torch.load(path, map_location=map_location)


