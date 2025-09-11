from __future__ import annotations               # future annotations
import os                                        # filesystem
import math                                      # math
from typing import Dict                          # typing
import torch                                     # tensors

class EMA:
    """Exponential moving average of model parameters."""
    def __init__(self, model, mu: float = 0.9995):
        self.mu = mu                                                 # decay factor
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}  # copy of weights

    def update(self, model):
        """Update EMA with current model parameters."""
        for k, v in model.state_dict().items():                      # iterate params/buffers
            if v.dtype.is_floating_point:                            # only float tensors
                self.shadow[k].mul_(self.mu).add_(v.detach(), alpha=1.0 - self.mu)  # EMA formula

    def copy_to(self, model):
        """Load EMA weights into the given model."""
        model.load_state_dict(self.shadow, strict=False)             # load shadow

def seed_everything(seed: int = 0):
    """Try to make runs deterministic-ish (CUDNN may remain nondeterministic)."""
    import random                                                    # stdlib RNG
    import numpy as np                                               # numpy RNG
    random.seed(seed)                                                # seed Python
    np.random.seed(seed)                                             # seed numpy
    torch.manual_seed(seed)                                          # seed torch CPU
    torch.cuda.manual_seed_all(seed)                                 # seed GPUs
    torch.backends.cudnn.benchmark = False                           # disable heuristics
    torch.backends.cudnn.deterministic = True                        # favor determinism

def cosine_warmup_lr(step: int, total: int, base_lr: float, warmup: int = 1000) -> float:
    """Cosine schedule with linear warmup for the first 'warmup' steps."""
    if step < warmup:                                                # warmup phase
        return base_lr * float(step + 1) / float(max(1, warmup))     # linear ramp
    t = (step - warmup) / float(max(1, total - warmup))              # normalized progress
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * t))             # cosine decay

def save_ckpt(path: str, model, ema: EMA | None, args: Dict, step: int):
    """Save model/EMA/args/step under the given path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)                # ensure dir
    torch.save({
        "model": model.state_dict(),                                 # raw weights
        "ema": (ema.shadow if ema is not None else None),            # shadow weights
        "args": args,                                                # training args
        "step": int(step),                                           # current step
    }, path)                                                         # write file

def load_ckpt(path: str, map_location="cpu"):
    """Load a checkpoint dict from a path."""
    return torch.load(path, map_location=map_location)               # read file

