# utils.py
# Small utilities: EMA, image grid saving, dirs, seeding.

import os
import math
import torch
import random
import numpy as np
from torchvision.utils import save_image


class EMA:
    """Exponential Moving Average of model weights."""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self):
        for k, v in self.model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=False)


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_grid(x: torch.Tensor, path: str, nrow: int = 6):
    """
    x in [-1,1], save as a grid.
    """
    x = (x + 1) / 2.0
    save_image(x, path, nrow=nrow)
