# utils.py
# Line-by-line commented utility functions for seeding, EMA, checkpointing, and image saving.
from __future__ import annotations  # Enable future annotations behavior for forward references

import os  # For interacting with the operating system (paths, folders)
import random  # Python's random for reproducible seeds alongside numpy/torch
from dataclasses import dataclass  # Convenient container for configuration objects
from typing import Dict  # Type hints for readability and static checking

import numpy as np  # For numeric utilities and seeding
import torch  # Core deep learning library
from torchvision.utils import save_image  # Utility to save image grids as PNG

def set_seed(seed: int = 42) -> None:
    """Set seeds for Python, NumPy, and PyTorch to improve reproducibility."""
    random.seed(seed)  # Seed Python's random
    np.random.seed(seed)  # Seed NumPy RNG
    torch.manual_seed(seed)  # Seed PyTorch RNG (CPU)
    torch.cuda.manual_seed_all(seed)  # Seed all CUDA devices if present
    torch.backends.cudnn.deterministic = True  # Make CuDNN deterministic (may slow down)
    torch.backends.cudnn.benchmark = False  # Disable benchmark to avoid nondeterministic algorithms

class EMA:
    """Exponential Moving Average for model parameters; stabilizes sampling."""
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.model = model  # Reference to the model to track
        self.decay = decay  # EMA decay factor
        self.shadow: Dict[str, torch.Tensor] = {}  # Dict storing moving averages
        self.backup: Dict[str, torch.Tensor] = {}  # Backup for temporary swaps
        # Initialize shadow weights as copies of current parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self) -> None:
        """Update EMA weights using current model parameters."""
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data

    def store(self) -> None:
        """Backup current model parameters before swapping to EMA weights."""
        self.backup = {name: p.data.clone() for name, p in self.model.named_parameters()}

    def copy_to(self) -> None:
        """Copy EMA (shadow) weights into the model for evaluation/sampling."""
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """Restore original model parameters after evaluation."""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def save_grid(tensor: torch.Tensor, path: str, nrow: int = 8, normalize: bool = True) -> None:
    """Save a grid of images; uses value_range (-1,1) when normalize=True."""
    ensure_dir(os.path.dirname(path) or ".")
    save_image(tensor, path, nrow=nrow, normalize=normalize, value_range=(-1, 1))

@dataclass
class TrainConfig:
    """Configuration values used by training and evaluation scripts."""
    data_root: str = "./CUB_200_2011"
    img_size: int = 64
    batch_size: int = 64
    epochs: int = 50
    lr: float = 2e-4
    ema_decay: float = 0.999
    num_steps: int = 1000
    cond_mode: str = "class"   # 'none' | 'class'
    num_classes: int = 200
    guidance_scale: float = 1.0
    p_uncond: float = 0.1
    schedule: str = "cosine"   # 'linear' | 'cosine'
    outdir: str = "runs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
