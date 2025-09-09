# utils.py
# Line-by-line commented utility functions for seeding, EMA, checkpointing, and image saving.
from __future__ import annotations  # Enable future annotations behavior for forward references

import os  # For interacting with the operating system (paths, folders)
import math  # Might be useful for numeric helpers
import random  # Python's random for reproducible seeds alongside numpy/torch
from dataclasses import dataclass  # Convenient container for configuration objects
from typing import Any, Dict  # Type hints for readability and static checking

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
    """Exponential Moving Average for model parameters, typically stabilizes sampling."""
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.model = model  # Reference to the model whose weights we track
        self.decay = decay  # EMA decay rate; larger means slower updates, more smoothing
        self.shadow: Dict[str, torch.Tensor] = {}  # Dict storing the moving-averaged weights
        self.backup: Dict[str, torch.Tensor] = {}  # Optional backup for swapping

        # Initialize shadow weights as a copy of current parameters
        for name, param in model.named_parameters():
            if param.requires_grad:  # Track only trainable parameters
                self.shadow[name] = param.data.clone()

    @torch.no_grad()  # No gradients needed when updating EMA weights
    def update(self) -> None:
        """Update EMA weights using the current model parameters."""
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue  # Skip frozen parameters
            assert name in self.shadow, "Unexpected parameter in EMA"
            new_avg = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
            self.shadow[name] = new_avg  # Store the updated moving average

    def store(self) -> None:
        """Save current model parameters to self.backup for a temporary swap."""
        self.backup = {name: p.data.clone() for name, p in self.model.named_parameters()}

    def copy_to(self) -> None:
        """Copy EMA (shadow) weights to the model for evaluation/sampling."""
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """Restore original model parameters from self.backup after evaluation."""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}  # Clear backup after restoring

def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)  # Recursively create folders as needed

def save_grid(tensor: torch.Tensor, path: str, nrow: int = 8, normalize: bool = True) -> None:
    """Save a grid of images to disk, normalizing from [-1,1] to [0,1] if needed."""
    ensure_dir(os.path.dirname(path) or ".")  # Make sure the output folder exists
    save_image(tensor, path, nrow=nrow, normalize=normalize, value_range=(-1, 1))  # Persist PNG grid

@dataclass
class TrainConfig:
    """Configuration values used by training and evaluation scripts."""
    data_root: str = "./CUB_200_2011"  # Root path to the dataset
    img_size: int = 64  # Target image resolution (square), e.g., 64 or 128
    batch_size: int = 64  # Batch size per step
    epochs: int = 50  # Number of training epochs
    lr: float = 2e-4  # Learning rate for AdamW
    ema_decay: float = 0.999  # EMA decay factor
    num_steps: int = 1000  # Number of diffusion steps T
    cond_mode: str = "class"  # Conditioning: 'none' | 'class'
    num_classes: int = 200  # CUB has 200 bird species
    guidance_scale: float = 1.0  # Classifier-free guidance scale
    p_uncond: float = 0.1  # Dropout rate for unconditional branch (CFG)
    schedule: str = "cosine"  # 'linear' or 'cosine'
    outdir: str = "runs"  # Where to store checkpoints and samples
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # Compute device
