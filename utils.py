import os                                # filesystem
import random                            # Python RNG
import numpy as np                       # NumPy RNG
import torch                             # PyTorch
from torchvision.utils import save_image as _save_image  # grid writer


def set_seed(seed: int):
    """Seed python/numpy/torch (and CUDA if available) for determinism."""
    random.seed(seed)                    # seed Python RNG
    np.random.seed(seed)                 # seed NumPy RNG
    torch.manual_seed(seed)              # seed CPU RNG
    if torch.cuda.is_available():        # also seed CUDA RNGs
        torch.cuda.manual_seed_all(seed)


def save_grid(x: torch.Tensor, path: str, nrow: int = 8):
    """Save a grid of images x∈[0,1] to PNG."""
    os.makedirs(os.path.dirname(path), exist_ok=True)  # ensure folder exists
    _save_image(x, path, nrow=nrow)                    # write file


class EMAHelper:
    """Simple Exponential Moving Average (EMA) of model parameters."""
    def __init__(self, mu=0.9995):
        self.mu = float(mu)             # decay factor close to 1
        self.shadow = None              # dict of averaged parameters

    def register(self, model):
        """Initialize EMA with the model's current parameters."""
        self.shadow = {name: p.clone().detach() for name, p in model.state_dict().items()}

    def update(self, model):
        """Update EMA after each optimizer step."""
        for name, p in model.state_dict().items():
            assert name in self.shadow
            self.shadow[name] = (1.0 - self.mu) * p.detach() + self.mu * self.shadow[name]

    def copy_to(self, model):
        """Copy EMA weights into model (for evaluation/sampling)."""
        state = model.state_dict()
        for name in state:
            if name in self.shadow:
                state[name].copy_(self.shadow[name])

