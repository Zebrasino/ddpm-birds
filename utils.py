import os  # OS utilities
import random  # Python RNG
import numpy as np  # NumPy RNG
import torch  # PyTorch core
from torchvision.utils import save_image as _save_image  # image grid saver


def set_seed(seed: int):  # determinism helper
    random.seed(seed)  # seed python rng
    np.random.seed(seed)  # seed numpy rng
    torch.manual_seed(seed)  # seed cpu torch
    if torch.cuda.is_available():  # if cuda present
        torch.cuda.manual_seed_all(seed)  # seed all gpus


def save_grid(x: torch.Tensor, path: str, nrow: int = 8):  # save a grid of images in [0,1]
    os.makedirs(os.path.dirname(path), exist_ok=True)  # ensure dir exists
    _save_image(x, path, nrow=nrow)  # write png


class EMAHelper:  # simple Exponential Moving Average tracker
    def __init__(self, mu=0.999):  # mu close to 1 → slower EMA
        self.mu = float(mu)  # decay factor
        self.shadow = None  # dict of parameter averages

    def register(self, model):  # initialize shadow from model
        self.shadow = {name: p.clone().detach() for name, p in model.state_dict().items()}  # copy weights

    def update(self, model):  # update shadow after each step
        for name, p in model.state_dict().items():  # iterate all tensors
            assert name in self.shadow  # must exist
            self.shadow[name] = (1.0 - self.mu) * p.detach() + self.mu * self.shadow[name]  # ema update

    def copy_to(self, model):  # load ema weights into model
        state = model.state_dict()  # current weights
        for name in state:  # iterate tensors
            if name in self.shadow:  # if tracked
                state[name].copy_(self.shadow[name])  # overwrite with ema

