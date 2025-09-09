import os  # import modules
import random  # import modules
import numpy as np  # import modules
import torch  # import modules
from torchvision.utils import save_image as _save_image  # import names from module


def set_seed(seed: int):  # define function set_seed
    random.seed(seed)  # statement
    np.random.seed(seed)  # statement
    torch.manual_seed(seed)  # statement
    if torch.cuda.is_available():  # control flow
        torch.cuda.manual_seed_all(seed)  # PyTorch operation


def save_grid(x: torch.Tensor, path: str, nrow: int = 8):  # define function save_grid
    os.makedirs(os.path.dirname(path), exist_ok=True)  # statement
    _save_image(x, path, nrow=nrow)  # statement


class EMAHelper:  # define class EMAHelper
    def __init__(self, mu=0.999):  # define function __init__
        self.mu = mu  # variable assignment
        self.shadow = None  # variable assignment

    def register(self, model):  # define function register
        self.shadow = {name: p.clone().detach() for name, p in model.state_dict().items()}  # variable assignment

    def update(self, model):  # define function update
        for name, p in model.state_dict().items():  # loop
            assert name in self.shadow  # statement
            new_average = (1.0 - self.mu) * p.detach() + self.mu * self.shadow[name]  # PyTorch operation
            self.shadow[name] = new_average.clone()  # variable assignment

    def copy_to(self, model):  # define function copy_to
        state = model.state_dict()  # variable assignment
        for name in state:  # loop
            if name in self.shadow:  # control flow
                state[name].copy_(self.shadow[name])  # PyTorch operation
