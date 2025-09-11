# src/ddpm_birds/__init__.py
# Top-level package file so you can do: `from ddpm_birds import UNet, Diffusion, EMA, ...`

from __future__ import annotations          # future annotations for nicer type hints
from typing import List                     # export type

# Re-export the most commonly used components so callers can import from the package root.
from .unet import UNet                      # U-Net backbone used by the DDPM
from .diffusion import Diffusion            # Schedules, loss, samplers (DDPM/DDIM)
from .data import CUBBBoxDataset, make_cub_bbox_dataset  # CUB dataset with bbox + mask
from .utils import (                        # Small set of utilities used across scripts
    EMA,                                    # Exponential Moving Average wrapper
    seed_everything,                        # Convenience seeding helper
    cosine_warmup_lr,                       # Cosine LR with warmup
    save_ckpt,                              # Serialize checkpoint dicts
    load_ckpt,                              # Load checkpoint dicts
)

__all__: List[str] = [                      # What `from ddpm_birds import *` will export
    "UNet",
    "Diffusion",
    "CUBBBoxDataset",
    "make_cub_bbox_dataset",
    "EMA",
    "seed_everything",
    "cosine_warmup_lr",
    "save_ckpt",
    "load_ckpt",
]

__version__ = "0.1.0"                       # Simple package version string
