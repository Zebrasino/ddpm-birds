# data.py
# Line-by-line commented dataset and dataloader utilities for CUB-200-2011.
from __future__ import annotations  # Future annotations for cleaner type hints

import os  # For filesystem operations
from typing import Tuple, Optional  # Type hints for function signatures

from PIL import Image  # Image loading
import torch  # For tensors
from torch.utils.data import Dataset, DataLoader  # Dataset/dataloader abstractions
import torchvision.transforms as T  # Common image transforms

class CUBDataset(Dataset):
    """Minimal CUB-200-2011 dataset with class-conditional labels.
    Assumes structure: root/class_x/*.jpg, root/class_y/*.jpg, ...
    """
    def __init__(self, root: str, img_size: int = 64, split: str = "train"):
        self.root = root  # Dataset root path
        self.img_size = img_size  # Target image size
        self.split = split  # Split name ('train' or 'test' if you use custom splits)

        self.samples = []  # List of (path, class_id) pairs
        self.class_to_idx = {}  # Map class name -> integer id
        self.idx_to_class = []  # Reverse map id -> class name

        # Index subfolders as classes
        for cls_name in sorted(os.listdir(root)):
            cls_path = os.path.join(root, cls_name)  # Path to a class folder
            if not os.path.isdir(cls_path):  # Skip files
                continue
            cls_idx = len(self.idx_to_class)  # Next available class id
            self.class_to_idx[cls_name] = cls_idx  # Record mapping
            self.idx_to_class.append(cls_name)  # Reverse mapping
            for fname in sorted(os.listdir(cls_path)):  # Files inside the class folder
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):  # Only images
                    self.samples.append((os.path.join(cls_path, fname), cls_idx))  # Add sample

        # Transformations: resize, center-crop, tensor, normalize to [-1,1]
        self.transform = T.Compose([
            T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self) -> int:
        return len(self.samples)  # Number of samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[int]]:
        path, cls = self.samples[idx]  # Unpack path and class id
        img = Image.open(path).convert('RGB')  # Load and ensure RGB
        x = self.transform(img)  # Apply transforms
        y = cls  # Label is the class id
        return x, y  # Return image tensor and label

def make_loader(root: str, img_size: int, batch_size: int, split: str = "train",
                num_workers: int = 2) -> DataLoader:
    """Create a DataLoader for the CUB dataset; defaults to 2 workers (Colab-friendly)."""
    ds = CUBDataset(root=root, img_size=img_size, split=split)  # Build dataset
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,   # Keep small (2) to avoid Colab warnings/freezes
        pin_memory=True,
        drop_last=True
    )
    return loader  # Return DataLoader
