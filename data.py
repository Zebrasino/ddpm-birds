# data.py
# Line-by-line commented dataset and dataloader utilities for CUB-200-2011.
from __future__ import annotations  # Future annotations for cleaner type hints

import os  # For filesystem operations
from typing import Tuple, Optional  # Type hints for function signatures

from PIL import Image  # Image loading for non-lazy datasets
import torch  # For tensors
from torch.utils.data import Dataset, DataLoader  # PyTorch dataset/dataloader abstractions
import torchvision.transforms as T  # Common image transforms

class CUBDataset(Dataset):
    """Minimal CUB-200-2011 dataset with class-conditional labels.

    Expects a directory structure with images under subfolders per class, or a flat folder with a CSV mapping.
    For didactic purposes, we only handle the common 'class-subfolders' layout.

    """
    def __init__(self, root: str, img_size: int = 64, split: str = "train"):
        self.root = root  # Dataset root path
        self.img_size = img_size  # Desired final image resolution
        self.split = split  # Split name, could be 'train' or 'test' depending on your setup

        # We assume the typical structure: root/class_x/*.jpg, root/class_y/*.jpg, ...
        # Collect (filepath, class_index) pairs.
        self.samples = []  # Will store tuples (path, class_id)
        self.class_to_idx = {}  # Map from class name (folder) to integer index
        self.idx_to_class = []  # Reverse map for convenience

        # Walk through the root directory and index class subfolders
        for cls_name in sorted(os.listdir(root)):
            cls_path = os.path.join(root, cls_name)  # Path to a specific class folder
            if not os.path.isdir(cls_path):  # Skip non-directories
                continue
            cls_idx = len(self.idx_to_class)  # Next available class index
            self.class_to_idx[cls_name] = cls_idx  # Record mapping
            self.idx_to_class.append(cls_name)  # Record reverse mapping
            for fname in sorted(os.listdir(cls_path)):  # Iterate files within class folder
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):  # Only image files
                    self.samples.append((os.path.join(cls_path, fname), cls_idx))  # Add sample

        # Define transformations: resize, center-crop, convert to tensor, and map to [-1,1]
        self.transform = T.Compose([
            T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),  # Resize shorter side
            T.CenterCrop(img_size),  # Center crop to a square
            T.ToTensor(),  # Convert PIL image to [0,1] tensor
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Map to [-1,1] per channel
        ])

    def __len__(self) -> int:
        return len(self.samples)  # Number of available image-label pairs

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[int]]:
        path, cls = self.samples[idx]  # Unpack file path and class id
        img = Image.open(path).convert('RGB')  # Load image and ensure RGB mode
        x = self.transform(img)  # Apply preprocessing transforms
        y = cls  # Our label is simply the class index
        return x, y  # Return image tensor and integer label

def make_loader(root: str, img_size: int, batch_size: int, split: str = "train",
                num_workers: int = 4) -> DataLoader:
    """Create a DataLoader for the CUB dataset with common defaults."""
    ds = CUBDataset(root=root, img_size=img_size, split=split)  # Instantiate dataset
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=(split == "train"),  # Shuffle during training
        num_workers=num_workers, pin_memory=True, drop_last=True  # Typical perf settings
    )
    return loader  # Return the configured DataLoader
