import os  # import modules
from torchvision import datasets  # import names from module


def make_dataset(root: str, transform):  # define function make_dataset
    # Expect ImageFolder structure: root/class_x/*.jpg, root/class_y/*.jpg, ...  # comment  # statement
    ds = datasets.ImageFolder(root=root, transform=transform)  # variable assignment
    num_classes = len(ds.classes)  # variable assignment
    return ds, num_classes  # return value
