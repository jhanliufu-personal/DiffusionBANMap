import os
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def find_images(cfd_dir: str, expressions: List[str]) -> List[str]:
    paths = []
    for root, _, files in os.walk(cfd_dir):
        for fname in files:
            if not fname.endswith(".jpg") or "Zone.Identifier" in fname:
                continue
            expr = fname.rsplit("-", 1)[-1].replace(".jpg", "")
            if expr in expressions:
                paths.append(os.path.join(root, fname))
    return sorted(paths)


class FaceDataset(Dataset):
    """Path-based dataset used during dataset preparation to apply transforms."""

    def __init__(self, image_paths: List[str], image_size: Tuple[int, int]):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),  # → [0, 1], shape [C, H, W]
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)


class PreloadedDataset(Dataset):
    """Dataset backed by a pre-processed tensor [N, C, H, W] already in memory."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __len__(self) -> int:
        return len(self.tensor)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tensor[idx]
