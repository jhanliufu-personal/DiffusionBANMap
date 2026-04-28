from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FaceDataset(Dataset):

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

    def remap_paths(self, old_prefix: str, new_prefix: str) -> None:
        """Rewrite stored image paths for cross-machine portability (e.g. Windows → Colab)."""
        self.image_paths = [
            new_prefix + p[len(old_prefix):].replace("\\", "/")
            for p in self.image_paths
        ]
