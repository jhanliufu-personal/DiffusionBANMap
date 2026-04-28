from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FaceDataset(Dataset):

    def __init__(self, image_paths: List[str], image_size: Tuple[int, int], eager: bool = False):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),  # → [0, 1], shape [C, H, W]
        ])
        # Pre-load all images into RAM as tensors to avoid per-batch file I/O
        self._cache: Optional[torch.Tensor] = None
        if eager:
            self._cache = torch.stack([
                self.transform(Image.open(p).convert("RGB"))
                for p in image_paths
            ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self._cache is not None:
            return self._cache[idx]
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)

    def remap_paths(self, old_prefix: str, new_prefix: str) -> None:
        """Rewrite stored image paths for cross-machine portability (e.g. Windows → Colab)."""
        self.image_paths = [
            new_prefix + p[len(old_prefix):].replace("\\", "/")
            for p in self.image_paths
        ]
