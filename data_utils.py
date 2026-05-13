import os
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def _find_images(cfd_dir: str, expressions: List[str]) -> List[str]:
    paths = []
    for root, _, files in os.walk(cfd_dir):
        for fname in files:
            if not fname.endswith(".jpg") or "Zone.Identifier" in fname:
                continue
            expr = fname.rsplit("-", 1)[-1].replace(".jpg", "")
            if expr in expressions:
                paths.append(os.path.join(root, fname))
    return sorted(paths)


class _PathDataset(Dataset):
    def __init__(self, paths: List[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.transform(Image.open(self.paths[idx]).convert("RGB"))


class _TensorDataset(Dataset):
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __len__(self) -> int:
        return len(self.tensor)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tensor[idx]


def _preload(paths: List[str], image_size: Tuple[int, int]) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    ds = _PathDataset(paths, transform)
    loader = DataLoader(ds, batch_size=128, num_workers=8, pin_memory=False)
    print(f"Preloading {len(ds)} images into RAM ...")
    return torch.cat([batch for batch in loader], dim=0)


def build_face_dataloaders(
    cfd_dir: str,
    expressions: List[str],
    image_size: Tuple[int, int],
    train_split: float = 0.8,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_dl, test_dl) that yield image tensors [B, C, H, W] in [0, 1].

    Images are loaded from cfd_dir once at startup and held in RAM so the
    training loop never touches disk.
    """
    paths = _find_images(cfd_dir, expressions)
    all_images = _preload(paths, image_size)

    n = len(all_images)
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng)
    n_train = int(n * train_split)
    train_tensor = all_images[perm[:n_train]]
    test_tensor = all_images[perm[n_train:]]

    train_dl = DataLoader(
        _TensorDataset(train_tensor),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    test_dl = DataLoader(
        _TensorDataset(test_tensor),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )
    print(
        f"Train: {n_train} images ({len(train_dl)} batches/epoch) | "
        f"Test: {n - n_train} images ({len(test_dl)} batches)"
    )
    return train_dl, test_dl
