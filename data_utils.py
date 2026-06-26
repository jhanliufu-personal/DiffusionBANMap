import os
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ── Internal helpers ──────────────────────────────────────────────────────────

def _find_images(data_dir: str, expressions: List[str]) -> List[str]:
    """Walk data_dir and return paths of images whose expression code is in expressions."""
    paths = []
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if not fname.endswith(".jpg") or "Zone.Identifier" in fname:
                continue
            expr = fname.rsplit("-", 1)[-1].replace(".jpg", "")
            if expr in expressions:
                paths.append(os.path.join(root, fname))
    return sorted(paths)


def _find_all_images(data_dir: str) -> List[str]:
    """Walk data_dir and return paths of all JPEG/PNG images."""
    paths = []
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if "Zone.Identifier" in fname:
                continue
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
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


def _split(n: int, train_split: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng)
    n_train = int(n * train_split)
    return perm[:n_train], perm[n_train:]


# ── Public API ────────────────────────────────────────────────────────────────

def build_face_dataloaders(
    data_dir: str,
    expressions: List[str],
    image_size: Tuple[int, int],
    train_split: float = 0.8,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_dl, test_dl) that yield image tensors [B, C, H, W] in [0, 1].

    All images are preloaded into RAM for fast training-loop IO.
    """
    paths = _find_images(data_dir, expressions)
    all_images = _preload(paths, image_size)

    train_idx, test_idx = _split(len(all_images), train_split, seed)
    train_dl = DataLoader(
        _TensorDataset(all_images[train_idx]),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    test_dl = DataLoader(
        _TensorDataset(all_images[test_idx]),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )
    n_train, n_test = len(train_idx), len(test_idx)
    print(
        f"Train: {n_train} images ({len(train_dl)} batches/epoch) | "
        f"Test: {n_test} images ({len(test_dl)} batches)"
    )
    return train_dl, test_dl


def build_celeba_dataloaders(
    data_dir: str,
    image_size: int = 256,
    train_split: float = 0.9,
    batch_size: int = 16,
    num_workers: int = 8,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_dl, test_dl) over CelebA that yield [B, C, H, W] in [0, 1].

    Images are loaded on-the-fly (dataset is too large to preload into RAM).
    Standard preprocessing: center-crop to 178×178, then resize to image_size×image_size.
    """
    paths = _find_all_images(data_dir)
    if not paths:
        raise FileNotFoundError(f"No images found in {data_dir}")

    train_idx, test_idx = _split(len(paths), train_split, seed)
    train_paths = [paths[i] for i in train_idx.tolist()]
    test_paths  = [paths[i] for i in test_idx.tolist()]

    transform = transforms.Compose([
        transforms.CenterCrop(178),                         # standard CelebA square crop
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    train_dl = DataLoader(
        _PathDataset(train_paths, transform),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    test_dl = DataLoader(
        _PathDataset(test_paths, transform),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )
    print(
        f"CelebA — Train: {len(train_paths)} images ({len(train_dl)} batches/epoch) | "
        f"Test: {len(test_paths)} images ({len(test_dl)} batches)"
    )
    return train_dl, test_dl


def build_tiny_imagenet_dataloaders(
    data_dir: str,
    image_size: int = 64,
    train_split: float = 0.9,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_dl, val_dl) over Tiny ImageNet yielding [B, C, H, W] in [0, 1].

    Scans train/ (100K images across 200 class subdirs) and val/images/ (10K flat).
    Labels are ignored — this is for VAE pretraining.
    Images are natively 64×64; image_size allows resizing if needed.
    """
    paths = (
        _find_all_images(os.path.join(data_dir, "train"))
        + _find_all_images(os.path.join(data_dir, "val", "images"))
    )
    if not paths:
        raise FileNotFoundError(f"No images found under {data_dir}")

    train_idx, test_idx = _split(len(paths), train_split, seed)
    train_paths = [paths[i] for i in train_idx.tolist()]
    test_paths  = [paths[i] for i in test_idx.tolist()]

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    train_dl = DataLoader(
        _PathDataset(train_paths, transform),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    test_dl = DataLoader(
        _PathDataset(test_paths, transform),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )
    print(
        f"Tiny ImageNet — Train: {len(train_paths)} images ({len(train_dl)} batches/epoch) | "
        f"Test: {len(test_paths)} images ({len(test_dl)} batches)"
    )
    return train_dl, test_dl


def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader]:
    """Dispatch to the right dataloader based on cfg.dataset_type."""
    dataset_type = getattr(cfg, "dataset_type", "cfd")
    # Resolve image size — BetaVAE config uses input_height/input_width; Diffusion uses image_size
    h = getattr(cfg, "input_height", getattr(cfg, "image_size", 64))
    w = getattr(cfg, "input_width", h)

    if dataset_type == "celeba":
        return build_celeba_dataloaders(
            data_dir=cfg.data_dir,
            image_size=h,
            train_split=cfg.train_split,
            batch_size=cfg.batch_size,
        )
    elif dataset_type == "tiny_imagenet":
        return build_tiny_imagenet_dataloaders(
            data_dir=cfg.data_dir,
            image_size=h,
            train_split=cfg.train_split,
            batch_size=cfg.batch_size,
        )
    else:
        return build_face_dataloaders(
            data_dir=cfg.data_dir,
            expressions=cfg.expressions,
            image_size=(h, w),
            train_split=cfg.train_split,
            batch_size=cfg.batch_size,
        )
