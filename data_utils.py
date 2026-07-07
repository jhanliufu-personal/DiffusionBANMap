import os
from typing import List, Optional, Tuple

import numpy as np
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
    """Walk data_dir and return paths of all JPEG/PNG/TIFF images."""
    paths = []
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if "Zone.Identifier" in fname:
                continue
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
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


class _ImageNet64ArrayDataset(Dataset):
    """Memory-maps a (N, 3, 64, 64) uint8 array produced by prepare_imagenet64_dataset.py.

    Backed by disk (np.load(mmap_mode='r')) rather than preloaded into RAM, since the
    full training set (~15 GB) can exceed a standard Colab runtime's memory.

    If latents_path is given (an (N, latent_dim) array whose row i is the precomputed
    conditioning latent for image row i — e.g. AlexNet-fc6-PCA latents written by
    notebooks/alexnet_pca_latents.ipynb), __getitem__ returns (image, latent) tuples
    instead of a plain image tensor.
    """

    def __init__(self, npy_path: str, image_size: int = 64, latents_path: Optional[str] = None):
        self.images = np.load(npy_path, mmap_mode='r')
        self._resize = transforms.Resize(image_size) if image_size != 64 else None
        self.latents = np.load(latents_path, mmap_mode='r') if latents_path is not None else None
        if self.latents is not None and len(self.latents) != len(self.images):
            raise ValueError(
                f"Latents/images count mismatch: {len(self.latents)} ({latents_path}) "
                f"vs {len(self.images)} ({npy_path})"
            )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = torch.from_numpy(np.array(self.images[idx])).float() / 255.0
        if self._resize is not None:
            img = self._resize(img)
        if self.latents is None:
            return img
        latent = torch.from_numpy(np.array(self.latents[idx])).float()
        return img, latent


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
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_dl, test_dl) that yield image tensors [B, C, H, W] in [0, 1].

    All images are preloaded into RAM for fast training-loop IO.

    shuffle_train=False makes train_dl unshuffled with no dropped batch (test_dl already
    behaves this way) — for a full, order-preserving pass instead of actual training.
    """
    paths = _find_images(data_dir, expressions)
    all_images = _preload(paths, image_size)

    train_idx, test_idx = _split(len(all_images), train_split, seed)
    train_dl = DataLoader(
        _TensorDataset(all_images[train_idx]),
        batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers,
        pin_memory=True, drop_last=shuffle_train,
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
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_dl, test_dl) over CelebA that yield [B, C, H, W] in [0, 1].

    Images are loaded on-the-fly (dataset is too large to preload into RAM).
    Standard preprocessing: center-crop to 178×178, then resize to image_size×image_size.

    shuffle_train=False makes train_dl unshuffled with no dropped batch (test_dl already
    behaves this way) — for a full, order-preserving pass instead of actual training.
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
        batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers,
        pin_memory=True, drop_last=shuffle_train,
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
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_dl, val_dl) over Tiny ImageNet yielding [B, C, H, W] in [0, 1].

    Scans train/ (100K images across 200 class subdirs) and val/images/ (10K flat).
    Labels are ignored — this is for VAE pretraining.
    Images are natively 64×64; image_size allows resizing if needed.

    shuffle_train=False makes train_dl unshuffled with no dropped batch (val_dl already
    behaves this way) — for a full, order-preserving pass instead of actual training.
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
        batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers,
        pin_memory=True, drop_last=shuffle_train,
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


def build_imagenet64_dataloaders(
    data_dir: str,
    image_size: int = 64,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_dl, val_dl) over the pre-downsampled ImageNet64 benchmark
    (van den Oord et al. pickled-batch format, consolidated into flat train_images.npy /
    val_images.npy arrays by scripts/prepare_imagenet64_dataset.py).

    Images are already 64×64 uint8 arrays — memory-mapped from disk rather than
    preloaded, since the full training set (~15 GB) can exceed a standard Colab
    runtime's RAM. Honors the benchmark's own train/val split.

    If train_latents.npy / val_latents.npy are also present in data_dir (written by
    notebooks/alexnet_pca_latents.ipynb, row-aligned with train_images.npy / val_images.npy),
    batches become (image, latent) tuples for precomputed-latent conditioning instead of
    plain image tensors — no config flag needed, this is detected purely from what's on disk.

    shuffle_train=False makes train_dl unshuffled with no dropped batch (val_dl already
    behaves this way) — for a full, order-preserving pass (row i in == row i out), e.g.
    extracting features to write those very latents files, rather than actual training.
    """
    train_path = os.path.join(data_dir, "train_images.npy")
    val_path = os.path.join(data_dir, "val_images.npy")
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(
            f"Expected {train_path} and {val_path} — run scripts/prepare_imagenet64_dataset.py first"
        )

    train_latents_path = os.path.join(data_dir, "train_latents.npy")
    val_latents_path = os.path.join(data_dir, "val_latents.npy")
    train_latents_path = train_latents_path if os.path.exists(train_latents_path) else None
    val_latents_path = val_latents_path if os.path.exists(val_latents_path) else None
    if (train_latents_path is None) != (val_latents_path is None):
        raise FileNotFoundError(
            f"Found latents for one split but not the other in {data_dir} — expected both "
            "train_latents.npy and val_latents.npy, or neither."
        )

    train_dl = DataLoader(
        _ImageNet64ArrayDataset(train_path, image_size, train_latents_path),
        batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers,
        pin_memory=True, drop_last=shuffle_train,
    )
    val_dl = DataLoader(
        _ImageNet64ArrayDataset(val_path, image_size, val_latents_path),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )
    conditioning = "precomputed latents" if train_latents_path else "none (unconditional / on-the-fly VAE)"
    print(
        f"ImageNet64 — Train: {len(train_dl.dataset):,} images ({len(train_dl)} batches/epoch) | "
        f"Val: {len(val_dl.dataset):,} images ({len(val_dl)} batches) | conditioning: {conditioning}"
    )
    return train_dl, val_dl


def build_encode_dataloader(
    data_dir: str,
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
) -> DataLoader:
    """Return a single DataLoader over all images in data_dir for encoding/inference."""
    paths = _find_all_images(data_dir)
    if not paths:
        raise FileNotFoundError(f"No images found in {data_dir}")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    print(f"Encode set: {len(paths)} images ({-(-len(paths) // batch_size)} batches)")
    return DataLoader(
        _PathDataset(paths, transform),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
    )


def build_stimuli_dataloaders(
    data_dir: str,
    image_size: int = 224,
    train_split: float = 0.9,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_dl, test_dl) over a flat directory of stimuli images.

    Images are loaded on-the-fly and converted to RGB (handles grayscale TIFF).

    shuffle_train=False makes train_dl unshuffled with no dropped batch (test_dl already
    behaves this way) — for a full, order-preserving pass instead of actual training.
    """
    paths = _find_all_images(data_dir)
    if not paths:
        raise FileNotFoundError(f"No images found in {data_dir}")

    train_idx, test_idx = _split(len(paths), train_split, seed)
    train_paths = [paths[i] for i in train_idx.tolist()]
    test_paths  = [paths[i] for i in test_idx.tolist()]

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    train_dl = DataLoader(
        _PathDataset(train_paths, transform),
        batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers,
        pin_memory=True, drop_last=shuffle_train,
    )
    test_dl = DataLoader(
        _PathDataset(test_paths, transform),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )
    print(
        f"Stimuli — Train: {len(train_paths)} images ({len(train_dl)} batches/epoch) | "
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
    elif dataset_type == "stimuli":
        return build_stimuli_dataloaders(
            data_dir=cfg.data_dir,
            image_size=h,
            train_split=cfg.train_split,
            batch_size=cfg.batch_size,
        )
    elif dataset_type == "imagenet64":
        return build_imagenet64_dataloaders(
            data_dir=cfg.data_dir,
            image_size=h,
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
