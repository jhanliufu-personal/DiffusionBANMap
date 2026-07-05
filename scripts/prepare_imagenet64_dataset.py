"""Stage the Downsampled ImageNet64 benchmark from Google Drive into a Colab runtime.

Expects you've already downloaded the three standard zips into a Google Drive
folder yourself (image-net.org requires a login, so this can't be automated):
    Imagenet64_train_part1.zip   (5 of the 10 train_data_batch_* pickle files)
    Imagenet64_train_part2.zip   (the other 5 train_data_batch_* pickle files)
    Imagenet64_val.zip           (the val_data pickle file)

Each pickle is a dict with a 'data' key: a (N, 12288) uint8 array, one row per
64×64 RGB image, stored channel-major (reshape(-1, 3, 64, 64) is already CHW).

Usage (inside a Colab notebook, after `drive.mount('/content/drive')`):
    python scripts/prepare_imagenet64_dataset.py \\
        --drive_train_part1_zip /content/drive/MyDrive/imagenet64/Imagenet64_train_part1.zip \\
        --drive_train_part2_zip /content/drive/MyDrive/imagenet64/Imagenet64_train_part2.zip \\
        --drive_val_zip         /content/drive/MyDrive/imagenet64/Imagenet64_val.zip \\
        --data_dir /content/data/imagenet64

Copies the zips to local disk first (a single large sequential copy is far faster
and more reliable than random-access reads against a Drive-mounted FUSE filesystem),
unzips them, then consolidates the per-batch pickles into two flat arrays:
    <data_dir>/train_images.npy   (1,281,167 × 3 × 64 × 64 uint8)
    <data_dir>/val_images.npy     (50,000 × 3 × 64 × 64 uint8)

Consolidation streams one batch at a time into a memory-mapped .npy file, so peak
RAM stays at one batch's size (~1.6 GB) regardless of total dataset size.

Set data_dir in your config's data_dir field after running this.
"""

import argparse
import os
import pickle
import shutil
import zipfile

import numpy as np


def _copy_if_needed(src: str, dst: str) -> None:
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
        print(f"Already staged: {dst}")
        return
    size_gb = os.path.getsize(src) / 1e9
    print(f"Copying {src} → {dst} ({size_gb:.1f} GB) ...")
    shutil.copy2(src, dst)


def _extract_zip(zip_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    marker = os.path.join(out_dir, f".extracted_{os.path.basename(zip_path)}")
    if os.path.exists(marker):
        print(f"Already extracted: {zip_path}")
        return
    print(f"Extracting {zip_path} → {out_dir} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    open(marker, "w").close()  # written last, so a crash mid-extract doesn't look "done"


def _find_batches(root: str, name_substr: str) -> list:
    matches = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if name_substr in fname:
                matches.append(os.path.join(dirpath, fname))
    return sorted(matches)


def _get_data_field(d: dict) -> np.ndarray:
    """'data' key is a plain str in this benchmark's pickles; fall back to bytes
    in case a differently-repickled mirror stores it as a legacy Python 2 key."""
    return np.asarray(d["data"] if "data" in d else d[b"data"])


def _batch_num_images(path: str) -> int:
    with open(path, "rb") as f:
        d = pickle.load(f)
    return _get_data_field(d).shape[0]


def _load_batch(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        d = pickle.load(f)
    data = _get_data_field(d).astype(np.uint8)
    return data.reshape(-1, 3, 64, 64)


def _consolidate(batch_paths: list, out_path: str) -> None:
    if os.path.exists(out_path):
        print(f"Already consolidated: {out_path}")
        return

    print(f"Consolidating {len(batch_paths)} batch file(s) → {out_path} ...")
    counts = [_batch_num_images(p) for p in batch_paths]
    total = sum(counts)

    # Write to a tmp path and rename into place at the end, so a crash mid-write
    # can't leave a partially-filled file sitting at out_path looking complete.
    tmp_path = out_path + ".tmp"
    out = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.uint8, shape=(total, 3, 64, 64))
    offset = 0
    for path, n in zip(batch_paths, counts):
        out[offset:offset + n] = _load_batch(path)
        offset += n
        print(f"  {offset:,}/{total:,} images written")
    out.flush()
    del out
    os.rename(tmp_path, out_path)
    print(f"  Done: {total:,} images ({total * 3 * 64 * 64 / 1e9:.1f} GB) → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage ImageNet64 pickled batches from Drive into a Colab runtime")
    parser.add_argument("--drive_train_part1_zip", required=True, help="Path to Imagenet64_train_part1.zip")
    parser.add_argument("--drive_train_part2_zip", required=True, help="Path to Imagenet64_train_part2.zip")
    parser.add_argument("--drive_val_zip", required=True, help="Path to Imagenet64_val.zip")
    parser.add_argument("--data_dir", required=True, help="Local staging/output directory (e.g. /content/data/imagenet64)")
    parser.add_argument("--cleanup", action="store_true", help="Delete staged zips + raw pickle batches after consolidation")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    train_out = os.path.join(args.data_dir, "train_images.npy")
    val_out = os.path.join(args.data_dir, "val_images.npy")
    if os.path.exists(train_out) and os.path.exists(val_out):
        print(f"Already fully prepared: {train_out}, {val_out}")
        return

    staging = os.path.join(args.data_dir, "_staging")
    os.makedirs(staging, exist_ok=True)

    local_part1 = os.path.join(staging, "Imagenet64_train_part1.zip")
    local_part2 = os.path.join(staging, "Imagenet64_train_part2.zip")
    local_val = os.path.join(staging, "Imagenet64_val.zip")
    _copy_if_needed(args.drive_train_part1_zip, local_part1)
    _copy_if_needed(args.drive_train_part2_zip, local_part2)
    _copy_if_needed(args.drive_val_zip, local_val)

    extract_dir = os.path.join(staging, "extracted")
    _extract_zip(local_part1, extract_dir)
    _extract_zip(local_part2, extract_dir)
    _extract_zip(local_val, extract_dir)

    train_batches = _find_batches(extract_dir, "train_data_batch")
    val_batches = _find_batches(extract_dir, "val_data")
    if not train_batches:
        raise FileNotFoundError(f"No train_data_batch_* files found under {extract_dir}")
    if not val_batches:
        raise FileNotFoundError(f"No val_data file found under {extract_dir}")
    print(f"Found {len(train_batches)} train batch file(s), {len(val_batches)} val batch file(s).")

    _consolidate(train_batches, train_out)
    _consolidate(val_batches, val_out)

    if args.cleanup:
        print("Removing staging directory (zips + raw pickle batches) ...")
        shutil.rmtree(staging)

    print(f"\nDone. Set in your config:\n  data_dir: {args.data_dir}")


if __name__ == "__main__":
    main()
