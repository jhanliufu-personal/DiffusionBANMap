"""Download the aligned CelebA dataset (img_align_celeba) via torchvision.

Usage:
    python scripts/download_celeba_dataset.py --data_dir /path/to/data

Images will land in:
    <data_dir>/celeba/img_align_celeba/

Set data_dir in your config to that folder after downloading.

Note: torchvision downloads from Google Drive using gdown. If the download
fails due to Google Drive quota limits, download manually:
  1. Visit https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
  2. Download img_align_celeba.zip under "Align&Cropped Images"
  3. Extract so that images are at <data_dir>/celeba/img_align_celeba/*.jpg
"""

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CelebA (aligned faces)")
    parser.add_argument(
        "--data_dir", required=True,
        help="Root directory; dataset lands in <data_dir>/celeba/",
    )
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    try:
        import torchvision.datasets as dsets
    except ImportError:
        print("torchvision is required:  pip install torchvision")
        sys.exit(1)

    print(f"Downloading CelebA → {args.data_dir}/celeba/ …")
    try:
        # split='all' downloads the full dataset in one shot
        dsets.CelebA(root=args.data_dir, split="all", download=True)
    except Exception as exc:
        print(f"\nAutomatic download failed: {exc}")
        print(__doc__)
        sys.exit(1)

    img_dir = os.path.join(args.data_dir, "celeba", "img_align_celeba")
    n = len([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    print(f"\nDone — {n:,} images in {img_dir}")
    print(f"\nSet in your config:\n  data_dir: {img_dir}")


if __name__ == "__main__":
    main()
