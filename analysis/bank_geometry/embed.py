"""Image -> DINOv2-B/14 meanpatch -> the repo's z50 space, matching scripts/extract_image_embeddings.py.

The fitted PCA joblib for meanpatch z50 is not on disk (only the scaler is), so the z50
projection is recovered exactly as a linear least-squares map from the saved raw 768-D
embeddings of the 15,901 bank images to their saved z50 latents. PCA after z-scoring is
affine, so the fit is exact up to float error (checked at load: max |resid| is printed).

python -m analysis.bank_geometry.embed --list paths.txt --out out.npz [--transform NAME]
"""

import argparse
import json
import os
from functools import lru_cache

import joblib
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(REPO, "data")
RAW = os.path.join(DATA, "stimuli_dinov2_vitb14_meanpatch_latents")
PCA_DIR = os.path.join(DATA, "stimuli_dinov2_vitb14_meanpatch_pca_latents")


@lru_cache(None)
def projector():
    """Returns f(raw [n,768]) -> z50 [n,50]."""
    scaler = joblib.load(os.path.join(PCA_DIR, "dinov2_vitb14_meanpatch_scaler_z50.joblib"))
    raw = np.load(os.path.join(RAW, "train_latents.npy")).astype(np.float64)
    z = np.load(os.path.join(PCA_DIR, "train_latents_z50.npy")).astype(np.float64)
    S = scaler.transform(raw)
    A = np.hstack([S, np.ones((len(S), 1))])
    W, *_ = np.linalg.lstsq(A, z, rcond=None)
    resid = np.abs(A @ W - z).max()
    assert resid < 1e-2, f"z50 projection not recovered (max resid {resid})"
    return lambda x: np.hstack([scaler.transform(np.asarray(x, np.float64)), np.ones((len(x), 1))]) @ W


@lru_cache(None)
def _dino():
    import torch
    from torchvision import transforms
    from transformers import AutoModel, AutoImageProcessor

    torch.set_num_threads(os.cpu_count())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    tf = transforms.Compose([
        transforms.Resize((224, 224)),  # as build_encode_dataloader
        transforms.ToTensor(),
        transforms.Normalize(mean=proc.image_mean, std=proc.image_std),
    ])
    n_prefix = 1 + getattr(m.config, "num_register_tokens", 0)
    return m, tf, n_prefix, dev


def embed_pil(images, batch_size=32, verbose=True):
    """List of PIL images -> (raw [n,768], z50 [n,50]). Images are converted to RGB first,
    as the bank pipeline does (grayscale TIFFs become 3 identical channels)."""
    import torch

    m, tf, n_prefix, dev = _dino()
    out = []
    for s in range(0, len(images), batch_size):
        x = torch.stack([tf(im.convert("RGB")) for im in images[s:s + batch_size]]).to(dev)
        with torch.inference_mode():
            tok = m(pixel_values=x).last_hidden_state
        out.append(tok[:, n_prefix:].mean(1).float().cpu().numpy())
        if verbose:
            print(f"  embedded {min(s + batch_size, len(images))}/{len(images)}", flush=True)
    raw = np.concatenate(out) if out else np.zeros((0, 768), np.float32)
    return raw, projector()(raw)


def embed_paths_cached(paths, out_path, transform=None, batch_size=32, chunk=256):
    """Embed image files (optionally through transform(PIL)->PIL) with resumable chunked
    caching to out_path (.npz with raw, z, paths)."""
    from PIL import Image

    done_raw, done_paths = [], []
    part = out_path + ".partial.npz"
    if os.path.exists(out_path):
        return dict(np.load(out_path, allow_pickle=True))
    if os.path.exists(part):
        d = np.load(part, allow_pickle=True)
        done_raw, done_paths = [d["raw"]], list(d["paths"])
    start = len(done_paths)
    for s in range(start, len(paths), chunk):
        ims = [Image.open(p) for p in paths[s:s + chunk]]
        if transform is not None:
            ims = [transform(im) for im in ims]
        raw, _ = embed_pil(ims, batch_size, verbose=False)
        done_raw.append(raw)
        done_paths += list(paths[s:s + chunk])
        np.savez(part, raw=np.concatenate(done_raw), paths=np.array(done_paths))
        print(f"{os.path.basename(out_path)}: {len(done_paths)}/{len(paths)}", flush=True)
    raw = np.concatenate(done_raw)
    np.savez(out_path, raw=raw, z=projector()(raw), paths=np.array(done_paths))
    if os.path.exists(part):
        os.remove(part)
    return dict(np.load(out_path, allow_pickle=True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="text file, one image path per line")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    paths = [l.strip() for l in open(a.list) if l.strip()]
    d = embed_paths_cached(paths, a.out)
    print(json.dumps({"n": len(d["z"]), "out": a.out}))
