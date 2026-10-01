"""
Standalone script version of notebooks/extract_image_embeddings*.ipynb: encode
every image in a dataset with a chosen pretrained encoder (AlexNet fc6, CLIP,
DINO/DINOv2, SD's VAE, ...) and stream the per-image embeddings to disk.

By default each encoder's native output is saved as-is. Pass --n-components N
to also z-score + PCA the embeddings down to their top N PCs (fit on a random
subsample of the train split, then applied to train and val) -- written as
train_latents_z{N}.npy / val_latents_z{N}.npy into {dataset}_{model}_pca_latents
next to the image directory, which is where the training pipeline looks for
them given `encoding_model: {model}` and `latent_dim: N` in the diffusion
config (see data_utils.resolve_latents_paths).

The full un-PCA'd embeddings are kept alongside them as train_embeddings.npy /
val_embeddings.npy, and PCA is always computed from those files. So the encoder
only ever runs once per dataset + model: re-running with a different
--n-components just re-fits PCA on the saved embeddings (no encoder load, no
image pass).

Exists as a plain script so this can run outside a Jupyter kernel -- useful if
the notebook's kernel crashes/runs out of memory (e.g. sd_vae at IMAGE_SIZE=512
is much heavier than the 64x64 case).

Deliberate deviations from the notebook (script context, not a Colab notebook):
  - No REPO_ROOT/sys.path.insert or importlib.metadata patch -- both were
    Colab-path/package-lookup workarounds; run this as `python -m
    scripts.extract_image_embeddings` from the repo root instead, which
    already resolves data_utils/etc. correctly.
  - Figures are saved to disk (plt.savefig) instead of plt.show()'d -- there's
    no inline display in a headless script run.

Run from repo root (the Config constants below are the CLI defaults):
  python -m scripts.extract_image_embeddings
  python -m scripts.extract_image_embeddings --model dinov2_vitb14 \
      --dataset imagenet64 --data-dir data/imagenet64 --n-components 200
  # -> data/imagenet64_dinov2_vitb14_pca_latents/{train,val}_latents_z200.npy
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import make_grid
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

from data_utils import build_encode_dataloader, build_imagenet64_dataloaders, latents_dir_for


# ---------------------------------------------------------------------------
# Config -- mirrors the notebook's Config cell. These are the CLI defaults; see
# parse_args() below for the flag that overrides each one.
# ---------------------------------------------------------------------------
# same values as the diffusion config's dataset_type:
# "stimuli": flat image directories (VAL_DATA_DIR, optionally TRAIN_DATA_DIR)
# "imagenet64": one directory holding train_images.npy / val_images.npy
DATASET = "stimuli"

TRAIN_DATA_DIR = None  # e.g. "data/15901Stimuli"; only needed to fit PCA in "stimuli" mode
VAL_DATA_DIR = "data/500Stimuli"

# which encoder to extract embeddings from -- see MODEL_REGISTRY below
MODEL_NAME = "sd_vae"

# fallback batch size; overridden per-model by MODEL_REGISTRY's "batch_size" where set
BATCH_SIZE = 64

# latent_dim to use in the diffusion config. None disables PCA and saves each encoder's
# native output directly (the only option for encoders like sd_vae with non-flat output)
N_COMPONENTS = None

# random train images used to fit PCA (fits comfortably in RAM)
SUBSAMPLE_SIZE = 15000
SEED = 42


# ---------------------------------------------------------------------------
# Load encoder model -- mirrors the "Load encoder model" section.
#
# Each encoder is registered in MODEL_REGISTRY below as image_size (native
# input resolution) plus a load_fn(device) that returns (extract_batch_fn,
# cleanup_fn):
#   - extract_batch_fn(batch) -> np.ndarray runs one batch through the model
#     and returns its raw (pre-PCA) embeddings.
#   - cleanup_fn() releases any resources (e.g. forward hooks) once extraction
#     is done.
# ---------------------------------------------------------------------------

def _load_alexnet_fc6(device):
    """AlexNet fc6 (relu6) activations, 4096-d. classifier is [Dropout,
    Linear(9216->4096) fc6, ReLU, Dropout, Linear(4096->4096) fc7, ReLU,
    Linear(4096->1000)]. "fc6 activations" conventionally means the
    post-ReLU output (relu6), i.e. the output of classifier[2] -- that's
    what the hook below captures."""
    alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).to(device).eval()
    for p in alexnet.parameters():
        p.requires_grad_(False)

    _activation = {"value": None}

    def _fc6_hook(module, inp, out):
        _activation["value"] = out.detach()

    hook_handle = alexnet.classifier[2].register_forward_hook(_fc6_hook)  # relu6 output

    # AlexNet's pretrained weights expect ImageNet-normalized input; build_encode_dataloader
    # only resizes and gives [0,1] tensors, so normalize per-batch here.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @torch.no_grad()
    def extract_batch(batch) -> np.ndarray:
        """Run one batch through AlexNet, return its fc6 (relu6) activations as a numpy array.
        O(1) memory regardless of dataset size — only ever holds one batch's activations."""
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        images = normalize(images).to(device)
        alexnet(images)
        return _activation["value"].cpu().numpy()

    def cleanup():
        hook_handle.remove()

    return extract_batch, cleanup


def _make_clip_loader(model_id: str):
    """Return a load_fn(device) for a given CLIP checkpoint (e.g.
    "openai/clip-vit-base-patch32"). Uses the HuggingFace transformers CLIP
    implementation, returns the post-projection image embedding (512-d for
    ViT-B/32, 768-d for ViT-L/14). Each CLIP checkpoint ships its own
    normalization stats (CLIPImageProcessor.image_mean/image_std), fetched
    here rather than hardcoded so different CLIP variants stay correct."""
    def load_fn(device):
        from transformers import CLIPModel, CLIPImageProcessor

        clip_model = CLIPModel.from_pretrained(model_id).to(device).eval()
        for p in clip_model.parameters():
            p.requires_grad_(False)

        processor = CLIPImageProcessor.from_pretrained(model_id)
        normalize = transforms.Normalize(mean=processor.image_mean, std=processor.image_std)

        @torch.no_grad()
        def extract_batch(batch) -> np.ndarray:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = normalize(images).to(device)
            features = clip_model.get_image_features(pixel_values=images)
            if not isinstance(features, torch.Tensor):
                # Look for 'image_embeds' (CLIPModel) or 'pooler_output' (CLIPVisionModel)
                features = getattr(features, "image_embeds", getattr(features, "pooler_output", features[0]))
            return features.cpu().numpy()

        def cleanup():
            pass

        return extract_batch, cleanup

    return load_fn


def _make_dino_loader(model_id: str):
    """Return a load_fn(device) for a given DINO/DINOv2 checkpoint (e.g.
    "facebook/dinov2-base"). Returns the final-layer CLS token after the last
    LayerNorm (384-d for ViT-S, 768-d for ViT-B, 1024-d for ViT-L) -- the
    standard DINO global image embedding. Read from last_hidden_state[:, 0]
    rather than pooler_output: the two are identical for DINOv2, but for the
    original DINO (a plain ViTModel) pooler_output goes through an extra
    dense+tanh head that isn't in the DINO checkpoint, i.e. randomly
    initialized. Input is resized straight to 224x224 like the other encoders
    (a multiple of both patch sizes, 14 and 16)."""
    def load_fn(device):
        from transformers import AutoModel, AutoImageProcessor

        dino_model = AutoModel.from_pretrained(model_id).to(device).eval()
        for p in dino_model.parameters():
            p.requires_grad_(False)

        processor = AutoImageProcessor.from_pretrained(model_id)
        normalize = transforms.Normalize(mean=processor.image_mean, std=processor.image_std)

        @torch.no_grad()
        def extract_batch(batch) -> np.ndarray:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = normalize(images).to(device)
            features = dino_model(pixel_values=images).last_hidden_state[:, 0]
            return features.cpu().numpy()

        def cleanup():
            pass

        return extract_batch, cleanup

    return load_fn


def _load_sd_vae(model_id: str = "stabilityai/sd-vae-ft-mse"):
    """Return a load_fn(device) for a pretrained SD VAE checkpoint. Returns the
    deterministic latent (posterior mean, not a stochastic sample) scaled by
    vae.config.scaling_factor -- the same convention SD's own UNet is trained
    on. Unlike the other encoders, this one also has a decoder: the returned
    extract_batch carries a `.decode` attribute (not part of the shared
    encoder interface) for the roundtrip spot-check at the end of this script."""
    def load_fn(device):
        from diffusers import AutoencoderKL

        vae = AutoencoderKL.from_pretrained(model_id).to(device).eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        scaling_factor = vae.config.scaling_factor

        @torch.no_grad()
        def extract_batch(batch) -> np.ndarray:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device) * 2.0 - 1.0  # build_encode_dataloader gives [0,1]; VAE expects [-1,1]
            latents = vae.encode(images).latent_dist.mean * scaling_factor
            return latents.cpu().numpy()

        @torch.no_grad()
        def decode_batch(latents: torch.Tensor) -> torch.Tensor:
            """Inverse of extract_batch: scaled latent -> [0,1] image batch."""
            images = vae.decode(latents.to(device) / scaling_factor).sample
            return ((images.clamp(-1, 1) + 1.0) / 2.0).cpu()

        extract_batch.decode = decode_batch

        def cleanup():
            pass

        return extract_batch, cleanup

    return load_fn


MODEL_REGISTRY = {
    "alexnet_fc6": {
        "display_name": "AlexNet fc6 (relu6)",
        "image_size": 224,
        "load_fn": _load_alexnet_fc6,
    },
    "clip_vit_b32": {
        "display_name": "CLIP ViT-B/32 (openai/clip-vit-base-patch32)",
        "image_size": 224,
        "load_fn": _make_clip_loader("openai/clip-vit-base-patch32"),
    },
    "clip_vit_l14": {
        "display_name": "CLIP ViT-L/14 (openai/clip-vit-large-patch14)",
        "image_size": 224,
        "load_fn": _make_clip_loader("openai/clip-vit-large-patch14"),
    },
    "dino_vitb16": {
        "display_name": "DINO ViT-B/16 (facebook/dino-vitb16)",
        "image_size": 224,
        "load_fn": _make_dino_loader("facebook/dino-vitb16"),
    },
    "dinov2_vits14": {
        "display_name": "DINOv2 ViT-S/14 (facebook/dinov2-small)",
        "image_size": 224,
        "load_fn": _make_dino_loader("facebook/dinov2-small"),
    },
    "dinov2_vitb14": {
        "display_name": "DINOv2 ViT-B/14 (facebook/dinov2-base)",
        "image_size": 224,
        "load_fn": _make_dino_loader("facebook/dinov2-base"),
    },
    "dinov2_vitl14": {
        "display_name": "DINOv2 ViT-L/14 (facebook/dinov2-large)",
        "image_size": 224,
        "load_fn": _make_dino_loader("facebook/dinov2-large"),
    },
    "sd_vae": {
        "display_name": "Stable Diffusion VAE (stabilityai/sd-vae-ft-mse)",
        "image_size": 512,
        # Global BATCH_SIZE (64) is sized for AlexNet/CLIP at 224x224; encoding at 512x512
        # through the VAE's conv+attention stack is far heavier per image and OOM'd at 64 --
        # override to a smaller batch size just for this encoder.
        "batch_size": 8,
        "load_fn": _load_sd_vae("stabilityai/sd-vae-ft-mse"),
    },
}


# ---------------------------------------------------------------------------
# Stream a full split through the encoder -- mirrors "Stream the full train/val
# sets through the encoder -> PCA" section, minus the PCA (see _project_latents).
#
# One batch of raw embeddings lives in memory at a time; each batch is written
# straight into a preallocated array on disk, so the full [N, D] (or
# [N, C, H, W]) embedding matrix is never materialized. Resumable: re-running
# skips any split whose file already exists (atomic tmp-then-rename, same
# pattern as prepare_imagenet64_dataset.py).
# ---------------------------------------------------------------------------

def _stream_embeddings(encode_dl: DataLoader, out_path: str, extract_batch, batch_size: int) -> None:
    n = len(encode_dl.dataset)
    tmp_path = out_path + ".tmp"
    out = None
    offset = 0
    for batch in encode_dl:
        feat_batch = extract_batch(batch).astype(np.float32)
        if out is None:
            # Preallocated lazily once the encoder's native output shape is known: a flat
            # vector for AlexNet/CLIP/DINO, or a (C, H, W) latent for sd_vae.
            out = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float32, shape=(n,) + feat_batch.shape[1:])
        out[offset:offset + feat_batch.shape[0]] = feat_batch
        offset += feat_batch.shape[0]
        if offset % (batch_size * 100) < batch_size or offset == n:
            print(f"  {offset:,}/{n:,} embeddings written → {out_path}")
    out.flush()
    del out
    os.rename(tmp_path, out_path)
    print(f"Done: {out_path}")


def _fit_pca(train_embeddings_path: str, args):
    """Fit StandardScaler + PCA on a random subsample of the saved train embeddings and
    return (scaler, pca).

    PCA's top-n_components subspace is stable with far fewer than all images, so fit it on
    a random subsample that fits comfortably in RAM. Raw embeddings can have very different
    per-dimension scales (e.g. AlexNet fc6 is post-ReLU and non-negative with uneven
    unit-wise variance), so z-score before PCA -- otherwise a handful of high-variance
    dimensions would dominate the top PCs.

    The fitted scaler/PCA are saved to args.output_dir -- for mapping any new image into
    the same latent space later -- and reused if already there, so a resumed run projects
    the remaining splits onto the same PCs as the ones already written.
    """
    scaler_path = os.path.join(args.output_dir, f"{args.model}_scaler_z{args.n_components}.joblib")
    pca_path = os.path.join(args.output_dir, f"{args.model}_pca_z{args.n_components}.joblib")
    if os.path.exists(scaler_path) and os.path.exists(pca_path):
        print(f"Reusing fitted scaler + PCA: {scaler_path}, {pca_path}")
        return joblib.load(scaler_path), joblib.load(pca_path)

    train_embeddings = np.load(train_embeddings_path, mmap_mode="r")
    assert train_embeddings.ndim == 2, f"PCA needs flat embeddings, got {train_embeddings.shape[1:]} per image"

    n_train = len(train_embeddings)
    rng = np.random.default_rng(SEED)
    subsample_idx = np.sort(rng.choice(n_train, size=min(args.subsample_size, n_train), replace=False))
    subsample_feats = train_embeddings[subsample_idx]
    print(f"Subsample {args.model} embeddings: {subsample_feats.shape}")

    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(subsample_feats)

    pca = PCA(n_components=args.n_components, random_state=SEED, svd_solver="randomized")
    pca.fit(feats_scaled)

    print(f"Fitted PCA on {subsample_feats.shape[0]:,} images")
    print(f"Explained variance (top {args.n_components} PCs): {pca.explained_variance_ratio_.sum():.3f}")

    plt.figure(figsize=(5, 3))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("# PCs")
    plt.ylabel("cumulative explained variance")
    plt.title(f"{args.model} → PCA")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"pca_explained_variance_z{args.n_components}.png"), dpi=150)
    plt.close()

    joblib.dump(scaler, scaler_path)
    joblib.dump(pca, pca_path)
    print(f"Saved scaler + PCA → {args.output_dir}")
    return scaler, pca


def _project_latents(embeddings_path: str, out_path: str, scaler, pca, chunk_size: int = 16384) -> None:
    """Apply the fitted scaler -> PCA to a saved embeddings file, chunk_size rows at a time
    (the embeddings stay memory-mapped), writing the [N, n_components] latents to out_path.
    Skips if out_path already exists (atomic tmp-then-rename, as in _stream_embeddings)."""
    if os.path.exists(out_path):
        print(f"Already projected: {out_path}")
        return
    embeddings = np.load(embeddings_path, mmap_mode="r")
    n = len(embeddings)
    tmp_path = out_path + ".tmp"
    out = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float32, shape=(n, pca.n_components_))
    for start in range(0, n, chunk_size):
        out[start:start + chunk_size] = pca.transform(scaler.transform(embeddings[start:start + chunk_size]))
    out.flush()
    del out
    os.rename(tmp_path, out_path)
    print(f"Done: {n:,} latents → {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract (optionally PCA-reduced) image embeddings from a pretrained encoder.")
    parser.add_argument("--model", default=MODEL_NAME, choices=list(MODEL_REGISTRY))
    parser.add_argument("--dataset", default=DATASET, choices=["stimuli", "imagenet64"],
                        help="'stimuli': flat image directories; 'imagenet64': train_images.npy/val_images.npy in --data-dir")
    parser.add_argument("--data-dir", default=None,
                        help=f"val image folder ('stimuli', default {VAL_DATA_DIR}) or ImageNet64 directory ('imagenet64')")
    parser.add_argument("--train-data-dir", default=TRAIN_DATA_DIR,
                        help="'stimuli' only: train image folder (the diffusion config's data_dir), also encoded and used to fit PCA")
    parser.add_argument("--n-components", type=int, default=N_COMPONENTS,
                        help="keep the top N PCs of the embeddings (default: no PCA, save raw embeddings)")
    parser.add_argument("--subsample-size", type=int, default=SUBSAMPLE_SIZE,
                        help="random train images used to fit PCA")
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"default: per-model value from MODEL_REGISTRY, else {BATCH_SIZE}")
    parser.add_argument("--output-dir", default=None,
                        help="default: {dataset}_{model}_pca_latents next to the image directory "
                             "({dataset}_{model}_latents without --n-components)")
    args = parser.parse_args()

    if args.data_dir is None:
        if args.dataset == "imagenet64":
            parser.error("--data-dir is required with --dataset imagenet64")
        args.data_dir = VAL_DATA_DIR
    if args.n_components is not None and args.dataset == "stimuli" and args.train_data_dir is None:
        parser.error("--n-components needs a train split to fit PCA on: pass --train-data-dir")
    if args.output_dir is None:
        # Same directory the training pipeline resolves from the diffusion config's data_dir
        # (the train folder for stimuli) + dataset_type + encoding_model.
        args.output_dir = latents_dir_for(args.train_data_dir or args.data_dir, args.dataset, args.model)
        if args.n_components is None:
            args.output_dir = args.output_dir.removesuffix("_pca_latents") + "_latents"
    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size = MODEL_REGISTRY[args.model]["image_size"]
    batch_size = args.batch_size or MODEL_REGISTRY[args.model].get("batch_size", BATCH_SIZE)

    # --- Build train/val dataloaders (unshuffled, for encoding) ---
    # Both loaders are unshuffled with no dropped batch, so index i here matches row i of
    # the row order the streaming pass below writes embeddings back into.
    if args.dataset == "imagenet64":
        train_encode_dl, val_encode_dl = build_imagenet64_dataloaders(
            args.data_dir, image_size=image_size, batch_size=batch_size, shuffle_train=False, num_workers=0,
        )
    else:
        train_encode_dl = build_encode_dataloader(
            args.train_data_dir, image_size=image_size, batch_size=batch_size, num_workers=0
        ) if args.train_data_dir is not None else None
        val_encode_dl = build_encode_dataloader(
            args.data_dir, image_size=image_size, batch_size=batch_size, num_workers=0
        )
    encode_dls = {"train": train_encode_dl, "val": val_encode_dl}
    encode_dls = {split: dl for split, dl in encode_dls.items() if dl is not None}

    # --- Raw embeddings: encode once, reuse on every later run ---
    # With PCA these are an intermediate ({split}_embeddings.npy) that every --n-components
    # is computed from; without PCA they are the output itself ({split}_latents.npy).
    raw_suffix = "embeddings.npy" if args.n_components is not None else "latents.npy"
    raw_paths = {split: os.path.join(args.output_dir, f"{split}_{raw_suffix}") for split in encode_dls}
    for split, path in raw_paths.items():
        if os.path.exists(path):
            n_saved, n_images = len(np.load(path, mmap_mode="r")), len(encode_dls[split].dataset)
            if n_saved != n_images:
                raise ValueError(f"{path} has {n_saved:,} rows but the {split} split has {n_images:,} images — "
                                 f"stale file from a different dataset? Delete it to re-encode.")
            print(f"Already extracted: {path}")
    to_encode = [split for split, path in raw_paths.items() if not os.path.exists(path)]

    # The encoder is only loaded if there is something left to encode (or, without PCA,
    # always -- the roundtrip spot-check below needs it), so re-running with a new
    # --n-components touches neither the encoder nor the images.
    extract_batch = None
    if to_encode or args.n_components is None:
        extract_batch, cleanup_model = MODEL_REGISTRY[args.model]["load_fn"](device)
        print(f"Loaded encoder: {MODEL_REGISTRY[args.model]['display_name']}  (image_size={image_size}, batch_size={batch_size})")
        for split in to_encode:
            _stream_embeddings(encode_dls[split], raw_paths[split], extract_batch, batch_size)
        cleanup_model()

    # --- PCA: fit on the saved train embeddings, project every split ---
    if args.n_components is not None:
        scaler, pca = _fit_pca(raw_paths["train"], args)
        # the dim suffix lets multiple PCA dimensionalities coexist in the same directory
        latents_paths = {split: os.path.join(args.output_dir, f"{split}_latents_z{args.n_components}.npy")
                         for split in raw_paths}
        for split in raw_paths:
            _project_latents(raw_paths[split], latents_paths[split], scaler, pca)

        with open(os.path.join(args.output_dir, f"manifest_z{args.n_components}.json"), "w") as f:
            json.dump({
                "model_name": args.model,
                "image_size": image_size,
                "dataset": args.dataset,
                "data_dir": args.data_dir,
                "train_data_dir": args.train_data_dir,
                "n_train": len(train_encode_dl.dataset),
                "n_val": len(val_encode_dl.dataset),
                "subsample_size": min(args.subsample_size, len(train_encode_dl.dataset)),
                "n_components": args.n_components,
                "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
                "train_embeddings_path": raw_paths["train"],
                "val_embeddings_path": raw_paths["val"],
                "train_latents_path": latents_paths["train"],
                "val_latents_path": latents_paths["val"],
            }, f, indent=2)

    # --- Spot-check VAE reconstruction (decode(encode(x))) ---
    # Only meaningful for encoders that have a decoder -- currently just sd_vae. Takes a
    # few real images straight from val_encode_dl, round-trips them through encode ->
    # decode, and saves original vs. reconstruction side by side plus a mean-pixel-MSE
    # number, as a sanity check that the VAE's latent space is a reasonable fit for these
    # stimuli before regressing neural data onto it.
    if extract_batch is not None and hasattr(extract_batch, "decode"):
        n_show = 6
        check_batch = next(iter(val_encode_dl))
        images = check_batch[0] if isinstance(check_batch, (list, tuple)) else check_batch
        images = images[:n_show]

        latents = torch.from_numpy(extract_batch(images))
        recon = extract_batch.decode(latents).clamp(0, 1)

        grid = make_grid(torch.cat([images, recon], dim=0), nrow=n_show)
        grid_np = grid.permute(1, 2, 0).numpy()

        plt.figure(figsize=(n_show * 1.6, 3.4))
        plt.imshow(grid_np.squeeze(-1) if grid_np.shape[-1] == 1 else grid_np,
                   cmap="gray" if grid_np.shape[-1] == 1 else None)
        plt.axis("off")
        plt.title(f"{MODEL_REGISTRY[args.model]['display_name']} — top: original, bottom: decode(encode(x))")
        plt.tight_layout()
        roundtrip_path = os.path.join(args.output_dir, "roundtrip_spotcheck.png")
        plt.savefig(roundtrip_path, dpi=150)
        plt.close()
        print(f"Saved roundtrip spot-check figure to {roundtrip_path}")

        mse = torch.mean((images - recon) ** 2).item()
        print(f"Mean per-pixel MSE over {n_show} roundtrip samples: {mse:.5f}")
    else:
        print(f"Skipping VAE roundtrip spot check — {args.model!r} has no decoder to check")


if __name__ == "__main__":
    main()
