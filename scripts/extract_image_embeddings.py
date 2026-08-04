"""
Standalone script mirroring notebooks/extract_image_embeddings.ipynb's current
(active) pipeline: encode every image in VAL_DATA_DIR with a chosen pretrained
encoder (AlexNet fc6, CLIP, SD's VAE, ...) and stream the raw per-image
embeddings to disk. PCA reduction is disabled here too, same as the notebook's
commented-out cells (kept below, commented, for parity) -- this just saves each
encoder's native output as-is.

Exists as a plain script so this can run outside a Jupyter kernel -- useful if
the notebook's kernel crashes/runs out of memory (e.g. sd_vae at IMAGE_SIZE=512
is much heavier than the 64x64 case).

Deliberate deviations from the notebook (script context, not a Colab notebook):
  - No REPO_ROOT/sys.path.insert or importlib.metadata patch -- both were
    Colab-path/package-lookup workarounds; run this as `python -m
    scripts.extract_image_embeddings` from the repo root instead, which
    already resolves data_utils/etc. correctly.
  - The roundtrip spot-check figure is saved to disk (plt.savefig) instead of
    plt.show()'d -- there's no inline display in a headless script run.

Run from repo root: python -m scripts.extract_image_embeddings
"""

import os
import sys
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from torchvision.utils import make_grid
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

from data_utils import build_encode_dataloader


# ---------------------------------------------------------------------------
# Config -- mirrors the notebook's Config cell
# ---------------------------------------------------------------------------
# holds train_images.npy / val_images.npy
# DATA_DIR = "/content/data/imagenet64"
# DATA_DIR = "data/tiny-imagenet-200"
# DATA_DIR = "data/500Stimuli"

# TRAIN_DATA_DIR = "data/15901Stimuli"
VAL_DATA_DIR = "data/500Stimuli"

# which encoder to extract embeddings from -- see MODEL_REGISTRY below
# options: "alexnet_fc6", "clip_vit_b32", "clip_vit_l14", "sd_vae"
MODEL_NAME = "sd_vae"

# default input resolution; overridden per-model once MODEL_REGISTRY is defined below
IMAGE_SIZE = 512
BATCH_SIZE = 64

# latent_dim to use in the diffusion config -- only used by the (currently disabled) PCA
# code below; ignored for encoders like sd_vae whose native output is saved directly
N_COMPONENTS = 50

# random train images used to fit PCA (fits comfortably in RAM)
SUBSAMPLE_SIZE = 15000
SEED = 42

# OUTPUT_DIR = f"data/imagenet64_alexnet_fc6_pca_latents"
OUTPUT_DIR = f"data/stimuli_{MODEL_NAME}_latents"


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
    "sd_vae": {
        "display_name": "Stable Diffusion VAE (stabilityai/sd-vae-ft-mse)",
        "image_size": 512,
        "load_fn": _load_sd_vae("stabilityai/sd-vae-ft-mse"),
    },
}


# ---------------------------------------------------------------------------
# Stream the full val set through the encoder -- mirrors "Stream the full
# train/val sets through the encoder -> PCA" section.
#
# One batch of raw embeddings lives in memory at a time; each batch is
# written straight into a preallocated latents array on disk, so the full
# [N, D] (or [N, C, H, W]) embedding matrix is never materialized. Resumable:
# re-running skips any split whose latents file already exists (atomic
# tmp-then-rename, same pattern as prepare_imagenet64_dataset.py).
# ---------------------------------------------------------------------------

def _stream_latents(encode_dl: DataLoader, out_path: str, extract_batch, batch_size: int) -> None:
    if os.path.exists(out_path):
        print(f"Already extracted: {out_path}")
        return
    n = len(encode_dl.dataset)
    tmp_path = out_path + ".tmp"
    out = None
    offset = 0
    for batch in encode_dl:
        feat_batch = extract_batch(batch)
        # z = pca.transform(scaler.transform(feat_batch)).astype(np.float32)
        z = feat_batch.astype(np.float32)
        if out is None:
            # Preallocated lazily once the encoder's native output shape is known -- PCA is
            # disabled above, so this is whatever extract_batch returns as-is: a flat vector
            # for AlexNet/CLIP, or a (C, H, W) latent for sd_vae.
            out = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float32, shape=(n,) + z.shape[1:])
        out[offset:offset + z.shape[0]] = z
        offset += z.shape[0]
        if offset % (batch_size * 100) < batch_size or offset == n:
            print(f"  {offset:,}/{n:,} latents written → {out_path}")
    out.flush()
    del out
    os.rename(tmp_path, out_path)
    print(f"Done: {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert MODEL_NAME in MODEL_REGISTRY, f"Unknown MODEL_NAME {MODEL_NAME!r}, choose from {list(MODEL_REGISTRY)}"
    image_size = MODEL_REGISTRY[MODEL_NAME]["image_size"]

    extract_batch, cleanup_model = MODEL_REGISTRY[MODEL_NAME]["load_fn"](device)
    print(f"Loaded encoder: {MODEL_REGISTRY[MODEL_NAME]['display_name']}  (image_size={image_size})")

    # --- Build train/val dataloaders (unshuffled, for encoding) ---
    # build_encode_dataloader returns a single, always-unshuffled DataLoader over a flat
    # image directory, so index i here matches row i of the row order the streaming pass
    # below writes latents back into.

    # train_encode_dl, val_encode_dl = build_imagenet64_dataloaders(
    #     DATA_DIR, image_size=image_size, batch_size=BATCH_SIZE, shuffle_train=False, num_workers=0,
    # )

    # train_encode_dl, val_encode_dl = build_tiny_imagenet_dataloaders(
    #     DATA_DIR, image_size=image_size, batch_size=BATCH_SIZE, shuffle_train=False, num_workers=0,
    # )

    # train_encode_dl = build_encode_dataloader(
    #     TRAIN_DATA_DIR, image_size=image_size, batch_size=BATCH_SIZE, num_workers=0
    # )

    val_encode_dl = build_encode_dataloader(
        VAL_DATA_DIR, image_size=image_size, batch_size=BATCH_SIZE, num_workers=0
    )

    # --- Fit PCA on a random subsample of train (disabled, mirrors notebook) ---
    # PCA's top-N_COMPONENTS subspace is stable with far fewer than all images. Fit it on
    # a random subsample that fits comfortably in RAM, then apply the fitted transform to
    # the full dataset in the streaming pass below -- the full [N, D] embedding matrix is
    # never materialized.
    #
    # n_train = len(train_encode_dl.dataset)
    # rng = np.random.default_rng(SEED)
    # subsample_idx = np.sort(rng.choice(n_train, size=min(SUBSAMPLE_SIZE, n_train), replace=False)).tolist()
    #
    # subsample_dl = DataLoader(
    #     Subset(train_encode_dl.dataset, subsample_idx),
    #     batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True,
    # )
    #
    # subsample_feats = np.concatenate([extract_batch(batch) for batch in subsample_dl], axis=0)
    # print(f"Subsample {MODEL_NAME} embeddings: {subsample_feats.shape}")

    # --- Standardize + PCA (disabled, mirrors notebook) ---
    # Raw embeddings can have very different per-dimension scales (e.g. AlexNet fc6 is
    # post-ReLU and non-negative with uneven unit-wise variance), so z-score before PCA --
    # otherwise a handful of high-variance dimensions would dominate the top PCs.
    #
    # scaler = StandardScaler()
    # feats_scaled = scaler.fit_transform(subsample_feats)
    #
    # pca = PCA(n_components=N_COMPONENTS, random_state=SEED, svd_solver="randomized")
    # pca.fit(feats_scaled)
    #
    # print(f"Fitted PCA on {subsample_feats.shape[0]:,} images")
    # print(f"Explained variance (top {N_COMPONENTS} PCs): {pca.explained_variance_ratio_.sum():.3f}")
    #
    # plt.figure(figsize=(5, 3))
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.xlabel("# PCs")
    # plt.ylabel("cumulative explained variance")
    # plt.title(f"{MODEL_NAME} → PCA")
    # plt.tight_layout()
    # plt.savefig(os.path.join(OUTPUT_DIR, "pca_explained_variance.png"), dpi=150)
    # plt.close()

    # train_latents_path = os.path.join(OUTPUT_DIR, "train_latents.npy")
    val_latents_path = os.path.join(OUTPUT_DIR, "val_latents.npy")

    # _stream_latents(train_encode_dl, train_latents_path, extract_batch, BATCH_SIZE)
    _stream_latents(val_encode_dl, val_latents_path, extract_batch, BATCH_SIZE)

    cleanup_model()

    # --- Save PCA/scaler artifacts + manifest (disabled, mirrors notebook) ---
    # Latents themselves already live in OUTPUT_DIR. This would save the fitted
    # StandardScaler/PCA -- for mapping any new image into the same latent space later --
    # and a manifest, to a repo-relative, more persistent location.
    #
    # joblib.dump(scaler, os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_scaler.joblib"))
    # joblib.dump(pca, os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_pca.joblib"))
    #
    # with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
    #     json.dump({
    #         "model_name": MODEL_NAME,
    #         "image_size": image_size,
    #         "val_data_dir": VAL_DATA_DIR,
    #         "n_val": len(val_encode_dl.dataset),
    #         "subsample_size": int(subsample_feats.shape[0]),
    #         "n_components": N_COMPONENTS,
    #         "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
    #         "val_latents_path": val_latents_path,
    #     }, f, indent=2)
    #
    # print(f"Saved scaler + PCA + manifest → {OUTPUT_DIR}")

    # --- Spot-check VAE reconstruction (decode(encode(x))) ---
    # Only meaningful for encoders that have a decoder -- currently just sd_vae. Takes a
    # few real images straight from val_encode_dl, round-trips them through encode ->
    # decode, and saves original vs. reconstruction side by side plus a mean-pixel-MSE
    # number, as a sanity check that the VAE's latent space is a reasonable fit for these
    # stimuli before regressing neural data onto it.
    if hasattr(extract_batch, "decode"):
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
        plt.title(f"{MODEL_REGISTRY[MODEL_NAME]['display_name']} — top: original, bottom: decode(encode(x))")
        plt.tight_layout()
        roundtrip_path = os.path.join(OUTPUT_DIR, "roundtrip_spotcheck.png")
        plt.savefig(roundtrip_path, dpi=150)
        plt.close()
        print(f"Saved roundtrip spot-check figure to {roundtrip_path}")

        mse = torch.mean((images - recon) ** 2).item()
        print(f"Mean per-pixel MSE over {n_show} roundtrip samples: {mse:.5f}")
    else:
        print(f"Skipping VAE roundtrip spot check — MODEL_NAME={MODEL_NAME!r} has no decoder to check")


if __name__ == "__main__":
    main()
