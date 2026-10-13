"""
Generate images from a trained conditional diffusion checkpoint, several samples per
stimulus, and save them in the same format as NeuralGuidance's scripts/generate.py --
so a from-scratch conditional UNet and the test-time steering methods there can be
scored by the same evaluation (NeuralGuidance: python -m scripts.evaluate --run <dir>).

scripts/inspect_diffusion.py stays the quick look at one checkpoint (one-step
predictions, one sample per image, a denoising progression). This script is the one
that produces the samples an experiment is scored on.

Where the conditioning latent z of each sample comes from is set by --targets:
  oracle   the real stimulus image's own latent: val_latents_z{latent_dim}.npy of the
           config's encoding_model (the same file training/inspection read)
  decoded  features decoded from neural firing rates: --decoded_path, the pred_val.npy
           written by NeuralGuidance's scripts/fit_neuron_map.py --direction decode.
           One row per val image, in val-folder (sorted path) order, NaN where a
           stimulus has no neural data.
             - If its width equals latent_dim, rows are used as z directly. They must
               then be in this model's PC basis (the scaler/PCA in the latents dir).
             - If its width equals the encoder's native dimension (e.g. 768 for a map
               fit on the un-PCA'd DINOv2 features), rows are projected with the
               scaler + PCA saved next to the latents this model was trained on.
  none     z = 0: the unconditional model (or the null condition of a CFG-trained one).
           The floor a conditional run is compared against.

Every (stimulus, seed) pair has its own fixed starting noise, so runs that differ only
in --targets (or in checkpoint) start from the same noise and can be compared sample
by sample.

Writes to --output_dir (default <run output_dir>/generation/<ckpt stem>_<targets>):
  samples.npy   uint8, [N stimuli, n_seeds, H, W, 3]
  meta.json     which stimuli (val-folder rows), seeds, checkpoint, sampler, cost
  grid.png      first 8 stimuli: top row the real image, then one row per seed

Run from repo root:
    # oracle: 100 random val stimuli x 4 seeds
    python -m scripts.generate --diffusion_config_path config/diffusion_full_stimuli_finetune_flow.yaml

    # decoded from firing rates, on the held-out stimuli of the NeuralGuidance split
    python -m scripts.generate --diffusion_config_path config/diffusion_full_stimuli_finetune_flow.yaml \
        --targets decoded --decoded_path neuron_maps/decode_dinov2_meanpatch/pred_val.npy \
        --indices splits/neural_holdout.npz:heldout
    # ... and the oracle run on the same stimuli, for the oracle / neural gap
    python -m scripts.generate --diffusion_config_path config/diffusion_full_stimuli_finetune_flow.yaml \
        --indices splits/neural_holdout.npz:heldout
"""

import os
import json
import time
import types
import argparse

import yaml
import joblib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from models.unet import UNet
from models.ema import ema_shadow_to_model_state_dict
from data_utils import _find_all_images, latents_dir_for, resolve_latents_paths
from utils import (
    discover_device, make_noise_schedule, diffusion_sample, flow_sample, vpsde_sample, make_run_tag,
    select_checkpoints,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--diffusion_config_path", type=str, required=True)
    p.add_argument("--ckpt", type=str, default="latest",
                    help="'latest' (default), a step number, a filename in the run's checkpoints dir, or a path")
    p.add_argument("--use_ema_weights", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--targets", type=str, default="oracle", choices=["oracle", "decoded", "none"])
    p.add_argument("--decoded_path", type=str, default=None,
                    help="--targets decoded: pred_val.npy of a NeuralGuidance decode map")
    p.add_argument("--indices", type=str, default=None,
                    help="val-folder rows to generate for: a .npy file or a key of an .npz split file "
                         "(e.g. splits/neural_holdout.npz:heldout). Default: all val images")
    p.add_argument("--n", type=int, default=100,
                    help="random subsample of --indices (same rule as NeuralGuidance, so the same --seed "
                         "picks the same stimuli); 0 = use all of them")
    p.add_argument("--n_seeds", type=int, default=4, help="samples per stimulus")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--guidance_scale", type=float, default=1.0,
                    help="> 1.0 requires a CFG-trained model (cfg_uncond_prob > 0)")
    p.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"],
                    help="DDPM-only; ignored when noise_process is flow/vpsde")
    p.add_argument("--eta", type=float, default=0.0, help="ddim only: 0=deterministic, 1~ddpm stochasticity")
    p.add_argument("--num_inference_steps", type=int, default=100,
                    help="Steps for ddim/flow/vpsde; ddpm always uses all T steps")
    p.add_argument("--output_dir", type=str, default=None,
                    help="Default: <diffusion run output_dir>/generation/<checkpoint stem>_<targets>")
    args = p.parse_args()
    if args.targets == "decoded" and args.decoded_path is None:
        p.error("--targets decoded needs --decoded_path")
    return args


def load_indices(spec: str) -> np.ndarray:
    """Indices from a .npy file or a key of a .npz split file (`path.npz:heldout`)."""
    if ".npz:" in spec:
        path, key = spec.rsplit(":", 1)
        return np.load(path)[key].astype(np.int64)
    return np.load(spec).astype(np.int64)


def load_decoded_latents(path: str, diff_cfg) -> np.ndarray:
    """Decoded features -> this model's conditioning space (see the module docstring)."""
    decoded = np.load(path).astype(np.float32)
    if decoded.shape[1] == diff_cfg.latent_dim:
        print(f"Decoded targets are {decoded.shape[1]}-d = latent_dim: used as z directly "
              f"(they must be in this model's PC basis)")
        return decoded
    latents_dir = latents_dir_for(diff_cfg.data_dir, diff_cfg.dataset_type, diff_cfg.encoding_model)
    scaler_path = os.path.join(latents_dir, f"{diff_cfg.encoding_model}_scaler_z{diff_cfg.latent_dim}.joblib")
    pca_path = os.path.join(latents_dir, f"{diff_cfg.encoding_model}_pca_z{diff_cfg.latent_dim}.joblib")
    if not (os.path.exists(scaler_path) and os.path.exists(pca_path)):
        raise FileNotFoundError(
            f"{path} is {decoded.shape[1]}-d but the model is conditioned on {diff_cfg.latent_dim} PCs, and "
            f"there is no scaler/PCA to project with ({scaler_path}, {pca_path})"
        )
    scaler, pca = joblib.load(scaler_path), joblib.load(pca_path)
    if decoded.shape[1] != scaler.n_features_in_:
        raise ValueError(
            f"{path} is {decoded.shape[1]}-d: neither latent_dim ({diff_cfg.latent_dim}) nor the "
            f"{diff_cfg.encoding_model} embedding dimension ({scaler.n_features_in_}). Was the neuron map "
            f"fit on another encoder?"
        )
    # scaler and PCA are affine, so rows without neural data (NaN) just stay NaN
    has_data = np.isfinite(decoded).all(axis=1)
    latents = np.full((len(decoded), diff_cfg.latent_dim), np.nan, dtype=np.float32)
    latents[has_data] = pca.transform(scaler.transform(decoded[has_data]))
    print(f"Projected {decoded.shape[1]}-d decoded features onto the model's top {diff_cfg.latent_dim} PCs "
          f"({pca_path})")
    return latents


def main():
    args = parse_args()
    device = torch.device(discover_device())
    print(f"Device: {device}")

    # --- Diffusion config + checkpoint (same resolution rules as inspect_diffusion.py) ---
    with open(args.diffusion_config_path) as f:
        raw_cfg = yaml.safe_load(f)
    diff_cfg = types.SimpleNamespace(**raw_cfg)
    diff_cfg.output_dir = f"{diff_cfg.output_dir}_{make_run_tag(diff_cfg)}"
    np_type = getattr(diff_cfg, "noise_process", "ddpm")

    if args.targets != "none" and getattr(diff_cfg, "encoding_model", None) is None:
        raise ValueError("--targets oracle/decoded need a model conditioned on precomputed latents "
                         "(encoding_model in the diffusion config); use --targets none otherwise")

    if args.ckpt == "all" or "," in args.ckpt:
        raise ValueError("--ckpt takes a single checkpoint")
    ckpt_path = select_checkpoints(os.path.join(diff_cfg.output_dir, "checkpoints"), args.ckpt)[0]
    ckpt_stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    output_dir = args.output_dir or os.path.join(diff_cfg.output_dir, "generation", f"{ckpt_stem}_{args.targets}")

    unet = UNet(
        in_channels=diff_cfg.in_channels,
        image_size=diff_cfg.image_size,
        model_channels=diff_cfg.model_channels,
        channel_mult=diff_cfg.channel_mult,
        num_res_blocks=diff_cfg.num_res_blocks,
        attention_resolutions=diff_cfg.attention_resolutions,
        dropout=0.0,
        latent_dim=diff_cfg.latent_dim,
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    unet.load_state_dict(ckpt["model_state_dict"])
    used_ema = args.use_ema_weights and "ema_state_dict" in ckpt
    if used_ema:
        unet.load_state_dict(ema_shadow_to_model_state_dict(unet, ckpt["ema_state_dict"]["shadow"]))
    elif args.use_ema_weights:
        print("use_ema_weights=True but checkpoint has no ema_state_dict -- using raw weights")
    unet = unet.to(device).eval()
    print(f"UNet loaded: {ckpt_path}  (step {ckpt['step']}, {'EMA' if used_ema else 'raw'} weights)")

    schedule = make_noise_schedule(diff_cfg, device) if np_type == "ddpm" else None

    # --- Val images: the stimuli the rows of every target file refer to ---
    # ImageNet64 layout: val_images.npy in data_dir. Stimuli layout: a separate folder,
    # val_data_dir (defaulting to data_dir's sibling "500Stimuli"), in sorted-path order --
    # same convention as inspect_diffusion.py / data_utils.build_stimuli_dataloaders.
    val_images_path = os.path.join(diff_cfg.data_dir, "val_images.npy")
    if os.path.exists(val_images_path):
        val_images, val_source = np.load(val_images_path, mmap_mode="r"), val_images_path
    else:
        val_source = getattr(diff_cfg, "val_data_dir", None) or os.path.join(
            os.path.dirname(os.path.normpath(diff_cfg.data_dir)), "500Stimuli"
        )
        val_images = _find_all_images(val_source)
    n_val = len(val_images)

    # --- Conditioning latents, one row per val image ---
    if args.targets == "oracle":
        latents_path = resolve_latents_paths(
            diff_cfg.data_dir, diff_cfg.dataset_type, diff_cfg.encoding_model, diff_cfg.latent_dim, splits=("val",),
        )[0]
        table = np.load(latents_path, mmap_mode="r")
    elif args.targets == "decoded":
        latents_path = args.decoded_path
        table = load_decoded_latents(args.decoded_path, diff_cfg)
    else:
        latents_path, table = None, None
    if table is not None and len(table) != n_val:
        raise ValueError(f"{latents_path} has {len(table)} rows but there are {n_val} val images ({val_source})")

    # --- Which stimuli ---
    indices = np.arange(n_val) if args.indices is None else load_indices(args.indices)
    if args.n and args.n < len(indices):
        indices = np.sort(np.random.default_rng(args.seed).choice(indices, args.n, replace=False))
    if table is not None:
        z_all = torch.from_numpy(np.asarray(table[indices], dtype=np.float32))
        missing = ~torch.isfinite(z_all).all(dim=1)
        if missing.any():
            raise SystemExit(
                f"{int(missing.sum())} of the selected stimuli have no target in {latents_path} "
                f"(e.g. index {int(indices[missing.numpy()][0])}). Restrict --indices to stimuli with "
                f"neural data, e.g. the heldout key of the NeuralGuidance split file"
            )
    else:
        z_all = torch.zeros(len(indices), diff_cfg.latent_dim)

    # --- Generate ---
    C, H = diff_cfg.in_channels, diff_cfg.image_size
    pairs = [(i, s) for i in range(len(indices)) for s in range(args.n_seeds)]
    n_steps = diff_cfg.num_timesteps if (np_type == "ddpm" and args.sampler == "ddpm") else args.num_inference_steps
    passes_per_step = 2 if args.guidance_scale != 1.0 else 1
    print(f"{len(indices)} stimuli x {args.n_seeds} seeds = {len(pairs)} samples, {n_steps} {np_type} steps, "
          f"targets: {args.targets}")

    images = np.zeros((len(indices), args.n_seeds, H, H, 3), dtype=np.uint8)
    start = time.time()
    for b in range(0, len(pairs), args.batch_size):
        batch = pairs[b:b + args.batch_size]
        # one fixed starting noise per (stimulus, seed), independent of batching and of --targets
        x_init = torch.cat([
            torch.randn(1, C, H, H, generator=torch.Generator().manual_seed(
                args.seed + 1_000_003 * s + int(indices[i])))
            for i, s in batch
        ]).to(device)
        z = z_all[[i for i, _ in batch]].to(device)
        torch.manual_seed(args.seed + b)  # the noise a stochastic sampler adds along the way
        with torch.no_grad():
            if np_type == "ddpm":
                generated = diffusion_sample(
                    z, unet, schedule, diff_cfg, device,
                    sampler=args.sampler, eta=args.eta, num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale, x_init=x_init,
                )
            elif np_type == "flow":
                generated = flow_sample(
                    z, unet, diff_cfg, device, num_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale, x_init=x_init,
                )
            elif np_type == "vpsde":
                generated = vpsde_sample(
                    z, unet, diff_cfg, device, num_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale, x_init=x_init,
                )
            else:
                raise ValueError(f"Unknown noise_process: {np_type!r}")
        if C == 1:
            generated = generated.expand(-1, 3, -1, -1)
        batch_images = (generated.permute(0, 2, 3, 1) * 255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        for k, (i, s) in enumerate(batch):
            images[i, s] = batch_images[k]
        print(f"  {min(b + args.batch_size, len(pairs))}/{len(pairs)} samples, "
              f"{(time.time() - start) / (b + len(batch)):.2f} s/sample")
    elapsed = time.time() - start

    # --- Save (same files and meta keys as NeuralGuidance's scripts/generate.py) ---
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "samples.npy"), images)
    meta = {
        "config": {
            "diffusion_config": raw_cfg,
            "diffusion_config_path": args.diffusion_config_path,
            "checkpoint": ckpt_path,
            "checkpoint_step": int(ckpt["step"]),
            "ema_weights": bool(used_ema),
            "targets": {"source": args.targets, "path": latents_path, "indices": args.indices, "n": args.n},
            "sampler": {"noise_process": np_type, "sampler": args.sampler if np_type == "ddpm" else np_type,
                        "eta": args.eta, "num_steps": n_steps, "guidance_scale": args.guidance_scale},
            "seed": args.seed,
        },
        "stage": {"oracle": "oracle", "decoded": "neural", "none": None}[args.targets],
        "indices": indices.tolist(),
        "split": "val",
        "val_images": val_source,
        "n_seeds": args.n_seeds,
        "forward_passes_per_sample": float(n_steps * passes_per_step),
        "backward_passes_per_sample": 0.0,
        "seconds_per_sample": elapsed / len(pairs),
        "final_energy_mean": None,
        "device": str(device),
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # grid: one column per stimulus; top row is the real image, then one row per seed
    n_show = min(8, len(indices))
    if isinstance(val_images, list):
        to_tensor = transforms.Compose([transforms.Resize((H, H)), transforms.ToTensor()])
        real = torch.stack([to_tensor(Image.open(val_images[i]).convert("RGB")) for i in indices[:n_show]])
    else:
        real = torch.from_numpy(np.array(val_images[indices[:n_show]])).float() / 255.0
        if real.shape[-1] != H:
            real = torch.nn.functional.interpolate(real, size=H, mode="bilinear", align_corners=False)
    rows = [real] + [torch.from_numpy(images[:n_show, s]).permute(0, 3, 1, 2).float() / 255.0
                     for s in range(args.n_seeds)]
    save_image(make_grid(torch.cat(rows), nrow=n_show, padding=2), os.path.join(output_dir, "grid.png"))
    print(f"Wrote {output_dir} ({elapsed / len(pairs):.2f} s/sample)")


if __name__ == "__main__":
    main()
