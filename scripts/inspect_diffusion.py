"""
Standalone script mirroring notebooks/inspect_diffusion.ipynb: load a trained diffusion
checkpoint (+ frozen beta-VAE if conditioned on one) and produce the same three
inspection figures -- one-step x0 predictions at a few noise levels, a final-sample
comparison against the originals/beta-VAE reconstruction, and a single sample's
denoising progression.

Deliberate deviations from the notebook (script context, not a Colab notebook):
  - No REPO_ROOT/sys.path.insert, Drive mount, or importlib.metadata patch -- those
    were Colab-path workarounds; run this as `python -m scripts.inspect_diffusion` from
    the repo root instead, which already resolves models/utils/config paths directly
    (config, betavae_config_path, betavae_ckpt_path, data_dir are all used as-is, same
    as scripts/diffusion_train.py).
  - Dropped the tiny_imagenet/imagenet64 dataset-staging cell -- that copied data onto
    Colab's local runtime disk for speed; a local/VM run already has data_dir on disk.
  - Every plt.show() becomes plt.savefig() into <diff_cfg.output_dir>/inspection/ --
    there's no inline display in a headless script run.
  - Notebook Config-cell variables became CLI flags (see parse_args below) so a VM run
    doesn't require editing the file -- defaults match the notebook's Config cell.

Run from repo root:
    python -m scripts.inspect_diffusion --diffusion_config_path config/diffusion_full_imagenet64_flow_uncond.yaml
"""

import os
import types
import random
import argparse

import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

from models.beta_vae import BetaVAE
from models.unet import UNet
from models.ema import ema_shadow_to_model_state_dict
from models.noise_process import DDPM, RectifiedFlow, VPSDE
from data_utils import resolve_latents_paths
from utils import discover_device, make_noise_schedule, diffusion_sample, flow_sample, vpsde_sample, make_run_tag


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--diffusion_config_path", type=str,
                    default="config/diffusion_full_imagenet64_flow_uncond.yaml")
    p.add_argument("--diffusion_ckpt_path", type=str, default=None,
                    help="None -> load best_ckpt.pt from the run's checkpoints dir")
    p.add_argument("--unconditional", action=argparse.BooleanOptionalAction, default=True,
                    help="True -> skip beta-VAE / precomputed latents, pass z=0")
    p.add_argument("--load_test_images", action=argparse.BooleanOptionalAction, default=None,
                    help="Defaults to `not unconditional`, same as the notebook's "
                         "LOAD_TEST_IMAGES = not UNCONDITIONAL")
    p.add_argument("--use_ema_weights", action=argparse.BooleanOptionalAction, default=True,
                    help="Sample from the checkpoint's EMA shadow if present, falling back "
                         "to raw weights with a warning otherwise")
    p.add_argument("--n_samples", type=int, default=8)
    p.add_argument("--guidance_scale", type=float, default=1.0,
                    help="> 1.0 requires a CFG-trained model (cfg_uncond_prob > 0)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"],
                    help="DDPM-only; ignored when noise_process is flow/vpsde")
    p.add_argument("--eta", type=float, default=0.0, help="ddim only: 0=deterministic, 1~ddpm stochasticity")
    p.add_argument("--num_inference_steps", type=int, default=100,
                    help="Steps for ddim/flow/vpsde; ddpm always uses all T steps")
    p.add_argument("--output_dir", type=str, default=None,
                    help="Where to save figures. Defaults to <diffusion run output_dir>/inspection")
    args = p.parse_args()
    if args.load_test_images is None:
        args.load_test_images = not args.unconditional
    return args


def _show(ax, img):
    img = img.detach().cpu().permute(1, 2, 0).float().clamp(0, 1).numpy()
    ax.imshow(img.squeeze(-1) if img.shape[-1] == 1 else img, cmap="gray" if img.shape[-1] == 1 else None)
    ax.axis("off")


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(discover_device())
    print(f"Device: {device}")

    # --- Load diffusion config ---
    with open(args.diffusion_config_path) as f:
        diff_cfg = types.SimpleNamespace(**yaml.safe_load(f))
    diff_cfg.output_dir = f"{diff_cfg.output_dir}_{make_run_tag(diff_cfg)}"

    inspect_dir = args.output_dir or os.path.join(diff_cfg.output_dir, "inspection")
    os.makedirs(inspect_dir, exist_ok=True)

    # --- frozen beta-VAE, only loaded when the diffusion model actually conditions on one ---
    if args.unconditional:
        vae = None
        print("Unconditional mode: skipping beta-VAE, z is a zero vector")
    elif hasattr(diff_cfg, "betavae_config_path"):
        with open(diff_cfg.betavae_config_path) as f:
            vae_cfg = types.SimpleNamespace(**yaml.safe_load(f))
        vae = BetaVAE(
            input_channels=vae_cfg.input_channels,
            input_height=vae_cfg.input_height,
            input_width=vae_cfg.input_width,
            latent_dim=vae_cfg.latent_dim,
            hidden_dim=vae_cfg.hidden_dim,
            beta=vae_cfg.beta,
            device=device,
        )
        vae_ckpt = torch.load(diff_cfg.betavae_ckpt_path, map_location=device)
        vae.load_state_dict(vae_ckpt["model_state_dict"])
        vae.eval()
        print(f"beta-VAE loaded (step {vae_ckpt['step']})")
    elif getattr(diff_cfg, "encoding_model", None) is not None:
        # Conditional on precomputed latents (e.g. AlexNet-fc6-PCA), row-aligned with the
        # val images, so z is looked up from val_latents_z{dim}.npy instead of vae.encode()
        # (see "Load images" below).
        vae = None
        print(f"Conditional on precomputed {diff_cfg.encoding_model} latents -- no VAE to load, z comes from val_latents")
    else:
        raise ValueError(
            "Conditional config with nothing to condition on: set encoding_model (precomputed "
            "latents), betavae_config_path (on-the-fly VAE latents), or pass --unconditional"
        )

    # --- UNet ---
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

    diffusion_ckpt_path = args.diffusion_ckpt_path
    if diffusion_ckpt_path is None:
        ckpt_dir = os.path.join(diff_cfg.output_dir, "checkpoints")
        diffusion_ckpt_path = os.path.join(ckpt_dir, "best_ckpt.pt")
        if not os.path.exists(diffusion_ckpt_path):
            raise FileNotFoundError(f"best_ckpt.pt not found in {ckpt_dir}")

    diff_ckpt = torch.load(diffusion_ckpt_path, map_location=device)
    unet.load_state_dict(diff_ckpt["model_state_dict"])

    if args.use_ema_weights and "ema_state_dict" in diff_ckpt:
        ema_sd = diff_ckpt["ema_state_dict"]
        unet.load_state_dict(ema_shadow_to_model_state_dict(unet, ema_sd["shadow"]))
        print(f"Sampling from EMA weights (EMA step {ema_sd['step']}, decay {ema_sd['decay']})")
    elif args.use_ema_weights:
        print("use_ema_weights=True but checkpoint has no ema_state_dict "
              "(older run, or ema_decay wasn't set during training) -- using raw weights")

    unet = unet.to(device).eval()
    print(f"UNet loaded: {diffusion_ckpt_path}  (step {diff_ckpt['step']})")

    # --- Load images ---
    if not args.unconditional and vae is not None and not args.load_test_images:
        raise ValueError("beta-VAE conditioning needs real images to encode a latent from -- "
                          "set --load_test_images")

    sampled_latents = None  # only populated in the precomputed-latents branch below
    originals = None

    if args.unconditional or vae is not None:
        if args.load_test_images:
            # beta-VAE conditioning (or an unconditional run that still wants images to
            # display) -- encode raw images from a folder dataset (e.g. data/500Stimuli)
            import glob
            from torchvision import transforms
            from PIL import Image

            exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
            all_paths = sorted(
                p for p in glob.glob(os.path.join(diff_cfg.data_dir, "**", "*"), recursive=True)
                if os.path.splitext(p)[1].lower() in exts and "Zone.Identifier" not in p
            )
            sampled_paths = random.sample(all_paths, min(args.n_samples, len(all_paths)))

            transform = transforms.Compose([
                transforms.Resize((diff_cfg.image_size, diff_cfg.image_size)),
                transforms.ToTensor(),
            ])
            originals = torch.stack([transform(Image.open(p).convert("RGB")) for p in sampled_paths]).to(device)
        else:
            print("load_test_images=False -- skipping all image loading (unconditional sampling only)")
    else:
        # Conditional on precomputed latents -- z always comes from val_latents_z{dim}.npy
        # in data_dir's sibling {dataset_type}_{encoding_model}_pca_latents directory (see
        # data_utils.resolve_latents_paths).
        # ImageNet64 layout: val_images.npy in data_dir is a consolidated (N,3,64,64) array,
        # row-aligned with the latents (_ImageNet64ArrayDataset convention). Stimuli layout
        # has no such array: the latents are row-aligned with a *separate* folder of files,
        # val_data_dir (defaulting to data_dir's sibling "500Stimuli") -- same convention as
        # data_utils.build_stimuli_dataloaders. Detected here purely from whether
        # val_images.npy exists on disk.
        val_latents_path = resolve_latents_paths(
            diff_cfg.data_dir, diff_cfg.dataset_type, diff_cfg.encoding_model, diff_cfg.latent_dim,
            splits=("val",),
        )[0]
        val_latents = np.load(val_latents_path, mmap_mode="r")

        val_images_path = os.path.join(diff_cfg.data_dir, "val_images.npy")
        val_images_are_paths = not os.path.exists(val_images_path)
        if not val_images_are_paths:
            val_images = np.load(val_images_path, mmap_mode="r")
        else:
            import glob
            exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
            val_data_dir = getattr(diff_cfg, "val_data_dir", None) or os.path.join(
                os.path.dirname(os.path.normpath(diff_cfg.data_dir)), "500Stimuli"
            )
            val_images = sorted(
                p for p in glob.glob(os.path.join(val_data_dir, "**", "*"), recursive=True)
                if os.path.splitext(p)[1].lower() in exts and "Zone.Identifier" not in p
            )
            if len(val_images) != len(val_latents):
                raise ValueError(
                    f"val_latents ({len(val_latents)} rows, {val_latents_path}) / val images "
                    f"({len(val_images)}, {val_data_dir}) count mismatch"
                )
        print("Loaded precomputed latents + matching real images")

    # --- Noise schedule ---
    np_type = getattr(diff_cfg, "noise_process", "ddpm")
    embed_scale = getattr(diff_cfg, "num_timesteps", 1000)

    if np_type == "ddpm":
        noise_proc = DDPM(
            num_timesteps=diff_cfg.num_timesteps,
            beta_schedule=diff_cfg.beta_schedule,
            min_snr_gamma=getattr(diff_cfg, "min_snr_gamma", 5.0),
            device=device,
        )
        schedule = make_noise_schedule(diff_cfg, device)
        print(f"DDPM  T={diff_cfg.num_timesteps}  beta_0={schedule['betas'][0]:.5f}  beta_T={schedule['betas'][-1]:.4f}")
    elif np_type == "flow":
        noise_proc = RectifiedFlow()
        schedule = None
        print("RectifiedFlow -- continuous ODE, t in [0, 1]")
    elif np_type == "vpsde":
        noise_proc = VPSDE(
            beta_min=getattr(diff_cfg, "vpsde_beta_min", 0.01),
            beta_max=getattr(diff_cfg, "vpsde_beta_max", 5.0),
        )
        schedule = None
        print(f"VPSDE  beta_min={noise_proc.beta_min}  beta_max={noise_proc.beta_max}")
    else:
        raise ValueError(f"Unknown noise_process: {np_type!r}")

    print(f"embed_scale={embed_scale}")

    # --- Get conditioning latents (z) ---
    if not args.unconditional and vae is None:
        idx = random.sample(range(len(val_latents)), min(args.n_samples, len(val_latents)))
        sampled_latents = torch.from_numpy(np.array(val_latents[idx])).float().to(device)

        if val_images_are_paths:
            from torchvision import transforms
            from PIL import Image

            transform = transforms.Compose([
                transforms.Resize((diff_cfg.image_size, diff_cfg.image_size)),
                transforms.ToTensor(),
            ])
            originals = torch.stack(
                [transform(Image.open(val_images[i]).convert("RGB")) for i in idx]
            ).to(device)
        else:
            originals = torch.from_numpy(np.array(val_images[idx])).float().to(device) / 255.0
            if diff_cfg.image_size != originals.shape[-1]:
                originals = torch.nn.functional.interpolate(
                    originals, size=diff_cfg.image_size, mode="bilinear", align_corners=False
                )

    with torch.no_grad():
        if args.unconditional:
            n_z = originals.shape[0] if originals is not None else args.n_samples
            z = torch.zeros(n_z, diff_cfg.latent_dim, device=device)
            vae_recon = None
            print("Diffusion model is unconditional, skip beta VAE")
        elif vae is not None:
            z, _ = vae.encode(originals)
            vae_recon = torch.sigmoid(vae.decode(z)).cpu()
        else:
            # Conditional on precomputed latents -- z was already looked up by index
            # alongside originals above, no VAE to reconstruct through.
            z = sampled_latents
            vae_recon = None
            print("Conditional on precomputed latents -- no VAE reconstruction to show")

    # --- One-step predictions ---
    if originals is None:
        print("Skipping one-step predictions -- no test images loaded (pass --load_test_images to enable)")
    else:
        n_vis = originals.shape[0]
        t_vals = noise_proc.vis_t_vals()
        n_rows = 1 + 2 * len(t_vals)

        fig, axes = plt.subplots(n_rows, n_vis, figsize=(n_vis * 2, n_rows * 2), squeeze=False)

        row_labels = ["x0"]
        for tv in t_vals:
            row_labels += [f"x_t  t={tv}", f"x_pred  t={tv}"]

        for col in range(n_vis):
            _show(axes[0, col], originals[col])

        with torch.no_grad():
            for ti, t_val in enumerate(t_vals):
                t_tensor = torch.full((n_vis,), t_val, device=device, dtype=noise_proc.dtype)
                x_t, _ = noise_proc.corrupt(originals[:n_vis], t_tensor)
                pred = unet(x_t, noise_proc.embed_t(t_tensor, embed_scale), z[:n_vis])
                x0_pred = noise_proc.predict_x0(x_t, pred, t_tensor)
                row_xt = 1 + 2 * ti
                for col in range(n_vis):
                    _show(axes[row_xt, col], x_t[col].clamp(0, 1))
                    _show(axes[row_xt + 1, col], x0_pred[col].clamp(0, 1))

        for row, label in enumerate(row_labels):
            axes[row, 0].set_ylabel(label, fontsize=7, rotation=0, ha="right", va="center", labelpad=55)
        fig.suptitle(f"One-step predictions ({np_type})", fontsize=9)
        plt.tight_layout()
        one_step_path = os.path.join(inspect_dir, "one_step_predictions.png")
        plt.savefig(one_step_path, dpi=150)
        plt.close()
        print(f"Saved one-step predictions figure to {one_step_path}")

    # --- Sample images and generate ---
    with torch.no_grad():
        if np_type == "ddpm":
            generated = diffusion_sample(
                z, unet, schedule, diff_cfg, device,
                sampler=args.sampler, eta=args.eta, num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            )
        elif np_type == "flow":
            generated = flow_sample(
                z, unet, diff_cfg, device,
                num_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            )
        elif np_type == "vpsde":
            generated = vpsde_sample(
                z, unet, diff_cfg, device,
                num_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            )

    generated = generated.cpu()
    originals_cpu = originals.cpu() if originals is not None else None

    n = generated.shape[0]
    if originals_cpu is None:
        rows = [generated]
        if args.unconditional:
            labels = ["diffusion (unconditional)"]
            title = "diffusion generated (unconditional) -- no test images loaded"
        else:
            labels = ["diffusion (conditioned on precomputed latents)"]
            title = "diffusion generated -- conditioned on precomputed latents, no test images loaded"
    elif vae_recon is not None:
        rows = [originals_cpu, vae_recon, generated]
        labels = ["original", "beta-VAE recon", "diffusion"]
        title = "original  |  beta-VAE reconstruction  |  diffusion generated"
    else:
        rows = [originals_cpu, generated]
        labels = ["original", "diffusion" if not args.unconditional else "diffusion (unconditional)"]
        title = "original  |  diffusion generated"

    fig, axes = plt.subplots(len(rows), n, figsize=(n * 2.5, len(rows) * 2.5), squeeze=False)
    fig.suptitle(title, fontsize=11)

    for row_idx, (imgs, label) in enumerate(zip(rows, labels)):
        axes[row_idx, 0].set_ylabel(label, fontsize=9)
        for col_idx in range(n):
            axes[row_idx, col_idx].imshow(imgs[col_idx].permute(1, 2, 0).clamp(0, 1))
            axes[row_idx, col_idx].axis("off")

    plt.tight_layout()
    samples_path = os.path.join(inspect_dir, "samples_comparison.png")
    plt.savefig(samples_path, dpi=150)
    plt.close()
    print(f"Saved samples comparison figure to {samples_path}")

    # --- Denoising progression (verbose) ---
    z_single = z[:1]

    with torch.no_grad():
        if np_type == "ddpm":
            _, snapshots = diffusion_sample(
                z_single, unet, schedule, diff_cfg, device,
                sampler=args.sampler, eta=args.eta, num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                verbose=True, n_snapshots=10,
            )
        elif np_type == "flow":
            _, snapshots = flow_sample(
                z_single, unet, diff_cfg, device,
                num_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                verbose=True, n_snapshots=10,
            )
        elif np_type == "vpsde":
            _, snapshots = vpsde_sample(
                z_single, unet, diff_cfg, device,
                num_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                verbose=True, n_snapshots=10,
            )

    n_snaps = len(snapshots)
    fig, axes = plt.subplots(1, n_snaps, figsize=(n_snaps * 2, 2.8))

    proc_label = args.sampler.upper() if np_type == "ddpm" else np_type.upper()
    steps_label = (f"[{diff_cfg.num_timesteps} steps]"
                   if np_type == "ddpm" and args.sampler == "ddpm"
                   else f"[{args.num_inference_steps} steps]")
    fig.suptitle(f"{proc_label} progression  (left: noise -> right: final)  {steps_label}", fontsize=11)

    for ax, (t_idx, img) in zip(axes, snapshots):
        ax.imshow(img[0].permute(1, 2, 0).clamp(0, 1))
        ax.set_title(f"t={t_idx:.3f}" if isinstance(t_idx, float) else f"t={t_idx}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    progression_path = os.path.join(inspect_dir, "denoising_progression.png")
    plt.savefig(progression_path, dpi=150)
    plt.close()
    print(f"Saved denoising progression figure to {progression_path}")


if __name__ == "__main__":
    main()
