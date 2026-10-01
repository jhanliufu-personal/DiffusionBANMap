"""
FID over a sweep of checkpoints, not just best_ckpt.pt.

Same pipeline as notebooks/calculate_fid.ipynb (Inception-v3 pool3 features + Frechet
distance), but instead of evaluating a single checkpoint it globs every checkpoint file
in checkpoints/ matching --ckpt_pattern (default "ckpt_step_*.pt", i.e. the periodic
saves from DiffusionTrainer's ckpt_interval -- see the "FID-vs-compute curve" comment in
config/diffusion_full_imagenet64_flow_uncond.yaml) and evaluates each one. Real reference
images are loaded once and reused across every checkpoint; all checkpoints share the same
sampling hyperparameters (N_FID_SAMPLES, FID_BATCH_SIZE, etc).

For each checkpoint <stem>.pt, writes into <output_dir>/fid_eval/:
    <stem>_sampled_images.pt        generated images, [N, C, H, W] float in [0, 1]
    <stem>_inception_features.npy   pool3 features of the sampled images, [N, 2048]
    <stem>_examples.png             grid of example sampled images
    <stem>_fid.yaml                 FID score + run info
plus a fid_summary.yaml mapping every checkpoint to {step, fid} for a FID-vs-compute plot.

Run from repo root:
    python -m scripts.calculate_fid_sweep --config config/diffusion_full_imagenet64_flow_uncond.yaml
"""

import os
import glob
import types
import yaml
import random
import argparse
import itertools
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import linalg
from tqdm import tqdm
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.utils import make_grid

from models.beta_vae import BetaVAE
from models.unet import UNet
from models.ema import ema_shadow_to_model_state_dict
from utils import make_noise_schedule, diffusion_sample, flow_sample, vpsde_sample, make_run_tag, discover_device
from data_utils import build_dataloaders, build_imagenet64_val_dataloader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--ckpt_dir", type=str, default=None,
                    help="Default: <output_dir>/checkpoints")
    p.add_argument("--ckpt_pattern", type=str, default="ckpt_step_*.pt",
                    help="glob pattern, relative to --ckpt_dir")
    p.add_argument("--eval_dir", type=str, default=None,
                    help="Default: <output_dir>/fid_eval")
    p.add_argument("--use_ema", dest="use_ema", action="store_true", default=True,
                    help="Sample from each checkpoint's EMA shadow when present (default: on)")
    p.add_argument("--no_ema", dest="use_ema", action="store_false")

    # FID sampling budget -- see notebooks/calculate_fid.ipynb for the real-vs-generated
    # sizing rationale. Shared across every checkpoint in the sweep.
    p.add_argument("--n_fid_samples", type=int, default=50000)
    p.add_argument("--fid_batch_size", type=int, default=128)

    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"],
                    help="DDPM-only; ignored when noise_process is 'flow' or 'vpsde'")
    p.add_argument("--eta", type=float, default=0.0, help="ddim only")
    p.add_argument("--num_inference_steps", type=int, default=100)
    p.add_argument("--n_show", type=int, default=8, help="Images in the example grid")
    return p.parse_args()


def frechet_distance(feat_real: np.ndarray, feat_fake: np.ndarray, eps: float = 1e-6) -> float:
    mu_r, mu_f = feat_real.mean(axis=0), feat_fake.mean(axis=0)
    sigma_r = np.cov(feat_real, rowvar=False)
    sigma_f = np.cov(feat_fake, rowvar=False)

    diff = mu_r - mu_f
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_f, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_r.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_r + offset) @ (sigma_f + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config) as f:
        diff_cfg = types.SimpleNamespace(**yaml.safe_load(f))
    diff_cfg.output_dir = f"{diff_cfg.output_dir}_{make_run_tag(diff_cfg)}"

    device = torch.device(discover_device())
    print(f"Device: {device}")

    unconditional = getattr(diff_cfg, "unconditional", False)
    if unconditional:
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
        # Conditional on precomputed latents -- z comes from the latents handed back by
        # the dataloader (see data_utils.resolve_latents_paths) instead of vae.encode().
        vae = None
        print(f"Conditional on precomputed {diff_cfg.encoding_model} latents -- no VAE to load, z comes from the dataloader")
    else:
        raise ValueError(
            "Conditional config with nothing to condition on: set encoding_model (precomputed "
            "latents), betavae_config_path (on-the-fly VAE latents), or unconditional: true"
        )

    ckpt_dir = args.ckpt_dir or os.path.join(diff_cfg.output_dir, "checkpoints")
    ckpt_paths = sorted(glob.glob(os.path.join(ckpt_dir, args.ckpt_pattern)))
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoints matching {args.ckpt_pattern!r} in {ckpt_dir}")
    print(f"Found {len(ckpt_paths)} checkpoints matching {args.ckpt_pattern!r} in {ckpt_dir}")

    eval_dir = args.eval_dir or os.path.join(diff_cfg.output_dir, "fid_eval")
    os.makedirs(eval_dir, exist_ok=True)

    # --- UNet skeleton, reloaded per checkpoint below ---
    unet = UNet(
        in_channels=diff_cfg.in_channels,
        image_size=diff_cfg.image_size,
        model_channels=diff_cfg.model_channels,
        channel_mult=diff_cfg.channel_mult,
        num_res_blocks=diff_cfg.num_res_blocks,
        attention_resolutions=diff_cfg.attention_resolutions,
        dropout=0.0,
        num_head_channels=getattr(diff_cfg, "num_head_channels", 64),
        latent_dim=diff_cfg.latent_dim,
    ).to(device).eval()

    # --- noise process / schedule -- shared across every checkpoint ---
    np_type = getattr(diff_cfg, "noise_process", "ddpm")
    if np_type == "ddpm":
        schedule = make_noise_schedule(diff_cfg, device)
        print(f"DDPM  T={diff_cfg.num_timesteps}  beta_0={schedule['betas'][0]:.5f}  beta_T={schedule['betas'][-1]:.4f}")
    elif np_type == "flow":
        schedule = None
        print("RectifiedFlow -- continuous ODE, t in [0, 1]")
    elif np_type == "vpsde":
        schedule = None
        print(f"VPSDE  beta_min={getattr(diff_cfg, 'vpsde_beta_min', 0.01)}  "
              f"beta_max={getattr(diff_cfg, 'vpsde_beta_max', 5.0)}")
    else:
        raise ValueError(f"Unknown noise_process: {np_type!r}")

    # --- real reference images -- loaded once, shared across every checkpoint ---
    if diff_cfg.dataset_type == "imagenet64":
        val_dl = build_imagenet64_val_dataloader(
            data_dir=diff_cfg.data_dir,
            image_size=diff_cfg.image_size,
            batch_size=diff_cfg.batch_size,
            latent_dim=getattr(diff_cfg, "latent_dim", None),
            unconditional=unconditional,
            encoding_model=getattr(diff_cfg, "encoding_model", None),
        )
    else:
        _, val_dl = build_dataloaders(diff_cfg)

    real_batches = []
    n_real = 0
    for batch in val_dl:
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        real_batches.append(imgs)
        n_real += imgs.shape[0]
        if n_real >= args.n_fid_samples:
            break
    real_images = torch.cat(real_batches, dim=0)[:args.n_fid_samples]

    if real_images.shape[0] < args.n_fid_samples:
        print(f"Real reference: only {real_images.shape[0]} images available in the val split "
              f"(< N_FID_SAMPLES={args.n_fid_samples}) -- using the full val split rather than "
              f"duplicating real images to pad it out.")
    else:
        print(f"Real reference: {real_images.shape[0]} images (capped at N_FID_SAMPLES)")

    # --- Inception feature extractor -- shared across every checkpoint ---
    inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    inception.fc = torch.nn.Identity()
    inception = inception.eval().to(device)

    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def get_inception_features(images: torch.Tensor, batch_size: int) -> np.ndarray:
        feats = []
        for i in tqdm(range(0, images.shape[0], batch_size), desc="Inception features", leave=False):
            batch = images[i:i + batch_size].to(device)
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)
            batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
            batch = (batch - imagenet_mean) / imagenet_std
            feats.append(inception(batch).cpu().numpy())
        return np.concatenate(feats, axis=0)

    feat_real = get_inception_features(real_images, batch_size=args.fid_batch_size)

    # Real images (and, for conditional models, their matching latents) also serve as the
    # source of z for generation, cycled if N_FID_SAMPLES exceeds the val split size --
    # same as notebooks/calculate_fid.ipynb.
    z_source = itertools.cycle(val_dl) if not unconditional else None

    summary = {}
    for ckpt_path in ckpt_paths:
        stem = os.path.splitext(os.path.basename(ckpt_path))[0]
        print(f"\n=== {stem} ===")

        sampled_images_path = os.path.join(eval_dir, f"{stem}_sampled_images.pt")
        feat_path = os.path.join(eval_dir, f"{stem}_inception_features.npy")
        fig_path = os.path.join(eval_dir, f"{stem}_examples.png")
        yaml_path = os.path.join(eval_dir, f"{stem}_fid.yaml")

        # All four target outputs already exist -- nothing to do for this checkpoint at
        # all, not even loading its weights.
        if all(os.path.exists(p) for p in (sampled_images_path, feat_path, fig_path, yaml_path)):
            with open(yaml_path) as f:
                info = yaml.safe_load(f)
            summary[stem] = {"step": info["step"], "fid": info["fid"]}
            print(f"All outputs already exist for {stem} -- skipping (fid={info['fid']:.3f})")
            continue

        # Loaded regardless (cheap -- no model forward) for its "step" metadata; weights
        # are only actually applied to unet below if generation turns out to be needed.
        diff_ckpt = torch.load(ckpt_path, map_location=device)
        used_ema = args.use_ema and "ema_state_dict" in diff_ckpt

        # --- generate samples for this checkpoint, checkpointing to disk periodically ---
        # (skipped entirely if sampled_images.pt already has >= N_FID_SAMPLES cached)
        generated_batches = []
        n_generated = 0
        if os.path.exists(sampled_images_path):
            cached = torch.load(sampled_images_path, map_location="cpu")
            generated_batches = [cached]
            n_generated = cached.shape[0]
            print(f"Loaded {n_generated} cached generated images from {sampled_images_path}")

        if n_generated >= args.n_fid_samples:
            generated_images = torch.cat(generated_batches, dim=0)[:args.n_fid_samples]
            print(f"Cache already has >= N_FID_SAMPLES={args.n_fid_samples} images -- nothing to generate")
        else:
            unet.load_state_dict(diff_ckpt["model_state_dict"])
            if used_ema:
                ema_sd = diff_ckpt["ema_state_dict"]
                unet.load_state_dict(ema_shadow_to_model_state_dict(unet, ema_sd["shadow"]))
                print(f"Sampling from EMA weights (EMA step {ema_sd['step']}, decay {ema_sd['decay']})")
            elif args.use_ema:
                print("use_ema=True but checkpoint has no ema_state_dict "
                      "(older run, or ema_decay wasn't set during training) -- using raw weights")
            unet.eval()

            print(f"Generating {args.n_fid_samples - n_generated} more images "
                  f"({n_generated} cached, target {args.n_fid_samples})")

            checkpoint_every = max(1, round(args.n_fid_samples * 0.05))
            n_since_checkpoint = 0

            with torch.no_grad():
                pbar = tqdm(total=args.n_fid_samples, initial=n_generated, desc=f"Sampling {stem}")
                while n_generated < args.n_fid_samples:
                    b = min(args.fid_batch_size, args.n_fid_samples - n_generated)

                    if unconditional:
                        z_batch = torch.zeros(b, diff_cfg.latent_dim, device=device)
                    else:
                        batch = next(z_source)
                        src_imgs, src_latents = batch if isinstance(batch, (list, tuple)) else (batch, None)
                        src_imgs = src_imgs[:b].to(device)
                        z_batch = vae.encode(src_imgs)[0] if vae is not None else src_latents[:b].to(device)

                    if np_type == "ddpm":
                        fake = diffusion_sample(
                            z_batch, unet, schedule, diff_cfg, device,
                            sampler=args.sampler, eta=args.eta, num_inference_steps=args.num_inference_steps,
                            guidance_scale=args.guidance_scale,
                        )
                    elif np_type == "flow":
                        fake = flow_sample(
                            z_batch, unet, diff_cfg, device,
                            num_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
                        )
                    elif np_type == "vpsde":
                        fake = vpsde_sample(
                            z_batch, unet, diff_cfg, device,
                            num_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
                        )

                    generated_batches.append(fake.cpu())
                    n_generated += fake.shape[0]
                    n_since_checkpoint += fake.shape[0]
                    pbar.update(fake.shape[0])

                    if n_since_checkpoint >= checkpoint_every:
                        torch.save(torch.cat(generated_batches, dim=0), sampled_images_path)
                        pbar.write(f"Checkpointed {n_generated}/{args.n_fid_samples} images to {sampled_images_path}")
                        n_since_checkpoint = 0
                pbar.close()

            generated_images = torch.cat(generated_batches, dim=0)[:args.n_fid_samples]
            torch.save(generated_images, sampled_images_path)
            print(f"Generated {generated_images.shape[0]} total images, saved to {sampled_images_path}")

        # --- example figure (skipped if already saved) ---
        if os.path.exists(fig_path):
            print(f"Example grid already exists at {fig_path} -- skipping")
        else:
            n_show = min(args.n_show, generated_images.shape[0])
            grid = make_grid(generated_images[:n_show].clamp(0, 1), nrow=n_show)
            grid_np = grid.permute(1, 2, 0).numpy()
            plt.figure(figsize=(n_show * 2, 2.2))
            plt.imshow(grid_np.squeeze(-1) if grid_np.shape[-1] == 1 else grid_np,
                       cmap="gray" if grid_np.shape[-1] == 1 else None)
            plt.axis("off")
            plt.title(f"{stem} -- example sampled images")
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()
            print(f"Saved example grid to {fig_path}")

        # --- inception features (loaded from cache if already computed) ---
        if os.path.exists(feat_path):
            feat_fake = np.load(feat_path)
            print(f"Loaded cached inception features from {feat_path}")
        else:
            feat_fake = get_inception_features(generated_images, batch_size=args.fid_batch_size)
            np.save(feat_path, feat_fake)
            print(f"Saved inception features to {feat_path}")

        # --- FID + per-checkpoint yaml (skipped if already recorded) ---
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                info = yaml.safe_load(f)
            print(f"FID already recorded in {yaml_path} -- reusing fid={info['fid']:.3f}")
        else:
            fid_score = frechet_distance(feat_real, feat_fake)
            print(f"FID ({stem}): {fid_score:.3f}  |  real: {feat_real.shape[0]}  generated: {feat_fake.shape[0]}")

            info = {
                "checkpoint": ckpt_path,
                "step": int(diff_ckpt.get("step", -1)),
                "used_ema": used_ema,
                "fid": fid_score,
                "n_real_images": int(feat_real.shape[0]),
                "n_generated_images": int(feat_fake.shape[0]),
                "n_fid_samples": args.n_fid_samples,
                "fid_batch_size": args.fid_batch_size,
                "guidance_scale": args.guidance_scale,
                "sampler": args.sampler,
                "eta": args.eta,
                "num_inference_steps": args.num_inference_steps,
                "noise_process": np_type,
                "seed": args.seed,
                "experiment_name": getattr(diff_cfg, "experiment_name", None),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            with open(yaml_path, "w") as f:
                yaml.safe_dump(info, f, sort_keys=False)
            print(f"Saved {yaml_path}")

        summary[stem] = {"step": info["step"], "fid": info["fid"]}

    summary_path = os.path.join(eval_dir, "fid_summary.yaml")
    with open(summary_path, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    print(f"\nSaved FID-vs-checkpoint summary to {summary_path}")


if __name__ == "__main__":
    main()
