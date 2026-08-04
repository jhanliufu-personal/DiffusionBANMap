"""
SDEdit-style inference with a pretrained SD 1.x checkpoint: partially re-noise a
batch of SD-VAE latents, then denoise them conditioned on CLIP ViT-L/14 image
embeddings instead of a text prompt.

This intentionally does NOT reuse models/unet.py or utils.py's
diffusion_sample/flow_sample/vpsde_sample -- those are built around this repo's
own trained UNet and noise_process classes (DDPM/RectifiedFlow/VPSDE, latent_dim
conditioning). A pretrained SD 1.x checkpoint is a different UNet architecture
(diffusers' UNet2DConditionModel, cross-attention conditioning) with its own
scheduler, so it's loaded and driven directly via `diffusers` here. `discover_device`
is reused from utils.py.

Expects:
  - --latents_path: [N, 4, 8, 8] float array/tensor, SD-VAE-encoded and already
    scaled by vae.config.scaling_factor -- e.g. the sd_vae encoder in
    notebooks/extract_image_embeddings.ipynb, or the regressed/unflattened
    y_pred_latent from analysis/aixs_tuning_analysis.ipynb.
  - --cond_path: [N, 768] float array/tensor, CLIP ViT-L/14 image embeddings --
    the clip_vit_l14 encoder in notebooks/extract_image_embeddings.ipynb's
    MODEL_REGISTRY. 768 is what the default --sd_model_id's cross-attention
    expects; a different encoder's output width will fail inside the UNet
    forward call with a shape mismatch.

The default --sd_model_id, lambdalabs/sd-image-variations-diffusers, matters: it's
a SD 1.x UNet specifically fine-tuned to take a single CLIP image embedding as
cross-attention context (encoder_hidden_states), unlike stock
runwayml/stable-diffusion-v1-5 which only ever saw CLIP *text* embeddings.

Caveat: SD 1.x's UNet was trained exclusively on 64x64 latents (512x512 images).
Running it on 8x8 latents (64x64 stimuli) is far outside its training
distribution -- treat this as an experiment, not an expected-to-work pipeline.

Usage (run from repo root, as `python -m scripts.<name>` per this repo's
convention):
    python -m scripts.sd_clip_img2img \
        --latents_path path/to/latents.npy \
        --cond_path path/to/clip_vit_l14_embeddings.npy \
        --output_dir outputs/sd_clip_img2img
"""

import os
import argparse

import numpy as np
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from utils import discover_device


def _load_array(path: str) -> torch.Tensor:
    if path.endswith(".npy"):
        return torch.from_numpy(np.load(path)).float()
    loaded = torch.load(path, map_location="cpu")
    return loaded.float() if torch.is_tensor(loaded) else torch.from_numpy(np.asarray(loaded)).float()


@torch.no_grad()
def denoise_batch(unet, scheduler, latents, cond, num_inference_steps, strength, guidance_scale, device,
                   pure_noise_start=False, null_cond=False):
    """SDEdit-style partial noise + denoise: re-noise `latents` back to the
    timestep set by `strength` (0 = return input unchanged, 1 = noise all the
    way to pure noise, i.e. plain text/image-to-image generation), then run the
    reverse process conditioned on `cond`. Mirrors diffusers'
    StableDiffusionImg2ImgPipeline loop, with `cond` (precomputed CLIP image
    embeddings) standing in for the usual text-prompt embeddings -- matches
    lambdalabs/sd-image-variations-diffusers' convention of a single-token
    cross-attention context.

    pure_noise_start=True is a debug control: even strength=1.0's add_noise call
    leaves a tiny residual of the input latent mixed in (alpha_cumprod at the top
    scheduler timestep isn't exactly 0), so it isn't bit-for-bit equivalent to a
    from-scratch generation. This bypasses that and starts from actual Gaussian
    noise, fully decoupled from `latents`, conditioned only on `cond`.

    null_cond=True is a debug control to isolate whether `cond` itself is the
    problem: overrides cond with the same zero embedding the model was trained
    to interpret as "no conditioning" (the CFG unconditional branch below),
    ignoring --cond_path entirely. If generation still looks bad with
    pure_noise_start + null_cond both set, the bug isn't in the conditioning
    vectors."""
    latents = latents.to(device)
    cond = cond.to(device)
    if cond.dim() == 2:
        cond = cond.unsqueeze(1)  # [B, 768] -> [B, 1, 768] single-token cross-attn context
    if null_cond:
        cond = torch.zeros_like(cond)

    scheduler.set_timesteps(num_inference_steps, device=device)
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]

    noise = torch.randn_like(latents)
    latent_timestep = timesteps[:1].repeat(latents.shape[0])
    if pure_noise_start:
        latents = torch.randn_like(latents) * scheduler.init_noise_sigma
    else:
        latents = scheduler.add_noise(latents, noise, latent_timestep)

    do_cfg = guidance_scale != 1.0
    uncond = torch.zeros_like(cond) if do_cfg else None

    for t in timesteps:
        model_input = torch.cat([latents, latents]) if do_cfg else latents
        model_input = scheduler.scale_model_input(model_input, t)
        context = torch.cat([uncond, cond]) if do_cfg else cond
        noise_pred = unet(model_input, t, encoder_hidden_states=context).sample
        if do_cfg:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents_path", type=str, required=True)
    parser.add_argument("--cond_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/sd_clip_img2img")
    parser.add_argument("--sd_model_id", type=str, default="lambdalabs/sd-image-variations-diffusers",
                         help="SD 1.x checkpoint (unet + scheduler subfolders) accepting CLIP image-embedding conditioning")
    parser.add_argument("--vae_model_id", type=str, default="stabilityai/sd-vae-ft-mse",
                         help="Should match whatever VAE encoded --latents_path (see notebooks/extract_image_embeddings.ipynb)")
    parser.add_argument("--strength", type=float, default=0.6,
                         help="Fraction of the noise schedule to re-noise input latents to before denoising")
    parser.add_argument("--pure_noise_start", action="store_true",
                         help="Debug control: ignore --latents_path content, start from actual Gaussian "
                              "noise instead (see denoise_batch docstring). Requires --strength=1.0 "
                              "(auto-corrected if not set) so the full reverse schedule runs.")
    parser.add_argument("--null_cond", action="store_true",
                         help="Debug control: ignore --cond_path content, condition on an all-zero "
                              "embedding instead (see denoise_batch docstring). Use with "
                              "--guidance_scale=1.0 for a clean unconditional test.")
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_show", type=int, default=8, help="How many images to include in the saved example grid")
    args = parser.parse_args()

    if args.pure_noise_start and args.strength != 1.0:
        print(f"--pure_noise_start requires the full schedule -- forcing --strength=1.0 (was {args.strength})")
        args.strength = 1.0

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(discover_device())
    torch.manual_seed(args.seed)
    print(f"Device: {device}")

    latents = _load_array(args.latents_path)
    cond = _load_array(args.cond_path)
    assert latents.shape[0] == cond.shape[0], \
        f"latents ({latents.shape[0]}) and cond ({cond.shape[0]}) must have the same N"
    # assert latents.shape[1:] == (4, 8, 8), f"Expected [N, 4, 8, 8] latents, got {tuple(latents.shape)}"
    print(f"Loaded {latents.shape[0]} latents {tuple(latents.shape)} and conditioning {tuple(cond.shape)}")

    from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler

    print(f"Loading SD UNet + scheduler from {args.sd_model_id}")
    unet = UNet2DConditionModel.from_pretrained(args.sd_model_id, subfolder="unet").to(device).eval()
    for p in unet.parameters():
        p.requires_grad_(False)
    scheduler = DDIMScheduler.from_pretrained(args.sd_model_id, subfolder="scheduler")

    print(f"Loading SD VAE from {args.vae_model_id}")
    vae = AutoencoderKL.from_pretrained(args.vae_model_id).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    scaling_factor = vae.config.scaling_factor

    n = latents.shape[0]
    denoised_batches = []
    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        print(f"Denoising {start}:{end} / {n}")
        denoised_batches.append(denoise_batch(
            unet, scheduler,
            latents[start:end], cond[start:end],
            num_inference_steps=args.num_inference_steps,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            device=device,
            pure_noise_start=args.pure_noise_start,
            null_cond=args.null_cond,
        ))
    denoised_latents = torch.cat(denoised_batches, dim=0)

    latents_out_path = os.path.join(args.output_dir, "denoised_latents.pt")
    torch.save(denoised_latents, latents_out_path)
    print(f"Saved denoised latents {tuple(denoised_latents.shape)} to {latents_out_path}")

    with torch.no_grad():
        images = vae.decode(denoised_latents.to(device) / scaling_factor).sample
        images = ((images.clamp(-1, 1) + 1.0) / 2.0).cpu()
    images_out_path = os.path.join(args.output_dir, "denoised_images.pt")
    torch.save(images, images_out_path)
    print(f"Saved decoded images {tuple(images.shape)} to {images_out_path}")

    n_show = min(args.n_show, images.shape[0])
    grid = make_grid(images[:n_show].clamp(0, 1), nrow=n_show)
    grid_np = grid.permute(1, 2, 0).numpy()
    plt.figure(figsize=(n_show * 2, 2.2))
    plt.imshow(grid_np)
    plt.axis("off")
    plt.title(f"SD img2img (strength={args.strength}, guidance={args.guidance_scale}) -- example outputs")
    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, "examples.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved example grid to {fig_path}")


if __name__ == "__main__":
    main()
