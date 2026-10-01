# DiffusionBANMap

Generating natural images from IT neuron activations with diffusion models. Images are encoded into a low-dimensional latent (top PCs of AlexNet fc6 / CLIP / DINO embeddings, or a beta-VAE latent), a pixel-space diffusion model is trained to generate images conditioned on that latent, and neuron firing rates are mapped linearly into the same latent space.

---

## Installation

Requires Python >= 3.11.

```bash
# with uv (uses pyproject.toml)
uv sync
source .venv/bin/activate

# or with pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For a specific CUDA build, install `torch` / `torchvision` from the matching PyTorch wheel index first. Training logs to Weights & Biases and needs `WANDB_API_KEY` set in the environment.

All scripts are run as modules from the repo root (`python -m scripts.<name>`), so that `models/`, `data_utils.py` and `utils.py` resolve.

---

## Directory structure

```
DiffusionBANMap/
├── config/            # YAML configs, one per experiment (diffusion_*.yaml, betavae_*.yaml)
├── models/            # unet.py, noise_process.py (DDPM / flow / VP-SDE), ema.py, beta_vae.py
├── scripts/           # entry points, see "Common commands"
├── notebooks/         # Colab versions of the scripts + inspection notebooks
├── analysis/          # axis-tuning and firing-rate -> latent analyses
├── data_utils.py      # dataloaders and conditioning-latent lookup
├── utils.py           # samplers, schedules, run naming
├── data/              # not tracked
└── outputs/           # not tracked
```

`data/` and `outputs/` are git-ignored. The code assumes this layout inside them:

```
data/
├── imagenet64/                              # data_dir for dataset_type: imagenet64
│   ├── train_images.npy                     #   (1,281,167, 3, 64, 64) uint8
│   └── val_images.npy                       #   (50,000, 3, 64, 64) uint8
├── imagenet64_{encoder}_pca_latents/        # conditioning latents, next to data_dir
│   ├── train_embeddings.npy                 #   full un-PCA'd embeddings (encode once)
│   ├── val_embeddings.npy
│   ├── train_latents_z{N}.npy               #   top-N PCs, row-aligned with the images
│   ├── val_latents_z{N}.npy
│   └── {encoder}_scaler_z{N}.joblib, {encoder}_pca_z{N}.joblib, manifest_z{N}.json
├── 15901Stimuli/                            # data_dir for dataset_type: stimuli (train images)
├── 500Stimuli/                              # held-out stimuli (val images)
├── stimuli_{encoder}_pca_latents/           # same contents as the imagenet64 one
└── stimuli_{encoder}_latents/               # raw (no PCA) embeddings, e.g. sd_vae, clip_vit_l14

outputs/
└── {output_dir}_{run tag}/                  # e.g. ..._T1000_flow_alexnet_fc6_z200
    ├── config.json
    ├── checkpoints/                         # best_ckpt.pt, ckpt_step_0025000.pt, ...
    ├── visualizations/
    ├── fid_eval/                            # written by calculate_fid_sweep
    └── inspection/                          # written by inspect_diffusion
```

The run tag is built from the config (image size, channels, noise process, latent dim), and includes the encoder name for runs conditioned on precomputed latents.

### How a config picks its conditioning

| Config keys | Conditioning |
|---|---|
| `unconditional: true` | none (z is a zero vector) |
| `encoding_model: <encoder>` + `latent_dim: N` | precomputed latents from `<data_dir>/../{dataset_type}_{encoder}_pca_latents/{train,val}_latents_z{N}.npy`; an error if the files are missing |
| `betavae_config_path` + `betavae_ckpt_path` | beta-VAE latents, encoded on the fly |

Available encoders: `alexnet_fc6`, `clip_vit_b32`, `clip_vit_l14`, `dino_vitb16`, `dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14`, `sd_vae` (no PCA; used for the Stable Diffusion experiments).

---

## Common commands

### 1. Prepare ImageNet64

Download the three Downsampled ImageNet 64x64 zips from image-net.org (login required), then consolidate them into `train_images.npy` / `val_images.npy`:

```bash
python scripts/prepare_imagenet64_dataset.py \
    --drive_train_part1_zip path/to/Imagenet64_train_part1.zip \
    --drive_train_part2_zip path/to/Imagenet64_train_part2.zip \
    --drive_val_zip         path/to/Imagenet64_val.zip \
    --data_dir data/imagenet64
```

### 2. Encode images into conditioning latents

```bash
# ImageNet64, DINOv2 ViT-B/14, top 200 PCs
python -m scripts.extract_image_embeddings \
    --model dinov2_vitb14 \
    --dataset imagenet64 --data-dir data/imagenet64 \
    --n-components 200 --subsample-size 50000

# Stimuli, AlexNet fc6, top 50 PCs (PCA fit on the train folder)
python -m scripts.extract_image_embeddings \
    --model alexnet_fc6 \
    --train-data-dir data/15901Stimuli --data-dir data/500Stimuli \
    --n-components 50
```

Output goes to `data/{dataset}_{model}_pca_latents/`. The full embeddings are saved there too, so running again with a different `--n-components` only re-fits PCA and does not re-encode the images. Without `--n-components` the raw embeddings are saved to `data/{dataset}_{model}_latents/{split}_latents.npy`.

### 3. Train a diffusion model

```bash
export WANDB_API_KEY=...

# conditional on precomputed latents (encoding_model + latent_dim in the config)
python -m scripts.diffusion_train --config config/diffusion_full_imagenet64_flow.yaml

# unconditional
python -m scripts.diffusion_train --config config/diffusion_full_imagenet64_flow_uncond.yaml

# resume
python -m scripts.diffusion_train --config config/diffusion_full_imagenet64_flow.yaml \
    --resume outputs/<run>/checkpoints/ckpt_step_0200000.pt
```

All visible GPUs are used automatically (DDP, no `torchrun` needed). `--run_name` and `--notes` annotate the wandb run. Finetuning on the stimuli from an ImageNet64 checkpoint is configured through `resume_ckpt_path` + `reset_step_on_resume: true`, see `config/diffusion_full_stimuli_finetune_flow.yaml`.

### 4. FID over a checkpoint sweep

```bash
python -m scripts.calculate_fid_sweep --config config/diffusion_full_imagenet64_flow.yaml \
    --n_fid_samples 50000 --num_inference_steps 100 --guidance_scale 1.0
```

Evaluates every `checkpoints/ckpt_step_*.pt` of the run (EMA weights by default, `--no_ema` to disable) and writes per-checkpoint samples, features and scores plus `fid_summary.yaml` into `<run>/fid_eval/`. `notebooks/fid_vs_step.ipynb` plots the result.

### 5. Inspect a checkpoint

```bash
# unconditional (default)
python -m scripts.inspect_diffusion \
    --diffusion_config_path config/diffusion_full_imagenet64_flow_uncond.yaml

# conditional: compare samples against the val images their latents came from
python -m scripts.inspect_diffusion \
    --diffusion_config_path config/diffusion_full_imagenet64_flow.yaml \
    --no-unconditional --guidance_scale 2.0
```

Loads `best_ckpt.pt` unless `--diffusion_ckpt_path` is given, and saves one-step predictions, final samples and a denoising progression into `<run>/inspection/`.

### 6. Stable Diffusion with CLIP-image conditioning

Uses a pretrained SD 1.x image-variations checkpoint instead of this repo's UNet: SD-VAE latents are partially re-noised and denoised conditioned on CLIP ViT-L/14 image embeddings.

```bash
# encode the stimuli: SD-VAE latents [N, 4, 64, 64] and CLIP ViT-L/14 embeddings [N, 768]
python -m scripts.extract_image_embeddings --model sd_vae       --data-dir data/500Stimuli
python -m scripts.extract_image_embeddings --model clip_vit_l14 --data-dir data/500Stimuli

# reference run: real latents + real CLIP embeddings
python -m scripts.sd_clip_img2img \
    --latents_path data/stimuli_sd_vae_latents/val_latents.npy \
    --cond_path data/stimuli_clip_vit_l14_latents/val_latents.npy \
    --strength 0.6 --output_dir outputs/sd_clip_img2img

# control: pure noise start, no conditioning
python -m scripts.sd_clip_img2img --pure_noise_start --null_cond --n 8 \
    --strength 1.0 --guidance_scale 1.0 --output_dir outputs/sd_clip_img2img_debug

# score every degraded setup against the reference run
python -m scripts.sd_conditioning_eval --output_dir outputs/sd_clip_img2img \
    --metrics mse,psnr,ssim,clip_cosine
```

Latents or CLIP embeddings regressed from firing rates (`analysis/firing_rate_to_latent.ipynb`) are passed through the same `--latents_path` / `--cond_path` flags. `sd_clip_img2img` always writes `denoised_images.pt`; for the eval, rename each non-reference run's file to `{tag}_denoised_images.pt` in the same directory (known tags are listed in `TAG_LABELS` in `scripts/sd_conditioning_eval.py`).

### 7. Beta-VAE (legacy)

```bash
python -m scripts.betavae_train --config config/betavae_full_stimuli.yaml
```

---

## Background

The project started from the observation that beta-VAE latent units and IT neurons are both **axis-tuned** in the feature space of pretrained networks (AlexNet fc6, VGG, DINO-ViT): each unit's response is well predicted by projecting the image embedding onto a single preferred axis, and the beta-VAE's strengthened KL penalty ($\beta > 1$) pushes those axes towards orthogonality. That shared coordinate system suggests the pipeline

$$\text{neuron firing rates} \xrightarrow{\text{linear map}} z \xrightarrow{p(x \mid z),\ \text{diffusion}} \text{natural image}$$

A beta-VAE decoder alone gives blurry images because a low-dimensional $z$ cannot carry high-frequency detail; a diffusion model conditioned on $z$ instead samples sharp images consistent with it. A **superstimulus** for neuron $i$ is then generated by increasing $z_i$ while holding the other dimensions fixed, and checked by re-encoding the output and confirming that only the targeted dimension moved. In practice the beta-VAE latent has largely been replaced by the top PCs of pretrained embeddings (AlexNet fc6, CLIP, DINO), which neurons are linearly mapped into in the same way.

Alternatives considered alongside this trained, latent-conditioned model: training-free **classifier guidance**, which backpropagates the neuron model $\hat{y}_i = \mathbf{a}_i \cdot f(x)$ through a pretrained diffusion model at each denoising step while penalizing changes in other neurons; **score distillation** with the same neuron objective as the driving term; and a lightweight **adapter** that maps neuron vectors into the conditioning space of a frozen pretrained diffusion model, which is the direction the Stable Diffusion + CLIP experiments above explore. The training-free methods cost nothing to set up but give only approximate selectivity; the conditioned model gives direct control by editing $z$.
