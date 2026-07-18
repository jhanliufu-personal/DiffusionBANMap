# Pretrained diffusion models conditioned on CLIP / AlexNet embeddings

Survey of Hugging Face-hosted pretrained checkpoints conditioned on image embeddings, for
comparison against this project's neuron-firing-rate → embedding → diffusion pipeline
(see `analysis/embedding_neuron_alignment.ipynb`).

None of the CLIP/AlexNet-conditioned checkpoints found use flow matching — they are all
classic epsilon-prediction diffusion (DDPM/DDIM).

## CLIP image-embedding-conditioned diffusion models

### 1. `stabilityai/stable-diffusion-2-1-unclip` / `-unclip-small`
[docs](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_unclip)

- Architecture: SD2.1 UNet finetuned to condition on CLIP *image* embeddings (unCLIP /
  DALL-E 2 style), classic diffusion, not flow matching.
- Two variants: `-small` uses OpenAI CLIP ViT-L/14 (768-dim, matches Karlo's prior), the
  full one uses OpenCLIP ViT-H/14 (1024-dim).
- Embedding processing: the raw CLIP embedding is passed through a learned
  `StableUnCLIPImageNormalizer` (an affine mean/std normalization — essentially a learned
  scaler), then optionally noised according to a `noise_level` param (noise-augmentation
  trick from the unCLIP paper — 0 by default), then un-normalized, then fed into the UNet
  added to the timestep embedding. **No PCA.**

### 2. Kandinsky 2.1 / 2.2
[docs](https://huggingface.co/docs/diffusers/en/api/pipelines/kandinsky_v22),
[2.2-prior](https://huggingface.co/kandinsky-community/kandinsky-2-2-prior)

- Also unCLIP-style: a diffusion *prior* maps text → CLIP-image-embedding space, then a
  diffusion decoder (MoVQ-based) generates the image conditioned on that CLIP image
  embedding.
- 2.1 uses CLIP ViT-L; 2.2 upgrades to CLIP-ViT-bigG. Standard classic diffusion (DDPM),
  not flow matching. Embeddings used directly (no PCA reported).

### 3. `lambdalabs/sd-image-variations-diffusers`
[model card](https://huggingface.co/lambdalabs/sd-image-variations-diffusers)

- SD1.4 UNet finetuned to take a CLIP ViT-L/14 image embedding (with projection, 768-dim)
  in place of the text embedding slot, via cross-attention. Classic diffusion. No
  PCA/scaling beyond CLIP's own normalization. Note: expects images resized *without*
  anti-aliasing (a training quirk called out in the model card).

### 4. MindEye2 — closest structural analog to this project's pipeline
[paper](https://arxiv.org/html/2403.11207v1),
[weights](https://huggingface.co/datasets/pscotti/mindeyev2/tree/main/train_logs)

- Pipeline: fMRI voxels → **ridge regression** → shared latent space → MLP + diffusion
  prior → OpenCLIP ViT-bigG/14 embedding → fed into **SDXL unCLIP** to reconstruct the
  image.
- Essentially the same shape as this project's neuron-firing-rate → Ridge/RidgeCV decoder
  → embedding → diffusion pipeline, just with fMRI voxels instead of single-unit firing
  rates. Worth revisiting as prior art — and since it already outputs OpenCLIP-bigG
  embeddings, its released SDXL-unCLIP decoder is a candidate for reuse instead of
  training a UNet from scratch, if this project's embedding space can be aligned to
  CLIP-bigG.

## DINO / DINOv2-conditioned diffusion models

Unlike CLIP, there is **no popular, widely-adopted pretrained image-generation checkpoint
on HF conditioned on DINO/DINOv2 embeddings**. The ecosystem here is much thinner:

- **IP-Adapter** (`h94/IP-Adapter`) — the most widely used "image prompt adapter" family
  for SD/SDXL — uses **CLIP** image encoders (ViT-H / ViT-bigG), not DINOv2. No official
  DINOv2 variant exists in the repo.
  [model repo](https://huggingface.co/h94/IP-Adapter)
- **Kandinsky 3 / 3.1** — checked specifically; uses Flan-UL2 for text and (in 3.1) adds
  IP-Adapter/ControlNet for image conditioning, but those still run on CLIP, not DINOv2.
  [docs](https://huggingface.co/docs/diffusers/api/pipelines/kandinsky3)
- **"Conditional Diffusion on Web-Scale Image Pairs leads to Diverse Image Variations"**
  (arXiv 2405.14857, aka *Semantica*) — a research paper that finds DINOv2 is empirically a
  *better* frozen image encoder than CLIP for image-variation diffusion. Directly relevant
  to the "how should embeddings be processed" question, but **no public HF weights found**
  — appears to be research-only, no released checkpoint.
  [paper](https://arxiv.org/abs/2405.14857)
- Scattered research work (e.g. "Generating metamers of human scene understanding",
  arXiv 2601.11675) integrates DINOv2 patch embeddings into SD's cross-attention via a
  Perceiver-style resampler that compresses 1024 DINOv2 tokens down to 32 conditioning
  tokens — architecturally like IP-Adapter-Plus but for DINOv2. Again, paper-level
  methodology, not a released general-purpose checkpoint.
- The only **flow-matching + DINOv2** combination found on HF is out of domain: robotics
  diffusion-policy checkpoints (`Ngseo/rh56f1_diffusion_dinov2s_flowmatch_multicam_dr`,
  `Ngseo/dg5f_diffusion_dinov2s_flowmatch_multicam_dr`) that predict robot actions from
  multi-camera DINOv2 features, not image synthesis — not usable for this project's image
  generation pipeline.

**Takeaway:** if this project were to switch from CLIP to DINO/DINOv2 as the embedding
space, there is no existing pretrained decoder to piggyback on (unlike CLIP, where
SD2.1-unclip / Kandinsky / lambdalabs give a menu of options). That conditioning would need
to be trained in-house, same situation as the AlexNet fc6 case below.

## Gaps found

- **No flow-matching model conditioned on image CLIP or DINO embeddings** turned up for
  image generation. SD3/3.5 and FLUX use flow matching but condition on *text* embeddings
  (CLIP+T5), not image embeddings — unCLIP-style image conditioning hasn't been ported to a
  flow-matching image-generation backbone on HF as far as this search found (the one
  flow-matching + DINOv2 pairing that exists is a robotics action-prediction model, not
  image synthesis).
- **No pretrained diffusion weights conditioned on AlexNet fc6 embeddings** exist on HF —
  that's a neuroscience-specific encoding-model feature space (Güçlü & van Gerven-style),
  not something anyone has released generative weights for. This conditioning would need
  to be trained in-house regardless.
- **No pretrained diffusion weights conditioned on DINO/DINOv2 embeddings** exist as a
  general-purpose released HF checkpoint either — see above. Same in-house-training
  conclusion as AlexNet fc6.

## Sources

- [Stable unCLIP docs](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_unclip)
- [stabilityai/stable-diffusion-2-1-unclip](https://hf.co/stabilityai/stable-diffusion-2-1-unclip)
- [Kandinsky 2.2 docs](https://huggingface.co/docs/diffusers/en/api/pipelines/kandinsky_v22)
- [kandinsky-community/kandinsky-2-2-prior](https://huggingface.co/kandinsky-community/kandinsky-2-2-prior)
- [lambdalabs/sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers)
- [MindEye2 paper](https://arxiv.org/html/2403.11207v1)
- [MindEye2 pretrained weights](https://huggingface.co/datasets/pscotti/mindeyev2/tree/main/train_logs)
- [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter)
- [Kandinsky 3 docs](https://huggingface.co/docs/diffusers/api/pipelines/kandinsky3)
- [Semantica / "Conditional Diffusion on Web-Scale Image Pairs..." (arXiv 2405.14857)](https://arxiv.org/abs/2405.14857)
- [DINOv2 docs](https://huggingface.co/docs/transformers/model_doc/dinov2)

_Search performed 2026-07-14. DINO section appended 2026-07-14._
