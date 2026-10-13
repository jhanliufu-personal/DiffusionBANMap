# Alternative conditioning routes for diffusion models

Notes from a literature search on 2026-10-08.

**Context.** This repo trains conditional diffusion models from scratch: a U-Net takes an encoding (AlexNet fc6, DINO CLS token, etc.) and applies it to each ResBlock's feature map via AdaGN. Two other routes exist:

1. Fine-tune a pretrained diffusion model on (image, new-embedding) pairs.
2. Training-free guidance: an off-the-shelf encoder supplies gradients to an unconditional model at sampling time.

**Provenance.** The 2025–2026 papers were found by web search and only their abstracts were read. The canonical papers and their arXiv IDs are from memory. Items marked *(inference)* are reasoning, not published results.

---

## Route 1: fine-tune a pretrained model on a new embedding

### Canonical

- **IP-Adapter** (Ye et al., 2023) — [arXiv 2308.06721](https://arxiv.org/abs/2308.06721), [project page](https://ip-adapter.github.io/)
  - Freezes the SD U-Net and adds a second, parallel set of cross-attention layers for the new embedding (about 22M trainable parameters).
  - The template most "swap in a different embedding" work copies.
- Background:
  - unCLIP / DALL-E 2 and Stable unCLIP: the base model is conditioned on image embeddings directly.
  - ControlNet ([arXiv 2302.05543](https://arxiv.org/abs/2302.05543)): spatial conditions, zero-initialized side branch.

### Recent (2025–2026)

- **OminiControl** (ICCV 2025) — [arXiv 2411.15098](https://arxiv.org/abs/2411.15098); sibling **EasyControl** (ICCV 2025) — [arXiv 2503.07027](https://arxiv.org/abs/2503.07027)
  - The shift from U-Net to DiT/flow models such as FLUX.
  - No separate adapter branch: the condition becomes extra tokens in the transformer's joint attention, and a LoRA is trained (about 0.1% extra parameters).
  - Built for image conditions; a vector embedding would need a small projector to tokens.
- **NeuroAdapter** (Feng, Adeli, Guo, Cheng, Hwang, Kriegeskorte; ICLR 2026) — [arXiv 2509.23566](https://arxiv.org/abs/2509.23566)
  - Closest to this project's setting.
  - Conditions a latent diffusion model directly on brain representations via cross-attention, skipping the intermediate CLIP-space mapping.
  - Uses cross-attention maps across denoising steps for interpretability (IBBI framework).
- **Brain-IT** (Oct 2025) — [arXiv 2510.25976](https://huggingface.co/papers/2510.25976)
  - Current state of the art in the MindEye2 lineage.
  - SDXL adapted to 256 OpenCLIP ViT-bigG/14 tokens; matches 40-hour results with 1 hour of data from a new subject.
- Related: **MindEye2** — [arXiv 2403.11207](https://arxiv.org/abs/2403.11207); **NeuroFlow** (2026, unified encoding/decoding via cross-modal flow matching) — [arXiv 2604.09817](https://arxiv.org/abs/2604.09817)

---

## Route 2: training-free guidance on an unconditional model

### Why CLIP does not work off the shelf

CLIP was trained on clean images, so its gradients on noisy intermediates are close to useless. Known fixes:

- Train a noise-aware CLIP (GLIDE).
- Average gradients over random cutouts/augmentations.
- Compute the loss on the predicted clean image instead of the noisy sample (the modern training-free line).

### Canonical

- **Classifier guidance** (Dhariwal & Nichol, 2021) — [arXiv 2105.05233](https://arxiv.org/abs/2105.05233). The origin, but needs a classifier trained on noisy images.
- **Universal Guidance** (Bansal et al., CVPR 2023) — [arXiv 2302.07121](https://arxiv.org/abs/2302.07121)
  - The canonical off-the-shelf version.
  - Applies any clean-image network (CLIP, segmenter, face recognition) to the predicted clean image.
  - Adds backward guidance and self-recurrence steps for stability.
- Close variants: FreeDoM (ICCV 2023); MPGD — [arXiv 2311.16424](https://arxiv.org/abs/2311.16424)

### Recent (2025–2026)

- **TFG: Unified Training-Free Guidance** (Ye et al., NeurIPS 2024) — arXiv 2409.15761 (ID from memory; verify)
  - Slightly before the window but the reference point; read first.
  - Shows Universal Guidance, FreeDoM, MPGD, DPS and LGD are points in one hyperparameter space, and benchmarks them.
- **FK steering** (Singhal et al., ICML 2025) — [arXiv 2501.06848](https://arxiv.org/abs/2501.06848)
  - Replaces gradients with particle resampling (sequential Monte Carlo): run several trajectories, resample by reward at intermediate steps.
  - Works with non-differentiable rewards; avoids off-manifold artifacts of gradient guidance.
  - **UniCoDe** (Dec 2025) combines it with gradients — [arXiv 2512.12339](https://arxiv.org/abs/2512.12339)
- **On the Guidance of Flow Matching** (Feng et al., ICML 2025) — [arXiv 2502.02150](https://arxiv.org/abs/2502.02150)
  - First general guidance framework for flow-matching models.
  - Recovers the gradient methods as approximations and adds an asymptotically exact training-free variant.
- **Not All Prediction Targets Keep Training-Free Diffusion Guidance on the Manifold** (Lee & Lee, Jul 2026) — [arXiv 2607.00647](https://arxiv.org/abs/2607.00647)
  - Argues x-prediction models tolerate guidance much better than ε- or v-prediction, because guidance relies on the clean-image estimate.

---

## Q1: Can fine-tuning hurt quality, and can the two conditioning pathways fight?

Yes on both counts, and LoRA is not an exemption.

### Generation quality

- **Forgetting.** Full fine-tuning on a small paired set erodes the pretrained prior (drift, loss of diversity, memorization). DreamBooth's prior-preservation loss exists for this.
- **LoRA is not immune.** The low-rank update still changes the effective weights. It limits drift but does not prevent overfitting a small set.
- **Frozen backbone.** Only adapters with a frozen base (IP-Adapter, ControlNet) leave the original model recoverable, by setting the adapter scale to zero.

### Pathway interference

- **Additive competition.** IP-Adapter sums the two cross-attention outputs (text + λ·new), so they compete for the same residual stream. A strong image prompt overrides the text; λ is a trade-off knob.
- **Evidence from MindEye2.** Text conditioning was dropped entirely when fine-tuning SDXL unCLIP, because any text conditioning worsened fidelity to the target embedding.
- **The new pathway can be ignored.** If the old condition already explains the image, a zero-initialized new pathway has little gradient pressure to learn. Standard fix: drop or empty the old condition during training (ControlNet drops prompts 50% of the time).
- **Guidance needs dropout.** To run classifier-free guidance on the new condition Y, Y must also be randomly dropped in training.

### Recommendation for this project

Replace the CLIP pathway instead of adding to it, since text is not needed at inference: train with an empty prompt, or feed projected AlexNet tokens into the existing cross-attention slots.

### Hallucination risk

- A strong prior produces realistic images even when the conditioning carries little information, so sample quality stops indicating how well the pathway works.
- **Spurious reconstruction from brain activity** (Shirakawa et al., Kamitani lab) — [arXiv 2405.10078](https://arxiv.org/abs/2405.10078): much of the apparent reconstruction quality on NSD was classification into training categories plus diffusion hallucination; it collapsed on a dataset without category overlap.
- The class-agreement analysis in this repo is the right kind of check, but should be run on held-out categories.

---

## Q2: Does the choice of encoder matter for training-free guidance?

Yes. The governing property has been formalized as smoothness/robustness of the guidance network. No head-to-head AlexNet-vs-DINO comparison was found.

### What has been formalized

- **Smoothness.** *Understanding and Improving Training-free Loss-based Diffusion Guidance* (Shen et al., NeurIPS 2024) — [arXiv 2403.12404](https://arxiv.org/abs/2403.12404)
  - Proves training-free guidance is more susceptible to adversarial gradients and converges more slowly than noise-trained classifier guidance.
  - Attributes both to the guidance network being less smooth.
  - Fix: average gradients over random augmentations (translation, resize, color, cutout). This is the theory behind the CLIP cutout trick.
- **Robustness.** Santurkar et al. 2019; Kawar, Ganz & Elad 2022 (from memory): adversarially robust classifiers have "perceptually aligned gradients" and give much better guidance than standard ones.
- **Clean-image estimate.** The x-prediction paper above: guidance is only as good as the predicted clean image the encoder sees.

Together: a guidance encoder is good when its input gradients are semantically meaningful and not adversarial, and when it behaves sensibly on the blurry early-step predictions.

### AlexNet vs DINO *(inference)*

- **AlexNet fc6** should be the harder case:
  - standard-trained with light augmentation, so non-robust with high-frequency gradients;
  - ReLU and max-pooling make gradients sparse;
  - has never seen blurry inputs.
- **DINO** should be better behaved:
  - trained with heavy augmentation (multi-crop, blur, color jitter);
  - CLS token is a smooth global summary;
  - ViT gradients can show patch-grid artifacts.
- **Both** need 64px samples upsampled to 224, which costs some gradient quality.
- **PCA helps.** Matching in the top-PC subspace drops low-variance directions, where adversarial solutions tend to live.

### Cheap diagnostic

Before any diffusion work, invert each encoder by plain gradient descent on pixels toward a target embedding, with and without augmentation averaging. If the result is noise-like, the gradients will not steer a diffusion model well either.

---

## Implications for this repo

- **Route 2** can be tried on the existing ImageNet64 unconditional checkpoint, with AlexNet fc6 or DINO as the guidance network and embedding distance as the loss. Guidance quality is bounded by the unconditional prior, and that checkpoint currently gives incoherent samples.
- **Route 1** means moving to a latent SD/FLUX base, which is the MindEye2 / NeuroAdapter path.
