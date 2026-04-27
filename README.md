# DiffusionBANMap

Generating natural images from IT neuron activations using diffusion models, leveraging the axis-tuning correspondence between beta-VAE latents and IT neurons.

---

## Beta-VAE Architecture

A convolutional VAE with a strengthened KL penalty (beta > 1).

**Encoder:** 4 stride-2 Conv layers → FC(256) → outputs μ and log σ² per latent dim  
**Decoder:** FC projection → 4 transposed Conv layers → reconstructed image  
**Latent:** z = μ + σ·ε via reparameterization; typically 10–50 dimensions  
**Loss:** `BCE_reconstruction + β × KL(q(z|x) || N(0,I))`

The β penalty (typically 4–10) forces latent dimensions to be statistically independent, pressuring each unit to align with a distinct, non-overlapping axis in representation space.

---

## Axis Tuning of Beta-VAE Latents

Each beta-VAE latent unit exhibits a **preferred axis** in the feature space of pretrained models (AlexNet fc6, VGG16/19, DINO-ViT): a linear direction that best predicts that unit's activation across images. This is quantified by:

- **Explained variance (EV):** how well the linear projection onto the preferred axis predicts the unit's response (leave-one-out cross-validated)
- **Selectivity index:** `(EV_preferred − EV_orthogonal) / (EV_preferred + EV_orthogonal)` — high values mean the unit is specifically tuned to its axis
- **Axis orthogonality:** low pairwise cosine similarity between preferred axes of different units, reflecting disentanglement

This property is expected from beta-VAE's training objective and would not hold as strongly in a standard VAE (β=1), which lacks the pressure to align each unit with an independent, orthogonal direction.

Critically, **IT cortex neurons show the same axis-tuning property in the same embedding spaces**, establishing a correspondence between beta-VAE latents and neural representations.

---

## Goal: Neuron-Guided Image Generation with Superstimuli

Given a target neuron i with preferred axis a_i in AlexNet space, generate a natural image that:
1. Activates neuron i more strongly than a baseline image
2. Leaves other neurons approximately unchanged

Because beta-VAE latent dimensions are axis-aligned with neurons, scaling latent dimension i moves the image's AlexNet embedding along a_i while orthogonal axes remain fixed — making superstimulus generation a natural operation.

---

## Methods

### Training-Free

**1. Classifier Guidance via Preferred Axis**

Uses the differentiable neuron model `ŷ_i = a_i · f(x)` (f = AlexNet) as a gradient signal into a pretrained diffusion model.

1. Start from noise or SDEdit (add noise to baseline image at intermediate t)
2. At each denoising step, estimate clean image x̂₀ from x_t via DDIM posterior
3. Compute guidance objective:  
   `L = −a_i · f(x̂₀) + λ · Σ_{j≠i} |a_j · f(x̂₀) − a_j · f(x_baseline)|²`  
   First term increases target neuron; second pins other neurons near baseline
4. Backprop `∇_{x_t} L` and add scaled gradient to the denoising update
5. For superstimuli: increase the coefficient on the first term

No training required — only the pretrained diffusion model and precomputed preferred axes.

**2. Score Distillation Sampling (SDS) + Neuron Objective**

1. Parameterize the output image via SD's VAE latent
2. Optimize: `L = −a_i · f(x) + λ · L_SDS(x)` where SDS keeps x on the natural image manifold
3. The SDS gradient acts as a naturalness regularizer; the neuron term drives selectivity

---

### Training-Required

**3. Latent Diffusion in Beta-VAE Space** *(most principled)*

Since beta-VAE latents and neurons share a coordinate system, superstimuli are exact by construction.

1. Encode all training images to beta-VAE latents z ∈ ℝ^N (N = 10–50)
2. Train a diffusion model (DDPM or LDM) over z — cheap due to low dimensionality
3. Train a high-quality decoder: finetune the beta-VAE decoder or train a pixel-space diffusion model conditioned on z
4. At inference: set z with dimension i scaled up, generate via the decoder
5. Superstimulus: z_i → z_i + Δ, z_j fixed for j≠i

Selectivity is structurally guaranteed by beta-VAE disentanglement, not just encouraged.

**4. Adapter / ControlNet Conditioned on Neuron Activation Vector**

Train a lightweight adapter mapping neuron activations into the conditioning space of a frozen pretrained diffusion model.

1. Collect paired data: (natural images, IT neuron response vectors) or use beta-VAE latents as proxy
2. Train a small MLP encoder E: ℝ^N → ℝ^{768} projecting neuron vectors to the diffusion model's cross-attention space (IP-Adapter or ControlNet style); keep diffusion U-Net frozen
3. At inference: pass neuron vector with dimension i amplified as the conditioning signal
4. Superstimulus: modify only z_i in the conditioning vector, run conditioned generation

---

## Comparison

| | Classifier Guidance | SDS | LDM in beta-VAE space | Adapter / ControlNet |
|---|---|---|---|---|
| Training cost | None | None | Moderate | Low |
| Image quality | Limited by guidance noise | Moderate | High | High (pretrained SD quality) |
| Superstimulus precision | Approximate | Approximate | Exact by construction | Good |
| Selectivity control | λ tuning | λ tuning | Direct: scale z_i | Conditioning vector |

**Recommended path:** Use classifier guidance to validate the concept; move to LDM-in-beta-VAE-space for principled, high-quality superstimulus generation.
