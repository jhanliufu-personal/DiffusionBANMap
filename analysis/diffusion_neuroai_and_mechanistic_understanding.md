# Diffusion models: NeuroAI parallels and mechanistic understanding

Notes from a literature survey on (1) whether diffusion models are studied the way
CNNs/VLMs (ResNet, CORNet) are studied in NeuroAI — brain alignment and unit-level
lesioning to model perceptual/psychiatric phenomena — and (2) the actual mechanistic
literature on how diffusion models generate images, independent of any brain framing.
For context on this project's own pipeline, see `analysis/embedding_neuron_alignment.ipynb`
and `analysis/pretrained_clip_diffusion_models.md`.

## 1. NeuroAI-style work on diffusion models

The CORNet/ResNet playbook — compare layer activations to neural recordings, then lesion
units to reproduce a deficit — has a much thinner analog for diffusion models. What exists
splits into five buckets:

- **Brain alignment / encoding**: [Interpreting V1 Population Activity via Image-Neural
  Latent Representation Alignment](https://arxiv.org/pdf/2605.04309) compares diffusion
  latents to V1 responses. [Brain2GAN](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11098503/)
  benchmarks diffusion (Stable Diffusion CLIP-latent) representations against a
  feature-disentangled GAN for explaining primate visual cortex responses — and finds the
  disentangled GAN features explain neural responses *better* in some ROIs, a useful
  negative result. See also the [survey on diffusion models in computational
  neuroimaging](https://arxiv.org/pdf/2502.06552).
- **Diffusion as decoder** (this project's own pipeline shape): brain-diffuser /
  "Semantic Alignment Brain Diffusion" line of work, and MindEye2 (already covered in
  `pretrained_clip_diffusion_models.md`).
- **Diffusion as a discovery tool, run in reverse** — the closest thing to "lesion to find
  function," but inverted: [BrainDiVE (NeurIPS 2023 oral)](https://arxiv.org/abs/2306.03089)
  uses a diffusion model's generative prior, guided by fMRI activity, to *synthesize* the
  images a cortical ROI prefers, then uses those synthesized images to discover functional
  subdivisions within category-selective regions (e.g. splitting FFA/PPA further). Discovery
  via synthesis rather than discovery via ablation.
- **Mechanistic interpretability inside diffusion models** (unit-level, ablatable, but not
  framed as symptom modeling): SAeUron, "Emergence and Evolution of Interpretable Concepts
  in Diffusion Models," "Residualized Temporal Sparse Autoencoders for Interpreting
  Diffusion Models," and [One-Step is Enough: Sparse Autoencoders for Text-to-Image
  Diffusion Models](https://arxiv.org/abs/2410.22366). These find causally-interpretable,
  ablatable SAE features in U-Net/DiT activations, but the goal has been concept
  erasure/unlearning (NSFW, style, copyright), not modeling a perceptual or psychiatric
  deficit.
- **The actual lesion → symptom paradigm, but on VLMs, not diffusion generators**:
  [Inducing Dyslexia in Vision Language Models](https://arxiv.org/abs/2509.24597) localizes
  visual-word-form-selective units (predicting human VWFA responses), ablates them, and
  shows selective reading deficits with intact general vision/language — an explicit
  localize → ablate → behaviorally-test platform, generalizable to other conditions.

**Gap found:** no paper turned up that lesions units inside a generative diffusion model
(e.g. Stable Diffusion's U-Net) specifically to reproduce a perceptual or psychiatric
phenomenon. The ingredients exist separately (SAE-based causal unit-finding for diffusion
internals; the localize/ablate/test template from the dyslexia-VLM paper) but haven't been
combined.

## 2. Coarse-to-fine generation and its (structural, not brain-first) parallels

Starting point: in video diffusion, early denoising steps establish global structure/motion
and later steps fill in high-frequency detail — a fact people already exploit by training on
blurry-but-motion-correct video to improve physical plausibility.

- **Mechanism, precisely**: this falls out of the noise schedule itself — high spatial
  frequencies are swamped by noise fastest in the forward process, so the reverse process
  necessarily resolves low-frequency structure before high-frequency detail has enough
  signal-to-noise to recover. See [A Fourier Space Perspective on Diffusion
  Models](https://arxiv.org/html/2505.11278v1) and [Towards Understanding the Working
  Mechanism of Text-to-Image Diffusion Models](https://arxiv.org/pdf/2405.15330).
- **Closest neuroscience parallel**: Moshe Bar's "gist first" model of object recognition —
  a fast, low-spatial-frequency signal reaches orbitofrontal cortex via the magnocellular
  pathway before the slower parvocellular/ventral stream resolves detail; OFC uses the
  coarse signal to generate a prediction that's projected back down to bias/narrow ongoing
  fine-detail processing in IT cortex. Structurally the same shape as coarse-then-detail
  diffusion generation, with attention/conditioning playing the role of the top-down
  prediction. [PNAS 2006](https://www.pnas.org/doi/10.1073/pnas.0507062103), [J Neurosci
  2007](https://www.jneurosci.org/content/27/48/13232).
- **General framing**: predictive coding / hierarchical-Bayesian-brain theories describe
  cortex as continually refining a top-down prediction against bottom-up error at every
  level — the same shape as a reverse diffusion sampler refining a guess against the learned
  score at every step. [CogDPM](https://arxiv.org/html/2405.02384v1) is an explicit early
  attempt to bridge the two computationally; treat this as a shared motif/inspiration, not
  an established empirical equivalence.
- **Imagination**: [Barry & Love, "A generative model of memory construction and
  consolidation" (Nature Human Behaviour 2023)](https://www.nature.com/articles/s41562-023-01799-z)
  proposes hippocampal replay trains a generative network (VAE-style, not diffusion) that
  reconstructs experience from latents, and that imagining/dreaming/future-thinking are just
  sampling from that trained model — supported by hippocampal damage knocking out dreaming
  and imagination alongside episodic memory. Real point of convergence with diffusion
  models: "imagining something that doesn't exist" is sampling/interpolating a learned
  manifold in both cases, not creation from nothing — neither system can produce a genuinely
  novel sensory primitive outside what it was trained on.
- **Where the analogy breaks**: a direct empirical comparison, [Stable Diffusion Models
  Reveal a Persisting Human-AI Gap in Visual Creativity](https://arxiv.org/pdf/2511.16814),
  finds SD output still falls short of human creative imagery. Likely reason: human
  imagination runs in a closed loop (continuous predictive coding against ongoing perception
  and goal-directed action, entangled with affect/motivation), while a diffusion sampler is
  a single open-loop forward pass with no feedback from an environment or task goal. Using
  Boden's creativity taxonomy, diffusion sampling looks like genuine *combinatorial* /
  *exploratory* creativity but not *transformational* creativity (changing the rules of the
  space itself).

**Testable hook for this project**: the Bar gist-first dissociation (coarse/low-SF →
magnocellular, fine/high-SF → parvocellular) predicts that early denoising steps in this
project's model should correlate more with coarse/low-spatial-frequency-selective neural
units, and later steps with fine-detail-selective units — worth checking directly against
the neuron recordings already in hand.

## 3. Seminal papers on mechanistically understanding diffusion (no brain framing)

**Foundational math — what the process is:**
- Sohl-Dickstein et al. 2015, [Deep Unsupervised Learning using Nonequilibrium
  Thermodynamics](https://arxiv.org/abs/1503.03585) — original formulation.
- Vincent 2011, [A Connection Between Score Matching and Denoising
  Autoencoders](https://www.iro.umontreal.ca/~vincentp/Publications/DenoisingScoreMatching_NeuralComp2011.pdf)
  — why training a denoiser learns the score ∇log p(x); everything else rests on this.
- Song & Ermon 2019, [Generative Modeling by Estimating Gradients of the Data
  Distribution](https://arxiv.org/abs/1907.05600) — score-based, multi-scale noise framing.
- Ho, Jain, Abbeel 2020, [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
  (DDPM) — made it practical.
- Song et al. 2021, [Score-Based Generative Modeling through Stochastic Differential
  Equations](https://arxiv.org/abs/2011.13456) — unifies DDPM and score matching as a
  continuous-time SDE; gives the vocabulary (probability flow ODE, forward/reverse SDE)
  used everywhere now.
- Karras, Aittala, Aila, Laine 2022, [Elucidating the Design Space of Diffusion-Based
  Generative Models (EDM)](https://arxiv.org/abs/2206.00364) — decomposes diffusion into
  orthogonal design choices (schedule, preconditioning, sampler) and shows which actually
  matter. Most useful single paper for understanding *why* diffusion models are built the
  way they are.

**What different timesteps/layers are doing (formalizes the coarse-to-fine observation):**
- Choi et al. 2022, [Perception Prioritized Training of Diffusion Models
  (P2)](https://arxiv.org/abs/2204.00227) — SNR at each timestep maps to a different
  perceptual job; reweights the loss accordingly.
- Balaji et al. 2022, [eDiff-I](https://arxiv.org/abs/2211.01324) — early vs. late steps
  rely on qualitatively different signals (text/layout vs. visual refinement), strongly
  enough to justify separate "expert" networks per stage.
- Si et al. 2023, [FreeU: Free Lunch in Diffusion U-Net](https://arxiv.org/abs/2309.11497)
  — U-Net backbone carries low-frequency/semantic content, skip connections inject
  high-frequency detail; reweighting the two at inference improves quality for free.

**Internal/semantic structure:**
- Kwon, Jeong, Uh 2022, [Diffusion Models Already Have a Semantic Latent Space
  (Asyrp)](https://arxiv.org/abs/2210.10960) — "h-space" (U-Net bottleneck activation) is a
  linear, homogeneous, directly-editable semantic space.
- Hertz et al. 2022, [Prompt-to-Prompt Image Editing with Cross-Attention
  Control](https://arxiv.org/abs/2208.01626) — cross-attention maps are the literal
  mechanism binding text tokens to spatial layout.

**Why diffusion models generalize instead of memorizing — the deepest mechanistic result:**
- Kadkhodaie, Guth, Simoncelli, Mallat 2023 (ICLR 2024 oral), [Generalization in Diffusion
  Models Arises from Geometry-Adaptive Harmonic
  Representations](https://arxiv.org/abs/2310.02557) — two networks trained on disjoint
  data converge to nearly the same score function; the learned denoiser implements
  shrinkage in a harmonic basis adapted to local image geometry. Explains the actual
  inductive bias behind generalization, not just that it empirically happens.
- Carlini et al. 2023, [Extracting Training Data from Diffusion
  Models](https://arxiv.org/abs/2301.13188) — empirical counterpoint: memorization does
  happen under specific conditions.

**Circuits-style interpretability (closest methodological analog to CORNet-style unit
analysis, applied to diffusion):**
- Surkov et al. 2024, [One-Step is Enough: Sparse Autoencoders for Text-to-Image Diffusion
  Models](https://arxiv.org/abs/2410.22366).

If reading only three: Song et al. (SDE) for the mathematical mechanism, Karras et al.
(EDM) for the engineering-level mechanistic breakdown, Kadkhodaie et al. for the deepest
"why this actually works" result.

## Sources

- [Interpreting V1 Population Activity via Image-Neural Latent Representation Alignment](https://arxiv.org/pdf/2605.04309)
- [Brain2GAN](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11098503/)
- [Diffusion Models for Computational Neuroimaging: A Survey](https://arxiv.org/pdf/2502.06552)
- [BrainDiVE](https://arxiv.org/abs/2306.03089)
- [Inducing Dyslexia in Vision Language Models](https://arxiv.org/abs/2509.24597)
- [One-Step is Enough: Sparse Autoencoders for Text-to-Image Diffusion Models](https://arxiv.org/abs/2410.22366)
- [A Fourier Space Perspective on Diffusion Models](https://arxiv.org/html/2505.11278v1)
- [Towards Understanding the Working Mechanism of Text-to-Image Diffusion Model](https://arxiv.org/pdf/2405.15330)
- [Bar et al., Top-down facilitation of visual recognition, PNAS 2006](https://www.pnas.org/doi/10.1073/pnas.0507062103)
- [Magnocellular Projections as the Trigger of Top-Down Facilitation in Recognition, J Neurosci 2007](https://www.jneurosci.org/content/27/48/13232)
- [CogDPM](https://arxiv.org/html/2405.02384v1)
- [A generative model of memory construction and consolidation, Nature Human Behaviour 2023](https://www.nature.com/articles/s41562-023-01799-z)
- [Stable Diffusion Models Reveal a Persisting Human-AI Gap in Visual Creativity](https://arxiv.org/pdf/2511.16814)
- [Sohl-Dickstein et al. 2015](https://arxiv.org/abs/1503.03585)
- [Vincent 2011](https://www.iro.umontreal.ca/~vincentp/Publications/DenoisingScoreMatching_NeuralComp2011.pdf)
- [Song & Ermon 2019](https://arxiv.org/abs/1907.05600)
- [Ho, Jain, Abbeel 2020 (DDPM)](https://arxiv.org/abs/2006.11239)
- [Song et al. 2021 (SDE)](https://arxiv.org/abs/2011.13456)
- [Karras et al. 2022 (EDM)](https://arxiv.org/abs/2206.00364)
- [Choi et al. 2022 (P2 weighting)](https://arxiv.org/abs/2204.00227)
- [Balaji et al. 2022 (eDiff-I)](https://arxiv.org/abs/2211.01324)
- [Si et al. 2023 (FreeU)](https://arxiv.org/abs/2309.11497)
- [Kwon et al. 2022 (Asyrp / h-space)](https://arxiv.org/abs/2210.10960)
- [Hertz et al. 2022 (Prompt-to-Prompt)](https://arxiv.org/abs/2208.01626)
- [Kadkhodaie et al. 2023](https://arxiv.org/abs/2310.02557)
- [Carlini et al. 2023](https://arxiv.org/abs/2301.13188)

_Research performed 2026-07-17._
