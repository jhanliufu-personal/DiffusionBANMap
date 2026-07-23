# World models applied to neuroscience / animal behavior

## Original prompt

I came across a work on using hidden markov model to study how mice navigate maze.
Basically, mice trajectories can be thought of making turns at different locations
in the maze, and these trajectories are modeled using markov model: u being the
mouse's location, z being the latent state (or strategy, say returning to home,
downward exploration etc.), and y as a function of z, u is the action / where to
turn at current location in current state. This methodology sounds similar to
recent works in latent world models, like LeWM. Has recent world model methods
been applied to neuroscience / modeling animal or human behavior?

## AI-generated response under review

> Modern world model methods are actively applied in neuroscience and behavioral
> modeling across three main areas:
>
> **1. Learning "Behavioral Digital Twins" of Animals**
> Instead of fitting simple HMM transition matrices, neuroscientists now train
> action-conditioned generative world models directly on animal pose-tracking data
> or high-dimensional behavioral video.
> - *Virtual Rodents & Imitation Learning*: Frameworks using architectures similar
>   to DeepMind's Dreamer or RSSMs train world models on animal movement (e.g.,
>   multi-camera tracking of rats or mice in complex arenas). A policy agent is
>   then trained inside the world model to clone the animal's navigation strategy,
>   allowing researchers to stress-test how the animal's "internal model"
>   generalizes to modified mazes or missing sensory cues.
> - *Continuous Latent Strategy Extraction*: Where a GLM-HMM forces a mouse into
>   discrete modes (e.g., "returning home" vs. "exploring"), latent world models
>   map behavior into a continuous strategy manifold, revealing subtle transitions
>   that discrete HMMs smooth over.
>
> **2. Comparing Neural Representations to Latent World Spaces**
> - *Hippocampal Cognitive Maps*: latent spaces learned by self-supervised world
>   models (JEPAs / LeWM, Successor Representation models) are compared directly
>   to neural recordings in the hippocampal-entorhinal system (place cells, grid
>   cells), suggesting spatial maps are optimized for planning, not just
>   localization.
> - *Prefrontal Cortex (PFC) & Internal Rollouts*: at decision points, PFC/
>   hippocampus "replay" future paths; latent world models provide the
>   mathematical framework to decode these neural rollouts as mental simulations.
> - *Perceptual Trajectory Straightening*: sensory cortices "straighten" temporal
>   trajectories in latent space; applying this constraint to AI world models
>   reduces trajectory curvature and speeds up planning.
>
> **3. Joint Neural-Behavioral Latent World Models**
> Rather than modeling behavior in isolation, recent approaches build joint world
> models predicting both future actions and neural activity (calcium imaging,
> Neuropixels) conditioned on a shared latent state, contrasted with classic
> discrete low-dimensional GLM-HMM approaches.

## Verification

**Likely source of the original prompt.** Rosenberg, Zhang, Perona & Meister
(2021), *"Mice in a labyrinth show rapid learning, sudden insight, and efficient
exploration,"* eLife. Mice explore a 127-node labyrinth; turn choices at each node
are modeled with Markov-chain / HMM-style models recovering latent strategies
(e.g., homing vs. exploring). This matches the (u, z, y) description almost
exactly.
https://elifesciences.org/articles/66175

**"LeWM" is real, but very recent and narrower than implied.**
*"LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from
Pixels"* (arXiv:2603.19312) is a JEPA-style world model (next-embedding
prediction + a Gaussian-regularizing loss, no reconstruction). Its neuroscience
connection is specific: latent trajectories are shown to straighten over
training as an *emergent* property, inspired by the temporal-straightening
hypothesis — not a general "world models are now used all over neuroscience"
claim.
https://arxiv.org/abs/2603.19312

**Virtual rodent — real work, but the Dreamer/RSSM framing is wrong.**
Merel et al., *"Deep neuroethology of a virtual rodent"* (arXiv:1911.09451,
2019) and the follow-up Merel et al., *"A virtual rodent predicts the structure
of neural activity across behaviours"* (Nature, 2024) train a biomechanical rat
body via deep RL + imitation learning with inverse dynamics models. This is
model-free policy cloning in a physics simulator, not a learned forward/world
model used for imagined rollouts (no Dreamer/RSSM architecture involved).
https://arxiv.org/abs/1911.09451
https://www.nature.com/articles/s41586-024-07633-4

**GLM-HMM strategy discovery — real, correctly characterized.**
Ashwood, Roy, Stone, IBL, Churchland, Pouget & Pillow (2022), *"Mice alternate
between discrete strategies during perceptual decision-making,"* Nature
Neuroscience, plus a 2025 Nature Communications follow-up on nonstationary
state switches. "Continuous latent strategy manifold" alternatives exist
(e.g., recurrent switching linear dynamical systems / rSLDS) but calling them
"latent world models" overstates/conflates two different literatures.
https://pillowlab.princeton.edu/pubs/Ashwood2022_NatNeurosci.pdf
https://www.nature.com/articles/s41467-025-66738-0

**Hippocampus as predictive map — real citation, but the JEPA/LeWM-vs-place-cell
comparison is unverified/likely fabricated.**
Stachenfeld, Botvinick & Gershman (2017), *"The hippocampus as a predictive
map,"* Nature Neuroscience — the successor-representation (SR) account of
hippocampal maps as planning-oriented, not just localization. Real and the
correct citation. No evidence was found of researchers directly comparing
JEPA/LeWM latent spaces to place-cell/grid-cell recordings; that specific claim
in the response looks like an extrapolation dressed up as an established line
of work.
https://gershmanlab.com/pubs/Stachenfeld17.pdf

**Replay / VTE as "mental simulation" — real, and there is a good formal
citation the response didn't give.**
Johnson & Redish (2007) on hippocampal theta sweeps during vicarious
trial-and-error; Pfeiffer & Foster (2013) on replay depicting future paths; and
most relevantly, **Mattar & Daw (2018), *"Prioritized memory access explains
planning and hippocampal replay,"* Nature Neuroscience** — the closest existing
formalization of replay as something like offline "imagination" rollouts for
model-based planning.
https://www.nature.com/articles/s41593-018-0232-z

**Perceptual trajectory straightening → AI world models — real, and the
strongest/most current item in the response.**
Hénaff, Goris & Simoncelli (2019), *"Perceptual straightening of natural
videos,"* Nature Neuroscience, is the neuroscience original. LeWM (above)
genuinely connects to it: straightening emerges during training without being
explicitly imposed as a constraint (the response's phrasing that it was
"applied ... as a constraint" is slightly wrong on mechanism, right on the
underlying link).

**Joint neural + behavioral latent models — real, and the most directly
relevant existing tool for this project's kind of setup.**
Schneider, Lee & Mathis (2023), *"Learnable latent embeddings for joint
behavioural and neural analysis"* (CEBRA), Nature; also pi-VAE (Zhou & Wei,
2020) and LFADS (Pandarinath et al., 2018, Nature Methods). These jointly embed
neural + behavioral data but do not do action-conditioned imagined rollouts the
way Dreamer/LeWM do — closer to structured latent dynamics models than to
"world models" in the planning sense.
https://www.nature.com/articles/s41586-023-06031-6

## Interpretation

The response oversold the connection: it attached real vocabulary ("Dreamer,"
"RSSM," "JEPA") to real neuroscience papers that don't actually use those
architectures, producing a plausible-sounding but inaccurate synthesis. The
genuinely solid throughline is narrower than advertised:

1. Discrete-state HMMs (the maze-paper lineage) are being pushed toward
   continuous/richer latent dynamics models (rSLDS-style), not literal world
   models.
2. Hippocampal replay is increasingly framed computationally as "offline
   rollouts for planning" (Mattar & Daw), which is the legitimate version of
   the response's vague "PFC internal rollouts" claim.
3. The straightening-hypothesis <-> world-model link (Hénaff -> LeWM) is a
   genuine, recent, and fairly narrow point of contact between world-model
   research and systems neuroscience — not yet a broad merger of the two
   fields as the response implies.

If moving from an HMM toward something more "world-model-like" for this kind
of (u, z, y) setup, CEBRA is the most directly relevant existing tool, since it
jointly embeds behavior + neural data without requiring a full RL/imagination-
rollout framework.
