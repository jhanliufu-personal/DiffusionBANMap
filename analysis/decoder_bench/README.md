# Firing rates → DINOv2 z50 decoder bench

Started 2026-10-29. Code: this folder. Logs, JSON results and out-of-fold predictions: `outputs/decoder_bench/`.

## Data (as found)

- **Neural** — `data/forJhan/data/MergedITCells_500Stim_AxisTuned.mat`
  - 386 axis-tuned cells from L/R FFA, 17 patients, 54 sessions.
  - Mean rate per stimulus in a 267 ms window starting at each cell's response latency.
  - 3–4 trials per stimulus; trial rates are rebuilt from the rasters and match `responses` exactly.
  - These are *viewing* (screening) responses to the 500 stimuli. It is a pseudo-population: cells were not recorded simultaneously.
- **Noise ceiling** — `ExV_500Stim_1000Reps_AxisTunedCells.mat`: split-half explainable variance per cell.
- **Targets** — `data/stimuli_dinov2_vitb14_{cls,meanpatch}_pca_latents/val_latents_z50.npy` (500×50).
- **Unpaired bank** — `train_latents_z50.npy`: the same projection for the 15,901 stimulus-set images.
  - The bank is disjoint from the 500: no byte-identical files, and no val latent sits closer to a bank latent than typical bank neighbours do.
  - The 500 are distribution-shifted relative to the bank: PC2 std is 7.4 vs 4.0.

## Protocol

- 10-fold CV over the 500 stimuli with a fixed seed. Every number is out-of-fold.
- Every tuned hyperparameter (alphas, k/τ, likelihood temperature, rank, kernel width…) is chosen by inner CV inside the training fold.

Metrics:

| metric | meaning |
|---|---|
| `r2_dim` | mean per-PC R² |
| `r2_tot` | variance-weighted R² over all 50 PCs |
| `cos` | cosine(pred, true) after subtracting the 500-stimulus mean |
| `top1`, `top5` | identification among all 500 true embeddings by centred cosine (chance 0.2% / 1%) |
| `pair2afc` | pairwise identification (chance 0.5) |
| `spread` | total variance of predictions / of targets (1 = no shrinkage) |
| `nn_bank` | median distance to the nearest bank embedding, relative to the same for the true embeddings. ≈1 = as on-manifold as real images; the constant mean prediction scores ≈1.6 |

## Plan

**0 Diagnostics**
- D1 noise ceiling per cell.
- D2 population split-half identification, a decoder-free upper bound.
- D3 ridge baselines.
- D4 per-PC decodability.
- D5 target-space comparison (DINO CLS / meanpatch / CLIP-B32 / AlexNet fc6).
- D6 encoding fit per space, normalized by the noise ceiling.
- D7 sensitivity to trial count.
- D8 learning curve in the number of stimuli.
- D9 learning curve in the number of cells.

**1 Target geometry**
- Variance and norm re-matching.
- Soft kNN snapping onto the unpaired bank, with an optional blend.
- Bayesian inversion of a per-cell encoding model, with these priors:
  - a Gaussian fit on the training stimuli;
  - a Gaussian fit on the bank;
  - a 32-component GMM fit on the bank;
  - an empirical bank prior, i.e. the posterior mean over the 16k bank images plus the training stimuli.

**2 Shared output structure**
- Reduced-rank ridge.
- PLS.

**3 Nonlinear models, ensembles and losses**
- RBF kernel ridge.
- Random-subspace bagged ridge.
- Linear map and MLP trained with MSE + InfoNCE and BiMixCo-style mixup (MindEye recipe).

**4 Neural data**
- Preprocessing: sqrt rates; dropping cells with ExpVar < 0.1.
- Pseudo-population single-trial augmentation, with the alpha chosen by stimulus-grouped CV.

**Not testable with these files**
- Viewing→imagery transfer: no imagery-period responses are exported here.
- Cross-subject alignment: the pseudo-population already pools all 17 patients on identical stimuli.
- Neural-likelihood guidance inside the image sampler: needs generation runs. `bayes_bank_posterior` is its offline analogue.

## Results

Logs: `outputs/decoder_bench/exp{0,1_cls,1_meanpatch,2,3}.log`; JSON next to them.

**Noise level.** Top-1 is out of 500, so its standard error is ±1.6 pts at 15%. Top-5 SE is ±2.2 pts. Differences under about 3 pts are within noise. The MLPs were run with a single seed.

### 0 Diagnostics (`exp0_diagnostics.py`, `exp2_encoding_ident.py`)

| | finding |
|---|---|
| D1 | Per-cell explainable variance: median 0.33 (IQR 0.16–0.50); 333/386 cells > 0.1. |
| D2 | **Population split-half identification, no decoder: top1 74%, top5 93%, 2AFC 99.7%** (77% top1 with cells at ExpVar > 0.1). The neural data carry far more stimulus information than any decoder below recovers. |
| D3 | Ridge baselines are about equal. CLS: r2_tot 0.365, top1 12–13%, top5 33–37%. Meanpatch: r2_tot 0.367, top5 40%. The notebook's PCA-100 step is slightly worse than no PCA. |
| D4 | Decodability is concentrated in the top PCs. CLS PC2 R² = 0.90, PC1–5 ≈ 0.4–0.9; 32/50 PCs have R² > 0.1. Hence r2_tot (0.37) ≫ r2_dim (0.17). |
| D5 | Target spaces with the same ridge: meanpatch ≥ CLIP-B32 ≈ CLS > AlexNet fc6. The differences are small. |
| D6 | Encoding (space → cell), CV R² / ExpVar, median over cells: CLS 0.45, meanpatch 0.47, CLIP-B32 0.45, AlexNet 0.40. DINO explains about half of each cell's explainable variance. |
| D6b | **Identification through the encoding model** (predict each candidate's response pattern, then match it in neural space): top1 18% / 22% / 21% / 17%, top5 49% / 55% / 58% / 49% for CLS / meanpatch / CLIP-B32 / AlexNet. This direction beats direct decoding (top5 33–40%) and is about the ceiling for anything that passes through the 50-D DINO space with a linear encoder. |
| D7 | Testing on half-trial means (~2 trials) instead of full means drops top1 12→9% and r2_tot 0.37→0.32. Trial noise matters, but it is not dominant. |
| D8 | Learning curve in stimuli (top1 among 100): 16% at n=100 → 32% at 300 → 34% at 400 (CLS). It is still rising slowly. |
| D9 | Learning curve in cells (top1 among 500): 25 cells 1% → 100 cells 5% → 200 cells 6% → 386 cells 12%. It is steep and not saturating. **More cells would help more than more stimuli.** |
| D3 | **Shrinkage:** ridge predictions have spread 0.34 (a third of the true variance). Their distance to the nearest real (bank) embedding is 1.7× that of true embeddings, worse than the constant mean prediction (1.6×). Ridge outputs are off-manifold, collapsed toward the centre. |
| ref | Oracle "snap the true embedding to its nearest bank image": r2_tot 0.88, top1 89%. The 16k bank covers the 500 stimuli well. |

### 1–4 Methods (`exp1_methods.py`), 10-fold out-of-fold

`nn_bank`: 1.0 = as close to real image embeddings as the true targets are. The best value in each column is bold, ignoring the oracle.

| method | CLS r2_tot | CLS cos | CLS top1 | CLS top5 | CLS spread | CLS nn_bank | MP r2_tot | MP cos | MP top1 | MP top5 | MP spread | MP nn_bank |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ridge (baseline) | 0.365 | 0.549 | 12.4 | 33.2 | 0.35 | 1.72 | 0.367 | 0.602 | 13.0 | 40.6 | 0.34 | 1.70 |
| ridge, shared alpha | 0.365 | 0.550 | 13.4 | 37.4 | 0.32 | 1.70 | 0.364 | 0.601 | 13.0 | 40.0 | 0.31 | 1.71 |
| *preprocessing* | | | | | | | | | | | | |
| sqrt rates | 0.348 | 0.532 | 10.2 | 31.6 | 0.32 | 1.71 | 0.348 | 0.584 | 9.8 | 35.2 | 0.32 | 1.71 |
| cells ExpVar > 0.1 | 0.365 | 0.549 | 11.8 | 34.2 | 0.35 | 1.72 | 0.367 | 0.602 | 13.6 | 39.8 | 0.34 | 1.70 |
| pseudo-trial aug., 1 trial | 0.369 | 0.555 | 13.4 | 35.6 | 0.33 | 1.71 | 0.368 | 0.604 | 14.4 | 43.2 | 0.32 | 1.72 |
| pseudo-trial aug., 2 trials | 0.370 | 0.555 | 13.0 | 38.8 | 0.34 | 1.73 | 0.371 | 0.605 | 13.6 | 40.8 | 0.32 | 1.71 |
| *geometry, post hoc* | | | | | | | | | | | | |
| per-dim variance rescale | 0.086 | 0.510 | 13.6 | 41.6 | 1.05 | 2.56 | 0.109 | 0.559 | 15.4 | 48.6 | 1.05 | 2.23 |
| global variance rescale | 0.202 | 0.550 | 12.8 | 34.0 | 1.02 | 2.22 | 0.202 | 0.603 | 13.0 | 41.0 | 1.03 | 1.98 |
| norm fix (direction only) | 0.128 | 0.551 | 12.8 | 34.2 | 0.94 | 2.63 | 0.221 | 0.603 | 13.0 | 41.0 | 0.91 | 2.11 |
| kNN snap onto bank | 0.381 | 0.595 | 9.8 | 30.4 | 0.68 | 1.42 | 0.426 | 0.657 | 9.6 | 34.6 | 0.67 | 1.08 |
| kNN snap, blended 0.75 | 0.394 | 0.599 | 9.8 | 32.4 | 0.69 | 1.56 | 0.433 | 0.661 | 12.2 | 37.2 | 0.67 | 1.27 |
| *Bayesian inversion of encoding model* | | | | | | | | | | | | |
| Gaussian prior (train stimuli) | 0.322 | 0.550 | 11.0 | 37.6 | 0.61 | 1.92 | 0.350 | 0.599 | 11.8 | 34.8 | 0.49 | 1.73 |
| Gaussian prior (bank) | 0.357 | 0.561 | 16.0 | 43.0 | 0.47 | 1.86 | 0.354 | 0.604 | 15.4 | 43.4 | 0.50 | 1.72 |
| GMM-32 prior (bank) | 0.454 | 0.643 | 15.6 | 47.6 | 0.71 | 1.67 | 0.480 | 0.690 | 14.4 | 45.4 | 0.66 | 1.37 |
| **empirical bank posterior (mean)** | **0.498** | **0.656** | 14.8 | 44.4 | 0.54 | 1.51 | 0.494 | 0.693 | 13.2 | 45.4 | 0.66 | 1.06 |
| bank posterior, bank only | 0.498 | 0.656 | 15.8 | 44.2 | 0.54 | 1.51 | 0.495 | **0.695** | 15.0 | 46.8 | 0.67 | **1.05** |
| bank posterior MAP (one real image) | 0.222 | 0.560 | 10.8 | 39.8 | 1.03 | 0.00 | 0.269 | 0.617 | 12.6 | 41.4 | 0.99 | 0.00 |
| *output structure* | | | | | | | | | | | | |
| reduced-rank ridge (CV picks rank 50) | 0.365 | 0.550 | 13.4 | 37.4 | 0.32 | 1.70 | 0.364 | 0.601 | 13.0 | 40.0 | 0.31 | 1.71 |
| PLS (CV picks 20 comps) | 0.331 | 0.513 | 7.4 | 28.4 | 0.40 | 1.78 | 0.331 | 0.566 | 8.0 | 32.4 | 0.39 | 1.75 |
| *nonlinear / ensembles / losses* | | | | | | | | | | | | |
| RBF kernel ridge | 0.399 | 0.576 | 13.4 | 42.0 | 0.33 | 1.76 | 0.404 | 0.626 | 14.2 | 47.2 | 0.34 | 1.72 |
| bagged ridge (half the cells) | 0.354 | 0.547 | 10.6 | 32.4 | 0.28 | 1.63 | 0.352 | 0.600 | 10.6 | 34.8 | 0.26 | 1.68 |
| linear, MSE + InfoNCE + mixup | 0.339 | 0.533 | 15.4 | 40.0 | 0.37 | 1.82 | 0.335 | 0.577 | 16.2 | 46.0 | 0.34 | 1.76 |
| MLP, MSE + InfoNCE + mixup | 0.377 | 0.565 | 13.4 | 44.4 | 0.38 | 1.85 | 0.372 | 0.608 | **17.8** | 47.8 | 0.35 | 1.72 |
| MLP, MSE + mixup | 0.421 | 0.586 | 12.0 | 40.2 | 0.51 | 1.82 | 0.427 | 0.639 | 12.2 | 42.6 | 0.48 | 1.66 |
| reference: oracle true → nearest bank | 0.884 | 0.929 | 89.0 | 99.4 | 1.01 | 0 | 0.872 | 0.927 | 89.2 | 99.6 | 1.00 | 0 |

### Fixed equal-weight averages of the above (`exp3_combos.py`)

| combo | CLS r2_tot | CLS cos | CLS top1 | CLS top5 | CLS nn_bank | MP r2_tot | MP cos | MP top1 | MP top5 | MP nn_bank |
|---|---|---|---|---|---|---|---|---|---|---|
| bank post. + GMM post. | 0.493 | 0.656 | 15.4 | 48.2 | 1.57 | **0.506** | **0.699** | 14.6 | 47.0 | **1.21** |
| bank post. + kernel ridge | 0.490 | 0.646 | 15.8 | 48.8 | 1.63 | 0.509 | 0.695 | 16.4 | 49.4 | 1.44 |
| bank post. + MLP (MSE+InfoNCE) | 0.485 | 0.641 | 16.2 | **49.6** | 1.64 | 0.500 | 0.691 | 18.8 | 49.0 | 1.42 |
| bank post. + GMM + kernel + MLP | 0.493 | 0.646 | **17.0** | 49.2 | 1.65 | 0.506 | 0.695 | **20.2** | **50.4** | 1.46 |
| kernel + MLP + linear-InfoNCE | 0.389 | 0.568 | 14.8 | 45.2 | 1.79 | 0.389 | 0.616 | 16.6 | 49.6 | 1.72 |

## Takeaways

1. **The target geometry helps, and the right way to use it is Bayesian.** Invert a per-cell encoding model (DINO z50 → rates) against the 16k unpaired stimulus-set embeddings as an empirical prior.
   - This is `bayes_bank_posterior`. It gives the biggest single gain: r2_tot 0.365 → 0.50, centred cosine 0.55 → 0.66 (CLS) and 0.60 → 0.70 (meanpatch).
   - Identification also improves: top5 33 → 44% (CLS), 41 → 47% (meanpatch).
   - For meanpatch the output is as on-manifold as real embeddings (nn_bank 1.05).
   - Why it works: the encoding direction fits 386 separate 50-input regressions, each well posed at n=450. The bank supplies the prior that 450 pairs cannot. D6b shows the encoding direction carries more identity information (top5 49–55%) than any direct decoder.
2. **A GMM prior on the bank is close behind** (r2_tot 0.45–0.48). A Gaussian bank prior helps identification (top5 43%) but not R². The non-Gaussian shape of the prior is what lifts R² and cosine.
3. **Post-hoc fixes for shrinkage are a trade-off, not a win.**
   - Variance rescaling raises identification (top5 +8 pts) but halves R² and pushes predictions further *off*-manifold (nn_bank 2.2–2.6). It inflates noise directions along with signal.
   - kNN snapping puts predictions on the manifold (nn_bank 1.08 for meanpatch) and improves R², but it lowers identification. With noisy predictions, the nearest neighbours of a ridge output are often wrong.
   - The Bayesian posterior gets both, because it weighs neighbours by their likelihood under every cell rather than by distance to a shrunk point.
4. **Nonlinear models and contrastive training help identification modestly.**
   - RBF kernel ridge and an MLP with MSE+InfoNCE+mixup reach top5 42–48%.
   - MSE-only MLPs give higher R² but lower top-1.
   - InfoNCE trades R² for identification, as in MindEye.
5. **No gain** from sqrt transforms, dropping low-ExpVar cells, reduced rank (CV always chose full rank), PLS or bagging. Pseudo-trial augmentation is within noise: +0–3 pts top5 vs the shared-alpha ridge it builds on.
6. **The best overall is an equal-weight average** of the bank posterior, GMM posterior, kernel ridge and contrastive MLP:
   - CLS: top1 17%, top5 49%, r2_tot 0.49, cos 0.65.
   - Meanpatch: top1 20%, top5 50%, r2_tot 0.51, cos 0.70.
   - This roughly matches the D6b ceiling of a linear-encoder-through-DINO model.
7. **Meanpatch is consistently a bit better than CLS** (cos +0.04, top5 +2–6 pts, and much closer to the manifold after the Bayesian step).
8. **The bottleneck is now the DINO-z50 link, and the cell count.**
   - The neural population supports 74% top-1 on its own (D2). Through DINO, even the encoding model reaches only about 20%.
   - DINO explains about half the explainable variance per cell (D6).
   - The cell learning curve (D9) is steep. Pooling more cells, if more sessions exist, should help more than anything above.
   - Richer targets would help too: more than 50 PCs, or multiple layers. That is what the encoding model limits.

## Suggested next steps

- **Generate** from `preds_meanpatch_bayes_bank_posterior.npy` and the 4-way combo. All `outputs/decoder_bench/preds_*.npy` files are out-of-fold `[500, 50]` in stimulus order, ready for `scripts/generate.py`. Compare them against the current ridge predictions. The question is whether on-manifold, less-shrunk conditions give more specific images.
- Use the encoding model + bank posterior idea *inside* the sampler, as a neural-likelihood potential (NeuralGuidance `encode` map). That is the continuous version of the empirical bank posterior.
- Improve the encoder, which is now the limiting part:
  - nonlinear or kernel encoding models;
  - more target PCs for the encoder (e.g. z200) while still decoding to z50;
  - concatenated CLS + meanpatch features.
- A learned diffusion/flow prior in z50 would generalise beyond the 16k bank. It is not tested here; the bank posterior is its nonparametric stand-in.
