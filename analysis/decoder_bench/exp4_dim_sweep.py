"""How does reachable stimulus information grow with the number of DINO PCs?
For d in DIMS, re-fit the same z-score -> PCA as scripts/extract_image_embeddings.py on the full
768-D DINOv2 embeddings (CLS and meanpatch), keep the top d PCs, and measure:
  - top1 identification among the 500 stimuli, two ways:
      encoding: exp2's encoding-model identification in neural space (pattern correlation)
      decoding: exp0's D3 ridge decoder rates -> z_d, matched by centred cosine in z_d
  - fraction of explainable variance explained per cell (exp0's D6 encoding fit, R2 / ExpVar,
    cells with ExpVar > 0.1), mean and median over cells. Plotted on raw PCs: D6 z-scores each
    PC, which gives noise PCs the same prior weight as signal PCs and collapses at large d.

python -m analysis.decoder_bench.exp4_dim_sweep          # compute + plot
python -m analysis.decoder_bench.exp4_dim_sweep cls      # one space only (merged into the JSON)
python -m analysis.decoder_bench.exp4_dim_sweep --plot   # replot from the saved JSON
"""

import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from .common import DATA, OUT, cv_predict, folds, load_neural, load_target, metrics
from .decoders import ALPHAS, ridge
from .methods import Encoder

EMB = {
    "cls": "stimuli_dinov2_vitb14_cls_pca_latents/{split}_embeddings.npy",
    "meanpatch": "stimuli_dinov2_vitb14_meanpatch_latents/{split}_latents.npy",
}
DIMS = [2, 5, 10, 20, 50, 100, 200, 400, 768]
SUBSAMPLE_SIZE, SEED = 15000, 42  # as in scripts/extract_image_embeddings.py
RESULTS = os.path.join(OUT, "exp4_results.json")


def pca_latents(name):
    """[500, 768] val latents in PC order, from the same scaler + PCA recipe as the saved z50."""
    train = np.load(os.path.join(DATA, EMB[name].format(split="train")), mmap_mode="r")
    val = np.load(os.path.join(DATA, EMB[name].format(split="val"))).astype(np.float64)
    idx = np.sort(np.random.default_rng(SEED).choice(len(train), min(SUBSAMPLE_SIZE, len(train)), replace=False))
    scaler = StandardScaler().fit(train[idx])
    pca = PCA(svd_solver="full").fit(scaler.transform(train[idx]))
    z = pca.transform(scaler.transform(val))
    # sanity: the top-50 subspace should match the saved z50 (single late PCs may swap order)
    z50, _ = load_target(name)
    fit = z[:, :50] @ np.linalg.lstsq(z[:, :50], z50, rcond=None)[0]
    print(f"[{name}] saved z50 explained by our top-50 PCs: R2={1 - ((z50 - fit) ** 2).sum() / ((z50 - z50.mean(0)) ** 2).sum():.4f}")
    return z, np.cumsum(pca.explained_variance_ratio_)


def encoding_top1(X, Y):
    """exp2 encoding-model identification (pattern correlation in neural space)."""
    C = np.zeros((500, 500))
    for tr, te in folds():
        enc = Encoder(Y[tr], X[tr])
        R = enc.r(X[te])
        F = Y @ enc.W.T + enc.b
        Rz = (R - R.mean(1, keepdims=True)) / R.std(1, keepdims=True)
        Fz = (F - F.mean(1, keepdims=True)) / F.std(1, keepdims=True)
        C[te] = Rz @ Fz.T / R.shape[1]
    rank = (C > np.diag(C)[:, None]).sum(1)
    return float((rank == 0).mean()), float((rank < 5).mean())


def fev(X, Y, expvar, standardize):
    """exp0 D6: per-cell CV R2 of z_d -> rate, normalized by explainable variance (ExpVar > 0.1)."""
    good = expvar > 0.1
    F = StandardScaler().fit_transform(Y) if standardize else Y
    pred = np.zeros_like(X)
    for tr, te in folds():
        pred[te] = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(F[tr], X[tr]).predict(F[te])
    r2 = 1 - ((X - pred) ** 2).sum(0) / ((X - X.mean(0)) ** 2).sum(0)
    norm = np.clip(r2[good] / expvar[good], 0, 1)
    return float(norm.mean()), float(np.median(norm))


def run(names):
    X, _, expvar, _, _ = load_neural()
    res = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
    res["dims"] = DIMS
    for name in names:
        Z, cumvar = pca_latents(name)
        out = {k: [] for k in ["enc_top1", "enc_top5", "dec_top1", "dec_top5",
                               "fev_mean", "fev_median", "fev_mean_raw", "fev_median_raw", "pca_var"]}
        for d in DIMS:
            Y = Z[:, :d]
            e1, e5 = encoding_top1(X, Y)
            m = metrics(cv_predict(ridge, X, Y), Y)
            fm, fmed = fev(X, Y, expvar, standardize=True)
            fmr, fmedr = fev(X, Y, expvar, standardize=False)
            for k, v in zip(out, [e1, e5, m["top1"], m["top5"], fm, fmed, fmr, fmedr, float(cumvar[d - 1])]):
                out[k].append(v)
            print(f"[{name:<9s}] d={d:3d}  enc top1={e1:.3f} top5={e5:.3f} | dec top1={m['top1']:.3f} "
                  f"top5={m['top5']:.3f} | FEV mean={fm:.3f} med={fmed:.3f} (unstd mean={fmr:.3f}) "
                  f"| PCA var={cumvar[d - 1]:.3f}", flush=True)
        res[name] = out
        json.dump(res, open(RESULTS, "w"), indent=1)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = json.load(open(RESULTS))
    d2_top1 = json.load(open(os.path.join(OUT, "exp0_results.json")))["D2_all"][0]
    dims = res["dims"]
    color = {"cls": "#2a78d6", "meanpatch": "#eb6834"}  # categorical slots 1-2 (dataviz palette)
    tex = {"cls": "CLS", "meanpatch": "meanpatch"}
    ink, muted = "#0b0b0b", "#52514e"

    def style(ax, ylabel, mark_z50=True):
        ax.set_xscale("log")
        ax.set_xticks(dims)
        ax.set_xticklabels([str(d) for d in dims])
        ax.minorticks_off()
        ax.set_xlabel("number of DINOv2 PCs (d)", color=ink)
        ax.set_ylabel(ylabel, color=ink)
        ax.grid(axis="y", color="#e6e5e1", lw=0.8)
        ax.set_axisbelow(True)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color("#b5b4ae")
        ax.tick_params(colors=muted)
        if mark_z50:
            ax.axvline(50, color="#b5b4ae", lw=1, ls=":")
            ax.text(50, ax.get_ylim()[1], " current z50", color=muted, fontsize=8, va="top")

    # 1: top1 vs d
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name in [n for n in EMB if n in res]:
        r = res[name]
        ax.plot(dims, np.array(r["enc_top1"]) * 100, color=color[name], lw=2, marker="o", ms=5,
                label=rf"$\tilde{{r}} = \mathrm{{encode}}(z_{{\mathrm{{{tex[name]}}}}})$")
        ax.plot(dims, np.array(r["dec_top1"]) * 100, color=color[name], lw=2, ls="--", marker="s", ms=4,
                label=rf"$\tilde{{z}}_{{\mathrm{{{tex[name]}}}}} = \mathrm{{decode}}(r)$")
    ax.axhline(d2_top1 * 100, color=muted, lw=1, ls="-.")
    ax.text(dims[0], d2_top1 * 100 - 1, "ground truth firing rates (r)", color=muted, fontsize=8, va="top")
    ax.axhline(0.2, color="#b5b4ae", lw=1)
    ax.set_ylim(0, max(80, d2_top1 * 100 + 5))
    style(ax, "top-1 identification among 500 (%)", mark_z50=False)
    ax.set_title("Top-1 identification vs. DINO dimensionality", color=ink, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=ink, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exp4_top1_vs_dim.png"), dpi=200)

    # 2: fraction of explainable variance vs d
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name in [n for n in EMB if n in res]:
        r = res[name]
        ax.plot(dims, r["fev_mean_raw"], color=color[name], lw=2, marker="o", ms=5, label=f"{name}")
        ax.annotate(name, (dims[-1], r["fev_mean_raw"][-1]), xytext=(4, 0), textcoords="offset points",
                    color=ink, fontsize=8, va="center")
    ax.set_ylim(0, 1)
    style(ax, "R² / explainable variance, mean over cells")
    ax.set_title("Explainable variance captured by DINO (cells with ExpVar > 0.1)", color=ink, loc="left",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=ink, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exp4_fev_vs_dim.png"), dpi=200)
    print("saved", os.path.join(OUT, "exp4_top1_vs_dim.png"), os.path.join(OUT, "exp4_fev_vs_dim.png"))


if __name__ == "__main__":
    if "--plot" not in sys.argv:
        run([a for a in sys.argv[1:] if a in EMB] or list(EMB))
    plot()
