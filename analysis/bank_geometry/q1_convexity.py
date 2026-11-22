"""Q1: is a weighted average of bank embeddings still on the image-embedding manifold?

A. Where do decoder outputs sit relative to the bank? precision / density / nn-ratio / classifier
   two-sample AUC for ridge, bank posterior (mean & MAP), GMM posterior, kNN snap, rescaled ridge and
   the true targets, plus the bank-posterior's own weight diagnostics (ESS, support radius) per stimulus.
B. Synthetic convex mixtures, decoder-free: average the k nearest bank points of held-out real anchors
   (and midpoints of pairs at increasing neighbour rank). Plot on-manifold metrics vs the mixture's
   support radius, measured against an independent half of the bank. This gives the radius at which
   local convexity breaks, and part A's posterior radii are overlaid on it.
C. Cycle consistency through the diffusion model needs a checkpoint: see dbm_cycle.py.

python -m analysis.bank_geometry.q1_convexity [cls|meanpatch ...]
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.decoder_bench.common import OUT as BENCH, load_neural, load_target
from . import geometry as G
from .posterior import cv_bank_posterior

OUT = os.path.join(G.__file__.rsplit("/analysis/", 1)[0], "outputs", "bank_geometry")
FIG = os.path.join(OUT, "figs")
os.makedirs(FIG, exist_ok=True)

DECODERS = {
    "true targets": None,
    "ridge": "ridge",
    "ridge, per-dim rescale": "rescale_per_dim",
    "kNN snap": "snap_bank",
    "GMM-32 posterior": "bayes_gmm32_bank",
    "bank posterior (mean)": "bayes_bank_posterior",
    "bank posterior (MAP)": "bayes_bank_posterior_map",
}


def manifold_row(Q, R, Yref, rad, scale):
    auc = G.c2st_auc(Q, Yref) if Q is not Yref else {"linear": 0.5, "gbt": 0.5}
    mu = R.mean(0)
    return {
        "precision": G.precision(Q, R, rad=rad),
        "density": G.density(Q, R, rad=rad),
        "nn_ratio": G.nn_ratio(Q, R, Yref),
        "nn_units": float(np.median(G.nn_dist(Q, R)) / scale),
        "auc_lin": auc["linear"], "auc_gbt": auc["gbt"],
        "norm_ratio": float(np.median(np.linalg.norm(Q - mu, axis=1)) / np.median(np.linalg.norm(Yref - mu, axis=1))),
    }


def part_a(target, X, Y, bank):
    rad = G.knn_radius(bank, 5)
    scale = G.bank_scale(bank)
    rows = {}
    for name, f in DECODERS.items():
        Q = Y if f is None else np.load(f"{BENCH}/preds_{target}_{f}.npy").astype(np.float64)
        rows[name] = manifold_row(Q, bank, Y, rad, scale)
        print(f"[{target}] A {name:<26s}", {k: round(v, 3) for k, v in rows[name].items()}, flush=True)
    rows["_true_vs_bank_auc"] = G.c2st_auc(Y, bank)
    rows["_bank_scale"] = scale

    P, D, hp, _ = cv_bank_posterior(X, Y, [bank], diag=True)
    inm = G.in_manifold(P, bank, rad=rad)
    nnr = G.nn_dist(P, bank) / np.median(G.nn_dist(Y, bank))
    np.savez(f"{OUT}/q1_{target}_posterior_diag.npz", ess=D["ess"], radius=D["radius"], wmax=D["wmax"],
             in_manifold=inm, nn_ratio=nnr, pred=P)
    rows["_posterior"] = {
        "kappa": [h[0] for h in hp],
        "ess_median": float(np.median(D["ess"])), "ess_iqr": np.percentile(D["ess"], [25, 75]).tolist(),
        "radius_units_median": float(np.median(D["radius"]) / scale),
        "radius_units_iqr": (np.percentile(D["radius"], [25, 75]) / scale).tolist(),
        "wmax_median": float(np.median(D["wmax"])),
        "spearman_radius_vs_nnratio": float(_spearman(D["radius"], nnr)),
        "spearman_ess_vs_nnratio": float(_spearman(D["ess"], nnr)),
    }
    print(f"[{target}] posterior diag", rows["_posterior"], flush=True)
    return rows, D, nnr, inm, scale


def _spearman(a, b):
    from scipy.stats import spearmanr
    return spearmanr(a, b).correlation


def part_b(target, bank, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(bank))
    anchors = bank[perm[:2000]]
    A = bank[perm[2000:2000 + (len(bank) - 2000) // 2]]  # mixtures are built from A
    B = bank[perm[2000 + (len(bank) - 2000) // 2:]]       # and judged against B
    radB = G.knn_radius(B, 5)
    scale = G.bank_scale(bank)
    I, _ = G.knn(anchors, A, 1000)
    rows = []

    def add(kind, param, Q, support):
        r = manifold_row(Q, B, anchors, radB, scale)
        r.update(kind=kind, param=param, radius_units=float(np.median(support) / scale))
        rows.append(r)
        print(f"[{target}] B {kind:<6s} {param:>6}", {k: (round(v, 3) if isinstance(v, float) else v)
                                                      for k, v in r.items() if k not in ("kind", "param")}, flush=True)

    # real points from A, judged against B: the "real" reference row
    add("real", 1, A[rng.choice(len(A), 2000, replace=False)], np.zeros(2000))
    for k in (2, 3, 5, 10, 20, 50, 100, 300, 1000):
        C = A[I[:, :k]]  # [n, k, d]
        Q = C.mean(1)
        sup = np.sqrt(((C - Q[:, None]) ** 2).sum(-1).mean(1))
        add("kNN", k, Q, sup)
    for j in (1, 4, 16, 64, 256, 999):
        C = np.stack([A[I[:, 0]], A[I[:, j]]], 1)
        Q = C.mean(1)
        sup = np.linalg.norm(C[:, 0] - C[:, 1], axis=1) / 2
        add("pair", j + 1, Q, sup)
    ri = rng.integers(0, len(A), (2000, 2))
    C = A[ri]
    add("pair", "random", C.mean(1), np.linalg.norm(C[:, 0] - C[:, 1], axis=1) / 2)
    return rows


def plot(target, rows_b, D, nnr, inm, scale):
    kn = [r for r in rows_b if r["kind"] == "kNN"]
    pr = [r for r in rows_b if r["kind"] == "pair"]
    real = [r for r in rows_b if r["kind"] == "real"][0]
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    for a, key, lab in zip(ax, ["precision", "nn_ratio", "auc_gbt", "norm_ratio"],
                           ["precision (inside a real 5-NN ball)", "NN distance / real", "C2ST AUC (GBT) vs real",
                            "norm ratio (shrinkage)"]):
        a.plot([r["radius_units"] for r in kn], [r[key] for r in kn], "o-", label="mean of k nearest")
        a.plot([r["radius_units"] for r in pr], [r[key] for r in pr], "s--", label="pair midpoint")
        a.axhline(real[key], color="k", lw=0.8, ls=":", label="real points")
        a.set_xscale("symlog", linthresh=0.5)
        a.set_xlabel("support radius (bank NN units)")
        a.set_title(lab, fontsize=10)
        lo, hi = np.percentile(D["radius"] / scale, [25, 75])
        a.axvspan(lo, hi, color="C3", alpha=0.15, label="bank posterior IQR")
    ax[0].legend(fontsize=7)
    fig.suptitle(f"Synthetic convex mixtures of bank embeddings ({target})")
    fig.tight_layout()
    fig.savefig(f"{FIG}/q1_mixture_curves_{target}.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].hist(np.log10(D["ess"]), 40, color="C0")
    ax[0].set_xlabel("log10 effective # candidates (ESS)")
    ax[0].set_title("posterior concentration")
    ax[1].hist(D["radius"] / scale, 40, color="C1")
    ax[1].set_xlabel("support radius (bank NN units)")
    ax[1].set_title("posterior spread")
    ax[2].scatter(D["radius"] / scale, nnr, s=6, c=np.where(inm, "C2", "C3"))
    ax[2].axhline(1, color="k", lw=0.8, ls=":")
    ax[2].set_xlabel("support radius (bank NN units)")
    ax[2].set_ylabel("NN distance to bank / real median")
    ax[2].set_title("green = inside a real 5-NN ball")
    fig.suptitle(f"Bank-posterior weights, 500 out-of-fold predictions ({target})")
    fig.tight_layout()
    fig.savefig(f"{FIG}/q1_posterior_diag_{target}.png", dpi=130)
    plt.close(fig)


def plot_a(all_rows):
    targets = list(all_rows)
    names = list(DECODERS)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, key, lab in zip(ax, ["precision", "nn_ratio", "auc_gbt"],
                           ["precision vs bank (true targets = reference)", "NN distance / true", "C2ST AUC vs true targets (0.5 = same)"]):
        w = 0.38
        for i, t in enumerate(targets):
            a.barh(np.arange(len(names)) + i * w, [all_rows[t]["A"][n][key] for n in names], w, label=t)
        a.set_yticks(np.arange(len(names)) + w / 2)
        a.set_yticklabels(names if a is ax[0] else [])
        a.set_title(lab, fontsize=10)
        a.invert_yaxis()
    ax[1].axvline(1, color="k", lw=0.8, ls=":")
    ax[2].axvline(0.5, color="k", lw=0.8, ls=":")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(f"{FIG}/q1_decoder_manifold.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    targets = sys.argv[1:] or ["meanpatch", "cls"]
    X, *_ = load_neural()
    res_path = f"{OUT}/q1_results.json"
    allr = json.load(open(res_path)) if os.path.exists(res_path) else {}
    for t in targets:
        Y, bank = load_target(t)
        rows_a, D, nnr, inm, scale = part_a(t, X, Y, bank)
        rows_b = part_b(t, bank)
        plot(t, rows_b, D, nnr, inm, scale)
        allr[t] = {"A": rows_a, "B": rows_b}
        json.dump(allr, open(res_path, "w"), indent=1)
    if all(t in allr for t in ("meanpatch", "cls")):
        plot_a(allr)
