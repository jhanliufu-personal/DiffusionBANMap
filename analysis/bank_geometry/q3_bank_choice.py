"""Q3: which candidate bank should the posterior use?

Every condition runs the same 10-fold bank posterior on the meanpatch targets with include_train=False,
so only the bank differs (the decoder_bench default also adds the ~450 training stimuli).

Conditions
  16k                       the stimulus-set bank (15,901)
  16k subset n (3 seeds)    bank-size curve: 300, 1000, 3000, 8000
  ImageNet raw / gray / styled   N_INET Tiny-ImageNet images as-is, grayscale, grayscale cut-out on white
  16k + ImageNet X (mixture)     union with prior weight pi on the 16k, pi tuned by inner CV
  16k + ImageNet-filtered   only ImageNet images inside the 16k manifold (a real 5-NN ball), uniform prior
  DBM bank (if present)     sets/dbm_bank.npz written by dbm_cycle.py

Bank-level descriptors: in-style fraction (precision against the 16k), C2ST AUC against the 16k,
coverage of the 500 targets (NN distance and convex-hull residual, in 16k-NN units), and the oracle
"nearest candidate" identification.

python -m analysis.bank_geometry.q3_bank_choice
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.decoder_bench.common import load_neural, load_target, metrics
from . import geometry as G
from .posterior import cv_bank_posterior

OUT = os.path.join(G.__file__.rsplit("/analysis/", 1)[0], "outputs", "bank_geometry")
FIG = os.path.join(OUT, "figs")
SETS = os.path.join(OUT, "sets")
PIS = (0.25, 0.5, 0.75, 0.9, 0.97, 1.0)


def load_set(name):
    p = f"{SETS}/{name}.npz"
    return np.load(p)["z"].astype(np.float64) if os.path.exists(p) else None


def describe(B, bank16, Y, scale, rad16):
    nn = G.nn_dist(Y, B) / scale
    oracle = B[G.knn(Y, B, 1)[0][:, 0]]
    mo = metrics(oracle, Y)
    d = {"n": int(len(B)),
         "in_style_precision": G.precision(B, bank16, rad=rad16) if len(B) else 0.0,
         "target_nn_units": float(np.median(nn)),
         "target_hull_units": float(np.median(G.hull_residual(Y, B, K=min(50, len(B)))) / scale),
         "oracle_top1": mo["top1"], "oracle_cos": mo["cos"]}
    return d


def run():
    X, *_ = load_neural()
    Y, bank16 = load_target("meanpatch")
    scale = G.bank_scale(bank16)
    rad16 = G.knn_radius(bank16, 5)
    res_path = f"{OUT}/q3_results.json"
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}

    def cond(name, banks, desc_bank, grid_pi=(None,), group=None, **meta):
        if name in res:
            print("cached", name)
            return
        P, _, hp, _ = cv_bank_posterior(X, Y, banks, include_train=False, grid_pi=grid_pi)
        m = metrics(P, Y, bank16)
        m.update(describe(desc_bank, bank16, Y, scale, rad16))
        m["kappa"] = [h[0] for h in hp]
        m["pi"] = [h[1] for h in hp]
        m["group"] = group or name
        m.update(meta)
        res[name] = m
        print(f"{name:<34s} cos={m['cos']:.3f} r2={m['r2_tot']:.3f} top1={m['top1']:.3f} top5={m['top5']:.3f} "
              f"nn_bank={m['nn_bank']:.2f} in_style={m['in_style_precision']:.2f} pi={m['pi']}", flush=True)
        json.dump(res, open(res_path, "w"), indent=1)

    cond("16k", [bank16], bank16, size=len(bank16))
    rng = np.random.default_rng(0)
    for n in (300, 1000, 3000, 8000):
        for s in range(3 if n < 8000 else 1):
            idx = np.random.default_rng(100 * n + s).choice(len(bank16), n, replace=False)
            cond(f"16k subset {n} s{s}", [bank16[idx]], bank16[idx], group="16k subset", size=n)

    inet = {v: load_set(f"inet_{v}") for v in ("raw", "gray", "styled")}
    for v, Z in inet.items():
        if Z is None:
            print(f"inet_{v} not embedded yet, skipping")
            continue
        cond(f"ImageNet {v}", [Z], Z, group="ImageNet alone", size=len(Z))
        cond(f"16k + ImageNet {v} (mixture)", [bank16, Z], np.vstack([bank16, Z]), grid_pi=PIS,
             group="16k + ImageNet mixture", size=len(bank16) + len(Z))
        keep = G.in_manifold(Z, bank16, rad=rad16)
        if keep.sum():
            U = np.vstack([bank16, Z[keep]])
            cond(f"16k + ImageNet {v} (filtered)", [U], U, group="16k + ImageNet filtered",
                 size=len(U), n_kept=int(keep.sum()))
    dbm = load_set("dbm_bank")
    if dbm is not None:
        cond("DBM samples", [dbm], dbm, group="DBM", size=len(dbm))
        cond("16k + DBM samples (mixture)", [bank16, dbm], np.vstack([bank16, dbm]), grid_pi=PIS,
             group="16k + DBM", size=len(bank16) + len(dbm))

    # bank-vs-16k two-sample AUCs (computed once per bank)
    for v, Z in list(inet.items()) + [("dbm", dbm)]:
        if Z is not None and f"_auc_{v}" not in res:
            res[f"_auc_{v}"] = G.c2st_auc(Z, bank16)
    res["_auc_targets_vs_16k"] = G.c2st_auc(Y, bank16)
    json.dump(res, open(res_path, "w"), indent=1)
    return res, Y, bank16, inet, dbm


def plots(res, Y, bank16, inet, dbm):
    conds = {k: v for k, v in res.items() if not k.startswith("_")}
    # 1) bank-size curve with the other banks as points at their size
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    sub = {}
    for k, v in conds.items():
        if v["group"] in ("16k subset", "16k"):
            sub.setdefault(v["size"], []).append(v)
    ns = sorted(sub)
    for a, key in zip(ax, ("cos", "top5", "nn_bank")):
        m = [np.mean([r[key] for r in sub[n]]) for n in ns]
        sd = [np.std([r[key] for r in sub[n]]) for n in ns]
        a.errorbar(ns, m, sd, fmt="o-", c="k", label="random 16k subsets")
        for i, (k, v) in enumerate((k, v) for k, v in conds.items() if v["group"] not in ("16k subset", "16k")):
            a.scatter(v["size"], v[key], s=40, marker="D", c=f"C{i}", label=k, zorder=3)
        a.set_xscale("log")
        a.set_xlabel("bank size")
        a.set_title(key)
    ax[0].legend(fontsize=6.5)
    fig.suptitle("Bank posterior (no training stimuli in the bank): bank size vs bank content")
    fig.tight_layout()
    fig.savefig(f"{FIG}/q3_bank_size_and_content.png", dpi=130)
    plt.close(fig)

    # 2) 2-D view: PC1/PC2 of the 16k (the z50 axes are already the stimulus-set PCs)
    fig, ax = plt.subplots(1, 1 + sum(z is not None for z in inet.values()), figsize=(16, 4), squeeze=False)
    ax = ax[0]
    ax[0].scatter(bank16[:, 0], bank16[:, 1], s=1, c="lightgray", label="16k")
    ax[0].scatter(Y[:, 0], Y[:, 1], s=4, c="C3", label="500 stimuli")
    ax[0].set_title("16k bank + 500 targets")
    ax[0].legend(markerscale=4, fontsize=7)
    j = 1
    for v, Z in inet.items():
        if Z is None:
            continue
        ax[j].scatter(bank16[:, 0], bank16[:, 1], s=1, c="lightgray")
        ax[j].scatter(Z[:, 0], Z[:, 1], s=2, c=f"C{j - 1}")
        ax[j].set_title(f"ImageNet {v}")
        j += 1
    for a in ax:
        a.set_xlabel("PC1")
        a.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(f"{FIG}/q3_bank_pc_scatter.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    res, Y, bank16, inet, dbm = run()
    plots(res, Y, bank16, inet, dbm)
