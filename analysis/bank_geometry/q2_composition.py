"""Q2: units of change, whether k units stay on-manifold, and whether the bank posterior can reach them.

Inputs: outputs/bank_geometry/sets/edits*.npz (z, base, family, k), written by
build_imagesets.py edits (controlled transforms) and nanobanana_edits.py (semantic edits).
Every family is a chain of k = 1..K identical edits of a base image ("one unit" = one step).

1. Unit size: ||z_k - z_0|| in bank-NN units, against the decoder's error and stimulus spacing.
2. Is a unit a global direction? Pairwise cosine of d_1 across base images; linearity of d_k vs k d_1;
   leave-one-base-out "vector arithmetic" z_0 + mean_d_k vs the true z_k.
3. On-manifold vs k: precision / NN ratio / convex-hull residual of (a) the real edited images,
   (b) the arithmetic prediction z_0 + k * mean_d_1.
4. Reachability under the noise model: simulate firing rates for each edited embedding with the fitted
   encoder + its residual noise, decode with the bank posterior (bank only, the 500 excluded) and with
   ridge; measure how much of the edit is recovered.
5. Leave-region-out with REAL neural data: remove every candidate within radius r of the true target
   and re-run the 10-fold bank posterior, to see how it degrades for images unlike anything in the bank.

python -m analysis.bank_geometry.q2_composition
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.decoder_bench.common import OUT as BENCH, load_neural, load_target, metrics
from analysis.decoder_bench.decoders import ridge_model
from . import geometry as G
from .posterior import _posterior, bank_posterior, cv_bank_posterior, get_encoder

OUT = os.path.join(G.__file__.rsplit("/analysis/", 1)[0], "outputs", "bank_geometry")
FIG = os.path.join(OUT, "figs")
TARGET = "meanpatch"  # edits are embedded in the meanpatch z50 space


def load_edits():
    sets = sorted(glob.glob(f"{OUT}/sets/edits*.npz"))
    zs, base, fam, k = [], [], [], []
    for p in sets:
        d = np.load(p, allow_pickle=True)
        tag = os.path.basename(p)[len("edits"):-4].strip("_")
        zs.append(d["z"]); base.append(d["base"]); k.append(d["k"])
        fam.append(np.array([f if (f == "orig" or not tag) else f"{tag}:{f}" for f in d["family"]]))
        print(f"loaded {p}: {len(d['z'])} images")
    return np.vstack(zs), np.concatenate(base), np.concatenate(fam), np.concatenate(k)


def chains(z, base, fam, k):
    """{family: {base: [z_0, z_1, ..., z_K]}} where z_0 is the base image's own (re-)embedding."""
    origin = {b: z[(base == b) & (fam == "orig")][0] for b in np.unique(base[fam == "orig"])}
    out = {}
    for f in np.unique(fam):
        if f == "orig":
            continue
        out[f] = {}
        for b in np.unique(base[fam == f]):
            if b not in origin:
                continue
            m = (base == b) & (fam == f)
            order = np.argsort(k[m])
            out[f][b] = np.vstack([origin[b], z[m][order]])
    return out


def cos(a, b):
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)).clip(1e-12)


def unit_stats(C, scale, err_scale, stim_scale):
    res = {}
    for f, ch in C.items():
        Z = np.stack(list(ch.values()))  # [n_base, K+1, d]
        D = Z[:, 1:] - Z[:, :1]  # [n_base, K, d]
        n, K = D.shape[:2]
        size = np.linalg.norm(D, axis=-1)
        d1 = D[:, 0]
        pc = cos(d1[:, None], d1[None])[np.triu_indices(n, 1)]
        # leave-one-base-out vector arithmetic: predict z_k = z_0 + mean_{other bases} d_k
        loo_err = np.zeros((n, K))
        for i in range(n):
            mdk = np.delete(D, i, 0).mean(0)  # [K, d]
            loo_err[i] = np.linalg.norm(Z[i, 1:] - (Z[i, :1] + mdk), axis=-1) / size[i].clip(1e-9)
        res[f] = {
            "K": int(K), "n_base": int(n),
            "size_nn_units": (np.median(size, 0) / scale).tolist(),
            "size_err_units": (np.median(size, 0) / err_scale).tolist(),
            "size_stim_units": (np.median(size, 0) / stim_scale).tolist(),
            "pairwise_cos_d1": float(np.mean(pc)),
            "cos_dk_d1": np.median(cos(D, d1[:, None]), 0).tolist(),
            "linearity_norm": (np.median(size / (np.arange(1, K + 1)[None] * size[:, :1]).clip(1e-9), 0)).tolist(),
            "arith_rel_err": np.median(loo_err, 0).tolist(),
        }
    return res


def manifold_vs_k(C, bank, Y, scale, rad):
    res = {}
    hull_ref_real = float(np.median(G.hull_residual(Y, bank)) / scale)
    for f, ch in C.items():
        Z = np.stack(list(ch.values()))
        n, K1 = Z.shape[:2]
        mean_d1 = (Z[:, 1] - Z[:, 0]).mean(0)
        r = {"k": list(range(K1)), "real": {}, "arith": {}}
        for kind in ("real", "arith"):
            for key in ("precision", "nn_ratio", "hull_units"):
                r[kind][key] = []
            for kk in range(K1):
                Q = Z[:, kk] if kind == "real" else Z[:, 0] + kk * mean_d1
                r[kind]["precision"].append(G.precision(Q, bank, rad=rad))
                r[kind]["nn_ratio"].append(G.nn_ratio(Q, bank, Y))
                r[kind]["hull_units"].append(float(np.median(G.hull_residual(Q, bank)) / scale))
        res[f] = r
    return res, hull_ref_real


def simulate_decoding(C, X, Y, bank, n_draws=10, seed=0):
    """Rates for each edited embedding drawn from the encoder's own noise model."""
    rng = np.random.default_rng(seed)
    enc = get_encoder(Y, X)
    _, (kappa, _), _ = bank_posterior(X, Y, X[:1], [bank], include_train=False)
    lp = np.zeros(len(bank))
    mu = Y.mean(0)
    rmodel = ridge_model().fit(X, Y)
    res = {}
    for f, ch in C.items():
        Z = np.stack(list(ch.values()))  # [n, K+1, d]
        n, K1, d = Z.shape
        F = Z.reshape(-1, d) @ enc.W.T + enc.b  # standardized-rate means
        out = {"bank_posterior": np.zeros((n_draws, n * K1, d)), "ridge": np.zeros((n_draws, n * K1, d))}
        for t in range(n_draws):
            R = F + rng.standard_normal(F.shape) * np.sqrt(enc.var)
            out["bank_posterior"][t] = _posterior(enc, R, bank, lp, kappa)
            out["ridge"][t] = rmodel.predict(enc.sx.inverse_transform(R))
        res[f] = {}
        for m, P in out.items():
            P = P.reshape(n_draws, n, K1, d)
            Dtrue = Z[:, 1:] - Z[:, :1]  # [n, K, d]
            Dpred = P[:, :, 1:] - P[:, :, :1]  # paired draws: same noise realisation index t
            Dpred_avg = P[:, :, 1:].mean(0) - P[:, :, :1].mean(0)  # noise-averaged
            frac = (Dpred * Dtrue[None]).sum(-1) / (Dtrue ** 2).sum(-1)[None]
            frac_avg = (Dpred_avg * Dtrue).sum(-1) / (Dtrue ** 2).sum(-1)
            # identification of k within the chain: is pred_k nearest to z_k among z_0..z_K?
            dist = ((P[:, :, :, None] - Z[None, :, None]) ** 2).sum(-1)  # [t, n, K1(pred), K1(true)]
            ident = (dist.argmin(-1) == np.arange(K1)[None, None]).mean((0, 1))
            res[f][m] = {
                "frac_recovered_single": np.median(frac, (0, 1)).tolist(),
                "frac_recovered_avg": np.median(frac_avg, 0).tolist(),
                "cos_edit_avg": np.median(cos(Dpred_avg, Dtrue), 0).tolist(),
                "cos_target": np.median(cos(P - mu, Z[None] - mu), (0, 1)).tolist(),
                "chain_ident": ident.tolist(),
            }
    return res, float(kappa)


def leave_region_out(X, Y, bank, scale, radii_units=(0, 1, 1.5, 2, 2.5, 3, 4)):
    res = {"radius_units": list(radii_units), "rows": []}
    d_true = G.knn(Y, bank, 1)[1][:, 0] / scale
    res["true_nn_units_median"] = float(np.median(d_true))
    for r in radii_units:
        P, _, hp, _ = cv_bank_posterior(X, Y, [bank], exclude_radius=r * scale if r else None)
        m = metrics(P, Y, bank)
        removed = float(np.mean([(np.linalg.norm(bank - y, axis=1) < r * scale).sum() for y in Y])) if r else 0.0
        m.update(radius_units=r, mean_removed=removed, frac_targets_affected=float((d_true < r).mean()),
                 kappa=[h[0] for h in hp])
        res["rows"].append(m)
        print(f"leave-region-out r={r}: cos={m['cos']:.3f} r2={m['r2_tot']:.3f} top5={m['top5']:.3f} "
              f"removed/target={removed:.1f}", flush=True)
    res["ridge"] = metrics(np.load(f"{BENCH}/preds_{TARGET}_ridge.npy").astype(np.float64), Y, bank)
    return res


# ----------------------------------------------------------------------------- plots

def plots(U, MK, hull_ref, SIM, LRO, scale_info):
    fams = list(U)
    cm = plt.get_cmap("tab10")
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for i, f in enumerate(fams):
        kk = np.arange(1, U[f]["K"] + 1)
        ax[0].plot(kk, U[f]["size_nn_units"], "o-", c=cm(i % 10), label=f)
        ax[1].plot(kk, U[f]["cos_dk_d1"], "o-", c=cm(i % 10))
        ax[2].plot(kk, U[f]["arith_rel_err"], "o-", c=cm(i % 10))
    ax[0].axhline(scale_info["err_in_nn"], c="k", ls="--", lw=0.8, label="bank-posterior error")
    ax[0].axhline(scale_info["stim_in_nn"], c="gray", ls=":", lw=0.8, label="nearest other stimulus")
    ax[0].set_ylabel("||z_k - z_0|| (bank NN units)")
    ax[1].set_ylabel("cos(d_k, d_1)")
    ax[2].set_ylabel("||z_k - (z_0 + mean d_k)|| / ||d_k||")
    ax[2].axhline(1, c="k", ls=":", lw=0.8)
    for a in ax:
        a.set_xlabel("k (units)")
    ax[0].legend(fontsize=7)
    ax[0].set_title("size of k units")
    ax[1].set_title("does the edit keep its direction?")
    ax[2].set_title("vector arithmetic error (<1 helps)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/q2_unit_size_linearity.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for i, f in enumerate(fams):
        kk = MK[f]["k"]
        for kind, ls in (("real", "-"), ("arith", "--")):
            for a, key in zip(ax, ("precision", "nn_ratio", "hull_units")):
                a.plot(kk, MK[f][kind][key], ls, marker="o", ms=3, c=cm(i % 10), label=f if kind == "real" else None)
    ax[2].axhline(hull_ref, c="k", ls=":", lw=0.8, label="500 real stimuli")
    ax[1].axhline(1, c="k", ls=":", lw=0.8)
    for a, t in zip(ax, ("precision vs bank", "NN distance / real stimuli", "convex-hull residual (NN units)")):
        a.set_title(t + "\nsolid = real edited image, dashed = z_0 + k mean d_1", fontsize=9)
        a.set_xlabel("k (units)")
    ax[0].legend(fontsize=7)
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{FIG}/q2_manifold_vs_k.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for i, f in enumerate(fams):
        kk = np.arange(1, U[f]["K"] + 1)
        for m, ls in (("bank_posterior", "-"), ("ridge", "--")):
            ax[0].plot(kk, SIM[f][m]["frac_recovered_avg"], ls, marker="o", ms=3, c=cm(i % 10),
                       label=f if m == "bank_posterior" else None)
            ax[1].plot(kk, SIM[f][m]["cos_edit_avg"], ls, marker="o", ms=3, c=cm(i % 10))
            ax[2].plot(np.arange(U[f]["K"] + 1), SIM[f][m]["chain_ident"], ls, marker="o", ms=3, c=cm(i % 10))
    ax[0].axhline(1, c="k", ls=":", lw=0.8)
    ax[0].set_title("fraction of the edit recovered\n(solid = bank posterior, dashed = ridge; noise-averaged)", fontsize=9)
    ax[1].set_title("cos(decoded edit, true edit), noise-averaged", fontsize=9)
    ax[2].set_title("identify k within the chain (single draw)", fontsize=9)
    for a in ax:
        a.set_xlabel("k (units)")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{FIG}/q2_simulated_decoding.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    r = [row["radius_units"] for row in LRO["rows"]]
    for a, key in zip(ax, ("cos", "r2_tot", "top5")):
        a.plot(r, [row[key] for row in LRO["rows"]], "o-", label="bank posterior, region removed")
        a.axhline(LRO["ridge"][key], c="C3", ls="--", label="ridge (unaffected)")
        a.axvline(LRO["true_nn_units_median"], c="gray", ls=":", label="median true->bank NN distance")
        a.set_xlabel("exclusion radius around true target (bank NN units)")
        a.set_title(key)
    ax[0].legend(fontsize=7)
    fig.suptitle("Leave-region-out: bank posterior on targets with no close bank neighbours (real neural data)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/q2_leave_region_out.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    X, *_ = load_neural()
    Y, bank = load_target(TARGET)
    scale = G.bank_scale(bank)
    rad = G.knn_radius(bank, 5)
    P_bp = np.load(f"{BENCH}/preds_{TARGET}_bayes_bank_posterior.npy").astype(np.float64)
    err_scale = float(np.median(np.linalg.norm(P_bp - Y, axis=1)))
    stim_scale = float(np.median(G.knn(Y, Y, 1, exclude_self=True)[1][:, 0]))
    scale_info = {"bank_nn": scale, "err": err_scale, "stim_nn": stim_scale,
                  "err_in_nn": err_scale / scale, "stim_in_nn": stim_scale / scale}
    print(scale_info)
    z, base, fam, k = load_edits()
    # sanity: re-embedded originals vs the saved val latents
    orig = fam == "orig"
    scale_info["reembed_err_nn_units"] = float(np.median(np.linalg.norm(z[orig] - Y[base[orig]], axis=1)) / scale)
    C = chains(z, base, fam, k)
    U = unit_stats(C, scale, err_scale, stim_scale)
    for f, u in U.items():
        print(f, {kk: (np.round(v, 2).tolist() if isinstance(v, list) else v) for kk, v in u.items()})
    MK, hull_ref = manifold_vs_k(C, bank, Y, scale, rad)
    SIM, kappa = simulate_decoding(C, X, Y, bank)
    LRO = leave_region_out(X, Y, bank, scale)
    json.dump({"scales": scale_info, "units": U, "manifold_vs_k": MK, "hull_ref_real_units": hull_ref,
               "simulated_decoding": SIM, "sim_kappa": kappa, "leave_region_out": LRO},
              open(f"{OUT}/q2_results.json", "w"), indent=1)
    plots(U, MK, hull_ref, SIM, LRO, scale_info)
