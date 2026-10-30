"""Shared data loading, metrics and cross-validation for the firing-rate -> DINO decoder bench.

Data
----
X      [500, 386]  mean firing rate (Hz) per stimulus and cell, in the 267 ms window starting
                   at each cell's response latency (`responses{1,:}` of MergedITCells_500Stim_AxisTuned).
trials list of 386 arrays, per cell: trial rates [n_trials] and stimulus index [n_trials],
                   rebuilt from the rasters with the same window (matches X to float precision).
expvar [386]       split-half explainable variance, mean over 1000 reps (ExV_500Stim_1000Reps).
Y[name]  [500, 50] z50 targets for the 500 stimuli (rows in sorted 500Stimuli order = image IDs 1..500).
BANK[name] [15901, 50] the same z50 projection of the 15901 unpaired stimulus-set images
                   (disjoint from the 500: no byte-identical files, no identical latents).
"""

import os
from functools import lru_cache

import h5py
import numpy as np
from sklearn.model_selection import KFold

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(REPO, "data")
NEURAL = os.path.join(DATA, "forJhan", "data")
OUT = os.path.join(REPO, "outputs", "decoder_bench")
os.makedirs(OUT, exist_ok=True)

TARGETS = {
    "cls": "stimuli_dinov2_vitb14_cls_pca_latents/{split}_latents_z50.npy",
    "meanpatch": "stimuli_dinov2_vitb14_meanpatch_pca_latents/{split}_latents_z50.npy",
    # other spaces, only for the target-space comparison
    "clip_b32": "stimuli_clip_vit_b32_pca_latents/{split}_latents_z50.npy",
    "alexnet_fc6": "stimuli_alexnet_fc6_pca_latents/{split}_latents.npy",
}
WINDOW_MS = 267  # ceil(266.625), as in Utils.computeExplainableVariance
OFFSET_MS = 170  # stimulus onset column of the raster


@lru_cache(None)
def load_neural():
    cache = os.path.join(OUT, "neural_cache.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return d["X"], list(d["trials"]), d["expvar"], d["session"], d["patient"]
    path = os.path.join(NEURAL, "MergedITCells_500Stim_AxisTuned.mat")
    recall = os.path.join(NEURAL, "Recall_Task", "P76CS", "ReScreenRecall_Session_1_20210917", "PsthandResponses.mat")
    with h5py.File(recall, "r") as g:
        recall_order = g["order"][()].ravel().astype(int) - 1
    with h5py.File(path, "r") as f:
        R, P, sc = f["responses"], f["psths"], f["strctCells"]
        s = lambda ref: "".join(map(chr, f[ref][()].ravel()))
        n = R.shape[1]
        X = np.stack([f[R[0, i]][()].ravel() for i in range(n)], axis=1)
        session = np.array([s(sc["SessionID"][i, 0]) for i in range(n)])
        patient = np.array([s(sc["PatientID"][i, 0]) for i in range(n)])
        trials = []
        for i in range(n):
            raster = f[P[0, i]][()].T  # [n_trials, 702] binary, 1 ms bins
            lat = int(round(f[R[1, i]][()].item()))
            if session[i] == "P76CSRec_ReScreen":
                order = recall_order
            else:
                order = np.repeat(np.arange(500), raster.shape[0] // 500)
            start = OFFSET_MS + lat - 1
            rate = raster[:, start:start + WINDOW_MS].sum(1) / (WINDOW_MS / 1000)
            trials.append(np.stack([rate, order]))
    with h5py.File(os.path.join(NEURAL, "ExV_500Stim_1000Reps_AxisTunedCells.mat"), "r") as f:
        expvar = f["explainableVar"][()].mean(0)
    trials_arr = np.empty(len(trials), dtype=object)
    trials_arr[:] = trials
    np.savez(cache, X=X, trials=trials_arr, expvar=expvar, session=session, patient=patient)
    return X, trials, expvar, session, patient


@lru_cache(None)
def load_target(name):
    y = np.load(os.path.join(DATA, TARGETS[name].format(split="val"))).astype(np.float64)
    bank = np.load(os.path.join(DATA, TARGETS[name].format(split="train"))).astype(np.float64)
    return y, bank


def trial_split_means(trials, rng, n_stim=500):
    """Per cell, split each stimulus's trials randomly in two halves; return the two
    [n_stim, n_cells] mean matrices (odd trial counts: the extra trial goes to half A)."""
    A = np.zeros((n_stim, len(trials)))
    B = np.zeros_like(A)
    for c, (rate, order) in enumerate(trials):
        order = order.astype(int)
        for k in range(n_stim):
            r = rng.permutation(rate[order == k])
            h = (len(r) + 1) // 2
            A[k, c] = r[:h].mean()
            B[k, c] = r[h:].mean() if len(r) > h else r[:h].mean()
    return A, B


def trial_lists(trials, n_stim=500):
    """[n_cells][n_stim] -> array of trial rates."""
    out = []
    for rate, order in trials:
        order = order.astype(int)
        out.append([rate[order == k] for k in range(n_stim)])
    return out


# ----------------------------------------------------------------------------- metrics

def _center_sim(pred, true, ref_mean):
    a = pred - ref_mean
    b = true - ref_mean
    a = a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)
    b = b / np.linalg.norm(b, axis=1, keepdims=True).clip(1e-12)
    return a @ b.T  # [n_pred, n_cand]


def metrics(pred, true, bank=None, ref_mean=None):
    """All predictions are out-of-fold. Identification: candidates are the 500 true
    embeddings, similarity = cosine after subtracting the 500-stimulus mean."""
    ref_mean = true.mean(0) if ref_mean is None else ref_mean
    sst = ((true - true.mean(0)) ** 2).sum(0)
    sse = ((true - pred) ** 2).sum(0)
    S = _center_sim(pred, true, ref_mean)
    diag = np.diag(S)
    # 0 = correct is best; ties count half (a constant prediction scores chance, not perfect)
    rank = (S > diag[:, None]).sum(1) + 0.5 * ((S == diag[:, None]).sum(1) - 1)
    out = {
        "r2_dim": float(np.mean(1 - sse / sst)),  # mean per-dim R^2
        "r2_tot": float(1 - sse.sum() / sst.sum()),  # variance-weighted R^2
        "cos": float(diag.mean()),  # centered cosine to true
        "top1": float((rank == 0).mean()),
        "top5": float((rank < 5).mean()),
        "pair2afc": float(1 - rank.mean() / (len(true) - 1)),
        "spread": float(((pred - pred.mean(0)) ** 2).sum() / sst.sum()),  # pred var / true var
    }
    if bank is not None:
        out["nn_bank"] = float(np.median(nn_dist(pred, bank)) / np.median(nn_dist(true, bank)))
    return out


def nn_dist(a, bank):
    d = (a ** 2).sum(1)[:, None] - 2 * a @ bank.T + (bank ** 2).sum(1)[None]
    return np.sqrt(d.min(1).clip(0))


def per_dim_r2(pred, true):
    return 1 - ((true - pred) ** 2).sum(0) / ((true - true.mean(0)) ** 2).sum(0)


# ----------------------------------------------------------------------------- CV

def folds(n=500, k=10, seed=0):
    return list(KFold(k, shuffle=True, random_state=seed).split(np.arange(n)))


def cv_predict(fit_predict, X, Y, k=10, seed=0, **kw):
    """fit_predict(Xtr, Ytr, Xte, tr_idx=..., te_idx=..., **kw) -> Yte_pred. Returns oof preds."""
    pred = np.full(Y.shape, np.nan)
    for tr, te in folds(len(Y), k, seed):
        pred[te] = fit_predict(X[tr], Y[tr], X[te], tr_idx=tr, te_idx=te, **kw)
    return pred


def fmt(name, m):
    keys = ["r2_dim", "r2_tot", "cos", "top1", "top5", "pair2afc", "spread", "nn_bank"]
    return f"{name:<44s} " + " ".join(f"{k}={m[k]:.3f}" for k in keys if k in m)
