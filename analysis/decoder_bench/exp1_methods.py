"""Parts 1-4: methods vs the ridge baseline, 10-fold CV over the 500 stimuli.

python -m analysis.decoder_bench.exp1_methods <target> [method ...]
Out-of-fold predictions go to outputs/decoder_bench/preds_<target>_<method>.npy (original stimulus order).
"""

import json
import os
import sys
import time
from functools import partial

import numpy as np

from . import methods as M
from .common import OUT, cv_predict, fmt, load_neural, load_target, metrics
from .decoders import ridge

target = sys.argv[1]
X, trials, expvar, _, _ = load_neural()
Y, bank = load_target(target)
TT = M.trial_tensor(trials)
good = expvar > 0.1


def on_cells(fn, cells):
    return lambda Xtr, Ytr, Xte, **kw: fn(Xtr[:, cells], Ytr, Xte[:, cells], **kw)


def on_transform(fn, f):
    return lambda Xtr, Ytr, Xte, **kw: fn(f(Xtr), Ytr, f(Xte), **kw)


METHODS = {
    # baseline + preprocessing (item 12)
    "ridge": (ridge, {}),
    "ridge_shared_alpha": (partial(ridge, per_target=False), {}),
    "ridge_sqrt": (on_transform(ridge, np.sqrt), {}),
    "ridge_ev>0.1": (on_cells(ridge, good), {}),
    # geometry, post hoc (items 1, 5)
    "rescale_per_dim": (partial(M.rescale, mode="per_dim"), {}),
    "rescale_global": (partial(M.rescale, mode="global"), {}),
    "rescale_norm": (partial(M.rescale, mode="norm"), {}),
    # bank snapping (item 2)
    "snap_bank": (partial(M.knn_snap, bank=bank, blend=False), {}),
    "snap_bank_blend": (partial(M.knn_snap, bank=bank, blend=True), {}),
    # Bayesian inversion of an encoding model (item 3)
    "bayes_gauss_trainprior": (partial(M.bayes_gauss, prior="train"), {}),
    "bayes_gauss_bankprior": (partial(M.bayes_gauss, bank=bank, prior="bank"), {}),
    "bayes_gmm32_bank": (partial(M.bayes_gmm, bank=bank, n_comp=32), {}),
    "bayes_bank_posterior": (partial(M.bayes_bank, bank=bank), {}),
    "bayes_bank_posterior_no_train": (partial(M.bayes_bank, bank=bank, include_train=False), {}),
    "bayes_bank_posterior_map": (partial(M.bayes_bank, bank=bank, grid_kappa=(1e-3,)), {}),
    # shared output structure (item 6)
    "rrr": (M.rrr, {}),
    "pls": (M.pls, {}),
    # nonlinear / ensembles / losses (items 7, 8)
    "kernel_ridge_rbf": (M.kernel_ridge, {}),
    "kernel_ridge_rbf_widegrid": (partial(M.kernel_ridge, grid_gamma=(0.3, 1, 3), grid_alpha=(0.001, 0.003, 0.01, 0.03)), {}),
    "bagged_ridge": (M.bagged_ridge, {}),
    "linear_mse+clip": (partial(M.mlp_contrastive, linear=True), {}),
    "mlp_mse+clip": (M.mlp_contrastive, {}),
    "mlp_mse_only": (partial(M.mlp_contrastive, lam_clip=0.0), {}),
    # trial-level data (item 9)
    "pseudo_trial_k1": (partial(M.pseudo_trial_ridge, ttensor=TT, k_trials=1), {}),
    "pseudo_trial_k2": (partial(M.pseudo_trial_ridge, ttensor=TT, k_trials=2), {}),
}

names = sys.argv[2:] or list(METHODS)
res_path = f"{OUT}/exp1_{target}.json"
results = json.load(open(res_path)) if os.path.exists(res_path) else {}
for name in names:
    fn, kw = METHODS[name]
    t0 = time.time()
    lasts = []

    def wrapped(Xtr, Ytr, Xte, **k):
        out = fn(Xtr, Ytr, Xte, **k)
        f = getattr(fn, "func", fn)
        if hasattr(f, "last"):
            lasts.append(f.last)
        return out

    P = cv_predict(wrapped, X, Y)
    np.save(f"{OUT}/preds_{target}_{name}.npy", P.astype(np.float32))
    m = metrics(P, Y, bank)
    m["hparams"] = [str(x) for x in lasts]
    results[name] = m
    print(fmt(f"[{target}] {name}", m), f"({time.time() - t0:.0f}s)", ("hp: " + " ".join(m["hparams"])) if lasts else "", flush=True)
    json.dump(results, open(res_path, "w"), indent=1)
