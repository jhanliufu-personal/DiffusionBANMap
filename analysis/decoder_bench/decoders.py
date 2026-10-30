"""Decoders: firing rates [n, n_cells] -> z50 target [n, 50]. Each is fit_predict(Xtr, Ytr, Xte, **kw)."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-1, 6, 30)


def ridge_model(n_pcs=None, per_target=True):
    steps = [StandardScaler()]
    if n_pcs:
        steps.append(PCA(n_pcs, random_state=0))
    steps.append(RidgeCV(alphas=ALPHAS, alpha_per_target=per_target))
    return make_pipeline(*steps)


def ridge(Xtr, Ytr, Xte, n_pcs=None, per_target=True, **_):
    return ridge_model(n_pcs, per_target).fit(Xtr, Ytr).predict(Xte)


def inner_oof(fit_predict, Xtr, Ytr, k=5, seed=1, **kw):
    """Out-of-fold predictions on the training set (for calibrating post-hoc steps)."""
    pred = np.zeros_like(Ytr)
    for a, b in KFold(k, shuffle=True, random_state=seed).split(Xtr):
        pred[b] = fit_predict(Xtr[a], Ytr[a], Xtr[b], **kw)
    return pred
