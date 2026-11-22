"""Empirical-bank posterior (decoder_bench.methods.bayes_bank), generalized:

  - any number of candidate banks with a mixture prior: candidate c in bank g has prior weight
    pi_g / N_g, pi chosen by inner CV jointly with the likelihood temperature kappa
  - leave-region-out: candidates within `exclude_radius` of a test item's TRUE embedding are
    removed (also inside the inner CV, so kappa is tuned under the same handicap)
  - returns per-item posterior diagnostics (ESS, support radius, max weight)

With one bank, include_train=True and no exclusion it reproduces bayes_bank exactly.
"""

import hashlib
import itertools
import os

import joblib

import numpy as np
from scipy.special import logsumexp
from sklearn.model_selection import KFold

from analysis.decoder_bench.common import REPO, folds
from analysis.decoder_bench.methods import Encoder, _cos_score

KAPPAS = (1, 2, 4, 8, 16, 32, 64)

# The encoding model depends only on the training rows, not on the bank, so it is cached
# (in memory and on disk) and shared by every bank condition.
_ENC = {}
_ENC_PATH = os.path.join(REPO, "outputs", "bank_geometry", "encoder_cache.joblib")


class _Lite:
    """What the posterior needs from an Encoder (W, b, var, scaler) without the sklearn models."""

    def __init__(self, e):
        self.W, self.b, self.var, self.sx = e.W, e.b, e.var, e.sx

    r = Encoder.r
    loglik = Encoder.loglik


def get_encoder(Ya, Xa):
    global _ENC
    if not _ENC and os.path.exists(_ENC_PATH):
        _ENC = joblib.load(_ENC_PATH)
    key = hashlib.sha1(np.ascontiguousarray(Ya).tobytes() + np.ascontiguousarray(Xa).tobytes()).hexdigest()
    if key not in _ENC:
        _ENC[key] = _Lite(Encoder(Ya, Xa))
        os.makedirs(os.path.dirname(_ENC_PATH), exist_ok=True)
        joblib.dump(_ENC, _ENC_PATH)
    return _ENC[key]


def _posterior(enc, Rte, cand, logprior, kappa, Yte=None, exclude_radius=None, diag=False):
    L = enc.loglik(Rte, cand, kappa) + logprior[None]
    if exclude_radius:
        d2 = (Yte ** 2).sum(1)[:, None] - 2 * Yte @ cand.T + (cand ** 2).sum(1)[None]
        L = np.where(d2 < exclude_radius ** 2, -np.inf, L)
    w = np.exp(L - logsumexp(L, axis=1, keepdims=True))
    mean = w @ cand
    if not diag:
        return mean
    # support radius: sqrt(E_w ||c - mean||^2) = sqrt(E_w||c||^2 - ||mean||^2)
    rad = np.sqrt(((w @ (cand ** 2).sum(1)) - (mean ** 2).sum(1)).clip(0))
    return mean, {"ess": 1 / (w ** 2).sum(1), "radius": rad, "wmax": w.max(1), "argmax": w.argmax(1),
                  "w": w}


def _cands(banks, Ytr, include_train, pi):
    groups = [np.vstack([banks[0], Ytr]) if include_train else banks[0]] + list(banks[1:])
    cand = np.vstack(groups)
    pis = (1.0,) if len(groups) == 1 else (pi, 1 - pi)
    with np.errstate(divide="ignore"):
        lp = np.concatenate([np.full(len(g), np.log(p) - np.log(len(g))) for g, p in zip(groups, pis)])
    return cand, lp


def bank_posterior(Xtr, Ytr, Xte, banks, include_train=True, grid_kappa=KAPPAS, grid_pi=(None,),
                   Yte=None, exclude_radius=None, diag=False, inner_seed=2):
    """fit on (Xtr, Ytr), predict Xte. banks: list of [N_g, 50] arrays (1 or 2)."""
    if len(banks) == 1:
        grid_pi = (None,)
    grid = list(itertools.product(grid_kappa, grid_pi))
    preds = {g: np.zeros_like(Ytr) for g in grid}
    for a, b in KFold(5, shuffle=True, random_state=inner_seed).split(Xtr):
        enc = get_encoder(Ytr[a], Xtr[a])
        R = enc.r(Xtr[b])
        for pi in grid_pi:
            cand, lp = _cands(banks, Ytr[a], include_train, pi)
            for k in grid_kappa:
                preds[(k, pi)][b] = _posterior(enc, R, cand, lp, k, Ytr[b], exclude_radius)
    mu = Ytr.mean(0)
    scores = {g: _cos_score(np.nan_to_num(preds[g], nan=0.0), Ytr, mu) for g in grid}
    best = max(scores, key=scores.get)
    enc = get_encoder(Ytr, Xtr)
    cand, lp = _cands(banks, Ytr, include_train, best[1])
    out = _posterior(enc, enc.r(Xte), cand, lp, best[0], Yte, exclude_radius, diag)
    return out, best, cand


def cv_bank_posterior(X, Y, banks, diag=False, keep_w=False, **kw):
    """10-fold OOF predictions (decoder_bench folds). Returns preds, diag arrays, chosen (kappa, pi) per fold."""
    P = np.zeros_like(Y)
    D = {}
    hp = []
    W = []
    for tr, te in folds(len(Y)):
        res, best, cand = bank_posterior(X[tr], Y[tr], X[te], banks, Yte=Y[te], diag=diag, **kw)
        hp.append(best)
        if diag:
            P[te], d = res
            for key in ("ess", "radius", "wmax"):
                D.setdefault(key, np.zeros(len(Y)))[te] = d[key]
            if keep_w:
                W.append((te, tr, d["w"]))
        else:
            P[te] = res
    return P, D, hp, W
