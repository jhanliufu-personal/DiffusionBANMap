"""Decoding methods beyond plain ridge. Every function is fit_predict(Xtr, Ytr, Xte, **kw) -> Yte_pred
and is self-contained (any hyperparameter it tunes is tuned by inner CV on the training fold only)."""

import numpy as np
from scipy.special import logsumexp
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from .decoders import ALPHAS, inner_oof, ridge


def _cos_score(P, Y, mu):
    a, b = P - mu, Y - mu
    return np.mean((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)).clip(1e-12))


def _r2_tot(P, Y):
    return 1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()


# ----------------------------------------------------------------------------- post-hoc geometry

def rescale(Xtr, Ytr, Xte, mode="per_dim", base=ridge, **kw):
    """Undo ridge shrinkage: scale predictions about the training mean so their spread matches
    the targets'. The scale is measured on inner out-of-fold predictions (in-sample predictions
    are less shrunk than test predictions)."""
    oof = inner_oof(base, Xtr, Ytr)
    mu = Ytr.mean(0)
    pred = base(Xtr, Ytr, Xte) - mu
    if mode == "per_dim":
        s = Ytr.std(0) / oof.std(0).clip(1e-9)
    elif mode == "global":
        s = np.sqrt(((Ytr - mu) ** 2).sum() / ((oof - oof.mean(0)) ** 2).sum())
    elif mode == "norm":  # keep the direction, set every prediction's norm to the median target norm
        return mu + pred / np.linalg.norm(pred, axis=1, keepdims=True) * np.median(np.linalg.norm(Ytr - mu, axis=1))
    return mu + pred * s


def snap_weights(P, bank, mu, k, tau):
    a = P - mu
    b = bank - mu
    S = (a / np.linalg.norm(a, axis=1, keepdims=True)) @ (b / np.linalg.norm(b, axis=1, keepdims=True)).T
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k] if k < bank.shape[0] else np.tile(np.arange(len(bank)), (len(P), 1))
    s = np.take_along_axis(S, idx, 1)
    w = np.exp((s - s.max(1, keepdims=True)) / tau)
    w /= w.sum(1, keepdims=True)
    return idx, w


def snap(P, bank, mu, k, tau):
    idx, w = snap_weights(P, bank, mu, k, tau)
    return (bank[idx] * w[..., None]).sum(1)


def knn_snap(Xtr, Ytr, Xte, bank=None, base=ridge, include_train=True, blend=True,
             grid_k=(1, 5, 20, 50, 200), grid_tau=(0.01, 0.03, 0.1), grid_lam=(0, 0.25, 0.5, 0.75, 1.0),
             score="cos", **kw):
    """Predict with `base`, then replace the prediction by a soft average of its k nearest
    (centered cosine) neighbours in the unpaired embedding bank, optionally blended with the
    variance-rescaled prediction. k, tau, blend weight chosen by inner CV."""
    B = np.vstack([bank, Ytr]) if include_train else bank
    mu = Ytr.mean(0)
    oof = inner_oof(base, Xtr, Ytr)
    s = Ytr.std(0) / oof.std(0).clip(1e-9)
    f = _cos_score if score == "cos" else (lambda P, Y, mu: _r2_tot(P, Y))
    best = (-np.inf, None)
    for k in grid_k:
        for tau in grid_tau:
            if k == 1 and tau != grid_tau[0]:
                continue
            # inner: bank must not contain the inner-held-out row itself
            Sn = np.zeros_like(oof)
            for a, b in KFold(5, shuffle=True, random_state=1).split(Xtr):
                Bi = np.vstack([bank, Ytr[a]]) if include_train else bank
                Sn[b] = snap(oof[b], Bi, mu, k, tau)
            for lam in (grid_lam if blend else (1.0,)):
                P = lam * Sn + (1 - lam) * (mu + (oof - mu) * s)
                sc = f(P, Ytr, mu)
                if sc > best[0]:
                    best = (sc, (k, tau, lam))
    k, tau, lam = best[1]
    knn_snap.last = best[1]
    pred = base(Xtr, Ytr, Xte)
    return lam * snap(pred, B, mu, k, tau) + (1 - lam) * (mu + (pred - mu) * s)


# ----------------------------------------------------------------------------- Bayesian inversion

class Encoder:
    """Per-cell linear encoding model r = W z + b + noise, noise variance from inner OOF residuals."""

    def __init__(self, Ytr, Xtr):
        self.sx = StandardScaler().fit(Xtr)
        R = self.sx.transform(Xtr)
        self.m = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(Ytr, R)
        oof = np.zeros_like(R)
        for a, b in KFold(5, shuffle=True, random_state=1).split(Ytr):
            oof[b] = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(Ytr[a], R[a]).predict(Ytr[b])
        self.var = ((R - oof) ** 2).mean(0)
        self.W = self.m.coef_  # [cells, 50]
        self.b = self.m.intercept_

    def r(self, X):
        return self.sx.transform(X)

    def loglik(self, Rte, Z, kappa=1.0):
        """[n_te, n_cand] Gaussian log-likelihood (diag noise, inflated by kappa)."""
        F = Z @ self.W.T + self.b
        iv = 1 / (self.var * kappa)
        return -0.5 * ((Rte ** 2) @ iv[:, None] - 2 * (Rte * iv) @ F.T + ((F ** 2) @ iv)[None])


def _post_bank(enc, Rte, cand, kappa):
    L = enc.loglik(Rte, cand, kappa)
    w = np.exp(L - logsumexp(L, axis=1, keepdims=True))
    return w @ cand


def bayes_bank(Xtr, Ytr, Xte, bank=None, include_train=True, grid_kappa=(1, 2, 4, 8, 16, 32, 64), **kw):
    """Empirical prior = the unpaired bank (+ training stimuli). Posterior mean over candidates
    under the encoding model. kappa (likelihood temperature) chosen by inner CV."""
    best = (-np.inf, None)
    preds = {k: np.zeros_like(Ytr) for k in grid_kappa}
    for a, b in KFold(5, shuffle=True, random_state=2).split(Xtr):
        enc = Encoder(Ytr[a], Xtr[a])
        cand = np.vstack([bank, Ytr[a]]) if include_train else bank
        for k in grid_kappa:
            preds[k][b] = _post_bank(enc, enc.r(Xtr[b]), cand, k)
    for k in grid_kappa:
        sc = _cos_score(preds[k], Ytr, Ytr.mean(0))
        if sc > best[0]:
            best = (sc, k)
    bayes_bank.last = best[1]
    enc = Encoder(Ytr, Xtr)
    cand = np.vstack([bank, Ytr]) if include_train else bank
    return _post_bank(enc, enc.r(Xte), cand, best[1])


def bayes_gauss(Xtr, Ytr, Xte, bank=None, prior="train", grid_kappa=(0.25, 0.5, 1, 2, 4, 8, 16), **kw):
    """Gaussian prior (fit on training stimuli or on the bank) + linear encoder -> closed-form
    posterior mean. A linear decoder whose regularizer is the target covariance."""
    P0 = Ytr if prior == "train" else bank

    def fit(Ya, Xa, kappa):
        enc = Encoder(Ya, Xa)
        mu0 = P0.mean(0) if prior != "train" else Ya.mean(0)
        S0 = np.cov((P0 if prior != "train" else Ya).T)
        iS0 = np.linalg.inv(S0)
        WtD = enc.W.T / (enc.var * kappa)
        A = np.linalg.inv(WtD @ enc.W + iS0)
        return lambda Xb: (A @ (WtD @ (enc.r(Xb) - enc.b).T + (iS0 @ mu0)[:, None])).T

    best = (-np.inf, None)
    for kappa in grid_kappa:
        P = np.zeros_like(Ytr)
        for a, b in KFold(5, shuffle=True, random_state=2).split(Xtr):
            P[b] = fit(Ytr[a], Xtr[a], kappa)(Xtr[b])
        sc = _cos_score(P, Ytr, Ytr.mean(0))
        if sc > best[0]:
            best = (sc, kappa)
    bayes_gauss.last = best[1]
    return fit(Ytr, Xtr, best[1])(Xte)


_GMM_CACHE = {}


def bayes_gmm(Xtr, Ytr, Xte, bank=None, n_comp=32, grid_kappa=(1, 2, 4, 8, 16), **kw):
    """GMM prior fit on the unpaired bank; posterior = mixture of per-component Gaussian
    posteriors; returns the posterior mean."""
    key = (id(bank), n_comp)
    if key not in _GMM_CACHE:
        _GMM_CACHE[key] = GaussianMixture(n_comp, covariance_type="full", reg_covar=1e-3, random_state=0).fit(bank)
    g = _GMM_CACHE[key]

    def fit(Ya, Xa, kappa):
        enc = Encoder(Ya, Xa)
        D = enc.var * kappa
        WtD = enc.W.T / D
        parts = []
        for k in range(n_comp):
            iS = np.linalg.inv(g.covariances_[k])
            A = np.linalg.inv(WtD @ enc.W + iS)
            # marginal of r under component k: N(W mu_k + b, W S_k W^T + D)
            C = enc.W @ g.covariances_[k] @ enc.W.T + np.diag(D)
            Ci = np.linalg.inv(C)
            _, logdet = np.linalg.slogdet(C)
            parts.append((A, iS @ g.means_[k], enc.W @ g.means_[k] + enc.b, Ci, logdet))

        def predict(Xb):
            R = enc.r(Xb)
            logw, means = [], []
            for k, (A, iSm, m_r, Ci, logdet) in enumerate(parts):
                d = R - m_r
                logw.append(np.log(g.weights_[k]) - 0.5 * (np.einsum("ij,jk,ik->i", d, Ci, d) + logdet))
                means.append((A @ (WtD @ (R - enc.b).T + iSm[:, None])).T)
            logw = np.array(logw)
            w = np.exp(logw - logsumexp(logw, axis=0))
            return np.einsum("kn,knd->nd", w, np.array(means))
        return predict

    best = (-np.inf, None)
    for kappa in grid_kappa:
        P = np.zeros_like(Ytr)
        for a, b in KFold(5, shuffle=True, random_state=2).split(Xtr):
            P[b] = fit(Ytr[a], Xtr[a], kappa)(Xtr[b])
        sc = _cos_score(P, Ytr, Ytr.mean(0))
        if sc > best[0]:
            best = (sc, kappa)
    bayes_gmm.last = best[1]
    return fit(Ytr, Xtr, best[1])(Xte)


# ----------------------------------------------------------------------------- shared output structure

def rrr(Xtr, Ytr, Xte, grid_rank=(3, 5, 10, 20, 30, 50), **kw):
    """Reduced-rank ridge: project ridge coefficients onto the top-r PCs of the fitted values."""
    def fit(Xa, Ya, r):
        sx = StandardScaler().fit(Xa)
        m = RidgeCV(alphas=ALPHAS, alpha_per_target=False).fit(sx.transform(Xa), Ya)
        F = m.predict(sx.transform(Xa))
        mf = F.mean(0)
        V = np.linalg.svd(F - mf, full_matrices=False)[2][:r].T
        return lambda Xb: mf + ((m.predict(sx.transform(Xb)) - mf) @ V) @ V.T
    best = (-np.inf, None)
    for r in grid_rank:
        P = np.zeros_like(Ytr)
        for a, b in KFold(5, shuffle=True, random_state=1).split(Xtr):
            P[b] = fit(Xtr[a], Ytr[a], r)(Xtr[b])
        sc = _r2_tot(P, Ytr)
        if sc > best[0]:
            best = (sc, r)
    rrr.last = best[1]
    return fit(Xtr, Ytr, best[1])(Xte)


def pls(Xtr, Ytr, Xte, grid_n=(2, 5, 10, 20, 40), **kw):
    best = (-np.inf, None)
    for n in grid_n:
        P = np.zeros_like(Ytr)
        for a, b in KFold(5, shuffle=True, random_state=1).split(Xtr):
            P[b] = PLSRegression(n, scale=True).fit(Xtr[a], Ytr[a]).predict(Xtr[b])
        sc = _r2_tot(P, Ytr)
        if sc > best[0]:
            best = (sc, n)
    pls.last = best[1]
    return PLSRegression(best[1], scale=True).fit(Xtr, Ytr).predict(Xte)


# ----------------------------------------------------------------------------- nonlinear

def kernel_ridge(Xtr, Ytr, Xte, grid_gamma=(0.1, 0.3, 1, 3), grid_alpha=(0.03, 0.1, 0.3, 1, 3), **kw):
    sx = StandardScaler().fit(Xtr)
    A, T = sx.transform(Xtr), sx.transform(Xte)
    d = A.shape[1]
    mu = Ytr.mean(0)
    best = (-np.inf, None)
    for g in grid_gamma:
        for al in grid_alpha:
            P = np.zeros_like(Ytr)
            for a, b in KFold(5, shuffle=True, random_state=1).split(A):
                P[b] = KernelRidge(alpha=al, kernel="rbf", gamma=g / d).fit(A[a], Ytr[a] - mu).predict(A[b]) + mu
            sc = _r2_tot(P, Ytr)
            if sc > best[0]:
                best = (sc, (g, al))
    kernel_ridge.last = best[1]
    g, al = best[1]
    return KernelRidge(alpha=al, kernel="rbf", gamma=g / d).fit(A, Ytr - mu).predict(T) + mu


def bagged_ridge(Xtr, Ytr, Xte, n_bags=30, cell_frac=0.5, **kw):
    """Random-subspace ensemble of ridges (each on half of the cells)."""
    rng = np.random.default_rng(0)
    out = 0
    for _ in range(n_bags):
        c = rng.choice(Xtr.shape[1], int(cell_frac * Xtr.shape[1]), replace=False)
        out = out + ridge(Xtr[:, c], Ytr, Xte[:, c])
    return out / n_bags


def mlp_contrastive(Xtr, Ytr, Xte, hidden=512, linear=False, lam_clip=1.0, mixup=True, epochs=400,
                    wd=0.05, dropout=0.5, temp=0.1, lr=1e-3, seed=0, **kw):
    """MindEye-style: MSE + symmetric InfoNCE between predicted and true (standardized) targets,
    BiMixCo-style mixup on the inputs. Fixed hyperparameters (no inner tuning)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    sx, sy = StandardScaler().fit(Xtr), StandardScaler().fit(Ytr)
    A = torch.tensor(sx.transform(Xtr), dtype=torch.float32)
    T = torch.tensor(sy.transform(Ytr), dtype=torch.float32)
    d = A.shape[1]
    if linear:
        net = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, T.shape[1]))
    else:
        net = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(),
                            nn.Dropout(dropout), nn.Linear(hidden, T.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    g = torch.Generator().manual_seed(seed)
    n = len(A)
    for ep in range(epochs):
        net.train()
        for idx in torch.randperm(n, generator=g).split(64):
            x, y = A[idx], T[idx]
            perm = torch.randperm(len(idx), generator=g)
            if mixup and ep < epochs * 0.66:
                beta = torch.distributions.Beta(0.4, 0.4).sample((len(idx), 1))
                x = beta * x + (1 - beta) * x[perm]
            p = net(x)
            pn, yn = F.normalize(p, dim=1), F.normalize(T[idx], dim=1)
            logits = pn @ yn.T / temp
            if mixup and ep < epochs * 0.66:
                b = beta.squeeze(1)
                tgt = torch.zeros_like(logits)
                tgt[torch.arange(len(idx)), torch.arange(len(idx))] += b
                tgt[torch.arange(len(idx)), perm] += 1 - b
                y_mix = beta * y + (1 - beta) * y[perm]
                clip = -(tgt * logits.log_softmax(1)).sum(1).mean()
                mse = F.mse_loss(p, y_mix)
            else:
                lbl = torch.arange(len(idx))
                clip = 0.5 * (F.cross_entropy(logits, lbl) + F.cross_entropy(logits.T, lbl))
                mse = F.mse_loss(p, y)
            loss = mse + lam_clip * clip
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        out = net(torch.tensor(sx.transform(Xte), dtype=torch.float32)).numpy()
    return sy.inverse_transform(out)


# ----------------------------------------------------------------------------- trial-level data

def trial_tensor(trials, n_stim=500):
    """[n_cells, n_stim, max_reps] trial rates (NaN-padded) and [n_cells, n_stim] counts."""
    reps = max(np.bincount(o.astype(int), minlength=n_stim).max() for _, o in trials)
    T = np.full((len(trials), n_stim, reps), np.nan)
    cnt = np.zeros((len(trials), n_stim), int)
    for c, (rate, order) in enumerate(trials):
        for s in range(n_stim):
            r = rate[order.astype(int) == s]
            T[c, s, :len(r)] = r
            cnt[c, s] = len(r)
    return T, cnt


def pseudo_trial_ridge(Xtr, Ytr, Xte, tr_idx=None, ttensor=None, n_draw=10, k_trials=1, include_means=True,
                       alphas=np.logspace(0, 6, 25), **kw):
    """Augment training with pseudo-population trials: for each training stimulus and draw, each
    cell contributes the mean of k random single trials (with replacement). Test on trial means.
    The (shared) alpha is chosen by stimulus-grouped inner CV scored on trial means, since rows of
    the same stimulus would leak through ridge's built-in leave-one-out."""
    from sklearn.linear_model import Ridge
    T, cnt = ttensor
    rng = np.random.default_rng(0)

    def augment(rows_idx, Xm, Ym):
        Ts, cs = T[:, rows_idx], cnt[:, rows_idx]  # [cells, n, reps]
        draws = []
        for _ in range(n_draw):
            pick = (rng.random((k_trials,) + cs.shape) * cs).astype(int)  # [k, cells, n]
            v = np.take_along_axis(Ts[None], pick[..., None], axis=3)[..., 0].mean(0)  # [cells, n]
            draws.append(v.T)
        Xa, Ya = np.vstack(draws), np.tile(Ym, (n_draw, 1))
        if include_means:
            Xa, Ya = np.vstack([Xa, Xm]), np.vstack([Ya, Ym])
        return Xa, Ya

    def fit(rows_idx, Xm, Ym, alpha):
        Xa, Ya = augment(rows_idx, Xm, Ym)
        sx = StandardScaler().fit(Xa)
        m = Ridge(alpha=alpha).fit(sx.transform(Xa), Ya)
        return lambda Xb: m.predict(sx.transform(Xb))

    tr_idx = np.asarray(tr_idx)
    scores = np.zeros(len(alphas))
    for a, b in KFold(5, shuffle=True, random_state=1).split(Xtr):
        Xa, Ya = augment(tr_idx[a], Xtr[a], Ytr[a])
        sx = StandardScaler().fit(Xa)
        for i, al in enumerate(alphas):
            P = Ridge(alpha=al).fit(sx.transform(Xa), Ya).predict(sx.transform(Xtr[b]))
            scores[i] -= ((P - Ytr[b]) ** 2).sum()
    pseudo_trial_ridge.last = alphas[np.argmax(scores)]
    return fit(tr_idx, Xtr, Ytr, pseudo_trial_ridge.last)(Xte)
