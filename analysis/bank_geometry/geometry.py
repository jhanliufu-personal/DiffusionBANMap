"""Manifold / on-manifold metrics for point sets in z50.

All metrics compare a query set Q against a reference set R of real-image embeddings (the bank).

  knn_radius(R, k)            distance from each real point to its k-th nearest other real point
  precision(Q, R, k)          Kynkaenniemi et al. 2019: fraction of Q inside the union of real k-NN balls
  recall(Q, R, k)             fraction of R inside the union of Q's k-NN balls (coverage of the real set)
  density(Q, R, k)            Naeem et al. 2020: mean number of real balls containing each query / k
                              (~1 for real-like queries; >1 = collapsed into dense regions, <1 = sparse/off)
  nn_ratio(Q, R, Qref)        median NN distance of Q to R / the same for real held-out points Qref
  c2st_auc(A, B)              classifier two-sample test: CV ROC-AUC of telling A from B
                              (0.5 = indistinguishable). Linear and gradient-boosted classifiers.
  hull_residual(Q, R, K)      distance from each query to the convex hull of its K nearest real points
                              (simplex-constrained least squares), the "can a convex combination of
                              bank points reach this?" test
"""

import numpy as np
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def sqdist(A, B):
    return ((A ** 2).sum(1)[:, None] - 2 * A @ B.T + (B ** 2).sum(1)[None]).clip(0)


def knn(Q, R, k, exclude_self=False, chunk=2048):
    """Indices and distances of the k nearest R points to each Q point."""
    kk = k + int(exclude_self)
    I = np.zeros((len(Q), k), int)
    D = np.zeros((len(Q), k))
    for s in range(0, len(Q), chunk):
        d = sqdist(Q[s:s + chunk], R)
        idx = np.argpartition(d, kk - 1, axis=1)[:, :kk]
        dd = np.take_along_axis(d, idx, 1)
        o = np.argsort(dd, 1)
        idx, dd = np.take_along_axis(idx, o, 1), np.take_along_axis(dd, o, 1)
        if exclude_self:
            idx, dd = idx[:, 1:], dd[:, 1:]
        I[s:s + chunk], D[s:s + chunk] = idx, np.sqrt(dd)
    return I, D


def knn_radius(R, k=5):
    return knn(R, R, k, exclude_self=True)[1][:, -1]


def nn_dist(Q, R):
    return knn(Q, R, 1)[1][:, 0]


def _ball_counts(Q, R, rad, chunk=2048):
    c = np.zeros(len(Q))
    r2 = rad ** 2
    for s in range(0, len(Q), chunk):
        c[s:s + chunk] = (sqdist(Q[s:s + chunk], R) <= r2[None]).sum(1)
    return c


def precision(Q, R, k=5, rad=None):
    rad = knn_radius(R, k) if rad is None else rad
    return float((_ball_counts(Q, R, rad) > 0).mean())


def in_manifold(Q, R, k=5, rad=None):
    """Per-query boolean: inside at least one real k-NN ball."""
    rad = knn_radius(R, k) if rad is None else rad
    return _ball_counts(Q, R, rad) > 0


def density(Q, R, k=5, rad=None):
    rad = knn_radius(R, k) if rad is None else rad
    return float(_ball_counts(Q, R, rad).mean() / k)


def recall(Q, R, k=5):
    radQ = knn_radius(Q, k)
    return float((_ball_counts(R, Q, radQ) > 0).mean())


def nn_ratio(Q, R, Qref):
    return float(np.median(nn_dist(Q, R)) / np.median(nn_dist(Qref, R)))


def c2st_auc(A, B, seed=0, kinds=("linear", "gbt")):
    """Balanced classifier two-sample test, 5-fold CV AUC."""
    rng = np.random.default_rng(seed)
    n = min(len(A), len(B))
    A = A[rng.choice(len(A), n, replace=False)]
    B = B[rng.choice(len(B), n, replace=False)]
    X = np.vstack([A, B])
    y = np.r_[np.zeros(n), np.ones(n)]
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    out = {}
    for kind in kinds:
        clf = (make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)) if kind == "linear"
               else HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed))
        p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        out[kind] = float(roc_auc_score(y, p))
    return out


def hull_residual(Q, R, K=50, M=1e3, return_weights=False):
    """min_w ||q - R_K w||, w >= 0, sum w = 1, over each query's K nearest real points.
    The sum-to-one constraint is enforced with a heavily weighted extra row (NNLS)."""
    I, D = knn(Q, R, K)
    res = np.zeros(len(Q))
    W = np.zeros((len(Q), K))
    for i in range(len(Q)):
        C = R[I[i]].T  # [d, K]
        A = np.vstack([C, M * np.ones((1, K))])
        b = np.r_[Q[i], M]
        w, _ = nnls(A, b, maxiter=50 * K)
        w = w / w.sum().clip(1e-12)
        res[i] = np.linalg.norm(Q[i] - C @ w)
        W[i] = w
    return (res, I, W) if return_weights else res


def bank_scale(R, sample=3000, seed=0):
    """The bank's local resolution: median distance from a real point to its nearest other real point."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(R), min(sample, len(R)), replace=False)
    return float(np.median(knn(R[idx], R, 1, exclude_self=True)[1][:, 0]))
