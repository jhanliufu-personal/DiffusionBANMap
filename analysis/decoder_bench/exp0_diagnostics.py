"""Part 0: diagnostics. Run from the repo root: python -m analysis.decoder_bench.exp0_diagnostics"""

import json

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler

from .common import OUT, cv_predict, fmt, folds, load_neural, load_target, metrics, per_dim_r2, trial_split_means
from .decoders import ALPHAS, ridge

X, trials, expvar, session, patient = load_neural()
results = {}
print(f"X {X.shape}, sessions {len(set(session))}, patients {len(set(patient))}")

# D1 noise ceiling per cell
q = np.percentile(expvar, [10, 25, 50, 75, 90])
print("D1 explainable variance percentiles 10/25/50/75/90:", np.round(q, 3), "| >0.1:", (expvar > 0.1).sum())
results["D1"] = {"pct": q.tolist(), "n_gt_0.1": int((expvar > 0.1).sum())}


# D2 population split-half identification: how identifiable is a stimulus from the
# neural population itself? (upper bound on any neural decoder's identification)
def ident(P, Q):
    P = (P - P.mean(0)) / P.std(0).clip(1e-9)
    Q = (Q - Q.mean(0)) / Q.std(0).clip(1e-9)
    P = P - P.mean(1, keepdims=True); Q = Q - Q.mean(1, keepdims=True)
    S = (P / np.linalg.norm(P, axis=1, keepdims=True)) @ (Q / np.linalg.norm(Q, axis=1, keepdims=True)).T
    rank = (S > np.diag(S)[:, None]).sum(1)
    return (rank == 0).mean(), (rank < 5).mean(), 1 - rank.mean() / (len(P) - 1)

rng = np.random.default_rng(0)
d2 = []
for rep in range(5):
    A, B = trial_split_means(trials, rng)
    for sel_name, sel in [("all", np.ones(len(expvar), bool)), ("ev>0.1", expvar > 0.1)]:
        d2.append((rep, sel_name, *ident(A[:, sel], B[:, sel])))
for sel_name in ["all", "ev>0.1"]:
    r = np.array([x[2:] for x in d2 if x[1] == sel_name]).mean(0)
    print(f"D2 neural half-vs-half identification ({sel_name}, ~2 trials each): top1={r[0]:.3f} top5={r[1]:.3f} 2afc={r[2]:.3f}")
    results[f"D2_{sel_name}"] = r.tolist()

# D3 baselines and D6 shrinkage, for both targets
for t in ["cls", "meanpatch"]:
    Y, bank = load_target(t)
    for name, kw in [("ridge std->PCA100->RidgeCV/target (notebook)", dict(n_pcs=100)),
                     ("ridge std->RidgeCV/target", dict()),
                     ("ridge std->RidgeCV shared alpha", dict(per_target=False))]:
        P = cv_predict(ridge, X, Y, **kw)
        m = metrics(P, Y, bank)
        print(fmt(f"D3 [{t}] {name}", m)); results[f"D3_{t}_{name}"] = m
    P = cv_predict(ridge, X, Y)
    pr = per_dim_r2(P, Y)
    print(f"D4 [{t}] per-PC R2 (PC1..10):", np.round(pr[:10], 2), "| #PCs R2>0.1:", (pr > 0.1).sum(),
          "| #PCs R2>0:", (pr > 0).sum())
    results[f"D4_{t}"] = pr.tolist()
    # oracle reference points
    m_mean = metrics(np.tile(Y.mean(0), (500, 1)), Y, bank)
    print(fmt(f"   [{t}] reference: predict train mean", m_mean))
    true_nn = metrics(bank[np.argmin(((Y[:, None] - bank[None]) ** 2).sum(-1), 1)], Y, bank)
    print(fmt(f"   [{t}] reference: oracle snap true->nearest bank", true_nn))
    results[f"oracle_snap_{t}"] = true_nn

# D5 target-space comparison (same decoder; spaces differ, so compare identification/R2 only)
for t in ["cls", "meanpatch", "clip_b32", "alexnet_fc6"]:
    Y, _ = load_target(t)
    m = metrics(cv_predict(ridge, X, Y), Y)
    print(fmt(f"D5 space={t}", m)); results[f"D5_{t}"] = m

# D6 encoding: per-cell CV R2 from each space, normalized by explainable variance
good = expvar > 0.1
for t in ["cls", "meanpatch", "clip_b32", "alexnet_fc6"]:
    Y, _ = load_target(t)
    F = StandardScaler().fit_transform(Y)
    pred = np.zeros_like(X)
    for tr, te in folds():
        pred[te] = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(F[tr], X[tr]).predict(F[te])
    r2 = 1 - ((X - pred) ** 2).sum(0) / ((X - X.mean(0)) ** 2).sum(0)
    norm = np.clip(r2[good] / expvar[good], 0, 1)
    print(f"D6 encoding {t:<12s} median CV R2 (ev>0.1 cells)={np.median(r2[good]):.3f}  "
          f"median R2/ExpVar={np.median(norm):.3f}  mean={norm.mean():.3f}")
    results[f"D6_{t}"] = {"median_r2": float(np.median(r2[good])), "median_norm": float(np.median(norm))}

# D7 SNR: decoder fit on full-trial means, tested on fewer-trial means vs full means
rng = np.random.default_rng(1)
A, B = trial_split_means(trials, rng)
for t in ["cls", "meanpatch"]:
    Y, _ = load_target(t)
    out = {}
    for test_name, Xtest in [("full means", X), ("half-trial means", B)]:
        P = np.zeros_like(Y)
        for tr, te in folds():
            P[te] = ridge(X[tr], Y[tr], Xtest[te])
        out[test_name] = metrics(P, Y)
        print(fmt(f"D7 [{t}] train full, test {test_name}", out[test_name]))
    # train on half A, test on half B (both noisier)
    P = np.zeros_like(Y)
    for tr, te in folds():
        P[te] = ridge(A[tr], Y[tr], B[te])
    out["half->half"] = metrics(P, Y); print(fmt(f"D7 [{t}] train half A, test half B", out["half->half"]))
    results[f"D7_{t}"] = out

# D8 learning curve (fixed 100-stimulus test set, 5 random test sets)
for t in ["cls", "meanpatch"]:
    Y, _ = load_target(t)
    curve = {}
    for n in [50, 100, 200, 300, 400]:
        ms = []
        for s in range(5):
            perm = np.random.default_rng(s).permutation(500)
            te, pool = perm[:100], perm[100:]
            tr = pool[:n]
            P = ridge(X[tr], Y[tr], X[te])
            ms.append(metrics(P, Y[te], ref_mean=Y.mean(0)))
        curve[n] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
        print(f"D8 [{t}] n_train={n:3d}  r2_tot={curve[n]['r2_tot']:.3f}  top1(of 100)={curve[n]['top1']:.3f}  "
              f"top5={curve[n]['top5']:.3f} 2afc={curve[n]['pair2afc']:.3f}")
    results[f"D8_{t}"] = curve

# D9 neuron-count curve (10-fold CV, 5 random subsets each)
for t in ["cls", "meanpatch"]:
    Y, _ = load_target(t)
    curve = {}
    for n in [25, 50, 100, 200, 300, 386]:
        ms = []
        for s in range(5 if n < 386 else 1):
            cells = np.random.default_rng(s).choice(386, n, replace=False)
            ms.append(metrics(cv_predict(ridge, X[:, cells], Y), Y))
        curve[n] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
        print(f"D9 [{t}] n_cells={n:3d}  r2_tot={curve[n]['r2_tot']:.3f}  top1={curve[n]['top1']:.3f}  "
              f"top5={curve[n]['top5']:.3f} 2afc={curve[n]['pair2afc']:.3f}")
    results[f"D9_{t}"] = curve

json.dump(results, open(f"{OUT}/exp0_results.json", "w"), indent=1)
