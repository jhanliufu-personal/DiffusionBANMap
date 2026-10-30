"""How much of the neural stimulus information is reachable through the DINO space?
Identification among the 500 stimuli done in *neural* space: predict each candidate's response
pattern with a per-cell encoding model (z50 -> rates), then rank candidates by Gaussian
log-likelihood or by pattern correlation with the observed held-out responses.

python -m analysis.decoder_bench.exp2_encoding_ident
"""

import json

import numpy as np

from .common import OUT, folds, load_neural, load_target
from .methods import Encoder

X, _, expvar, _, _ = load_neural()
res = {}
for t in ["cls", "meanpatch", "clip_b32", "alexnet_fc6"]:
    Y, _ = load_target(t)
    L = np.zeros((500, 500))
    C = np.zeros((500, 500))
    for tr, te in folds():
        enc = Encoder(Y[tr], X[tr])
        R = enc.r(X[te])
        L[te] = enc.loglik(R, Y)  # candidates: all 500 embeddings
        F = Y @ enc.W.T + enc.b
        Rz = (R - R.mean(1, keepdims=True)) / R.std(1, keepdims=True)
        Fz = (F - F.mean(1, keepdims=True)) / F.std(1, keepdims=True)
        C[te] = Rz @ Fz.T / R.shape[1]
    out = {}
    for name, S in [("loglik", L), ("corr", C)]:
        d = np.diag(S)
        rank = (S > d[:, None]).sum(1)
        out[name] = {"top1": float((rank == 0).mean()), "top5": float((rank < 5).mean()),
                     "pair2afc": float(1 - rank.mean() / 499)}
        print(f"[{t:<11s}] encoding-model identification ({name:6s}): top1={out[name]['top1']:.3f} "
              f"top5={out[name]['top5']:.3f} 2afc={out[name]['pair2afc']:.3f}")
    res[t] = out
json.dump(res, open(f"{OUT}/exp2_results.json", "w"), indent=1)
