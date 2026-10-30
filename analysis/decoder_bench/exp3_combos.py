"""Combinations of saved out-of-fold predictions with fixed (untuned, equal) weights, so no
test-fold information is used. python -m analysis.decoder_bench.exp3_combos"""

import json

import numpy as np

from .common import OUT, fmt, load_target, metrics

COMBOS = {
    "bank_post + gmm_post": ["bayes_bank_posterior", "bayes_gmm32_bank"],
    "bank_post + kernel_ridge": ["bayes_bank_posterior", "kernel_ridge_rbf"],
    "bank_post + mlp_mse+clip": ["bayes_bank_posterior", "mlp_mse+clip"],
    "bank_post + gmm_post + kernel_ridge + mlp_mse+clip": ["bayes_bank_posterior", "bayes_gmm32_bank",
                                                           "kernel_ridge_rbf", "mlp_mse+clip"],
    "kernel_ridge + mlp_mse+clip + linear_mse+clip": ["kernel_ridge_rbf", "mlp_mse+clip", "linear_mse+clip"],
}
res = {}
for t in ["cls", "meanpatch"]:
    Y, bank = load_target(t)
    for name, parts in COMBOS.items():
        P = np.mean([np.load(f"{OUT}/preds_{t}_{p}.npy") for p in parts], axis=0)
        m = metrics(P, Y, bank)
        res[f"{t}:{name}"] = m
        np.save(f"{OUT}/preds_{t}_combo_{'+'.join(parts)}.npy", P.astype(np.float32))
        print(fmt(f"[{t}] {name}", m))
json.dump(res, open(f"{OUT}/exp3_results.json", "w"), indent=1)
