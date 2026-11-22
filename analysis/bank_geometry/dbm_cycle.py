"""Cycle consistency through the conditional diffusion model (DBM), and a DBM-sampled bank.

For a set of conditioning vectors z (z50, meanpatch), generate G(z) with the finetuned flow model,
re-embed it with DINOv2 meanpatch -> z50 (analysis.bank_geometry.embed), and measure
    cycle error  e(z) = ||z - E(G(z))||  (in bank-NN units; also centred cosine)
A z the generator can realise (on the *generator's* manifold) cycles about as well as a real image's
embedding does. Compare the cycle-error distributions of:
    real      bank embeddings and the 500 true targets             (the reference)
    posterior bank-posterior means (OOF, outputs/decoder_bench/preds_meanpatch_bayes_bank_posterior.npy)
    ridge     ridge predictions (shrunk, off-manifold)
    mixK      uniform mixtures of the K nearest bank points (K = 2, 10, 50, 300) of real anchors
    arith     z_0 + k * mean edit direction (from sets/edits*.npz), k = 1..K
The re-embedded E(G(z)) of the `bank_sample` conditions is saved as sets/dbm_bank.npz, which
q3_bank_choice.py then evaluates as a "generator-realisable" bank.

Needs a checkpoint (none is on this machine) and ideally a GPU:
    python -m analysis.bank_geometry.dbm_cycle \
        --config config/diffusion_full_stimuli_finetune_flow.yaml --ckpt latest \
        --n 200 --n_seeds 2 --steps 50 [--bank_sample 5000]
"""

import argparse
import json
import os
import types

import numpy as np
import torch
import yaml
from PIL import Image

from analysis.decoder_bench.common import OUT as BENCH, load_target
from . import geometry as G
from .embed import embed_pil

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "outputs", "bank_geometry")


def load_model(config_path, ckpt_spec, use_ema=True):
    from models.ema import ema_shadow_to_model_state_dict
    from models.unet import UNet
    from utils import discover_device, make_run_tag, select_checkpoints

    cfg = types.SimpleNamespace(**yaml.safe_load(open(config_path)))
    assert getattr(cfg, "noise_process", "ddpm") == "flow", "this script uses the flow sampler"
    assert cfg.encoding_model == "dinov2_vitb14_meanpatch" and cfg.latent_dim == 50, \
        "cycle consistency is measured in the meanpatch z50 space"
    cfg.output_dir = f"{cfg.output_dir}_{make_run_tag(cfg)}"
    path = select_checkpoints(os.path.join(cfg.output_dir, "checkpoints"), ckpt_spec)[0]
    dev = torch.device(discover_device())
    unet = UNet(in_channels=cfg.in_channels, image_size=cfg.image_size, model_channels=cfg.model_channels,
                channel_mult=cfg.channel_mult, num_res_blocks=cfg.num_res_blocks,
                attention_resolutions=cfg.attention_resolutions, dropout=0.0, latent_dim=cfg.latent_dim)
    ck = torch.load(path, map_location=dev)
    unet.load_state_dict(ck["model_state_dict"])
    if use_ema and "ema_state_dict" in ck:
        unet.load_state_dict(ema_shadow_to_model_state_dict(unet, ck["ema_state_dict"]["shadow"]))
    print(f"loaded {path} (step {ck['step']})")
    return unet.to(dev).eval(), cfg, dev, path


@torch.no_grad()
def generate(unet, cfg, dev, Z, n_seeds, steps, batch=64, seed=0):
    from utils import flow_sample

    C, H = cfg.in_channels, cfg.image_size
    jobs = [(i, s) for i in range(len(Z)) for s in range(n_seeds)]
    imgs = [None] * len(jobs)
    for b in range(0, len(jobs), batch):
        jb = jobs[b:b + batch]
        x0 = torch.cat([torch.randn(1, C, H, H, generator=torch.Generator().manual_seed(seed + 1_000_003 * s + i))
                        for i, s in jb])
        z = torch.from_numpy(Z[[i for i, _ in jb]].astype(np.float32)).to(dev)
        x = flow_sample(z, unet, cfg, dev, num_steps=steps, x_init=x0)
        if C == 1:
            x = x.expand(-1, 3, -1, -1)
        arr = (x.permute(0, 2, 3, 1) * 255).round().clamp(0, 255).byte().cpu().numpy()
        for j, a in enumerate(arr):
            imgs[b + j] = Image.fromarray(a)
        print(f"  generated {min(b + batch, len(jobs))}/{len(jobs)}", flush=True)
    return imgs, jobs


def conditions(n, rng):
    Y, bank = load_target("meanpatch")
    sel = rng.choice(500, min(n, 500), replace=False)
    out = {
        "real: bank": bank[rng.choice(len(bank), n, replace=False)],
        "real: 500 targets": Y[sel],
        "bank posterior": np.load(f"{BENCH}/preds_meanpatch_bayes_bank_posterior.npy")[sel],
        "ridge": np.load(f"{BENCH}/preds_meanpatch_ridge.npy")[sel],
    }
    anchors = bank[rng.choice(len(bank), n, replace=False)]
    I, _ = G.knn(anchors, bank, 301)
    for K in (2, 10, 50, 300):
        out[f"mix{K}"] = bank[I[:, 1:K + 1]].mean(1)
    ep = os.path.join(OUT, "sets", "edits.npz")
    if os.path.exists(ep):
        d = np.load(ep, allow_pickle=True)
        z0 = {b: d["z"][(d["base"] == b) & (d["family"] == "orig")][0] for b in np.unique(d["base"])}
        for fam in ("rotate", "scale"):
            m1 = (d["family"] == fam) & (d["k"] == 1)
            mean_d1 = np.mean([d["z"][m1][j] - z0[b] for j, b in enumerate(d["base"][m1])], 0)
            for k in (1, 3, 5):
                out[f"arith {fam} k={k}"] = np.stack([z0[b] + k * mean_d1 for b in list(z0)[:n]])
    return out, bank, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/diffusion_full_stimuli_finetune_flow.yaml")
    ap.add_argument("--ckpt", default="latest")
    ap.add_argument("--n", type=int, default=200, help="conditions per group")
    ap.add_argument("--n_seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--bank_sample", type=int, default=0, help="also build a DBM bank from this many bank z")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    unet, cfg, dev, path = load_model(a.config, a.ckpt)
    conds, bank, Y = conditions(a.n, rng)
    scale = G.bank_scale(bank)
    mu = Y.mean(0)
    res = {"checkpoint": path, "steps": a.steps, "n_seeds": a.n_seeds, "bank_nn_scale": scale, "groups": {}}
    os.makedirs(os.path.join(OUT, "cycle"), exist_ok=True)
    for name, Z in conds.items():
        imgs, jobs = generate(unet, cfg, dev, Z, a.n_seeds, a.steps)
        _, Zc = embed_pil(imgs, verbose=False)
        Zin = Z[[i for i, _ in jobs]]
        err = np.linalg.norm(Zc - Zin, axis=1) / scale
        c = ((Zc - mu) * (Zin - mu)).sum(1) / (np.linalg.norm(Zc - mu, axis=1) * np.linalg.norm(Zin - mu, axis=1))
        res["groups"][name] = {"cycle_err_units_median": float(np.median(err)),
                               "cycle_err_units_iqr": np.percentile(err, [25, 75]).tolist(),
                               "cycle_cos_median": float(np.median(c)),
                               "regen_precision": G.precision(Zc, bank)}
        np.savez(os.path.join(OUT, "cycle", f"{name.replace(' ', '_').replace(':', '')}.npz"), z=Zin, zc=Zc, err=err)
        print(name, res["groups"][name], flush=True)
    json.dump(res, open(os.path.join(OUT, "cycle_results.json"), "w"), indent=1)
    if a.bank_sample:
        Z = bank[rng.choice(len(bank), a.bank_sample, replace=False)]
        imgs, _ = generate(unet, cfg, dev, Z, 1, a.steps)
        raw, Zc = embed_pil(imgs, verbose=False)
        np.savez(os.path.join(OUT, "sets", "dbm_bank.npz"), raw=raw, z=Zc, cond=Z)
        print(f"wrote DBM bank ({len(Zc)}) -> sets/dbm_bank.npz; re-run q3_bank_choice")


if __name__ == "__main__":
    main()
