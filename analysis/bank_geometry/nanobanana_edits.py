"""Semantic "one unit" edits with an image-editing model (Gemini 2.5 Flash Image, a.k.a. NanoBanana),
embedded into z50 for q2_composition.py.

API KEY — either
    export GEMINI_API_KEY=...                         (preferred), or
    put the key alone in outputs/bank_geometry/GEMINI_API_KEY   (outputs/ is git-ignored)

    python -m analysis.bank_geometry.nanobanana_edits --dry-run      # list the requests, no API calls
    python -m analysis.bank_geometry.nanobanana_edits --max-bases 10 # a small paid run first
    python -m analysis.bank_geometry.nanobanana_edits                # full spec
    python -m analysis.bank_geometry.nanobanana_edits --embed-only   # (re-)embed what is on disk
    python -m analysis.bank_geometry.q2_composition                  # picks up sets/edits_nb.npz

Edit families live in nanobanana_edits.json (prompt, number of units, base stimuli). Each unit edits the
previous unit's output, so a family is a chain z_0 -> z_1 -> ... -> z_K. Outputs are converted to
grayscale and resized to 224 to match the stimulus pipeline. Images:
outputs/bank_geometry/images/nanobanana/<family>/<base>_k<k>.png ; requests are resumable (existing files
are skipped) and every response is logged to manifest.jsonl.

Costs: (#bases x total units) image generations. With the default spec and 100 bases that is
100 x 18 = 1,800 images; start with --max-bases.
"""

import argparse
import base64
import io
import json
import os
import time

import numpy as np
import requests
from PIL import Image

from .build_imagesets import N_EDIT_BASE, OUT, SETS, STIM500

HERE = os.path.dirname(__file__)
IMG = os.path.join(OUT, "images", "nanobanana")
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def api_key():
    k = os.environ.get("GEMINI_API_KEY")
    p = os.path.join(OUT, "GEMINI_API_KEY")
    if not k and os.path.exists(p):
        k = open(p).read().strip()
    if not k:
        raise SystemExit("No API key: export GEMINI_API_KEY=... or write it to " + p)
    return k


def edit_image(im, prompt, model, key, retries=5):
    """One generateContent call: image + instruction in, edited image out."""
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="PNG")
    body = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(buf.getvalue()).decode()}},
        ]}],
        "generationConfig": {"responseModalities": ["IMAGE"]},
    }
    for attempt in range(retries):
        r = requests.post(ENDPOINT.format(model=model), json=body, timeout=120,
                          headers={"x-goog-api-key": key, "Content-Type": "application/json"})
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt * 5)
            continue
        r.raise_for_status()
        js = r.json()
        for cand in js.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                blob = part.get("inline_data") or part.get("inlineData")
                if blob and blob.get("data"):
                    return Image.open(io.BytesIO(base64.b64decode(blob["data"]))), js
        return None, js  # refused / text-only answer: logged, chain stops
    raise RuntimeError(f"gave up after {retries} retries: {r.status_code} {r.text[:300]}")


def to_style(im):
    return im.convert("L").resize((224, 224), Image.BICUBIC)


def default_bases(n):
    files = sorted(os.listdir(STIM500))
    return np.sort(np.random.default_rng(0).choice(len(files), n, replace=False)).tolist()  # same as edits()


def run(spec, max_bases, dry):
    files = sorted(os.listdir(STIM500))
    key = None if dry else api_key()
    man = open(os.path.join(IMG, "manifest.jsonl"), "a") if not dry else None
    n_req = 0
    for fam, cfg in spec["families"].items():
        bases = cfg["bases"] if cfg["bases"] is not None else default_bases(N_EDIT_BASE)
        bases = bases[:max_bases] if max_bases else bases
        prompt = cfg["prompt"] + spec.get("style_suffix", "")
        os.makedirs(os.path.join(IMG, fam), exist_ok=True)
        for b in bases:
            prev = Image.open(os.path.join(STIM500, files[b]))
            for k in range(1, cfg["n_units"] + 1):
                dst = os.path.join(IMG, fam, f"{b:03d}_k{k}.png")
                if os.path.exists(dst):
                    prev = Image.open(dst)
                    continue
                n_req += 1
                if dry:
                    print(f"[dry] {fam} base={b} ({files[b]}) k={k}: {prompt[:80]}...")
                    continue
                out, js = edit_image(prev, prompt, spec["model"], key)
                man.write(json.dumps({"family": fam, "base": b, "k": k, "ok": out is not None,
                                      "finish": [c.get("finishReason") for c in js.get("candidates", [])],
                                      "usage": js.get("usageMetadata")}) + "\n")
                man.flush()
                if out is None:
                    print(f"{fam} base={b} k={k}: no image returned, stopping this chain")
                    break
                out = to_style(out)
                out.save(dst)
                prev = out
                print(f"{fam} base={b} k={k} ok", flush=True)
    print(f"{n_req} {'requests planned' if dry else 'requests made'}")


def embed(spec):
    """Embed every chain on disk -> sets/edits_nb.npz in the format of sets/edits.npz."""
    from .embed import embed_pil

    files = sorted(os.listdir(STIM500))
    ims, meta = [], []
    bases_seen = set()
    for fam in spec["families"]:
        d = os.path.join(IMG, fam)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            b, k = int(f[:3]), int(f.split("_k")[1].split(".")[0])
            ims.append(Image.open(os.path.join(d, f)))
            meta.append((b, fam, k))
            bases_seen.add(b)
    for b in sorted(bases_seen):  # the unedited originals (k = 0), through the same style conversion
        ims.append(to_style(Image.open(os.path.join(STIM500, files[b]))))
        meta.append((b, "orig", 0))
    if not meta:
        raise SystemExit("no edited images on disk yet")
    raw, z = embed_pil(ims)
    np.savez(os.path.join(SETS, "edits_nb.npz"), raw=raw, z=z, base=np.array([m[0] for m in meta]),
             family=np.array([m[1] for m in meta]), k=np.array([m[2] for m in meta]))
    print(f"embedded {len(meta)} images -> {SETS}/edits_nb.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=os.path.join(HERE, "nanobanana_edits.json"))
    ap.add_argument("--max-bases", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--embed-only", action="store_true")
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    os.makedirs(IMG, exist_ok=True)
    if not a.embed_only:
        run(spec, a.max_bases, a.dry_run)
    if not a.dry_run:
        embed(spec)
