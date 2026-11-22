"""Build and embed the image sets the bank-geometry analyses need. Every step is resumable.

  python -m analysis.bank_geometry.build_imagesets edits      # controlled one-factor edits of 100 stimuli
  python -m analysis.bank_geometry.build_imagesets inet_list  # sample N Tiny-ImageNet images
  python -m analysis.bank_geometry.build_imagesets inet_raw   # embed them as-is (colour, background)
  python -m analysis.bank_geometry.build_imagesets inet_gray  # grayscale only
  uv run --with "rembg[cpu]" python -m analysis.bank_geometry.build_imagesets inet_cutout
                                                             # grayscale + background removed (needs rembg)
  python -m analysis.bank_geometry.build_imagesets inet_styled  # embed the cut-outs

Env: N_INET (default 3000), N_EDIT_BASE (default 100). Outputs: outputs/bank_geometry/sets/*.npz
"""

import os
import sys

import numpy as np
from PIL import Image, ImageOps

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "outputs", "bank_geometry")
SETS = os.path.join(OUT, "sets")
CUT = os.path.join(OUT, "images", "inet_cutout")
os.makedirs(SETS, exist_ok=True)
N_INET = int(os.environ.get("N_INET", 3000))
N_EDIT_BASE = int(os.environ.get("N_EDIT_BASE", 100))
STIM500 = os.path.join(REPO, "data", "500Stimuli")
INET = os.path.join(REPO, "data", "tiny-imagenet-200", "train")

# ----------------------------------------------------------------------------- one-factor edits
# Each family is a sequence of k = 1..K "units" of one factor. Background is white like the stimuli.
WHITE = (255, 255, 255)


def _rot(im, deg):
    return im.convert("RGB").rotate(-deg, resample=Image.BICUBIC, fillcolor=WHITE)


def _scale(im, s):
    im = im.convert("RGB")
    w, h = im.size
    small = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)
    canvas = Image.new("RGB", (w, h), WHITE)
    canvas.paste(small, ((w - small.width) // 2, (h - small.height) // 2))
    return canvas


def _shift(im, frac):
    im = im.convert("RGB")
    w, h = im.size
    canvas = Image.new("RGB", (w, h), WHITE)
    canvas.paste(im, (int(round(frac * w)), 0))
    return canvas


def _contrast(im, c):
    a = np.asarray(im.convert("RGB")).astype(np.float32)
    a = 255 - (255 - a) * c  # fade toward the white background
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


EDIT_FAMILIES = {
    # name: (unit description, list of (k, fn))
    "hflip": ("mirror", [(1, lambda im: ImageOps.mirror(im.convert("RGB")))]),
    "rotate": ("15 deg in-plane rotation", [(k, (lambda k: lambda im: _rot(im, 15 * k))(k)) for k in range(1, 7)]),
    "scale": ("x0.85 size", [(k, (lambda k: lambda im: _scale(im, 0.85 ** k))(k)) for k in range(1, 6)]),
    "shift": ("8% rightward translation", [(k, (lambda k: lambda im: _shift(im, 0.08 * k))(k)) for k in range(1, 6)]),
    "fade": ("x0.8 contrast toward white", [(k, (lambda k: lambda im: _contrast(im, 0.8 ** k))(k)) for k in range(1, 6)]),
}


def edits():
    from .embed import embed_pil

    out = os.path.join(SETS, "edits.npz")
    if os.path.exists(out):
        print("exists", out)
        return
    files = sorted(os.listdir(STIM500))
    rng = np.random.default_rng(0)
    base_idx = np.sort(rng.choice(len(files), N_EDIT_BASE, replace=False))  # rows of the 500-stimulus order
    ims, meta = [], []
    for i in base_idx:
        im = Image.open(os.path.join(STIM500, files[i]))
        ims.append(im.convert("RGB"))
        meta.append((i, "orig", 0))
        for fam, (_, steps) in EDIT_FAMILIES.items():
            for k, fn in steps:
                ims.append(fn(im))
                meta.append((i, fam, k))
    print(f"{len(ims)} edited images")
    raw, z = embed_pil(ims)
    np.savez(out, raw=raw, z=z, base=np.array([m[0] for m in meta]), family=np.array([m[1] for m in meta]),
             k=np.array([m[2] for m in meta]))
    # small visual sheet of the edit families for one base image
    sheet = [ims[j] for j in range(len(meta)) if meta[j][0] == base_idx[0]]
    W = Image.new("RGB", (96 * len(sheet), 96), WHITE)
    for j, s in enumerate(sheet):
        W.paste(s.resize((96, 96)), (96 * j, 0))
    os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
    W.save(os.path.join(OUT, "figs", "edit_families_example.png"))


# ----------------------------------------------------------------------------- Tiny-ImageNet banks

def _inet_paths():
    lst = os.path.join(SETS, "inet_paths.txt")  # repo-relative, so the list works on another machine
    return [os.path.join(REPO, os.path.relpath(l.strip(), REPO) if os.path.isabs(l.strip()) else l.strip())
            for l in open(lst) if l.strip()]


def inet_list():
    lst = os.path.join(SETS, "inet_paths.txt")
    if os.path.exists(lst):
        print("exists", lst)
        return
    paths = []
    for wn in sorted(os.listdir(INET)):
        d = os.path.join(INET, wn, "images")
        paths += [os.path.join(d, f) for f in sorted(os.listdir(d))]
    rng = np.random.default_rng(0)
    sel = sorted(rng.choice(len(paths), N_INET, replace=False))
    open(lst, "w").write("\n".join(os.path.relpath(paths[i], REPO) for i in sel))
    print(f"sampled {N_INET} of {len(paths)}")


def inet_embed(variant):
    from .embed import embed_paths_cached

    paths = _inet_paths()
    if variant == "raw":
        tf = None
    elif variant == "gray":
        tf = lambda im: im.convert("L")
    elif variant == "styled":
        paths = [os.path.join(CUT, os.path.basename(p).rsplit(".", 1)[0] + ".png") for p in paths]
        tf = None
    embed_paths_cached(paths, os.path.join(SETS, f"inet_{variant}.npz"), transform=tf)


def inet_cutout():
    """Grayscale + rembg (u2netp) foreground on white, recentred and scaled so the object's bounding
    box fills ~85% of the canvas -- the stimulus-set style."""
    from rembg import new_session, remove

    os.makedirs(CUT, exist_ok=True)
    sess = new_session("u2netp")
    paths = _inet_paths()
    for n, p in enumerate(paths):
        dst = os.path.join(CUT, os.path.basename(p).rsplit(".", 1)[0] + ".png")
        if os.path.exists(dst):
            continue
        im = Image.open(p).convert("RGB")
        rgba = remove(im, session=sess)
        alpha = np.asarray(rgba)[..., 3].astype(np.float32) / 255
        gray = np.asarray(im.convert("L")).astype(np.float32)
        comp = gray * alpha + 255 * (1 - alpha)
        out = Image.fromarray(comp.clip(0, 255).astype(np.uint8))
        ys, xs = np.where(alpha > 0.5)
        if len(xs) > 20:
            box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
            crop = out.crop(box)
            S = 224
            s = 0.85 * S / max(crop.size)
            crop = crop.resize((max(1, int(crop.width * s)), max(1, int(crop.height * s))), Image.BICUBIC)
            out = Image.new("L", (S, S), 255)
            out.paste(crop, ((S - crop.width) // 2, (S - crop.height) // 2))
        else:  # segmentation failed: keep the grayscale image
            out = out.resize((224, 224), Image.BICUBIC)
        out.save(dst)
        if n % 200 == 0:
            print(f"cutout {n}/{len(paths)}", flush=True)
    print("cutout done")


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"edits": edits, "inet_list": inet_list, "inet_cutout": inet_cutout,
     "inet_raw": lambda: inet_embed("raw"), "inet_gray": lambda: inet_embed("gray"),
     "inet_styled": lambda: inet_embed("styled")}[cmd]()
