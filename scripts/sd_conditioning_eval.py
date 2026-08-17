"""
SD conditioning ablation eval: scores degraded-conditioning generations (pure-noise
latent start, null CLIP cond, or regressed VAE latents/CLIP embeddings from firing
rates) against the reference generation that used the *real* stimulus VAE latent + real
CLIP embedding for the same stimuli. Comparing against that reference run (rather than
the raw stimulus images) isolates how much conditioning quality costs, separate from
whatever fidelity loss the SD img2img pipeline itself introduces even under perfect
conditioning.

GPU/VM port of analysis/sd_conditioning_eval.ipynb -- moved out of the notebook because
loading every setup's [500, 3, 512, 512] image tensor into RAM at once (~1.5GB/setup)
OOM'd on a laptop. This script instead loads one setup at a time, scores it against the
(once-loaded) reference, then frees it before moving to the next.

Expects scripts/sd_clip_img2img.py to have been run once per setup, with each run's
denoised_images.pt saved into --output_dir under a tag prefix: {tag}_denoised_images.pt
(bare denoised_images.pt for the reference/perfect-latent+perfect-CLIP run). See
TAG_LABELS below for the tags this script knows human-readable descriptions for --
unrecognized tags still get scored, just plotted under a prettified version of the raw
tag.

For each requested --metric, writes into <output_dir>/eval/:
    <metric>_scores.npz     per-setup arrays of per-image scores (tag -> [N] float)
    <metric>_comparison.png bar chart of mean +/- SEM per setup, ranked best-to-worst
    <metric>_distribution.png violin plot of the per-image score distribution per setup
plus eval_summary.yaml mapping every (tag, metric) to {mean, sem, n}.

Run from repo root:
    python -m scripts.sd_conditioning_eval --output_dir outputs/sd_clip_img2img \
        --metrics mse,psnr,ssim,clip_cosine
"""

import os
import re
import gc
import argparse
from glob import glob

import yaml
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import discover_device

# Matches "denoised_images.pt" (tag="") and "{tag}_denoised_images.pt" (tag=anything
# before the trailing underscore) -- deliberately does NOT match "*_denoised_latents.pt".
FILENAME_RE = re.compile(r"^(?:(?P<tag>.+)_)?denoised_images\.pt$")

# Human-readable setup description per tag, carried through into plots/summary. A tag
# found on disk that isn't listed here still gets scored, just under a lightly
# prettified version of the raw tag (see label_for_tag) -- extend as new setups/tags
# are generated.
TAG_LABELS = {
    "": "perfect latent + perfect CLIP (reference)",
    "null_cond": "perfect latent + null CLIP",
    "noise_start": "noise start + perfect CLIP",
    "noise_start_regressed_cond": "noise start + regressed CLIP",
    "regressed_latent_0.3": "regressed latent (s=0.3) + perfect CLIP",
    "regressed_latent_0.6": "regressed latent (s=0.6) + perfect CLIP",
    "regressed_latent_0.9": "regressed latent (s=0.9) + perfect CLIP",
    "regressed_latent_0.3_regressed_cond": "regressed latent (s=0.3) + regressed CLIP",
    "regressed_latent_0.6_regressed_cond": "regressed latent (s=0.6) + regressed CLIP",
    "regressed_latent_0.9_regressed_cond": "regressed latent (s=0.9) + regressed CLIP",
}

# Which metrics are available and their comparison direction (True = higher means more
# similar to the reference). All are paired, per-image scores -- not a distributional
# metric like FID, since we have known image-to-image correspondence here.
HIGHER_IS_BETTER = {"mse": False, "psnr": True, "ssim": True, "clip_cosine": True}


def label_for_tag(tag):
    return TAG_LABELS.get(tag, tag.replace("_", " "))


def discover_setups(output_dir):
    setups = {}
    for path in sorted(glob(os.path.join(output_dir, "*denoised_images.pt"))):
        m = FILENAME_RE.match(os.path.basename(path))
        if not m:
            continue
        tag = m.group("tag") or ""
        setups[tag] = path
    return setups


def metric_mse(ref, comp):
    return ((ref - comp) ** 2).flatten(1).mean(dim=1).cpu().numpy()


def metric_psnr(ref, comp):
    mse = ((ref - comp) ** 2).flatten(1).mean(dim=1).clamp_min(1e-10)
    return (10 * torch.log10(1.0 / mse)).cpu().numpy()


def _gaussian_window(window_size, sigma, channels, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    window_2d = g.T @ g
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def metric_ssim(ref, comp, window_size=11, sigma=1.5):
    # Standard single-scale SSIM (Wang et al. 2004), implemented directly to avoid an
    # extra scikit-image/torchmetrics dependency -- images are assumed in [0, 1].
    C = ref.shape[1]
    window = _gaussian_window(window_size, sigma, C, ref.device, ref.dtype)
    pad = window_size // 2
    mu_ref, mu_comp = F.conv2d(ref, window, padding=pad, groups=C), F.conv2d(comp, window, padding=pad, groups=C)
    mu_ref_sq, mu_comp_sq, mu_ref_comp = mu_ref ** 2, mu_comp ** 2, mu_ref * mu_comp
    sigma_ref_sq = F.conv2d(ref * ref, window, padding=pad, groups=C) - mu_ref_sq
    sigma_comp_sq = F.conv2d(comp * comp, window, padding=pad, groups=C) - mu_comp_sq
    sigma_ref_comp = F.conv2d(ref * comp, window, padding=pad, groups=C) - mu_ref_comp
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_ref_comp + C1) * (2 * sigma_ref_comp + C2)) / (
        (mu_ref_sq + mu_comp_sq + C1) * (sigma_ref_sq + sigma_comp_sq + C2)
    )
    return ssim_map.flatten(1).mean(dim=1).cpu().numpy()


_clip_embed_fn_cache = {}


def _get_clip_embed_fn(device, model_id="openai/clip-vit-large-patch14"):
    # Same encoder/normalization convention as scripts/extract_image_embeddings.py's
    # clip_vit_l14 entry, so scores are comparable to that pipeline's embeddings.
    if device not in _clip_embed_fn_cache:
        from transformers import CLIPModel, CLIPImageProcessor

        print(f"Loading {model_id} for clip_cosine scoring")
        clip_model = CLIPModel.from_pretrained(model_id).to(device).eval()
        for p in clip_model.parameters():
            p.requires_grad_(False)
        processor = CLIPImageProcessor.from_pretrained(model_id)
        normalize = transforms.Normalize(mean=processor.image_mean, std=processor.image_std)

        @torch.no_grad()
        def embed(images):
            resized = F.interpolate(images, size=(224, 224), mode="bicubic", align_corners=False)
            normed = normalize(resized.clamp(0, 1))
            feats = clip_model.get_image_features(pixel_values=normed)
            if not isinstance(feats, torch.Tensor):
                # newer transformers versions return BaseModelOutputWithPooling instead
                # of a bare tensor -- same fallback as scripts/extract_image_embeddings.py
                feats = getattr(feats, "image_embeds", getattr(feats, "pooler_output", feats[0]))
            return F.normalize(feats, dim=-1)

        _clip_embed_fn_cache[device] = embed
    return _clip_embed_fn_cache[device]


def metric_clip_cosine(ref, comp, device):
    embed = _get_clip_embed_fn(device)
    return (embed(ref) * embed(comp)).sum(dim=1).cpu().numpy()


@torch.no_grad()
def compute_metrics_for_setup(metric_names, ref_images, comp_images, device, batch_size):
    """Score one setup's images against the reference, one batch at a time, computing
    every requested metric per batch so ref/comp only need to be resident once."""
    assert ref_images.shape == comp_images.shape, \
        f"shape mismatch: {tuple(ref_images.shape)} vs {tuple(comp_images.shape)}"
    scores = {m: [] for m in metric_names}
    for start in range(0, ref_images.shape[0], batch_size):
        r = ref_images[start:start + batch_size].to(device)
        c = comp_images[start:start + batch_size].to(device)
        for m in metric_names:
            if m == "mse":
                scores[m].append(metric_mse(r, c))
            elif m == "psnr":
                scores[m].append(metric_psnr(r, c))
            elif m == "ssim":
                scores[m].append(metric_ssim(r, c))
            elif m == "clip_cosine":
                scores[m].append(metric_clip_cosine(r, c, device))
            else:
                raise ValueError(f"Unknown metric {m!r}, choose from {list(HIGHER_IS_BETTER)}")
    return {m: np.concatenate(v) for m, v in scores.items()}


def plot_comparison(scores_by_tag, metric, out_path):
    tags = list(scores_by_tag.keys())
    labels = [label_for_tag(t) for t in tags]
    means = np.array([scores_by_tag[t].mean() for t in tags])
    sems = np.array([scores_by_tag[t].std(ddof=1) / np.sqrt(len(scores_by_tag[t])) for t in tags])
    order = np.argsort(-means if HIGHER_IS_BETTER[metric] else means)
    direction = "higher = closer to reference" if HIGHER_IS_BETTER[metric] else "lower = closer to reference"

    plt.figure(figsize=(1.6 * len(tags) + 2, 4.5))
    plt.bar(np.arange(len(tags)), means[order], yerr=sems[order], capsize=4, color="steelblue")
    plt.xticks(np.arange(len(tags)), [labels[i] for i in order], rotation=20, ha="right")
    plt.ylabel(f"{metric} ({direction})")
    plt.title(f"Generation similarity to perfect-latent + perfect-CLIP reference\nmetric={metric}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return tags, order


def plot_distribution(scores_by_tag, metric, tags, order, out_path):
    labels = [label_for_tag(t) for t in tags]
    direction = "higher = closer to reference" if HIGHER_IS_BETTER[metric] else "lower = closer to reference"

    plt.figure(figsize=(1.6 * len(tags) + 2, 4.5))
    plt.violinplot([scores_by_tag[tags[i]] for i in order], showmeans=True)
    plt.xticks(np.arange(1, len(tags) + 1), [labels[i] for i in order], rotation=20, ha="right")
    plt.ylabel(f"{metric} ({direction})")
    plt.title(f"Per-image {metric} distribution by setup")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/sd_clip_img2img",
                         help="Directory to glob for *denoised_images.pt (each setup's generated images)")
    parser.add_argument("--eval_dir", type=str, default=None,
                         help="Where to write scores/plots/summary. Defaults to <output_dir>/eval")
    parser.add_argument("--reference_tag", type=str, default="",
                         help="Tag of the reference run (bare 'denoised_images.pt' = ''); every other "
                              "setup is scored against it, matched by index")
    parser.add_argument("--metrics", type=str, default="mse,psnr,ssim,clip_cosine",
                         help="Comma-separated list from mse,psnr,ssim,clip_cosine")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]
    for m in metric_names:
        assert m in HIGHER_IS_BETTER, f"Unknown metric {m!r}, choose from {list(HIGHER_IS_BETTER)}"

    eval_dir = args.eval_dir or os.path.join(args.output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    device = torch.device(discover_device())
    print(f"Device: {device}")

    setup_paths = discover_setups(args.output_dir)
    print(f"Found {len(setup_paths)} setup(s) in {args.output_dir}:")
    for tag, path in setup_paths.items():
        if tag not in TAG_LABELS:
            print(f"  (unrecognized tag {tag!r} -- add it to TAG_LABELS for a nicer plot label)")
        print(f"  tag={tag!r:36} label={label_for_tag(tag)!r:45} -> {os.path.basename(path)}")

    assert args.reference_tag in setup_paths, (
        f"No reference run found (tag={args.reference_tag!r}). Run scripts/sd_clip_img2img.py with "
        f"real latents + real CLIP cond and save its output as "
        f"{os.path.join(args.output_dir, 'denoised_images.pt')} first."
    )
    assert len(setup_paths) >= 2, "Need the reference run plus at least one other setup to compare."

    print(f"Loading reference ({label_for_tag(args.reference_tag)!r}) from {setup_paths[args.reference_tag]}")
    ref_images = torch.load(setup_paths[args.reference_tag], map_location="cpu").float()
    assert ref_images.dim() == 4 and ref_images.shape[1] == 3, \
        f"expected [N, 3, H, W] images, got {tuple(ref_images.shape)}"

    # tag -> metric -> per-image scores
    all_scores = {m: {} for m in metric_names}
    other_tags = [t for t in setup_paths if t != args.reference_tag]
    for tag in tqdm(other_tags, desc="Scoring setups"):
        comp_images = torch.load(setup_paths[tag], map_location="cpu").float()
        assert comp_images.shape[0] == ref_images.shape[0], (
            f"setup {tag!r} has {comp_images.shape[0]} images, reference has {ref_images.shape[0]} -- "
            "per-image pairing assumes matching stimulus order/count across setups"
        )
        scores = compute_metrics_for_setup(metric_names, ref_images, comp_images, device, args.batch_size)
        for m in metric_names:
            all_scores[m][tag] = scores[m]
            print(f"  {label_for_tag(tag)!r}: {m} mean={scores[m].mean():.4f} "
                  f"sem={scores[m].std(ddof=1) / np.sqrt(len(scores[m])):.4f}")
        del comp_images
        gc.collect()

    summary = {}
    for m in metric_names:
        scores_path = os.path.join(eval_dir, f"{m}_scores.npz")
        np.savez(scores_path, **all_scores[m])
        print(f"Saved {m} per-image scores to {scores_path}")

        tags, order = plot_comparison(all_scores[m], m, os.path.join(eval_dir, f"{m}_comparison.png"))
        plot_distribution(all_scores[m], m, tags, order, os.path.join(eval_dir, f"{m}_distribution.png"))

        for tag in tags:
            s = all_scores[m][tag]
            summary[f"{m}/{tag or '(reference tag)'}"] = {
                "mean": float(s.mean()), "sem": float(s.std(ddof=1) / np.sqrt(len(s))), "n": int(len(s)),
            }

    summary_path = os.path.join(eval_dir, "eval_summary.yaml")
    with open(summary_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=True)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
