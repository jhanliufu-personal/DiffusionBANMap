"""
Class agreement over a sweep of checkpoints of a conditional diffusion run.

For a fixed random subset of the run's val images: take each image's precomputed
conditioning embedding (the same val_latents_z{N}.npy row the model was trained/validated
with), generate one image from it, and classify both the original and the generated
image with ImageNet classifiers. A conditional model that captures the semantics of its
conditioning embedding should generate an image of the same class as the original, so
the main number is the top-1 agreement rate. No ground-truth labels are needed, so this
works for the stimuli as well as ImageNet64.

Per classifier, each checkpoint's yaml records:
    top1_agreement            argmax(generated) == argmax(original)
    top5_agreement            argmax(original) is in top-5(generated)
    shuffled_top1_agreement   chance baseline: generated image i vs original image perm(i),
                              i.e. how often unrelated pairs agree given each label distribution
    original_top1 / generated_top1   per-image predicted class indices, for later analysis

Caveat: with alexnet_fc6 conditioning, the alexnet classifier is partly circular -- its
logits are a function of fc6. Report an independent classifier (resnet50, vit_b_16) too.

The run is identified by its output folder (not its config, since several runs can share
one config file): its saved config.json gives everything needed to rebuild the model and
the val data. By default only the run's latest periodic checkpoint is evaluated; --ckpts
picks others (specific steps, best_ckpt.pt, or all of them). Outputs go into
<run_dir>/class_agreement/, per checkpoint <stem>.pt:
    <stem>_g{guidance}.yaml            agreement scores + run info
    <stem>_g{guidance}_generated.pt    generated images [N, C, H, W] in [0, 1] (cache, lets
                                       a later run add classifiers without regenerating)
    <stem>_g{guidance}_examples.png    originals vs generated, with predicted labels

Run from repo root:
    python -m scripts.class_agreement_sweep \\
        --run_dir outputs/diffusion_full_imagenet64_flow_grad_accum_ema_h64_mc192_ch1x2x3x4_T1000_flow_alexnet_fc6_z200 \\
        --classifiers alexnet,resnet50 --n_images 1000 --guidance_scale 1.0

    # specific checkpoints / every checkpoint instead of just the latest
    ... --ckpts 100000,200000,best_ckpt.pt
    ... --ckpts all
"""

import os
import json
import hashlib
import types
import argparse
from datetime import datetime

import yaml
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import models
from tqdm import tqdm

from models.unet import UNet
from models.ema import ema_shadow_to_model_state_dict
from utils import (
    make_noise_schedule, diffusion_sample, flow_sample, vpsde_sample, discover_device, select_checkpoints,
)
from data_utils import build_dataloaders, build_imagenet64_val_dataloader


# name -> (torchvision constructor, pretrained ImageNet-1k weights)
CLASSIFIERS = {
    "alexnet": (models.alexnet, models.AlexNet_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "vit_b_16": (models.vit_b_16, models.ViT_B_16_Weights.IMAGENET1K_V1),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True,
                   help="Run output folder (<config output_dir>_<run tag>) holding config.json and checkpoints/")
    p.add_argument("--ckpts", type=str, default="latest",
                   help="'latest' (highest-step ckpt_step_*.pt, default), 'all' (every ckpt_step_*.pt), or a "
                        "comma-separated list of steps and/or filenames in <run_dir>/checkpoints, "
                        "e.g. '100000,200000,best_ckpt.pt'")
    p.add_argument("--classifiers", type=str, default="alexnet,resnet50",
                   help=f"comma-separated, from {list(CLASSIFIERS)}")
    p.add_argument("--n_images", type=int, default=1000,
                   help="val images to evaluate (capped at the val split size); same subset for every checkpoint")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--use_ema", dest="use_ema", action="store_true", default=True,
                   help="Sample from each checkpoint's EMA shadow when present (default: on)")
    p.add_argument("--no_ema", dest="use_ema", action="store_false")
    p.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"],
                   help="DDPM-only; ignored when noise_process is 'flow' or 'vpsde'")
    p.add_argument("--eta", type=float, default=0.0, help="ddim only")
    p.add_argument("--num_inference_steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_show", type=int, default=8, help="Image pairs in the example figure")
    # Overrides for config.json fields, e.g. runs trained before encoding_model existed, or
    # trained on a machine where data_dir lived somewhere else.
    p.add_argument("--encoding_model", type=str, default=None,
                   help="Override/fill in config.json's encoding_model (e.g. alexnet_fc6 for older runs)")
    p.add_argument("--data_dir", type=str, default=None, help="Override config.json's data_dir")
    return p.parse_args()


def build_val_subset_loader(cfg, n_images: int, batch_size: int, seed: int):
    """Unshuffled loader of (image, conditioning latent) pairs over a fixed random subset of
    the run's val split -- the same images for every checkpoint and every rerun. Also returns
    the subset's row indices and the conditioning-latents file they come from."""
    if cfg.dataset_type == "imagenet64":
        # val-only: doesn't need train_images.npy on disk
        val_dl = build_imagenet64_val_dataloader(
            data_dir=cfg.data_dir, image_size=cfg.image_size, batch_size=batch_size,
            latent_dim=cfg.latent_dim, encoding_model=cfg.encoding_model,
        )
    else:
        _, val_dl = build_dataloaders(cfg)
    dataset = val_dl.dataset
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(dataset), size=min(n_images, len(dataset)), replace=False))
    latents_path = os.path.abspath(dataset.latents.filename)   # np.load(mmap_mode="r") memmap
    loader = DataLoader(Subset(dataset, idx.tolist()), batch_size=batch_size, shuffle=False, num_workers=4)
    return loader, idx, latents_path


def load_classifier(name: str, device):
    """Return (classify_fn, category_names). classify_fn maps [B, C, H, W] images in [0, 1]
    at any resolution to [B, 1000] logits; originals and generated images go through the
    exact same resize + normalization, so neither side is favoured."""
    ctor, weights = CLASSIFIERS[name]
    model = ctor(weights=weights).eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def classify(images: torch.Tensor, batch_size: int) -> torch.Tensor:
        logits = []
        for i in range(0, images.shape[0], batch_size):
            batch = images[i:i + batch_size].to(device)
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)
            batch = F.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)
            logits.append(model((batch - mean) / std).cpu())
        return torch.cat(logits, dim=0)

    return classify, weights.meta["categories"]


def agreement_scores(orig_logits: torch.Tensor, gen_logits: torch.Tensor, seed: int) -> dict:
    orig_top1 = orig_logits.argmax(dim=1)
    gen_top1 = gen_logits.argmax(dim=1)
    gen_top5 = gen_logits.topk(5, dim=1).indices
    perm = torch.from_numpy(np.random.default_rng(seed).permutation(len(orig_top1)))
    return {
        "top1_agreement": round(float((gen_top1 == orig_top1).float().mean()), 4),
        "top5_agreement": round(float((gen_top5 == orig_top1[:, None]).any(dim=1).float().mean()), 4),
        "shuffled_top1_agreement": round(float((gen_top1[perm] == orig_top1).float().mean()), 4),
        "original_top1": orig_top1.tolist(),
        "generated_top1": gen_top1.tolist(),
    }


def save_examples(originals, generated, orig_top1, gen_top1, categories, classifier, title, path, n_show):
    n_show = min(n_show, originals.shape[0])
    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 1.8, 4.4), squeeze=False)
    for row, (images, labels, name) in enumerate([(originals, orig_top1, "original"),
                                                  (generated, gen_top1, "generated")]):
        for j in range(n_show):
            ax = axes[row, j]
            img = images[j].clamp(0, 1).permute(1, 2, 0).numpy()
            ax.imshow(img.squeeze(-1) if img.shape[-1] == 1 else img, cmap="gray" if img.shape[-1] == 1 else None)
            ax.set_xticks([]), ax.set_yticks([])
            match = gen_top1[j] == orig_top1[j]
            ax.set_title(categories[labels[j]][:18], fontsize=8,
                         color="#0b0b0b" if row == 0 or match else "#b42318")
            if j == 0:
                ax.set_ylabel(name, fontsize=9)
    fig.suptitle(f"{title}\n{classifier} top-1 labels (red: generated label differs from original)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    classifier_names = [c.strip() for c in args.classifiers.split(",") if c.strip()]
    unknown = set(classifier_names) - set(CLASSIFIERS)
    if unknown:
        raise ValueError(f"Unknown classifiers {sorted(unknown)}, choose from {list(CLASSIFIERS)}")

    config_path = os.path.join(args.run_dir, "config.json")
    with open(config_path) as f:
        cfg = types.SimpleNamespace(**json.load(f))
    if args.encoding_model is not None:
        cfg.encoding_model = args.encoding_model
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir

    if getattr(cfg, "unconditional", False):
        raise ValueError(f"{args.run_dir} is an unconditional run -- class agreement needs a conditional model")
    if getattr(cfg, "encoding_model", None) is None:
        raise ValueError(
            f"{config_path} has no encoding_model (run predates the key, or is beta-VAE-conditioned) -- "
            "pass --encoding_model <encoder> for a run conditioned on precomputed latents"
        )
    if args.guidance_scale != 1.0 and not getattr(cfg, "cfg_uncond_prob", 0.0):
        print(f"WARNING: guidance_scale={args.guidance_scale} but the run was trained with cfg_uncond_prob=0 -- "
              "its unconditional branch was never trained, so guidance is not meaningful")

    device = torch.device(discover_device())
    print(f"Device: {device}")

    ckpt_paths = select_checkpoints(os.path.join(args.run_dir, "checkpoints"), args.ckpts)
    print(f"Evaluating {len(ckpt_paths)} checkpoint(s) (--ckpts {args.ckpts}): "
          + ", ".join(os.path.basename(p) for p in ckpt_paths))

    eval_dir = os.path.join(args.run_dir, "class_agreement")
    os.makedirs(eval_dir, exist_ok=True)

    # --- val subset: originals + their conditioning latents, shared across checkpoints ---
    subset_dl, subset_idx, latents_path = build_val_subset_loader(cfg, args.n_images, args.batch_size, args.seed)
    print(f"Conditioning latents: {latents_path}")
    originals, latents = [], []
    for imgs, z in subset_dl:
        originals.append(imgs)
        latents.append(z)
    originals, latents = torch.cat(originals), torch.cat(latents)
    print(f"Val subset: {originals.shape[0]} images, latents {tuple(latents.shape)}")

    # --- classifiers + their predictions on the originals, shared across checkpoints ---
    classifiers = {name: load_classifier(name, device) for name in classifier_names}
    orig_logits = {name: classify(originals, args.batch_size) for name, (classify, _) in classifiers.items()}

    # --- UNet skeleton + noise process, reused for every checkpoint ---
    unet = UNet(
        in_channels=cfg.in_channels,
        image_size=cfg.image_size,
        model_channels=cfg.model_channels,
        channel_mult=cfg.channel_mult,
        num_res_blocks=cfg.num_res_blocks,
        attention_resolutions=cfg.attention_resolutions,
        dropout=0.0,
        num_head_channels=getattr(cfg, "num_head_channels", 64),
        latent_dim=cfg.latent_dim,
    ).to(device).eval()
    np_type = getattr(cfg, "noise_process", "ddpm")
    schedule = make_noise_schedule(cfg, device) if np_type == "ddpm" else None

    @torch.no_grad()
    def generate(z: torch.Tensor) -> torch.Tensor:
        if np_type == "ddpm":
            return diffusion_sample(
                z, unet, schedule, cfg, device, sampler=args.sampler, eta=args.eta,
                num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
            )
        if np_type == "flow":
            return flow_sample(z, unet, cfg, device, num_steps=args.num_inference_steps,
                               guidance_scale=args.guidance_scale)
        if np_type == "vpsde":
            return vpsde_sample(z, unet, cfg, device, num_steps=args.num_inference_steps,
                                guidance_scale=args.guidance_scale)
        raise ValueError(f"Unknown noise_process: {np_type!r}")

    def fingerprint(ckpt_path: str) -> dict:
        """Everything the generated images depend on. A cached _generated.pt / yaml is only
        reused if its fingerprint matches, so changing the latents file, image subset, sampling
        settings, or overwriting a checkpoint can never silently pair stale generations with
        the current originals."""
        stat = os.stat(ckpt_path)
        return {
            "latents_path": latents_path,
            "latents_mtime": int(os.stat(latents_path).st_mtime),
            "subset_sha1": hashlib.sha1(subset_idx.tobytes()).hexdigest()[:16],
            "ckpt_size": int(stat.st_size),
            "ckpt_mtime": int(stat.st_mtime),
            "use_ema": args.use_ema,
            "guidance_scale": args.guidance_scale,
            "sampler": args.sampler,
            "eta": args.eta,
            "num_inference_steps": args.num_inference_steps,
            "seed": args.seed,
        }

    g_tag = f"g{args.guidance_scale:g}"
    for ckpt_path in ckpt_paths:
        stem = os.path.splitext(os.path.basename(ckpt_path))[0]
        yaml_path = os.path.join(eval_dir, f"{stem}_{g_tag}.yaml")
        generated_path = os.path.join(eval_dir, f"{stem}_{g_tag}_generated.pt")
        fig_path = os.path.join(eval_dir, f"{stem}_{g_tag}_examples.png")
        print(f"\n=== {stem} ===")

        fp = fingerprint(ckpt_path)
        info = {}
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                info = yaml.safe_load(f)
            if info.get("fingerprint") != fp:
                print(f"{yaml_path} was computed with different inputs/settings -- recomputing from scratch")
                info = {}
            elif set(classifier_names) <= set(info.get("classifiers", {})):
                print(f"All requested classifiers already scored in {yaml_path} -- skipping")
                continue

        diff_ckpt = torch.load(ckpt_path, map_location=device)
        used_ema = args.use_ema and "ema_state_dict" in diff_ckpt

        # --- generate (or reuse cached) images for this checkpoint ---
        cache = torch.load(generated_path) if os.path.exists(generated_path) else None
        if isinstance(cache, dict) and cache.get("fingerprint") == fp:
            generated = cache["images"]
            print(f"Loaded {generated.shape[0]} cached generated images from {generated_path}")
        else:
            if cache is not None:
                print(f"Ignoring stale cache {generated_path} (different inputs/settings) -- regenerating")
            unet.load_state_dict(diff_ckpt["model_state_dict"])
            if used_ema:
                unet.load_state_dict(ema_shadow_to_model_state_dict(unet, diff_ckpt["ema_state_dict"]["shadow"]))
            elif args.use_ema:
                print("use_ema=True but checkpoint has no ema_state_dict -- using raw weights")
            unet.eval()

            # Same starting noise for every checkpoint, so checkpoints differ only in weights.
            torch.manual_seed(args.seed)
            batches = []
            for i in tqdm(range(0, latents.shape[0], args.batch_size), desc=f"Sampling {stem}"):
                batches.append(generate(latents[i:i + args.batch_size].to(device)).cpu())
            generated = torch.cat(batches)
            torch.save({"images": generated, "fingerprint": fp}, generated_path)
            print(f"Saved {generated.shape[0]} generated images to {generated_path}")

        # --- classify + score ---
        scores = dict(info.get("classifiers", {}))
        for name, (classify, _) in classifiers.items():
            scores[name] = agreement_scores(orig_logits[name], classify(generated, args.batch_size), args.seed)
            print(f"  {name:>9}: top-1 agreement {scores[name]['top1_agreement']:.3f}  "
                  f"top-5 {scores[name]['top5_agreement']:.3f}  "
                  f"(shuffled baseline {scores[name]['shuffled_top1_agreement']:.3f})")

        first = classifier_names[0]
        save_examples(originals, generated, scores[first]["original_top1"], scores[first]["generated_top1"],
                      classifiers[first][1], first, f"{os.path.basename(os.path.normpath(args.run_dir))} / {stem}",
                      fig_path, args.n_show)

        info = {
            "run_dir": os.path.normpath(args.run_dir),
            "checkpoint": ckpt_path,
            "step": int(diff_ckpt.get("step", -1)),
            "used_ema": used_ema,
            "dataset_type": cfg.dataset_type,
            "data_dir": cfg.data_dir,
            "encoding_model": cfg.encoding_model,
            "latent_dim": cfg.latent_dim,
            "noise_process": np_type,
            "guidance_scale": args.guidance_scale,
            "sampler": args.sampler,
            "eta": args.eta,
            "num_inference_steps": args.num_inference_steps,
            "n_images": int(originals.shape[0]),
            "seed": args.seed,
            "experiment_name": getattr(cfg, "experiment_name", None),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "fingerprint": fp,
            "classifiers": scores,
        }
        with open(yaml_path, "w") as f:
            yaml.safe_dump(info, f, sort_keys=False, default_flow_style=None, width=1_000_000)
        print(f"Saved {yaml_path}")


if __name__ == "__main__":
    main()
