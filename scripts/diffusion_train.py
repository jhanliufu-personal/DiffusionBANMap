import os
import argparse
import types

import wandb
import yaml
import torch

from data_utils import build_dataloaders
from models.beta_vae import BetaVAE
from models.unet import UNet
from scripts.diffusion_trainer import DiffusionTrainer
from utils import discover_device, count_model_params, make_run_tag, build_lr_scheduler


def _load_frozen_vae(cfg, device: torch.device) -> BetaVAE:
    with open(cfg.betavae_config_path) as f:
        vae_cfg = types.SimpleNamespace(**yaml.safe_load(f))
    vae = BetaVAE(
        input_channels=vae_cfg.input_channels,
        input_height=vae_cfg.input_height,
        input_width=vae_cfg.input_width,
        latent_dim=vae_cfg.latent_dim,
        hidden_dim=vae_cfg.hidden_dim,
        beta=vae_cfg.beta,
        device=device,
    )
    ckpt = torch.load(cfg.betavae_ckpt_path, map_location=device)
    vae.load_state_dict(ckpt["model_state_dict"])
    vae.eval()
    print(f"Loaded frozen β-VAE from {cfg.betavae_ckpt_path} (step {ckpt['step']})")
    return vae


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--run_name", type=str, default=None, help="Appended to cfg.experiment_name for the wandb run name")
    parser.add_argument("--notes", type=str, default=None, help="wandb run notes")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = types.SimpleNamespace(**yaml.safe_load(f))

    cfg.output_dir = f"{cfg.output_dir}_{make_run_tag(cfg)}"

    resume_ckpt_path = args.resume or getattr(cfg, "resume_ckpt_path", None)
    if resume_ckpt_path:
        print(f"Will resume training from {resume_ckpt_path}")

    device = torch.device(discover_device())
    print(f"Device: {device}")

    unconditional = getattr(cfg, "unconditional", False)
    if unconditional:
        vae = None
        print("Train for unconditional generation")
        print("Unconditional — no VAE, z is a zero vector")
    elif hasattr(cfg, "betavae_config_path"):
        vae = _load_frozen_vae(cfg, device)
    else:
        # Conditional, but no VAE configured — the dataloader is expected to already
        # provide precomputed latents (e.g. AlexNet-fc6-PCA, see
        # notebooks/alexnet_pca_latents.ipynb) as (image, latent) pairs.
        vae = None
        print("Conditional on precomputed latents — no VAE to load, expecting (image, latent) pairs from the dataloader")

    model = UNet(
        in_channels=cfg.in_channels,
        image_size=cfg.image_size,
        model_channels=cfg.model_channels,
        channel_mult=cfg.channel_mult,
        num_res_blocks=cfg.num_res_blocks,
        attention_resolutions=cfg.attention_resolutions,
        dropout=cfg.dropout,
        num_head_channels=getattr(cfg, "num_head_channels", 64),
        latent_dim=cfg.latent_dim,
    )
    count_model_params(model)

    train_loader, val_loader = build_dataloaders(cfg)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = build_lr_scheduler(optimizer, cfg)

    run_name = f"{cfg.experiment_name}_{args.run_name}" if args.run_name else cfg.experiment_name
    wandb.login(key=os.environ["WANDB_API_KEY"])
    wandb.init(project="diffusion_ban_map", name=run_name, notes=args.notes, config=vars(cfg))

    trainer = DiffusionTrainer(
        model=model,
        vae_model=vae,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=cfg,
        device=device,
    )
    trainer.train(resume_ckpt_path=resume_ckpt_path)
