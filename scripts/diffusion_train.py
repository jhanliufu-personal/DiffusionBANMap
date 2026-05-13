import argparse
import types

import wandb
import yaml
import torch

from data_utils import build_face_dataloaders
from models.beta_vae import BetaVAE
from models.unet import UNet
from scripts.diffusion_trainer import DiffusionTrainer
from utils import discover_device, count_model_params


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
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = types.SimpleNamespace(**yaml.safe_load(f))

    device = torch.device(discover_device())

    vae = _load_frozen_vae(cfg, device)

    model = UNet(
        in_channels=cfg.in_channels,
        image_size=cfg.image_size,
        model_channels=cfg.model_channels,
        channel_mult=cfg.channel_mult,
        num_res_blocks=cfg.num_res_blocks,
        attention_resolutions=cfg.attention_resolutions,
        dropout=cfg.dropout,
        latent_dim=cfg.latent_dim,
    )
    count_model_params(model)

    train_loader, val_loader = build_face_dataloaders(
        cfd_dir=cfg.cfd_dir,
        expressions=cfg.expressions,
        image_size=(cfg.image_size, cfg.image_size),
        train_split=cfg.train_split,
        batch_size=cfg.batch_size,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    wandb.init(project="diffusion_ban_map", name=cfg.experiment_name, config=vars(cfg))

    trainer = DiffusionTrainer(
        model=model,
        vae_model=vae,
        optimizer=optimizer,
        scheduler=None,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=cfg,
        device=device,
    )
    trainer.train(resume_ckpt_path=args.resume)
