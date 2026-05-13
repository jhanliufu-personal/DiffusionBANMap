import argparse
import types

import wandb
import yaml
import torch

from data_utils import build_dataloaders
from models.beta_vae import BetaVAE
from scripts.betavae_trainer import BetaVAETrainer
from utils import discover_device, count_model_params, make_run_tag


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = types.SimpleNamespace(**yaml.safe_load(f))

    cfg.output_dir = f"{cfg.output_dir}_{make_run_tag(cfg)}"
    device = torch.device(discover_device())

    train_loader, val_loader = build_dataloaders(cfg)

    model = BetaVAE(
        input_channels=cfg.input_channels,
        input_height=cfg.input_height,
        input_width=cfg.input_width,
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        beta=cfg.beta,
        device=device,
    )
    count_model_params(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    wandb.init(project="diffusion_ban_map", name=cfg.experiment_name, config=vars(cfg))

    trainer = BetaVAETrainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=cfg,
        device=device,
    )
    trainer.train()
