import json
import wandb
import torch
import argparse
import pickle as pkl
from dataclasses import asdict
from typing import Optional, Tuple
from torch.utils.data import DataLoader

from data_utils import PreloadedDataset
from models.beta_vae import BetaVAE
from models.unet import UNet
from scripts.config import BetaVAEConfig, DiffusionConfig
from scripts.diffusion_trainer import DiffusionTrainer
from utils import discover_device, count_model_params


def _load_frozen_vae(config: DiffusionConfig, device: torch.device) -> BetaVAE:
    with open(config.betavae_config_path) as f:
        vae_cfg = BetaVAEConfig(**json.load(f))
    vae = BetaVAE(
        input_channels=vae_cfg.input_channels,
        input_height=vae_cfg.input_height,
        input_width=vae_cfg.input_width,
        latent_dim=vae_cfg.latent_dim,
        hidden_dim=vae_cfg.hidden_dim,
        beta=vae_cfg.beta,
        device=device,
    )
    ckpt = torch.load(config.betavae_ckpt_path, map_location=device)
    vae.load_state_dict(ckpt["model_state_dict"])
    vae.eval()
    print(f"Loaded frozen β-VAE from {config.betavae_ckpt_path} (step {ckpt['step']})")
    return vae


def _setup_data_loaders(
    train_path: Optional[str],
    val_path: Optional[str],
    batch_size: int,
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:

    def make_loader(path: Optional[str], shuffle: bool) -> Optional[DataLoader]:
        if not path:
            return None
        with open(path, 'rb') as f:
            tensor = pkl.load(f)
        dataset = PreloadedDataset(tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
        print(f"{path}: {len(dataset)} images, {len(loader)} batches/step-cycle")
        return loader

    return make_loader(train_path, shuffle=True), make_loader(val_path, shuffle=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        config = DiffusionConfig(**json.load(f))

    device = torch.device(discover_device())

    vae = _load_frozen_vae(config, device)

    model = UNet(
        in_channels=config.in_channels,
        image_size=config.image_size,
        model_channels=config.model_channels,
        channel_mult=config.channel_mult,
        num_res_blocks=config.num_res_blocks,
        attention_resolutions=config.attention_resolutions,
        dropout=config.dropout,
        latent_dim=config.latent_dim,
    )
    count_model_params(model)

    train_loader, val_loader = _setup_data_loaders(
        config.train_dataset_path, config.test_dataset_path, config.batch_size
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    wandb.init(
        project="diffusion_ban_map",
        name=f"{config.experiment_name}_run1",
        config=asdict(config),
    )

    trainer = DiffusionTrainer(
        model=model,
        vae_model=vae,
        optimizer=optimizer,
        scheduler=None,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=config,
        device=device,
    )
    trainer.train(resume_ckpt_path=args.resume)
