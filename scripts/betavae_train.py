import json
import wandb
import torch
import argparse
import pickle as pkl
from dataclasses import asdict
from typing import Tuple, Optional
from torch.utils.data import DataLoader

from data_utils import PreloadedDataset
from models.beta_vae import BetaVAE
from scripts.config import BetaVAEConfig
from scripts.betavae_trainer import BetaVAETrainer
from utils import discover_device, count_model_params


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
        print(f"{path}: {len(dataset)} images, {len(loader)} batches/epoch")
        return loader

    return make_loader(train_path, shuffle=True), make_loader(val_path, shuffle=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = BetaVAEConfig(**json.load(f))

    device = torch.device(discover_device())

    train_loader, val_loader = _setup_data_loaders(
        config.train_dataset_path, config.test_dataset_path, config.batch_size
    )

    model = BetaVAE(
        input_channels=config.input_channels,
        input_height=config.input_height,
        input_width=config.input_width,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        beta=config.beta,
        device=device,
    )
    count_model_params(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    wandb.init(
        project="diffusion_ban_map",
        name=config.experiment_name,
        config=asdict(config),
    )

    trainer = BetaVAETrainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=config,
        device=device,
    )
    trainer.train()
