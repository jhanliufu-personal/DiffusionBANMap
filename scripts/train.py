import json
import wandb
import torch
import argparse
import pickle as pkl
from dataclasses import asdict
from typing import Tuple, Optional
from data_utils import FaceDataset  # noqa: F401 — needed for pickle to resolve FaceDataset
from models.beta_vae import BetaVAE
from scripts.config import TrainingConfig
from torch.utils.data import DataLoader
from scripts.trainer import BetaVAETrainer
from utils import discover_device, count_model_params



def _setup_data_loaders(
    train_dataset_path: Optional[str], 
    test_dataset_path: Optional[str], 
    batch_size: int = 32
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:

    train_loader = None
    test_loader = None

    if train_dataset_path:
        with open(train_dataset_path, 'rb') as f:
            train_dataset = pkl.load(f)

        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4
        )

        print(f"Train dataset: {len(train_dataset)} images")
        print(f"Train batches per epoch: {len(train_loader)}")
    
    if test_dataset_path:
        with open(test_dataset_path, 'rb') as f:
            test_dataset = pkl.load(f)

        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4
        )
    
        print(f"Test dataset: {len(test_dataset)} images")      
        print(f"Test batches: {len(test_loader)}")

    return train_loader, test_loader


if __name__ == "__main__":

    # Read args from json
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = TrainingConfig(**json.load(f))

    device = torch.device(discover_device())

    train_dataloader, val_dataloader = _setup_data_loaders(
        config.train_dataset_path, config.test_dataset_path, batch_size=config.batch_size
    )

    beta_vae = BetaVAE(
        input_channels=config.input_channels,
        input_height=config.input_height,
        input_width=config.input_width,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        beta=config.beta,
        device=device
    )
    count_model_params(beta_vae)

    optimizer = torch.optim.AdamW(beta_vae.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    scheduler = None

    # Init wandb session
    wandb.init(
        project="diffusion_ban_map",
        name=f"{config.experiment_name}_run1",
        notes="Overfit beta VAE on mini batch to debug",
        config=asdict(config)
    )

    # Create trainer
    trainer = BetaVAETrainer(
        model=beta_vae, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader, 
        config=config, 
        device=device
    )
    trainer.train()