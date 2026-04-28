"""
β-VAE Trainer Class
Handles training, validation, and visualization for β-VAE models.
"""

import os
import json
import time
import torch
import wandb
from typing import Optional, Dict
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# from torchvision.utils import save_image, make_grid
from utils import beta_vae_loss #, visualize_reconstructions
from scripts.config import TrainingConfig


class BetaVAETrainer:
    """
    Trainer class for β-VAE model with config-based training.
    """
    
    def __init__(self, 
        model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler], 
        train_dataloader: Optional[DataLoader], val_dataloader: Optional[DataLoader],
        config: TrainingConfig, device: torch.device
    ):
        """
        Initialize trainer with configuration file.
        
        Args:
            config_path: Path to JSON configuration file
        """
        self.config = config        
        self.device = torch.device(device)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.start_time = None
        self.step = 0

        # Create output directories        
        self.ckpt_dir = os.path.join(self.config.output_dir, 'checkpoints')
        self.vis_dir = os.path.join(self.config.output_dir, 'visualizations')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)
                
        # Save config to output directory
        config_save_path = os.path.join(self.config.output_dir, 'config.json')
        with open(config_save_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Trainer initialized. Output directory: {self.config.output_dir}")
    

    def train(self, resume_ckpt_path: Optional[str] = None):
        # Elapsed time accumulated in previous runs (non-zero when resuming)
        elapsed_time_offset = 0.0

        if resume_ckpt_path:
            ckpt = torch.load(resume_ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if self.scheduler is not None and "scheduler_state_dict" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.step = ckpt["step"]
            elapsed_time_offset = ckpt["elapsed_time"]

        print("\n=== Starting β-VAE Training ===\n")
        print(f"Training for {self.config.num_epochs} epochs")

        self.start_time = time.time()
        self.model = self.model.to(self.device)
        self.model.train()
        while self.step <= self.config.num_epochs:

            train_metrics = self.train_epoch()
            # Total elapsed time = time accumulated before this run + time elapsed in this session
            elapsed_time = elapsed_time_offset + (time.time() - self.start_time)

            # Log metrics with wandb
            if not self.step % self.config.log_interval:
                wandb.log({
                    "train/total_loss": train_metrics['total_loss'],
                    "train/recon_loss": train_metrics['reconstruction_loss'],
                    "train/beta_kl_loss": train_metrics['beta_weighted_kl'],
                    "train/grad_norm_mean": train_metrics['grad_norm_mean'],
                    "train/grad_norm_max": train_metrics['grad_norm_max'],
                    "train/grad_clip_frac": train_metrics['grad_clip_frac']
                }, step=self.step)

            # Run eval
            if not self.step % self.config.eval_interval:
                val_metrics = self.evaluate()
                wandb.log({
                    "val/total_loss": val_metrics['total_loss'],
                    "val/recon_loss": val_metrics['reconstruction_loss'],
                    "val/beta_kl_loss": val_metrics['beta_weighted_kl']
                }, step=self.step)
                # self._visualize_reconstructions(epoch)
                # self._generate_samples(epoch)

            # Save checkpoint
            if not self.step % self.config.ckpt_interval:
                ckpt = {
                    "step": self.step,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "elapsed_time": elapsed_time
                }
                if self.scheduler is not None:
                    ckpt["scheduler_state_dict"] = self.scheduler.state_dict()
                torch.save(ckpt, os.path.join(self.ckpt_dir, f"{self.config.experiment_name}_step{self.step}.pt"))

            self.step += 1

            # # Special handling for final epoch - save individual reconstructions
            # if epoch == num_epochs:
            #     print("\n=== Final Epoch: Saving detailed reconstructions ===\n")
            #     self._visualize_reconstructions(
            #         epoch, num_samples=16, save_individual=True
            #     )

            #     # Generate extra samples
            #     self._generate_samples(epoch, num_samples=32)

            #     print("Final visualizations saved!")

        print("\n=== Training Complete ===\n")


    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_metrics = {
            "total_loss": 0.0,
            "reconstruction_loss": 0.0,
            "kl_loss": 0.0,
            "beta_weighted_kl": 0.0,
            "grad_norm_mean": 0.0,
            "grad_norm_max": 0.0,
            # Fraction of steps where clipping actually fired
            "grad_clip_frac": 0.0
        }

        if self.train_dataloader is None:
            raise RuntimeError("train_dataloader is required for training")

        num_batches = 0
        for _, data_batch in enumerate(self.train_dataloader):
            num_batches += 1
            data_batch = data_batch.to(self.device)
            self.optimizer.zero_grad()
            x_recon, mu, logvar = self.model(data_batch)

            loss, loss_dict = beta_vae_loss(x_recon, data_batch, mu, logvar, self.config.beta)
            loss.backward()
            # Prevent loss spike and weight update overshoot
            grad_norm = clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            # if self.scheduler:
            #     self.scheduler.step()

            # Accumulate losses
            for key, value in loss_dict.items():
                epoch_metrics[key] += value

            epoch_metrics["grad_norm_mean"] += grad_norm.item()
            epoch_metrics["grad_norm_max"] = max(epoch_metrics.get("grad_norm_max", 0), grad_norm.item())
            epoch_metrics["clip_frac"] += float(grad_norm.item() > self.config.max_grad_norm)

        # Average metrics over epoch
        for key in epoch_metrics:
            if key == "grad_norm_max":
                continue
            epoch_metrics[key] /= num_batches

        return epoch_metrics

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        
        self.model.eval()
        epoch_metrics = {
            "total_loss": 0.0,
            "reconstruction_loss": 0.0,
            "kl_loss": 0.0,
            "beta_weighted_kl": 0.0,
        }

        if self.val_dataloader is None:
            raise RuntimeError("val_dataloader is required for evaluation")

        num_batches = 0
        with torch.no_grad():
            for data_batch in self.val_dataloader:
                num_batches += 1
                data_batch = data_batch.to(self.device)
                x_recon, mu, logvar = self.model(data_batch)

                _, loss_dict = beta_vae_loss(x_recon, data_batch, mu, logvar, self.config.beta)
            
                # Accumulate losses
                for key, value in loss_dict.items():
                    epoch_metrics[key] += value
        
        # Average losses over epoch
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        self.model.train()
        return epoch_metrics


    # def _visualize_reconstructions(self, epoch: int, num_samples: int = 8, save_individual: bool = False):
    #     "Create and save reconstruction visualizations"
    #     self.model.eval()
    #     with torch.no_grad():
    #         # Get a batch from test set
    #         test_batch = next(iter(self.test_loader))[:num_samples]
    #         test_batch = test_batch.to(self.device)
            
    #         # Get reconstructions
    #         x_recon, mu, logvar = self.model(test_batch)
    #         # Create comparison grid
    #         comparison = torch.cat([test_batch, x_recon], dim=0)
    #         grid = make_grid(comparison.cpu(), nrow=num_samples, normalize=True, pad_value=1.0)
            
    #         # Save grid image
    #         grid_path = os.path.join(self.vis_dir, f'reconstructions_epoch_{epoch}.png')
    #         save_image(grid, grid_path)
            
    #     # Save individual reconstructions if requested
    #     if save_individual:
    #         for i in range(num_samples):
    #             orig_path = os.path.join(self.vis_dir, f'epoch_{epoch}_orig_{i}.png')                  
    #             recon_path = os.path.join(self.vis_dir, f'epoch_{epoch}_recon_{i}.png')
    #             save_image(test_batch[i].cpu(), orig_path, normalize=True)
    #             save_image(x_recon[i].cpu(), recon_path, normalize=True)
        
    #     print(f"Reconstructions saved to {grid_path}")              
    #     return grid_path
        
    # def _generate_samples(self, epoch: int, num_samples: int = 16):
    #     "Generate and save samples from the model"      
    #     self.model.eval()
    #     with torch.no_grad():
    #         # Generate samples
    #         samples = self.model.sample(num_samples, self.device)
    #         # Create grid
    #         grid = make_grid(samples.cpu(), nrow=4, normalize=True, pad_value=1.0)
            
    #     # Save samples
    #     samples_path = os.path.join(self.vis_dir, f'samples_epoch_{epoch}.png')
    #     save_image(grid, samples_path)
    #     print(f"Generated samples saved to {samples_path}")