import time
import torch
import wandb
from typing import Optional, Dict
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from utils import beta_vae_loss
from scripts.config import BetaVAEConfig
from scripts.base_trainer import BaseTrainer


class BetaVAETrainer(BaseTrainer):

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        train_dataloader: Optional[DataLoader],
        val_dataloader: Optional[DataLoader],
        config: BetaVAEConfig,
        device: torch.device,
    ):
        super().__init__(model, optimizer, scheduler, config, device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.start_time = None

    def train(self, resume_ckpt_path: Optional[str] = None):
        elapsed_time_offset = 0.0
        if resume_ckpt_path:
            elapsed_time_offset = self._load_checkpoint(resume_ckpt_path)

        print(f"\n=== Starting β-VAE Training ({self.config.num_epochs} epochs) ===\n")
        self.start_time = time.time()
        self.model = self.model.to(self.device)
        self.model.train()

        while self.step <= self.config.num_epochs:
            train_metrics = self.train_epoch()
            elapsed_time = elapsed_time_offset + (time.time() - self.start_time)

            if not self.step % self.config.log_interval:
                print(f"Epoch {self.step}/{self.config.num_epochs}")
                wandb.log({
                    "train/total_loss": train_metrics['total_loss'],
                    "train/recon_loss": train_metrics['reconstruction_loss'],
                    "train/beta_kl_loss": train_metrics['beta_weighted_kl'],
                    "train/grad_norm_mean": train_metrics['grad_norm_mean'],
                    "train/grad_norm_max": train_metrics['grad_norm_max'],
                    "train/grad_clip_frac": train_metrics['grad_clip_frac'],
                }, step=self.step)

            if not self.step % self.config.eval_interval:
                val_metrics = self.evaluate()
                wandb.log({
                    "val/total_loss": val_metrics['total_loss'],
                    "val/recon_loss": val_metrics['reconstruction_loss'],
                    "val/beta_kl_loss": val_metrics['beta_weighted_kl'],
                }, step=self.step)

            if not self.step % self.config.ckpt_interval:
                self._save_checkpoint(elapsed_time)

            self.step += 1

        print("\n=== Training Complete ===\n")

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        metrics = {
            "total_loss": 0.0,
            "reconstruction_loss": 0.0,
            "kl_loss": 0.0,
            "beta_weighted_kl": 0.0,
            "grad_norm_mean": 0.0,
            "grad_norm_max": 0.0,
            "grad_clip_frac": 0.0,
        }
        if self.train_dataloader is None:
            raise RuntimeError("train_dataloader is required for training")

        num_batches = 0
        for batch in self.train_dataloader:
            num_batches += 1
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            x_recon, mu, logvar = self.model(batch)
            loss, loss_dict = beta_vae_loss(x_recon, batch, mu, logvar, self.config.beta)
            loss.backward()
            grad_norm = clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            for key, val in loss_dict.items():
                metrics[key] += val
            metrics["grad_norm_mean"] += grad_norm.item()
            metrics["grad_norm_max"] = max(metrics["grad_norm_max"], grad_norm.item())
            metrics["grad_clip_frac"] += float(grad_norm.item() > self.config.max_grad_norm)

        for key in metrics:
            if key != "grad_norm_max":
                metrics[key] /= num_batches
        return metrics

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        if self.val_dataloader is None:
            raise RuntimeError("val_dataloader is required for evaluation")
        self.model.eval()
        metrics = {
            "total_loss": 0.0,
            "reconstruction_loss": 0.0,
            "kl_loss": 0.0,
            "beta_weighted_kl": 0.0,
        }
        num_batches = 0
        for batch in self.val_dataloader:
            num_batches += 1
            batch = batch.to(self.device)
            x_recon, mu, logvar = self.model(batch)
            _, loss_dict = beta_vae_loss(x_recon, batch, mu, logvar, self.config.beta)
            for key, val in loss_dict.items():
                metrics[key] += val
        for key in metrics:
            metrics[key] /= num_batches
        self.model.train()
        return metrics
