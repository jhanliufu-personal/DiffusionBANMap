"""
Trainer for the conditional pixel-space diffusion model.

Training objective: ε-prediction MSE (Ho et al. 2020).
Conditioning: β-VAE posterior mean μ, obtained by passing the clean image
through a frozen encoder at every training step.
"""

import math
import time
import torch
import torch.nn.functional as F
import wandb
from typing import Optional, Dict
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from scripts.config import DiffusionConfig
from scripts.base_trainer import BaseTrainer


def _cosine_betas(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Improved DDPM cosine noise schedule (Nichol & Dhariwal 2021)."""
    t = torch.arange(num_timesteps + 1) / num_timesteps
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(max=0.999)


def _linear_betas(num_timesteps: int) -> torch.Tensor:
    return torch.linspace(1e-4, 0.02, num_timesteps)


class DiffusionTrainer(BaseTrainer):

    def __init__(
        self,
        model: torch.nn.Module,
        vae_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        train_dataloader: Optional[DataLoader],
        val_dataloader: Optional[DataLoader],
        config: DiffusionConfig,
        device: torch.device,
    ):
        super().__init__(model, optimizer, scheduler, config, device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Frozen beta-VAE encoder — never updated
        self.vae_model = vae_model.to(device).eval()
        for p in self.vae_model.parameters():
            p.requires_grad_(False)

        # Noise schedule buffers (registered on device)
        T = config.num_timesteps
        if config.beta_schedule == 'cosine':
            betas = _cosine_betas(T)
        elif config.beta_schedule == 'linear':
            betas = _linear_betas(T)
        else:
            raise ValueError(f"Unknown beta_schedule: {config.beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer = lambda name, val: setattr(self, name, val.to(device))
        self.register_buffer('betas', betas)
        self.register_buffer('sqrt_alphas_cumprod', alphas_cumprod.sqrt())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', (1 - alphas_cumprod).sqrt())
        # SNR_t = ᾱ_t / (1 - ᾱ_t); used for Min-SNR-γ loss weighting
        self.register_buffer('snr', alphas_cumprod / (1.0 - alphas_cumprod))

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self, resume_ckpt_path: Optional[str] = None):
        elapsed_offset = 0.0
        if resume_ckpt_path:
            elapsed_offset = self._load_checkpoint(resume_ckpt_path)

        print(f"\n=== Starting Diffusion Training ({self.config.num_steps} steps) ===\n")
        start_time = time.time()
        self.model = self.model.to(self.device).train()

        loader_iter = iter(self.train_dataloader)

        while self.step < self.config.num_steps:
            # Cycle through dataset indefinitely
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.train_dataloader)
                batch = next(loader_iter)

            loss, grad_norm = self._train_step(batch)
            elapsed = elapsed_offset + (time.time() - start_time)

            if not self.step % self.config.log_interval:
                print(f"Step {self.step}/{self.config.num_steps}")
                wandb.log({
                    "train/loss": loss,
                    "train/grad_norm": grad_norm,
                }, step=self.step)

            if self.val_dataloader is not None and not self.step % self.config.eval_interval:
                val_loss = self._evaluate()
                wandb.log({"val/loss": val_loss}, step=self.step)

            if not self.step % self.config.ckpt_interval:
                self._save_checkpoint(elapsed)

            self.step += 1

        print("\n=== Training Complete ===\n")

    def _train_step(self, batch: torch.Tensor):
        x0 = batch.to(self.device)

        # Conditioning: frozen encoder → posterior mean μ
        with torch.no_grad():
            z, _ = self.vae_model.encode(x0)   # mu, logvar

        # Sample t and noise
        B = x0.shape[0]
        t = torch.randint(0, self.config.num_timesteps, (B,), device=self.device)
        eps = torch.randn_like(x0)

        # Forward diffusion: x_t = sqrt(ᾱ_t) * x0 + sqrt(1-ᾱ_t) * ε
        sqrt_a = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_1ma = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        x_t = sqrt_a * x0 + sqrt_1ma * eps

        self.optimizer.zero_grad()
        eps_pred = self.model(x_t, t, z)

        # Min-SNR-γ weighting: downweight easy high-noise timesteps so the model
        # is forced to learn low-noise denoising that actually drives sample quality.
        # weight_t = min(SNR_t, γ) / SNR_t  (Hang et al. 2023, γ=5 is standard)
        snr_t = self.snr[t]                                              # [B]
        weight = (snr_t.clamp(max=self.config.min_snr_gamma) / snr_t).detach()
        loss_per_sample = F.mse_loss(eps_pred, eps, reduction='none').mean(dim=[1, 2, 3])
        loss = (loss_per_sample * weight).mean()
        loss.backward()
        grad_norm = clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm).item()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return loss.item(), grad_norm

    @torch.no_grad()
    def _evaluate(self) -> float:
        self.model.eval()
        total, n = 0.0, 0
        for batch in self.val_dataloader:
            x0 = batch.to(self.device)
            z, _ = self.vae_model.encode(x0)
            B = x0.shape[0]
            t = torch.randint(0, self.config.num_timesteps, (B,), device=self.device)
            eps = torch.randn_like(x0)
            sqrt_a = self.sqrt_alphas_cumprod[t][:, None, None, None]
            sqrt_1ma = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
            x_t = sqrt_a * x0 + sqrt_1ma * eps
            eps_pred = self.model(x_t, t, z)
            snr_t = self.snr[t]
            weight = (snr_t.clamp(max=self.config.min_snr_gamma) / snr_t).detach()
            loss_per_sample = F.mse_loss(eps_pred, eps, reduction='none').mean(dim=[1, 2, 3])
            total += ((loss_per_sample * weight).sum()).item()
            n += B
        self.model.train()
        return total / n
