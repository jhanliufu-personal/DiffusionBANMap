"""
Trainer for conditional pixel-space diffusion models.

The noise process is selected via config.noise_process (default 'ddpm') and
encapsulated in a single self.np object.  All three processes (DDPM, flow,
VPSDE) expose the same interface, so _train_step, _evaluate, and _visualize
contain no per-process branching.

Config keys:
    noise_process:   'ddpm' | 'flow' | 'vpsde'  (default: 'ddpm')

    ddpm:   num_timesteps, beta_schedule ('cosine'|'linear'), min_snr_gamma
    flow:   num_timesteps used as sinusoidal embedding scale (default 1000)
    vpsde:  vpsde_beta_min (default 0.01), vpsde_beta_max (default 5.0),
            min_snr_gamma (default 0 = unweighted MSE),
            num_timesteps used as sinusoidal embedding scale (default 1000)

    mixed_precision: 'no' | 'bf16' | 'fp16'  (default: 'no')
        'bf16' needs no loss scaling (safe default on Ampere+ GPUs).
        'fp16' uses a GradScaler for loss scaling — needed on older GPUs
        without native bf16 support.
"""

import os
import time
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
from typing import Optional
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from scripts.base_trainer import BaseTrainer
from models.noise_process import DDPM, RectifiedFlow, VPSDE


class DiffusionTrainer(BaseTrainer):

    def __init__(
        self,
        model: torch.nn.Module,
        vae_model: Optional[torch.nn.Module],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        train_dataloader: Optional[DataLoader],
        val_dataloader: Optional[DataLoader],
        config,
        device: torch.device,
    ):
        super().__init__(model, optimizer, scheduler, config, device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.best_val_loss = float("inf")

        self.unconditional = getattr(config, 'unconditional', False)
        if self.unconditional:
            print("Train for unconditional generation")

        # Frozen beta-VAE encoder — never updated. Unconditional runs need no VAE at all.
        if vae_model is not None:
            self.vae_model = vae_model.to(device).eval()
            for p in self.vae_model.parameters():
                p.requires_grad_(False)
        else:
            self.vae_model = None

        self.np, self._embed_scale = self._build_noise_process(config, device)
        print(f"Noise process: {type(self.np).__name__}")

        self.mixed_precision, self._amp_dtype, self.scaler = self._build_amp(config, device)
        if self.mixed_precision != 'no':
            active = "active" if self._amp_dtype is not None else f"inactive — {device.type} isn't cuda"
            print(f"Mixed precision: {self.mixed_precision} ({active})")

    @staticmethod
    def _build_amp(config, device: torch.device):
        """Set up autocast dtype + GradScaler. 'bf16' needs no scaler (disabled no-op);
        'fp16' does. Autocast/scaling only apply on CUDA — elsewhere this is a no-op."""
        mode = getattr(config, 'mixed_precision', 'no')
        if mode not in ('no', 'bf16', 'fp16'):
            raise ValueError(f"Unknown mixed_precision: {mode!r}")

        use_amp = mode != 'no' and device.type == 'cuda'
        amp_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16}.get(mode) if use_amp else None
        scaler = torch.amp.GradScaler('cuda', enabled=(mode == 'fp16' and use_amp))
        return mode, amp_dtype, scaler

    @staticmethod
    def _build_noise_process(config, device):
        """Construct the noise process object and its sinusoidal embedding scale."""
        np_type       = getattr(config, 'noise_process', 'ddpm')
        embed_scale   = getattr(config, 'num_timesteps', 1000)
        min_snr_gamma = getattr(config, 'min_snr_gamma', 5.0)

        if np_type == 'ddpm':
            return DDPM(
                num_timesteps=config.num_timesteps,
                beta_schedule=config.beta_schedule,
                min_snr_gamma=min_snr_gamma,
                device=device,
            ), embed_scale

        if np_type == 'flow':
            return RectifiedFlow(), embed_scale

        if np_type == 'vpsde':
            return VPSDE(
                beta_min=getattr(config, 'vpsde_beta_min', 0.01),
                beta_max=getattr(config, 'vpsde_beta_max', 5.0),
                min_snr_gamma=getattr(config, 'min_snr_gamma', 0.0),
            ), embed_scale

        raise ValueError(f"Unknown noise_process: {np_type!r}")

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
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.train_dataloader)
                batch = next(loader_iter)

            loss, grad_norm = self._train_step(batch)
            elapsed = elapsed_offset + (time.time() - start_time)

            if not self.step % self.config.log_interval:
                print(f"Step {self.step}/{self.config.num_steps}")
                wandb.log({"train/loss": loss, "train/grad_norm": grad_norm}, step=self.step)

            if self.val_dataloader is not None and not self.step % self.config.eval_interval:
                val_loss = self._evaluate()
                wandb.log({"val/loss": val_loss}, step=self.step)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(elapsed)
                    wandb.log({"val/best_loss": self.best_val_loss}, step=self.step)
                    print(f"  ↓ best val_loss={self.best_val_loss:.4f} → saved best_ckpt.pt")
                self._visualize(self.step)

            self.step += 1

        print("\n=== Training Complete ===\n")

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _get_conditioning(self, x0: torch.Tensor, B: int) -> torch.Tensor:
        """Return z [B, latent_dim], with CFG dropout applied if configured."""
        if self.unconditional:
            return torch.zeros(B, self.config.latent_dim, device=self.device)
        with torch.no_grad():
            z, _ = self.vae_model.encode(x0)
        if self.config.cfg_uncond_prob > 0.0:
            null_mask = torch.rand(B, device=self.device) < self.config.cfg_uncond_prob
            z = z.masked_fill(null_mask.unsqueeze(1), 0.0)
        return z

    # ── Single unified train / eval step ─────────────────────────────────────

    def _train_step(self, batch: torch.Tensor):
        x0 = batch.to(self.device)
        B  = x0.shape[0]
        z  = self._get_conditioning(x0, B)

        t        = self.np.sample_t(B, self.device)
        x_t, eps = self.np.corrupt(x0, t)
        target   = self.np.get_target(x0, eps)
        t_embed  = self.np.embed_t(t, self._embed_scale)

        self.optimizer.zero_grad()
        with torch.autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=self._amp_dtype is not None):
            pred = self.model(x_t, t_embed, z)
            loss = self.np.loss(pred, target, t)
        self.scaler.scale(loss).backward()

        self.scaler.unscale_(self.optimizer)
        grad_norm = clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm).item()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()

        return loss.item(), grad_norm

    @torch.no_grad()
    def _evaluate(self) -> float:
        self.model.eval()
        total, n = 0.0, 0
        for batch in self.val_dataloader:
            x0 = batch.to(self.device)
            B  = x0.shape[0]
            z  = self._get_conditioning(x0, B)

            t        = self.np.sample_t(B, self.device)
            x_t, eps = self.np.corrupt(x0, t)
            target   = self.np.get_target(x0, eps)
            t_embed  = self.np.embed_t(t, self._embed_scale)

            with torch.autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=self._amp_dtype is not None):
                pred = self.model(x_t, t_embed, z)
                loss = self.np.loss(pred, target, t)
            total += loss.item() * B
            n += B

        self.model.train()
        return total / n

    # ── Visualization ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def _visualize(self, step: int):
        self.model.eval()
        batch = next(iter(self.val_dataloader))
        n  = min(8, batch.shape[0])
        x0 = batch[:n].to(self.device)
        z  = self._get_conditioning(x0, n)
        path = os.path.join(self.vis_dir, f'step_{step:07d}.png')
        self._visualize_one_step(x0, z, path, title=f'Step {step}')
        self.model.train()
        print(f"  Saved visualization → {path}")

    @torch.no_grad()
    def _visualize_one_step(
        self,
        x0: torch.Tensor,
        z: torch.Tensor,
        save_path: str,
        title: Optional[str] = None,
    ) -> None:
        """
        One-step reconstruction diagnostic for all noise processes.

        Shows x0, x_t, and x0_pred = np.predict_x0(x_t, model(x_t,t,z), t)
        at each of np.vis_t_vals().
        """
        n      = x0.shape[0]
        t_vals = self.np.vis_t_vals()
        n_rows = 1 + 2 * len(t_vals)

        fig, axes = plt.subplots(n_rows, n, figsize=(n * 2, n_rows * 2), squeeze=False)

        def _show(ax, img):
            img = img.cpu().permute(1, 2, 0).float().numpy().clip(0, 1)
            if img.shape[-1] == 1:
                ax.imshow(img.squeeze(-1), cmap='gray', vmin=0, vmax=1)
            else:
                ax.imshow(img)
            ax.axis('off')

        row_labels = ['x0']
        for tv in t_vals:
            row_labels += [f'x_t  t={tv}', f'x_pred  t={tv}']

        for col in range(n):
            _show(axes[0, col], x0[col])

        for ti, t_val in enumerate(t_vals):
            t_tensor = torch.full((n,), t_val, device=self.device, dtype=self.np.dtype)
            x_t, eps = self.np.corrupt(x0, t_tensor)
            t_embed  = self.np.embed_t(t_tensor, self._embed_scale)
            with torch.autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=self._amp_dtype is not None):
                pred = self.model(x_t, t_embed, z)
            x0_pred  = self.np.predict_x0(x_t, pred, t_tensor)

            row_xt = 1 + 2 * ti
            for col in range(n):
                _show(axes[row_xt,     col], x_t[col].clamp(0, 1))
                _show(axes[row_xt + 1, col], x0_pred[col].clamp(0, 1))

        for row, label in enumerate(row_labels):
            axes[row, 0].set_ylabel(label, fontsize=7, rotation=0,
                                    ha='right', va='center', labelpad=55)
        if title:
            fig.suptitle(title, fontsize=9)
        plt.tight_layout()
        fig.savefig(save_path, dpi=80, bbox_inches='tight')
        plt.close(fig)
