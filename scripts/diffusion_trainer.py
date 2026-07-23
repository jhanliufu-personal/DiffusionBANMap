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

    grad_accum_steps: micro-batches accumulated per optimizer step (default 1).
        Effective batch size = batch_size * grad_accum_steps. self.step / num_steps /
        log_interval / eval_interval all count optimizer updates, not micro-batches.

    ema_decay: EMA decay rate for a shadow copy of the model's parameters (default: None,
        i.e. no EMA). Updated once per optimizer step, checkpointed under "ema_state_dict"
        alongside the raw weights — see models/ema.py. Sample from the EMA weights instead
        of the raw ones for meaningfully sharper/more coherent generations (this is what
        e.g. inspect_diffusion.ipynb does when a checkpoint has ema_state_dict).

Per-noise-level loss logging: train/val loss is also broken out by noise_level (each
noise process's own [0,1]-normalized corruption fraction, see models/noise_process.py)
into low/mid/high thirds and logged to wandb as train|val/loss_{low,mid,high}_noise, in
addition to the overall loss. This is purely diagnostic — the scalar actually
backpropagated (loss/train/loss) is unchanged, computed exactly as before via
loss_per_sample(...).mean(); the buckets are just a different aggregation of the same
per-sample losses that were already available before the final mean.
"""

import os
import time
import contextlib
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
from typing import Optional
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from scripts.base_trainer import BaseTrainer
from models.noise_process import DDPM, RectifiedFlow, VPSDE
from models.ema import EMA


class DiffusionTrainer(BaseTrainer):

    # (name, lo, hi) thirds of noise_level ∈ [0, 1] — 0 = clean data, 1 = pure noise.
    # hi on the "high" bucket is nudged past 1.0 so noise_level==1.0 samples are included.
    _NOISE_BUCKETS = (("low", 0.0, 1 / 3), ("mid", 1 / 3, 2 / 3), ("high", 2 / 3, 1.0 + 1e-6))

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
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__(model, optimizer, scheduler, config, device, rank=rank, world_size=world_size)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.best_val_loss = float("inf")
        # Set in train(): self.model stays the raw module (checkpoint/EMA keys stay clean
        # and unprefixed) while self.ddp_model is what forward passes actually go through.
        self.ddp_model = None

        self.unconditional = getattr(config, 'unconditional', False)
        if self.unconditional and self.is_main:
            print("Train for unconditional generation")

        # Frozen beta-VAE encoder — never updated. Unconditional runs need no VAE at all.
        if vae_model is not None:
            self.vae_model = vae_model.to(device).eval()
            for p in self.vae_model.parameters():
                p.requires_grad_(False)
        else:
            self.vae_model = None

        self.np, self._embed_scale = self._build_noise_process(config, device)
        if self.is_main:
            print(f"Noise process: {type(self.np).__name__}")

        self.mixed_precision, self._amp_dtype, self.scaler = self._build_amp(config, device)
        if self.mixed_precision != 'no' and self.is_main:
            active = "active" if self._amp_dtype is not None else f"inactive — {device.type} isn't cuda"
            print(f"Mixed precision: {self.mixed_precision} ({active})")

        self.grad_accum_steps = max(1, getattr(config, 'grad_accum_steps', 1))
        batch_size = getattr(config, 'batch_size', None)
        if batch_size is not None and self.is_main:
            effective_bs = batch_size * self.grad_accum_steps * self.world_size
            factors = [f"per-GPU batch {batch_size}"]
            if self.grad_accum_steps > 1:
                factors.append(f"grad_accum {self.grad_accum_steps}")
            if self.world_size > 1:
                factors.append(f"{self.world_size} GPUs")
            print(f"Effective batch size: {effective_bs} ({' x '.join(factors)})")

        ema_decay = getattr(config, 'ema_decay', None)
        self.ema = EMA(model, decay=ema_decay) if ema_decay else None
        if self.is_main:
            print(f"EMA: decay={ema_decay}" if self.ema is not None else "EMA: disabled")

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
        # Model must be on self.device *before* optimizer state is loaded below:
        # Optimizer.load_state_dict casts each loaded tensor (exp_avg, exp_avg_sq, step)
        # to the *current* param.device at call time, ignoring torch.load's map_location.
        # Loading while the model is still on CPU pins that state to CPU, which later
        # mismatches the CUDA grads once the model is moved — "Expected all tensors to be
        # on the same device" inside torch._foreach_lerp_ in Adam's step.
        self.model = self.model.to(self.device).train()

        elapsed_offset = 0.0
        if resume_ckpt_path:
            reset_step = getattr(self.config, 'reset_step_on_resume', False)
            elapsed_offset = self._load_checkpoint(resume_ckpt_path, reset_step=reset_step)
            if self.ema is not None and self._last_loaded_ckpt is not None \
                    and "ema_state_dict" in self._last_loaded_ckpt and not reset_step:
                self.ema.load_state_dict(self._last_loaded_ckpt["ema_state_dict"])
                if self.is_main:
                    print(f"Resumed EMA shadow (step {self.ema.step})")

        # Every rank already loaded the same checkpoint file independently above (or
        # started from the same init, since model construction is deterministic from
        # config), so DDP's constructor-time parameter broadcast from rank 0 is a no-op
        # here — just the standard, safe way to guarantee replicas can never drift.
        if self.world_size > 1:
            self.ddp_model = DistributedDataParallel(
                self.model, device_ids=[self.device.index], output_device=self.device.index,
            )
        else:
            self.ddp_model = self.model

        if self.is_main:
            print(f"\n=== Starting Diffusion Training ({self.config.num_steps} steps) ===\n")
        start_time = time.time()
        if self.ema is not None:
            self.ema.to(self.device)

        loader_iter = iter(self.train_dataloader)
        epoch = 0

        def _next_batch():
            nonlocal loader_iter, epoch
            try:
                return next(loader_iter)
            except StopIteration:
                epoch += 1
                sampler = getattr(self.train_dataloader, 'sampler', None)
                if isinstance(sampler, DistributedSampler):
                    sampler.set_epoch(epoch)
                loader_iter = iter(self.train_dataloader)
                return next(loader_iter)

        while self.step < self.config.num_steps:
            micro_batches = [_next_batch() for _ in range(self.grad_accum_steps)]

            loss, grad_norm, noise_bucket_loss = self._train_step(micro_batches)
            if self.world_size > 1:
                loss = self._allreduce_mean(loss)
                grad_norm = self._allreduce_mean(grad_norm)
                noise_bucket_loss = {k: self._allreduce_mean(v) for k, v in noise_bucket_loss.items()}
            elapsed = elapsed_offset + (time.time() - start_time)

            if not self.step % self.config.log_interval and self.is_main:
                print(f"Step {self.step}/{self.config.num_steps}")
                wandb.log({
                    "train/loss": loss,
                    "train/grad_norm": grad_norm,
                    **{f"train/loss_{name}_noise": v for name, v in noise_bucket_loss.items()},
                }, step=self.step)

            if self.val_dataloader is not None and not self.step % self.config.eval_interval:
                # Collective — _evaluate() all-reduces internally, so every rank must call
                # it in lockstep even though only rank 0 acts on the result below.
                val_loss, val_noise_bucket_loss = self._evaluate()
                if self.is_main:
                    wandb.log({
                        "val/loss": val_loss,
                        **{f"val/loss_{name}_noise": v for name, v in val_noise_bucket_loss.items()},
                    }, step=self.step)
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        extra = {"ema_state_dict": self.ema.state_dict()} if self.ema is not None else {}
                        self._save_checkpoint(elapsed, extra=extra)
                        wandb.log({"val/best_loss": self.best_val_loss}, step=self.step)
                        print(f"  ↓ best val_loss={self.best_val_loss:.4f} → saved best_ckpt.pt")
                    self._visualize(self.step)

            self.step += 1

        if self.is_main:
            print("\n=== Training Complete ===\n")

    def _allreduce_mean(self, value: float) -> float:
        """Average a python scalar across ranks — used to turn each rank's local loss/
        grad_norm (computed over its own DistributedSampler shard) into the true global
        value for logging, rather than silently logging only rank 0's local slice."""
        t = torch.tensor(value, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return (t / self.world_size).item()

    # ── Shared helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _unpack_batch(batch):
        """Dataloaders yield either a plain image tensor, or (image, latent) tuples when
        conditioning on precomputed latents (e.g. AlexNet-fc6-PCA) instead of an on-the-fly
        VAE encode — see _ImageNet64ArrayDataset. Returns (x0, provided_z_or_None)."""
        if isinstance(batch, (list, tuple)):
            x0, z = batch
            return x0, z
        return batch, None

    def _get_conditioning(self, x0: torch.Tensor, B: int, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return z [B, latent_dim], with CFG dropout applied if configured.

        z is already-provided (precomputed) conditioning when the dataloader yields
        (image, latent) pairs; otherwise falls back to unconditional zeros or an
        on-the-fly VAE encode of x0.
        """
        if z is None:
            if self.unconditional:
                z = torch.zeros(B, self.config.latent_dim, device=self.device)
            else:
                with torch.no_grad():
                    z, _ = self.vae_model.encode(x0)
        else:
            z = z.to(self.device)

        if self.config.cfg_uncond_prob > 0.0:
            null_mask = torch.rand(B, device=self.device) < self.config.cfg_uncond_prob
            z = z.masked_fill(null_mask.unsqueeze(1), 0.0)
        return z

    @classmethod
    def _update_noise_buckets(cls, sums: dict, counts: dict, loss_per: torch.Tensor, noise_level: torch.Tensor) -> None:
        """Accumulate (sum, count) of loss_per into the low/mid/high noise_level buckets
        defined by cls._NOISE_BUCKETS. Diagnostic bookkeeping only — loss_per/noise_level
        should already be detached (or this is called under torch.no_grad(), as in
        _evaluate) since these sums never participate in the backward pass."""
        for name, lo, hi in cls._NOISE_BUCKETS:
            mask = (noise_level >= lo) & (noise_level < hi)
            n = int(mask.sum().item())
            if n:
                sums[name] += loss_per[mask].sum().item()
                counts[name] += n

    @classmethod
    def _finalize_noise_buckets(cls, sums: dict, counts: dict) -> dict:
        return {name: (sums[name] / counts[name] if counts[name] else float('nan'))
                for name, _, _ in cls._NOISE_BUCKETS}

    # ── Single unified train / eval step ─────────────────────────────────────

    def _train_step(self, micro_batches):
        """One optimizer step, accumulating gradients over len(micro_batches) micro-batches
        so the effective batch size is batch_size * grad_accum_steps without raising peak
        activation memory past a single micro-batch's footprint.

        Also buckets the (already-computed, pre-mean) per-sample loss by noise level for
        diagnostics — purely additional bookkeeping on a .detach()'d copy, so the scalar
        actually backpropagated (loss_per.mean() / accum, same as before) is unaffected.
        """
        accum = len(micro_batches)
        self.optimizer.zero_grad()
        total_loss = 0.0
        bucket_sums = {name: 0.0 for name, _, _ in self._NOISE_BUCKETS}
        bucket_counts = {name: 0 for name, _, _ in self._NOISE_BUCKETS}

        for i, batch in enumerate(micro_batches):
            x0, z_provided = self._unpack_batch(batch)
            x0 = x0.to(self.device)
            B  = x0.shape[0]
            z  = self._get_conditioning(x0, B, z_provided)

            t        = self.np.sample_t(B, self.device)
            x_t, eps = self.np.corrupt(x0, t)
            target   = self.np.get_target(x0, eps)
            t_embed  = self.np.embed_t(t, self._embed_scale)

            # DDP all-reduces gradients on every .backward() by default; with grad
            # accumulation we only need that sync once the full effective batch is in,
            # so suppress it on every micro-batch but the last via no_sync().
            is_last_micro_batch = i == accum - 1
            sync_ctx = (
                contextlib.nullcontext()
                if self.world_size == 1 or is_last_micro_batch
                else self.ddp_model.no_sync()
            )
            with sync_ctx:
                with torch.autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=self._amp_dtype is not None):
                    pred = self.ddp_model(x_t, t_embed, z)
                    loss_per = self.np.loss_per_sample(pred, target, t)
                    loss = loss_per.mean() / accum
                self.scaler.scale(loss).backward()
            total_loss += loss.item()

            noise_level = self.np.noise_level(t)
            self._update_noise_buckets(bucket_sums, bucket_counts, loss_per.detach(), noise_level)

        self.scaler.unscale_(self.optimizer)
        grad_norm = clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm).item()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()
        if self.ema is not None:
            self.ema.update(self.model)

        return total_loss, grad_norm, self._finalize_noise_buckets(bucket_sums, bucket_counts)

    @torch.no_grad()
    def _evaluate(self):
        """Collective when world_size > 1: every rank must call this (it all-reduces at
        the end), even though only rank 0 acts on the returned value — see train()."""
        self.model.eval()
        total, n = 0.0, 0
        bucket_sums = {name: 0.0 for name, _, _ in self._NOISE_BUCKETS}
        bucket_counts = {name: 0 for name, _, _ in self._NOISE_BUCKETS}

        for batch in self.val_dataloader:
            x0, z_provided = self._unpack_batch(batch)
            x0 = x0.to(self.device)
            B  = x0.shape[0]
            z  = self._get_conditioning(x0, B, z_provided)

            t        = self.np.sample_t(B, self.device)
            x_t, eps = self.np.corrupt(x0, t)
            target   = self.np.get_target(x0, eps)
            t_embed  = self.np.embed_t(t, self._embed_scale)

            with torch.autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=self._amp_dtype is not None):
                pred = self.model(x_t, t_embed, z)
                loss_per = self.np.loss_per_sample(pred, target, t)
                loss = loss_per.mean()
            total += loss.item() * B
            n += B

            noise_level = self.np.noise_level(t)
            self._update_noise_buckets(bucket_sums, bucket_counts, loss_per, noise_level)

        if self.world_size > 1:
            # Each rank only saw its own DistributedSampler shard of val data, so total/n
            # (and the per-bucket sums/counts) need summing across ranks before dividing —
            # otherwise "val loss" and the best-checkpoint decision would silently only
            # reflect rank 0's slice of the validation set.
            stats = torch.tensor([total, float(n)], device=self.device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total, n = stats[0].item(), stats[1].item()
            for name, _, _ in self._NOISE_BUCKETS:
                bstats = torch.tensor([bucket_sums[name], float(bucket_counts[name])], device=self.device)
                dist.all_reduce(bstats, op=dist.ReduceOp.SUM)
                bucket_sums[name], bucket_counts[name] = bstats[0].item(), int(bstats[1].item())

        self.model.train()
        return total / n, self._finalize_noise_buckets(bucket_sums, bucket_counts)

    # ── Visualization ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def _visualize(self, step: int):
        self.model.eval()
        batch = next(iter(self.val_dataloader))
        x0, z_provided = self._unpack_batch(batch)
        n  = min(8, x0.shape[0])
        x0 = x0[:n].to(self.device)
        z_provided = z_provided[:n] if z_provided is not None else None
        z  = self._get_conditioning(x0, n, z_provided)
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
