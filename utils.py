"""
Utility functions for β-VAE training and evaluation.
Implements the exact loss function from Higgins et al. (2017) and training utilities.
"""

import math
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Literal, Optional, List
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from models.noise_process import _cosine_betas  # noqa: F401  (re-exported for callers)


def make_run_tag(cfg) -> str:
    """Short hyperparam summary appended to output_dir to uniquely identify a run."""
    if hasattr(cfg, 'hidden_dim'):  # BetaVAE
        return (f"b{cfg.beta:g}_h{cfg.input_height}x{cfg.input_width}"
                f"_z{cfg.latent_dim}_hd{cfg.hidden_dim}")
    else:  # Diffusion
        ch = 'x'.join(str(c) for c in cfg.channel_mult)
        np_type = getattr(cfg, 'noise_process', 'ddpm')
        T = getattr(cfg, 'num_timesteps', 1000)
        # For DDPM show the schedule; for continuous processes show the process name
        sched = getattr(cfg, 'beta_schedule', np_type) if np_type == 'ddpm' else np_type
        
        uncond = "_uncond" if getattr(cfg, "unconditional", False) else ""
        return f"h{cfg.image_size}_mc{cfg.model_channels}_ch{ch}_T{T}_{sched}{uncond}_z{cfg.latent_dim}"


def discover_device() -> Literal["cuda", "mps", "cpu"]:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


def count_model_params(model: torch.nn.Module) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {total_params:,} total parameters ({trainable_params:,} trainable)")


def build_lr_scheduler(optimizer: torch.optim.Optimizer, cfg) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """Build the LR schedule named by cfg.lr_schedule (default 'constant' — no scheduler,
    i.e. returns None and DiffusionTrainer just leaves the optimizer's LR untouched).

    The Flow Matching paper (Lipman et al. 2023) reuses the Dhariwal & Nichol (2021) ADM
    U-Net + training recipe for its ImageNet experiments, which — per that lineage's usual
    convention — trains at a constant LR and relies on EMA (models/ema.py, decay 0.9999)
    rather than LR decay for stabilization; that's why 'constant' is this project's default
    for the imagenet64 flow config. The Flow Matching paper's own Table 3 also lists
    'polynomial decay' for some of its other experiments, so that option is implemented
    here too in case you want to compare against it directly.

    cfg.lr_schedule:      'constant' (default) | 'polynomial'
    cfg.lr_warmup_steps:  linear warmup length in optimizer steps (default 0)
    cfg.lr_poly_power:    polynomial decay power (default 1.0 — linear decay to 0 by num_steps)
    """
    schedule = getattr(cfg, 'lr_schedule', 'constant')
    warmup_steps = getattr(cfg, 'lr_warmup_steps', 0)

    if schedule == 'constant':
        base_scheduler = None
    elif schedule == 'polynomial':
        power = getattr(cfg, 'lr_poly_power', 1.0)
        base_scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=cfg.num_steps - warmup_steps, power=power,
        )
    else:
        raise ValueError(f"Unknown lr_schedule: {schedule!r}")

    if warmup_steps <= 0:
        return base_scheduler

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps,
    )
    if base_scheduler is None:
        return warmup
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, base_scheduler], milestones=[warmup_steps],
    )


def compute_gaussian_kl(z_mean: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence between input Gaussian and Standard Normal.
    Mirrors compute_gaussian_kl() from disentanglement_lib/methods/unsupervised/vae.py

    KL(q(z|x) || p(z)) where q(z|x) = N(z_mean, exp(z_logvar)) and p(z) = N(0, I)

    Args:
        z_mean: Mean parameters [batch_size, latent_dim]
        z_logvar: Log variance parameters [batch_size, latent_dim]

    Returns:
        kl_loss: Scalar KL divergence averaged over batch
    """
    # KL divergence per sample: 0.5 * sum(mu^2 + exp(logvar) - logvar - 1)
    # Sum over latent dimensions (dim=1), then mean over batch
    kl_per_sample = 0.5 * torch.sum(
        z_mean.pow(2) + z_logvar.exp() - z_logvar - 1,
        dim=1
    )
    return torch.mean(kl_per_sample)


def bernoulli_loss(
    true_images: torch.Tensor,
    reconstructed_images: torch.Tensor,
    activation: str = 'logits'
    ) -> torch.Tensor:
    """
    Compute Bernoulli reconstruction loss.
    Mirrors bernoulli_loss() from disentanglement_lib/methods/shared/losses.py

    Args:
        true_images: Original images [batch_size, channels, height, width]
        reconstructed_images: Reconstructed images [batch_size, channels, height, width]
        activation: 'logits' or 'tanh'
            - 'logits': reconstructed_images are raw logits (pre-sigmoid)
            - 'tanh': reconstructed_images are tanh outputs

    Returns:
        loss: Per-sample reconstruction loss [batch_size]
    """
    batch_size = true_images.size(0)
    flattened_dim = true_images[0].numel()  # Product of [channels, height, width]

    # Flatten images: [batch_size, channels, height, width] -> [batch_size, flattened_dim]
    reconstructed_images = reconstructed_images.view(batch_size, flattened_dim)
    true_images = true_images.view(batch_size, flattened_dim)

    if activation == 'logits':
        # Binary cross entropy with logits: sum over pixels (dim=1)
        loss = F.binary_cross_entropy_with_logits(
            reconstructed_images,
            true_images,
            reduction='none'
        ).sum(dim=1)
        # loss = F.binary_cross_entropy_with_logits(
        #     reconstructed_images,
        #     true_images,
        #     reduction='none'
        # ).mean(dim=1)
    elif activation == 'tanh':
        # Convert tanh output to [0, 1] range and clip
        reconstructed_images = torch.clamp(
            torch.tanh(reconstructed_images) / 2 + 0.5,
            1e-6,
            1 - 1e-6
        )
        # Manual binary cross entropy: -sum(x*log(p) + (1-x)*log(1-p))
        loss = -torch.sum(
            true_images * torch.log(reconstructed_images) +
            (1 - true_images) * torch.log(1 - reconstructed_images),
            dim=1
        )
    else:
        raise NotImplementedError(f"Activation '{activation}' not supported.")

    return loss


def beta_vae_loss(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 4.0,
    activation: str = 'logits'
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the β-VAE loss function from Higgins et al. (2017).

    Loss function: L_β(θ, φ, β; x, z) = E[reconstruction_loss] + β · KL_loss

    Args:
        x_recon: Reconstructed images [batch_size, channels, height, width]
        x: Original images [batch_size, channels, height, width]
        mu: Mean parameters from encoder [batch_size, latent_dim]
        logvar: Log variance parameters from encoder [batch_size, latent_dim]
        beta: β hyperparameter controlling reconstruction-disentanglement trade-off
        activation: 'logits' (default) or 'tanh' for reconstruction loss

    Returns:
        total_loss: Total β-VAE loss (scalar)
        loss_dict: Dictionary containing individual loss components
    """
    # Reconstruction loss: mean of per-sample Bernoulli loss
    recon_loss_per_sample = bernoulli_loss(x, x_recon, activation=activation)
    recon_loss = torch.mean(recon_loss_per_sample)

    # KL divergence: D_KL(q_φ(z|x) || p(z))
    kl_loss = compute_gaussian_kl(mu, logvar)

    # Total β-VAE loss
    total_loss = recon_loss + beta * kl_loss

    # Create loss dictionary for monitoring
    loss_dict = {
        'total_loss': total_loss.item(),
        'reconstruction_loss': recon_loss.item(),
        'kl_loss': kl_loss.item(),
        'beta_weighted_kl': (beta * kl_loss).item(),
    }

    return total_loss, loss_dict


def mse_beta_vae_loss(x_recon: torch.Tensor,
                      x: torch.Tensor,
                      mu: torch.Tensor,
                      logvar: torch.Tensor,
                      beta: float = 4.0) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Alternative β-VAE loss using MSE for reconstruction (useful for continuous pixel values).

    Args:
        x_recon: Reconstructed images [batch_size, channels, height, width]
        x: Original images [batch_size, channels, height, width]
        mu: Mean parameters from encoder [batch_size, latent_dim]
        logvar: Log variance parameters from encoder [batch_size, latent_dim]
        beta: β hyperparameter controlling reconstruction-disentanglement trade-off

    Returns:
        total_loss: Total β-VAE loss
        loss_dict: Dictionary containing individual loss components
    """
    # Reconstruction loss using MSE (mean over all dimensions)
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')

    # KL divergence using consistent implementation
    kl_loss = compute_gaussian_kl(mu, logvar)

    # Total β-VAE loss
    total_loss = recon_loss + beta * kl_loss

    loss_dict = {
        'total_loss': total_loss.item(),
        'reconstruction_loss': recon_loss.item(),
        'kl_loss': kl_loss.item(),
        'beta_weighted_kl': (beta * kl_loss).item(),
        'beta': beta
    }

    return total_loss, loss_dict


# def compute_disentanglement_metric(model: torch.nn.Module, 
#                                    data_loader: DataLoader, 
#                                    device: torch.device = torch.device('cpu'),
#                                    num_samples: int = 1000) -> Dict[str, float]:
#     """
#     Compute disentanglement metrics for the trained β-VAE model.
#     This implementation provides a basic mutual information-based metric.
    
#     Args:
#         model: Trained β-VAE model
#         data_loader: DataLoader for evaluation data
#         device: Device to run computation on
#         num_samples: Number of samples to use for metric computation
        
#     Returns:
#         metrics: Dictionary containing disentanglement metrics
#     """
#     model.eval()
    
#     latent_codes = []
#     with torch.no_grad():
#         samples_collected = 0
#         for data_batch in data_loader:
#             if samples_collected >= num_samples:
#                 break
                
#             data_batch = data_batch.to(device)
#             mu, _ = model.encode(data_batch)
#             latent_codes.append(mu.cpu())
#             samples_collected += data_batch.size(0)
    
#     # Concatenate all latent codes
#     latent_codes = torch.cat(latent_codes, dim=0)[:num_samples]
    
#     # Compute basic statistics
#     latent_std = torch.std(latent_codes, dim=0)
#     latent_mean = torch.mean(latent_codes, dim=0)
    
#     # Disentanglement metric based on latent variance
#     # Higher variance in individual dimensions suggests better disentanglement
#     disentanglement_score = torch.mean(latent_std).item()
    
#     metrics = {
#         'disentanglement_score': disentanglement_score,
#         'avg_latent_std': torch.mean(latent_std).item(),
#         'max_latent_std': torch.max(latent_std).item(),
#         'min_latent_std': torch.min(latent_std).item(),
#     }
    
#     return metrics


# def visualize_reconstructions(model: torch.nn.Module, 
#                               data_batch: torch.Tensor, 
#                               device: torch.device = torch.device('cpu'),
#                               num_images: int = 8) -> plt.Figure:
#     """
#     Visualize original images and their reconstructions.
    
#     Args:
#         model: β-VAE model
#         data_batch: Batch of images [batch_size, channels, height, width]
#         device: Device to run model on
#         num_images: Number of image pairs to visualize
        
#     Returns:
#         fig: Matplotlib figure containing the visualization
#     """
#     model.eval()
    
#     # Select subset of images
#     data_batch = data_batch[:num_images].to(device)
    
#     with torch.no_grad():
#         x_recon, _, _ = model(data_batch)
    
#     # Create comparison grid
#     comparison = torch.cat([data_batch, x_recon], dim=0)
#     grid = make_grid(comparison.cpu(), nrow=num_images, normalize=True, pad_value=1.0)
    
#     # Create plot
#     fig, ax = plt.subplots(1, 1, figsize=(15, 6))
#     ax.imshow(grid.permute(1, 2, 0))
#     ax.axis('off')
#     ax.set_title('Top: Original Images, Bottom: Reconstructions')
    
#     return fig


# def visualize_latent_traversal(model: torch.nn.Module, 
#                                image: torch.Tensor, 
#                                device: torch.device = torch.device('cpu'),
#                                latent_dim: int = 0,
#                                traversal_range: Tuple[float, float] = (-3.0, 3.0),
#                                num_steps: int = 10) -> plt.Figure:
#     """
#     Visualize latent space traversal by varying one latent dimension.
    
#     Args:
#         model: β-VAE model
#         image: Single input image [1, channels, height, width]
#         device: Device to run model on
#         latent_dim: Index of latent dimension to traverse
#         traversal_range: Range of values to traverse (min, max)
#         num_steps: Number of steps in traversal
        
#     Returns:
#         fig: Matplotlib figure containing the traversal visualization
#     """
#     model.eval()
#     image = image.to(device)
    
#     with torch.no_grad():
#         # Get latent representation
#         mu, _ = model.encode(image)
        
#         # Create traversal values
#         traversal_values = torch.linspace(traversal_range[0], traversal_range[1], num_steps)
        
#         # Generate images for each traversal step
#         traversal_images = []
#         for val in traversal_values:
#             # Modify the selected latent dimension
#             z_modified = mu.clone()
#             z_modified[0, latent_dim] = val
            
#             # Decode modified latent code
#             x_recon = model.decode(z_modified)
#             traversal_images.append(x_recon)
        
#         # Stack images
#         traversal_images = torch.cat(traversal_images, dim=0)
    
#     # Create visualization grid
#     grid = make_grid(traversal_images.cpu(), nrow=num_steps, normalize=True, pad_value=1.0)
    
#     # Create plot
#     fig, ax = plt.subplots(1, 1, figsize=(15, 3))
#     ax.imshow(grid.permute(1, 2, 0))
#     ax.axis('off')
#     ax.set_title(f'Latent Dimension {latent_dim} Traversal '
#                 f'(from {traversal_range[0]:.1f} to {traversal_range[1]:.1f})')
    
#     return fig


def create_training_config(dataset_name: str = 'faces') -> Dict[str, Any]:
    """
    Create training configuration based on literature recommendations.
    
    Args:
        dataset_name: Name of dataset ('faces', 'celeba', 'chairs', 'dsprites')
        
    Returns:
        config: Training configuration dictionary
    """
    base_config = {
        'learning_rate': 1e-4,
        'batch_size': 64,
        'max_epochs': 100,
        'optimizer': 'adam',
        'loss_type': 'bce',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Dataset-specific configurations from literature
    if dataset_name.lower() in ['faces', 'celeba']:
        config = {
            **base_config,
            'beta': 10.0,  # Higher β for face images as used in Nature Communications 2021
            'latent_dim': 10,
            'input_channels': 3,
            'image_size': 64
        }
    elif dataset_name.lower() == 'chairs':
        config = {
            **base_config,
            'beta': 4.0,  # β=4 for 3D chairs dataset
            'latent_dim': 10,
            'input_channels': 3,
            'image_size': 64
        }
    elif dataset_name.lower() == 'dsprites':
        config = {
            **base_config,
            'beta': 6.0,
            'latent_dim': 10,
            'input_channels': 1,
            'image_size': 64
        }
    else:
        # Default configuration
        config = {
            **base_config,
            'beta': 4.0,
            'latent_dim': 10,
            'input_channels': 3,
            'image_size': 64
        }
    
    return config


# ── Diffusion noise schedule & sampling ───────────────────────────────────────

def make_noise_schedule(diff_config, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Precompute all noise schedule tensors needed for diffusion training and inference.
    Returns a dict of [T]-shaped tensors on the given device.
    """
    T = diff_config.num_timesteps
    if diff_config.beta_schedule == 'cosine':
        betas = _cosine_betas(T).to(device)
    elif diff_config.beta_schedule == 'linear':
        betas = torch.linspace(1e-4, 0.02, T, device=device)
    else:
        raise ValueError(f"Unknown beta_schedule: {diff_config.beta_schedule}")

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), alphas_cumprod[:-1]])

    return {
        'betas':                     betas,
        'alphas':                    alphas,
        'alphas_cumprod':            alphas_cumprod,
        'alphas_cumprod_prev':       alphas_cumprod_prev,
        'sqrt_alphas_cumprod':       alphas_cumprod.sqrt(),
        'sqrt_one_minus_alphas_cumprod': (1.0 - alphas_cumprod).sqrt(),
        'posterior_var':             betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        'snr':                       alphas_cumprod / (1.0 - alphas_cumprod),
    }


@torch.no_grad()
def visualize_unet_one_step(
    model: torch.nn.Module,
    x0: torch.Tensor,
    z: torch.Tensor,
    schedule: Dict[str, torch.Tensor],
    num_timesteps: int,
    device: torch.device,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """
    One-step reconstruction diagnostic.

    Shows x0, x_t, and x_pred = (x_t - sqrt(1-ᾱ_t)·ε_pred) / sqrt(ᾱ_t) at three
    noise levels (T/8, T/2, 7T/8) for up to 8 images.  If save_path is given the
    figure is saved and closed; otherwise plt.show() is called.

    Args:
        model:          UNet (set to eval before calling if needed).
        x0:             Clean images [B, C, H, W] in [0, 1], on device.
        z:              Conditioning latents [B, latent_dim]; pass zeros for unconditional.
        schedule:       Dict from make_noise_schedule — needs sqrt_alphas_cumprod and
                        sqrt_one_minus_alphas_cumprod.
        num_timesteps:  T (length of the noise schedule).
        device:         Torch device.
        save_path:      Where to write the PNG (None → display inline).
        title:          Optional figure suptitle (e.g. "Step 1000").
    """
    model.eval()
    n = min(8, x0.shape[0])
    x0 = x0[:n]
    z  = z[:n]

    sqrt_a_all = schedule['sqrt_alphas_cumprod']
    sqrt_1ma_all = schedule['sqrt_one_minus_alphas_cumprod']

    t_vals = [num_timesteps // 8, num_timesteps // 2, num_timesteps * 7 // 8]
    n_rows = 1 + 2 * len(t_vals)

    fig, axes = plt.subplots(n_rows, n, figsize=(n * 2, n_rows * 2), squeeze=False)

    def _show(ax, img_tensor):
        img = img_tensor.cpu().permute(1, 2, 0).float().numpy().clip(0, 1)
        if img.shape[-1] == 1:
            ax.imshow(img.squeeze(-1), cmap='gray', vmin=0, vmax=1)
        else:
            ax.imshow(img)
        ax.axis('off')

    row_labels = ['x0']
    for t in t_vals:
        row_labels += [f'x_t  t={t}', f'x_pred  t={t}']

    for col in range(n):
        _show(axes[0, col], x0[col])

    for ti, t_val in enumerate(t_vals):
        t_tensor = torch.full((n,), t_val, device=device, dtype=torch.long)
        eps = torch.randn_like(x0)
        sqrt_a = sqrt_a_all[t_tensor][:, None, None, None]
        sqrt_1ma = sqrt_1ma_all[t_tensor][:, None, None, None]
        x_t = sqrt_a * x0 + sqrt_1ma * eps
        eps_pred = model(x_t, t_tensor, z)
        x_pred = (x_t - sqrt_1ma * eps_pred) / sqrt_a

        row_xt = 1 + 2 * ti
        row_xp = row_xt + 1
        for col in range(n):
            _show(axes[row_xt, col], x_t[col].clamp(0, 1))
            _show(axes[row_xp, col], x_pred[col].clamp(0, 1))

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=7, rotation=0,
                                ha='right', va='center', labelpad=55)

    if title:
        fig.suptitle(title, fontsize=9)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=80, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


@torch.no_grad()
def diffusion_sample(
    z: torch.Tensor,
    unet: torch.nn.Module,
    schedule: Dict[str, torch.Tensor],
    diff_config,
    device: torch.device,
    sampler: str = 'ddpm',
    eta: float = 0.0,
    num_inference_steps: Optional[int] = None,
    guidance_scale: float = 1.0,
    verbose: bool = False,
    n_snapshots: int = 10,
):
    """
    Generate images conditioned on beta-VAE latents z [B, latent_dim].

    sampler='ddpm' : stochastic DDPM reverse process, always uses all T steps.
    sampler='ddim' : DDIM (Song et al. 2020).
        eta=0.0  → fully deterministic; same seed = same output every run.
        eta=1.0  → stochasticity matching DDPM posterior variance.
        num_inference_steps < T → accelerated sampling (e.g. 50 steps instead of 1000).

    guidance_scale > 1 applies classifier-free guidance (requires CFG-trained model).

    verbose=True: returns (image, snapshots) where snapshots is a list of
    (t, cpu_tensor [B,C,H,W]) captured at n_snapshots evenly-spaced steps,
    ordered noisy → clean.
    """
    T   = diff_config.num_timesteps
    B   = z.shape[0]
    C, H, W = diff_config.in_channels, diff_config.image_size, diff_config.image_size

    acp      = schedule['alphas_cumprod']
    alphas   = schedule['alphas']
    betas    = schedule['betas']
    sqrt_acp = schedule['sqrt_alphas_cumprod']
    sqrt_omacp = schedule['sqrt_one_minus_alphas_cumprod']
    post_var = schedule['posterior_var']

    # ── Build ordered timestep list (high t → low t) ──────────────────────────
    if sampler == 'ddim':
        n_steps = num_inference_steps or T
        # Uniformly space n_steps indices across [T-1, 0]
        timesteps = torch.linspace(T - 1, 0, n_steps).round().long().tolist()
    else:
        timesteps = list(range(T - 1, -1, -1))

    # ── Snapshot bookkeeping ──────────────────────────────────────────────────
    if verbose:
        stride = max(1, len(timesteps) // n_snapshots)
        capture_at = set(range(0, len(timesteps), stride))
        capture_at.add(len(timesteps) - 1)
        snapshots: List[Tuple[int, torch.Tensor]] = []

    x      = torch.randn(B, C, H, W, device=device)
    z_null = torch.zeros_like(z)

    for i, t_idx in enumerate(timesteps):
        if verbose and i in capture_at:
            snapshots.append((t_idx, x.clone().clamp(0.0, 1.0).cpu()))

        t_batch = torch.full((B,), t_idx, device=device, dtype=torch.long)

        # ── Noise prediction (with optional CFG) ──────────────────────────────
        if guidance_scale != 1.0:
            eps_both = unet(
                torch.cat([x, x]),
                torch.cat([t_batch, t_batch]),
                torch.cat([z, z_null]),
            )
            eps_cond, eps_uncond = eps_both.chunk(2)
            eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps_pred = unet(x, t_batch, z)

        # ── Predict x_0 ───────────────────────────────────────────────────────
        pred_x0 = (x - sqrt_omacp[t_idx] * eps_pred) / sqrt_acp[t_idx]
        pred_x0 = pred_x0.clamp(0.0, 1.0)

        # ── Reverse step ──────────────────────────────────────────────────────
        if sampler == 'ddpm':
            acp_prev_t = schedule['alphas_cumprod_prev'][t_idx]
            coef1 = acp_prev_t.sqrt() * betas[t_idx] / (1.0 - acp[t_idx])
            coef2 = alphas[t_idx].sqrt() * (1.0 - acp_prev_t) / (1.0 - acp[t_idx])
            mean  = coef1 * pred_x0 + coef2 * x
            x = mean + (post_var[t_idx].sqrt() * torch.randn_like(x) if t_idx > 0 else 0)

        else:  # ddim
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else None
            acp_t      = acp[t_idx]
            acp_t_prev = acp[t_prev] if t_prev is not None else torch.ones(1, device=device).squeeze()

            # σ_t interpolates between deterministic (η=0) and stochastic (η=1)
            sigma = eta * (
                ((1 - acp_t_prev) / (1 - acp_t)).sqrt()
                * (1 - acp_t / acp_t_prev).clamp(min=0).sqrt()
            )
            dir_xt = (1 - acp_t_prev - sigma ** 2).clamp(min=0).sqrt() * eps_pred
            noise  = sigma * torch.randn_like(x) if (t_prev is not None and eta > 0) else 0
            x = acp_t_prev.sqrt() * pred_x0 + dir_xt + noise

    final = x.clamp(0.0, 1.0)
    if verbose:
        snapshots.append((0, final.cpu()))
        seen: set = set()
        snapshots = [(t, img) for t, img in snapshots if not (t in seen or seen.add(t))]
        return final, snapshots
    return final


@torch.no_grad()
def flow_sample(
    z: torch.Tensor,
    model: torch.nn.Module,
    config,
    device: torch.device,
    num_steps: int = 100,
    guidance_scale: float = 1.0,
    verbose: bool = False,
    n_snapshots: int = 10,
):
    """
    Euler ODE sampler for rectified flow.

    Integrates v_θ(x_t, t) from t=0 (noise) to t=1 (data) with uniform
    step size Δt = 1/num_steps.

    Args:
        z:              β-VAE conditioning latents [B, latent_dim].
        model:          Trained velocity network; called as model(x, t_embed, z).
        config:         Config object; must have in_channels, image_size.
                        num_timesteps is used as the sinusoidal embedding scale
                        (default 1000 if absent).
        device:         Target device.
        num_steps:      Number of Euler steps (fewer → faster, lower quality).
        guidance_scale: CFG scale; > 1 requires a CFG-trained model.
        verbose:        If True, returns (image, snapshots) where snapshots is a
                        list of (t_float, cpu_tensor [B,C,H,W]) at n_snapshots
                        evenly-spaced steps.

    Returns:
        Generated images [B, C, H, W] in [0, 1], or (images, snapshots) if verbose.
    """
    B = z.shape[0]
    C, H, W = config.in_channels, config.image_size, config.image_size
    embed_scale = getattr(config, 'num_timesteps', 1000)
    z_null = torch.zeros_like(z)

    x = torch.randn(B, C, H, W, device=device)
    dt = 1.0 / num_steps

    if verbose:
        stride = max(1, num_steps // n_snapshots)
        capture_at = set(range(0, num_steps, stride))
        capture_at.add(num_steps - 1)
        snapshots: List[Tuple[float, torch.Tensor]] = []

    for i in range(num_steps):
        t_val = i * dt
        t_batch = torch.full((B,), t_val, device=device)
        t_embed = t_batch * embed_scale

        if verbose and i in capture_at:
            snapshots.append((t_val, x.clone().clamp(0.0, 1.0).cpu()))

        if guidance_scale != 1.0:
            v_both = model(
                torch.cat([x, x]),
                torch.cat([t_embed, t_embed]),
                torch.cat([z, z_null]),
            )
            v_cond, v_uncond = v_both.chunk(2)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            v = model(x, t_embed, z)

        x = x + v * dt

    final = x.clamp(0.0, 1.0)
    if verbose:
        snapshots.append((1.0, final.cpu()))
        seen_t: set = set()
        snapshots = [(t, img) for t, img in snapshots if not (t in seen_t or seen_t.add(t))]
        return final, snapshots
    return final


@torch.no_grad()
def vpsde_sample(
    z: torch.Tensor,
    model: torch.nn.Module,
    config,
    device: torch.device,
    num_steps: int = 1000,
    guidance_scale: float = 1.0,
    verbose: bool = False,
    n_snapshots: int = 10,
):
    """
    Euler-Maruyama reverse SDE sampler for VP SDE.

    Integrates the reverse VP SDE from t=1 (prior) to t=0 (data):
        dx = [½β(t)x + β(t)·score] dt + √β(t) dB̄_t
    where score = -ε_θ / σ(t).

    Args:
        z:              β-VAE conditioning latents [B, latent_dim].
        model:          Trained score network (ε-prediction); called as model(x, t_embed, z).
        config:         Config object; must have in_channels, image_size.
                        vpsde_beta_min / vpsde_beta_max override defaults (0.01 / 5.0).
                        num_timesteps used as sinusoidal embedding scale (default 1000).
        device:         Target device.
        num_steps:      Number of EM steps.
        guidance_scale: CFG scale; > 1 requires a CFG-trained model.
        verbose:        If True, returns (image, snapshots).

    Returns:
        Generated images [B, C, H, W] in [0, 1], or (images, snapshots) if verbose.
    """
    from models.noise_process import VPSDE

    beta_min    = getattr(config, 'vpsde_beta_min', 0.01)
    beta_max    = getattr(config, 'vpsde_beta_max', 5.0)
    embed_scale = getattr(config, 'num_timesteps', 1000)
    vpsde = VPSDE(beta_min=beta_min, beta_max=beta_max)

    B = z.shape[0]
    C, H, W = config.in_channels, config.image_size, config.image_size
    dt = 1.0 / num_steps
    z_null = torch.zeros_like(z)

    # Initialize from prior: x(1) ~ N(0, σ(1)² I)
    t1 = torch.ones(B, device=device)
    sigma1 = vpsde.sigma(t1)[:, None, None, None]
    x = sigma1 * torch.randn(B, C, H, W, device=device)

    timesteps = torch.linspace(1.0, dt, num_steps, device=device)

    if verbose:
        stride = max(1, num_steps // n_snapshots)
        capture_at = set(range(0, num_steps, stride))
        capture_at.add(num_steps - 1)
        snapshots: List[Tuple[float, torch.Tensor]] = []

    for i, t_val in enumerate(timesteps):
        t_batch = torch.full((B,), t_val.item(), device=device)
        t_embed = t_batch * embed_scale

        if verbose and i in capture_at:
            snapshots.append((t_val.item(), x.clone().clamp(0.0, 1.0).cpu()))

        if guidance_scale != 1.0:
            eps_both = model(
                torch.cat([x, x]),
                torch.cat([t_embed, t_embed]),
                torch.cat([z, z_null]),
            )
            eps_cond, eps_uncond = eps_both.chunk(2)
            eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps_pred = model(x, t_embed, z)

        sigma_t = vpsde.sigma(t_batch)[:, None, None, None]
        beta_t  = vpsde.beta(t_batch)[:, None, None, None]
        score   = -eps_pred / (sigma_t + 1e-8)

        # Reverse-SDE drift: ½β·x + β·score
        drift  = 0.5 * beta_t * x + beta_t * score
        sde_noise = torch.randn_like(x)
        x = x + drift * dt + beta_t.sqrt() * (dt ** 0.5) * sde_noise

    final = x.clamp(0.0, 1.0)
    if verbose:
        snapshots.append((0.0, final.cpu()))
        seen_t: set = set()
        snapshots = [(t, img) for t, img in snapshots if not (t in seen_t or seen_t.add(t))]
        return final, snapshots
    return final