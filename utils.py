"""
Utility functions for β-VAE training and evaluation.
Implements the exact loss function from Higgins et al. (2017) and training utilities.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Literal
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


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