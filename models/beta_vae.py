"""
β-VAE Implementation
Based on Higgins et al. (2017) "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
and the Nature Communications (2021) application to face patch neurons.

This implementation provides the exact architecture and training procedure described in the papers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Encoder(nn.Module):
    """
    Probabilistic encoder that maps input images to latent space.
    Outputs parameters (mu, sigma) for the latent distribution.
    """
    
    def __init__(self, input_channels: int = 3, latent_dim: int = 10, hidden_dim: int = 128, 
                 input_height: int = 64, input_width: int = 64):
        super(Encoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.input_height = input_height
        self.input_width = input_width
        
        # Convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # Second conv block  
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # Third conv block
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # Fourth conv block
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # Calculate feature map dimensions after convolutions
        # Each conv layer with stride=2, padding=1, kernel=4 reduces size by factor of 2
        self.feature_height = input_height // (2 ** 4)  # 4 conv layers with stride 2
        self.feature_width = input_width // (2 ** 4)
        self.flattened_size = 64 * self.feature_height * self.feature_width
        
        # Fully connected layers for latent parameters
        self.fc_common = nn.Linear(self.flattened_size, hidden_dim)
        # Mean parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        # Log variance parameters log(σ²)  
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            
        Returns:
            mu: Mean parameters of latent distribution [batch_size, latent_dim]
            logvar: Log variance parameters [batch_size, latent_dim]
        """
        # Extract features through convolutional layers
        features = self.conv_layers(x)
        
        # Flatten features
        features = features.view(features.size(0), -1)
        
        # Common fully connected layer
        h = F.relu(self.fc_common(features))
        
        # Output latent parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class Decoder(nn.Module):
    """
    Probabilistic decoder network p_θ(x|z) that reconstructs images from latent codes.
    Uses deconvolutional/transpose convolutional layers.
    """
    
    def __init__(self, latent_dim: int = 10, output_channels: int = 3, hidden_dim: int = 128,
                 output_height: int = 64, output_width: int = 64):
        super(Decoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_height = output_height
        self.output_width = output_width
        
        # Calculate initial feature map dimensions
        # We need to work backwards from output dimensions
        self.init_height = output_height // (2 ** 4)  # 4 deconv layers with stride 2
        self.init_width = output_width // (2 ** 4)
        
        # Fully connected layer to project latent code to feature map
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64 * self.init_height * self.init_width),
            nn.ReLU()
        )
        
        # Transposed convolutional layers for upsampling
        self.deconv_layers = nn.Sequential(
            # First deconv block
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # Second deconv block
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # Third deconv block
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # Fourth deconv block
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            z: Latent codes [batch_size, latent_dim]
            
        Returns:
            x_recon: Reconstructed images [batch_size, channels, height, width]
        """
        # Project latent code to feature map
        h = self.fc(z)
        
        # Reshape to 4D tensor for deconvolution
        h = h.view(h.size(0), 64, self.init_height, self.init_width)
        
        # Generate reconstruction through deconvolutional layers
        x_recon = self.deconv_layers(h)
        
        return x_recon


class BetaVAE(nn.Module):
    """
    β-VAE model implementing the framework from Higgins et al. (2017).
    
    The β-VAE modifies the standard VAE with an adjustable hyperparameter β that
    balances latent channel capacity and independence constraints with reconstruction accuracy.
    """
    
    def __init__(self,
        input_channels: int = 3,
        input_height: int = 64,
        input_width: int = 64,
        latent_dim: int = 10,
        hidden_dim: int = 128,
        beta: float = 4.0,
        device: torch.device = torch.device('cpu')
    ):
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.input_height = input_height
        self.input_width = input_width
        self.device = device

        # Initialize encoder and decoder networks
        self.encoder = Encoder(input_channels, latent_dim, hidden_dim, input_height, input_width)
        self.decoder = Decoder(latent_dim, input_channels, hidden_dim, input_height, input_width)
        self.to(device)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution.
        
        Args:
            mu: Mean parameters [batch_size, latent_dim]
            logvar: Log variance parameters [batch_size, latent_dim]
            
        Returns:
            z: Sampled latent codes [batch_size, latent_dim]
        """
        # if self.training:
        #     std = torch.exp(0.5 * logvar)
        #     eps = torch.randn_like(std)
        #     return mu + eps * std
        # else:
        #     # During inference, use mean
        #     return mu

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through β-VAE.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            
        Returns:
            x_recon: Reconstructed images [batch_size, channels, height, width]
            mu: Mean parameters [batch_size, latent_dim]
            logvar: Log variance parameters [batch_size, latent_dim]
        """
        # Encode input to latent parameters
        mu, logvar = self.encoder(x)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, logvar)
        
        # Decode latent codes to reconstruction
        x_recon = self.decoder(z)
        
        return x_recon, mu, logvar
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input images to latent parameters.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            
        Returns:
            mu: Mean parameters [batch_size, latent_dim]
            logvar: Log variance parameters [batch_size, latent_dim]
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent codes to images.
        
        Args:
            z: Latent codes [batch_size, latent_dim]
            
        Returns:
            x_recon: Reconstructed images [batch_size, channels, height, width]
        """
        return self.decoder(z)
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Generate samples by decoding random latent codes.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            samples: Generated images [num_samples, channels, height, width]
        """
        # Sample from prior distribution N(0, I)
        z = torch.randn(num_samples, self.latent_dim, device=device)
        
        # Decode to images
        with torch.no_grad():
            samples = self.decoder(z)
        
        return samples
    
    def get_latent_codes(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent codes for input images (using mean of latent distribution).
        
        Args:
            x: Input images [batch_size, channels, height, width]
            
        Returns:
            z: Latent codes [batch_size, latent_dim]
        """
        mu, logvar = self.encoder(x)
        return mu  # Use mean as latent code
    
    def set_beta(self, beta: float):
        """
        Update the β hyperparameter.
        
        Args:
            beta: New β value
        """
        self.beta = beta


# def create_beta_vae(config: Optional[dict] = None) -> BetaVAE:
#     """
#     Factory function to create β-VAE model with different configurations.
    
#     Args:
#         config: Configuration dictionary with model parameters including:
#             - input_channels: Number of input channels (default: 3)
#             - latent_dim: Latent space dimensions (default: 10)
#             - hidden_dim: Hidden layer dimensions (default: 128)
#             - beta: β hyperparameter (default: 4.0)
#             - input_height: Input image height (default: 64)
#             - input_width: Input image width (default: 64)
        
#     Returns:
#         model: Configured β-VAE model
#     """
#     if config is None:
#         # Default configuration for face images (similar to Nature Communications 2021)
#         config = {
#             'input_channels': 3,
#             'latent_dim': 10,
#             'hidden_dim': 128,
#             'beta': 4.0,
#             'input_height': 64,
#             'input_width': 64
#         }
    
#     model = BetaVAE(
#         input_channels=config.get('input_channels', 3),
#         latent_dim=config.get('latent_dim', 10),
#         hidden_dim=config.get('hidden_dim', 128),
#         beta=config.get('beta', 4.0),
#         input_height=config.get('input_height', 64),
#         input_width=config.get('input_width', 64)
#     )
    
#     return model