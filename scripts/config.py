from dataclasses import dataclass
from typing import List


@dataclass
class BetaVAEConfig:
    # Experiment
    experiment_name: str
    output_dir: str

    # Data
    train_dataset_path: str
    test_dataset_path: str
    batch_size: int

    # Model
    input_channels: int
    input_height: int
    input_width: int
    latent_dim: int
    hidden_dim: int
    beta: float

    # Optimizer
    lr: float
    weight_decay: float
    max_grad_norm: float

    # Training
    num_epochs: int
    log_interval: int
    eval_interval: int
    ckpt_interval: int


@dataclass
class DiffusionConfig:
    # Experiment
    experiment_name: str
    output_dir: str

    # Data
    train_dataset_path: str
    test_dataset_path: str
    batch_size: int

    # UNet
    in_channels: int
    image_size: int
    model_channels: int
    channel_mult: List[int]
    num_res_blocks: int
    attention_resolutions: List[int]
    dropout: float

    # Conditioning (frozen beta-VAE)
    betavae_config_path: str
    betavae_ckpt_path: str
    latent_dim: int

    # Diffusion
    num_timesteps: int
    beta_schedule: str      # 'linear' or 'cosine'
    min_snr_gamma: float    # Min-SNR-γ loss weighting (Hang et al. 2023); 5.0 is standard
    cfg_uncond_prob: float  # CFG conditioning dropout probability; 0.0 = disabled, 0.15 = standard

    # Optimizer
    lr: float
    weight_decay: float
    max_grad_norm: float

    # Training
    num_steps: int
    log_interval: int
    eval_interval: int
    ckpt_interval: int
