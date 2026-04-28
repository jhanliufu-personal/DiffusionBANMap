from dataclasses import dataclass


@dataclass
class TrainingConfig:
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
