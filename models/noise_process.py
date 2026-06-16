"""
Noise process implementations for conditional diffusion training.

Three processes, all sharing the same interface (sample_t, corrupt, get_target,
loss, predict_x0, embed_t, vis_t_vals, dtype) so the trainer dispatches to any
of them without branching:

    DDPM         — discrete ε-prediction, precomputed cosine/linear schedule
    RectifiedFlow — continuous ODE, velocity-prediction, t ∈ [0, 1]
    VPSDE        — continuous SDE, ε-prediction, linear β schedule, t ∈ [0, 1]

z conditioning is handled externally by the UNet; these classes define only the
forward noising process, the training target, and the loss.
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor


# ── Shared schedule utility ───────────────────────────────────────────────────

def _cosine_betas(num_timesteps: int, s: float = 0.008) -> Tensor:
    """Cosine noise schedule (Nichol & Dhariwal 2021)."""
    steps = torch.arange(num_timesteps + 1) / num_timesteps
    f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    acp = f / f[0]
    return (1 - acp[1:] / acp[:-1]).clamp(max=0.999)


# ── DDPM ─────────────────────────────────────────────────────────────────────

class DDPM:
    """
    Discrete DDPM forward process (Ho et al. 2020).

    Forward:  x_t = √ᾱ_t · x0 + √(1-ᾱ_t) · ε
    Target:   ε  (ε-prediction)
    Loss:     MSE with Min-SNR-γ weighting (Hang et al. 2023)

    All schedule tensors are precomputed and stored on device.

    Args:
        num_timesteps: T, number of discrete diffusion steps.
        beta_schedule: 'cosine' (Nichol & Dhariwal 2021) or 'linear'.
        min_snr_gamma: γ for Min-SNR-γ loss weighting (default 5.0).
        device:        Device for schedule tensors.
    """

    dtype = torch.long  # t is an integer index into the schedule

    def __init__(
        self,
        num_timesteps: int,
        beta_schedule: str,
        min_snr_gamma: float = 5.0,
        device='cpu',
    ):
        self.T = num_timesteps
        self.min_snr_gamma = min_snr_gamma

        if beta_schedule == 'cosine':
            betas = _cosine_betas(num_timesteps)
        elif beta_schedule == 'linear':
            betas = torch.linspace(1e-4, 0.02, num_timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule!r}")

        alphas = 1.0 - betas
        acp    = torch.cumprod(alphas, dim=0)
        self._sqrt_acp    = acp.sqrt().to(device)
        self._sqrt_1macp  = (1 - acp).sqrt().to(device)
        self._snr         = (acp / (1.0 - acp)).to(device)

    def sample_t(self, B: int, device) -> Tensor:
        return torch.randint(0, self.T, (B,), device=device)

    def corrupt(self, x0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        sqrt_a   = self._sqrt_acp[t][:, None, None, None]
        sqrt_1ma = self._sqrt_1macp[t][:, None, None, None]
        eps = torch.randn_like(x0)
        return sqrt_a * x0 + sqrt_1ma * eps, eps

    def get_target(self, x0: Tensor, eps: Tensor) -> Tensor:
        return eps

    def loss(self, pred: Tensor, target: Tensor, t: Tensor) -> Tensor:
        snr_t  = self._snr[t]
        weight = (snr_t.clamp(max=self.min_snr_gamma) / snr_t).detach()
        loss_per = F.mse_loss(pred, target, reduction='none').mean(dim=[1, 2, 3])
        return (loss_per * weight).mean()

    def predict_x0(self, x_t: Tensor, eps_pred: Tensor, t: Tensor) -> Tensor:
        sqrt_a   = self._sqrt_acp[t][:, None, None, None].clamp(min=1e-8)
        sqrt_1ma = self._sqrt_1macp[t][:, None, None, None]
        return (x_t - sqrt_1ma * eps_pred) / sqrt_a

    def embed_t(self, t: Tensor, scale: int = 1) -> Tensor:
        # t is already in {0, …, T-1}, matching the UNet's sinusoidal embedding range
        return t.float()

    def vis_t_vals(self) -> list:
        return [self.T // 8, self.T // 2, self.T * 7 // 8]


# ── Rectified Flow ────────────────────────────────────────────────────────────

class RectifiedFlow:
    """
    Rectified Flow / Flow Matching (Liu et al. 2023).

    Forward:  x_t = t·x0 + (1-t)·ε,   ε ~ N(0, I),   t ∈ [0, 1]
              t=0 → pure noise, t=1 → pure data.
    Target:   v* = x0 - ε  (velocity)
    Loss:     unweighted MSE

    Sampler:  Euler ODE from t=0 to t=1,  x_{t+dt} = x_t + v_θ · dt
    """

    dtype = torch.float32

    def sample_t(self, B: int, device) -> Tensor:
        return torch.rand(B, device=device)

    def corrupt(self, x0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        eps = torch.randn_like(x0)
        t_b = t[:, None, None, None]
        return t_b * x0 + (1.0 - t_b) * eps, eps

    def get_target(self, x0: Tensor, eps: Tensor) -> Tensor:
        return x0 - eps

    def loss(self, pred: Tensor, target: Tensor, t: Tensor) -> Tensor:
        return F.mse_loss(pred, target)

    def predict_x0(self, x_t: Tensor, v_pred: Tensor, t: Tensor) -> Tensor:
        """x0 = x_t + (1-t)·v  (exact inversion when v = v*)."""
        t_b = t[:, None, None, None]
        return x_t + (1.0 - t_b) * v_pred

    def embed_t(self, t: Tensor, scale: int = 1000) -> Tensor:
        """Scale t ∈ [0,1] to [0, scale] to match DDPM's sinusoidal embedding range."""
        return t * scale

    def vis_t_vals(self) -> list:
        return [0.125, 0.5, 0.875]


# ── VP SDE ────────────────────────────────────────────────────────────────────

class VPSDE:
    """
    Variance-Preserving SDE (Song et al. 2021).

    Forward SDE:  dx = -½ β(t) x dt + √β(t) dB_t
    Linear β:     β(t) = β_min + (β_max - β_min) · t
    Marginal:     x_t = c(t)·x0 + σ(t)·ε
                  c(t) = exp(-½ ∫₀ᵗ β(s)ds),  σ(t) = √(1 - c(t)²)
    Target:       ε  (same as DDPM, but t is continuous)
    Loss:         MSE, optional Min-SNR-γ weighting

    Sampler:  Euler-Maruyama reverse SDE from t=1 to t=0.

    Args:
        beta_min:      Minimum of the linear noise schedule (default 0.01).
        beta_max:      Maximum of the linear noise schedule (default 5.0).
        min_snr_gamma: γ for Min-SNR-γ weighting; 0 disables weighting (default 0).
    """

    dtype = torch.float32

    def __init__(
        self,
        beta_min: float = 0.01,
        beta_max: float = 5.0,
        min_snr_gamma: float = 0.0,
    ):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.min_snr_gamma = min_snr_gamma

    def beta(self, t: Tensor) -> Tensor:
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def c(self, t: Tensor) -> Tensor:
        integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2
        return torch.exp(-0.5 * integral)

    def sigma(self, t: Tensor) -> Tensor:
        return (1.0 - self.c(t) ** 2).clamp(min=0.0).sqrt()

    def snr(self, t: Tensor) -> Tensor:
        ct = self.c(t)
        return ct ** 2 / (1.0 - ct ** 2).clamp(min=1e-8)

    def sample_t(self, B: int, device) -> Tensor:
        return torch.rand(B, device=device)

    def corrupt(self, x0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        ct = self.c(t)[:, None, None, None]
        st = self.sigma(t)[:, None, None, None]
        eps = torch.randn_like(x0)
        return ct * x0 + st * eps, eps

    def get_target(self, x0: Tensor, eps: Tensor) -> Tensor:
        return eps

    def loss(self, pred: Tensor, target: Tensor, t: Tensor) -> Tensor:
        if self.min_snr_gamma > 0.0:
            snr_t  = self.snr(t)
            weight = (snr_t.clamp(max=self.min_snr_gamma) / snr_t).detach()
            loss_per = F.mse_loss(pred, target, reduction='none').mean(dim=[1, 2, 3])
            return (loss_per * weight).mean()
        return F.mse_loss(pred, target)

    def predict_x0(self, x_t: Tensor, eps_pred: Tensor, t: Tensor) -> Tensor:
        ct = self.c(t)[:, None, None, None].clamp(min=1e-8)
        st = self.sigma(t)[:, None, None, None]
        return (x_t - st * eps_pred) / ct

    def embed_t(self, t: Tensor, scale: int = 1000) -> Tensor:
        """Scale t ∈ [0,1] to [0, scale] to match DDPM's sinusoidal embedding range."""
        return t * scale

    def vis_t_vals(self) -> list:
        return [0.125, 0.5, 0.875]
