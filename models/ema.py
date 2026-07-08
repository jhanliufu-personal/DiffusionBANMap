"""Exponential moving average of model parameters.

The Flow Matching (Lipman et al. 2023) ImageNet experiments reuse the Dhariwal & Nichol
(2021) ADM U-Net and training recipe, which trains at a constant learning rate and relies
on EMA (decay 0.9999) rather than LR decay to smooth the noisy raw training weights before
sampling from them — see scripts/diffusion_trainer.py and utils.build_lr_scheduler.
"""

from typing import Dict

import torch


class EMA:
    """Tracks an exponential moving average of a model's parameters. Buffers (e.g. any
    non-parameter running stats) aren't tracked — they're read straight off the live model
    when an inference-ready snapshot is built via ema_shadow_to_model_state_dict.

    Uses the standard bias-corrected ramp for the first several hundred/thousand steps
    (as in e.g. HF diffusers' EMAModel): effective decay is
    min(decay, (1 + step) / (10 + step)), so the shadow doesn't stay pinned near the
    near-random initial weights for the whole first ~1/(1-decay) steps of training. That
    ramp matters here specifically — a flat decay=0.9999 wouldn't have converged away
    from init within a training run of only a few tens of thousands of steps.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.step = 0
        self.shadow: Dict[str, torch.Tensor] = {
            name: p.detach().clone() for name, p in model.named_parameters()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for name, p in model.named_parameters():
            self.shadow[name].mul_(d).add_(p.detach(), alpha=1 - d)

    def to(self, device) -> "EMA":
        """Move the shadow to `device`. Needed because __init__ snapshots whatever device
        `model` is on at construction time — if the caller moves the model afterward (e.g.
        DiffusionTrainer.train() does model.to(device) after __init__), the shadow is left
        behind unless this is called too."""
        self.shadow = {name: t.to(device) for name, t in self.shadow.items()}
        return self

    def state_dict(self) -> dict:
        return {"decay": self.decay, "step": self.step, "shadow": self.shadow}

    def load_state_dict(self, sd: dict) -> None:
        self.decay = sd["decay"]
        self.step = sd["step"]
        self.shadow = sd["shadow"]


def ema_shadow_to_model_state_dict(model: torch.nn.Module, shadow: Dict[str, torch.Tensor]) -> dict:
    """Merge an EMA shadow (EMA.state_dict()['shadow'], or a loaded checkpoint's
    ckpt['ema_state_dict']['shadow']) over model.state_dict() — buffers pass through
    unchanged since EMA only tracks parameters. Load the result with
    model.load_state_dict(...) to sample from the EMA weights instead of the raw ones.
    """
    sd = model.state_dict()
    sd.update(shadow)
    return sd
