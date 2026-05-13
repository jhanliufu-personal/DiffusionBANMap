import os
import json
import torch
from typing import Optional


class BaseTrainer:
    """Shared machinery: output dirs, config persistence, checkpoint save/load."""

    def __init__(self, model, optimizer, scheduler, config, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = torch.device(device)
        self.step = 0

        self.ckpt_dir = os.path.join(config.output_dir, 'checkpoints')
        self.vis_dir = os.path.join(config.output_dir, 'visualizations')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)

        with open(os.path.join(config.output_dir, 'config.json'), 'w') as f:
            json.dump(vars(config), f, indent=2)

        print(f"Trainer initialized. Output directory: {config.output_dir}")

    def _save_checkpoint(self, elapsed_time: float, extra: dict = {}):
        ckpt = {
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "elapsed_time": elapsed_time,
            **extra,
        }
        if self.scheduler is not None:
            ckpt["scheduler_state_dict"] = self.scheduler.state_dict()
        path = os.path.join(
            self.ckpt_dir, f"{self.config.experiment_name}_step{self.step}.pt"
        )
        torch.save(ckpt, path)

    def _load_checkpoint(self, path: str) -> float:
        """Load checkpoint state; returns elapsed_time offset."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.step = ckpt["step"]
        return ckpt.get("elapsed_time", 0.0)
