import os
import json
import torch
from typing import Optional


class BaseTrainer:
    """Shared machinery: output dirs, config persistence, checkpoint save/load."""

    def __init__(self, model, optimizer, scheduler, config, device, rank: int = 0, world_size: int = 1):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = torch.device(device)
        self.step = 0
        self._last_loaded_ckpt: Optional[dict] = None
        self.rank = rank
        self.world_size = world_size
        self.is_main = rank == 0

        self.ckpt_dir = os.path.join(config.output_dir, 'checkpoints')
        self.vis_dir = os.path.join(config.output_dir, 'visualizations')

        # Only rank 0 touches disk here — every rank computed the same output_dir/paths
        # independently (deterministic from config), but N processes concurrently
        # creating dirs / writing config.json is a pointless race. Other ranks wait at
        # the barrier so they can't read ckpt_dir before rank 0 has created it.
        if self.is_main:
            os.makedirs(self.ckpt_dir, exist_ok=True)
            os.makedirs(self.vis_dir, exist_ok=True)
            with open(os.path.join(config.output_dir, 'config.json'), 'w') as f:
                json.dump(vars(config), f, indent=2)
            print(f"Trainer initialized. Output directory: {config.output_dir}")
        if self.world_size > 1:
            torch.distributed.barrier()

    def _save_checkpoint(self, elapsed_time: float, extra: dict = {}, filename: str = "best_ckpt.pt"):
        if not self.is_main:
            return
        ckpt = {
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "elapsed_time": elapsed_time,
            **extra,
        }
        if self.scheduler is not None:
            ckpt["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(ckpt, os.path.join(self.ckpt_dir, filename))

    def _load_checkpoint(self, path: str, reset_step: bool = False) -> float:
        """Load checkpoint state; returns elapsed_time offset. The full checkpoint dict is
        kept on self._last_loaded_ckpt so subclasses can pull out extra keys (e.g. EMA
        state) they passed into _save_checkpoint's `extra`, without a second torch.load.

        reset_step=True treats the checkpoint as a weights-only initialization for a new
        experiment (e.g. finetuning from a different run's checkpoint) rather than resuming
        this same experiment after an interruption: self.step stays 0 and elapsed_time is
        not carried over, so config.num_steps/num_epochs is interpreted as this experiment's
        own budget instead of an absolute target the foreign checkpoint's step count might
        already exceed.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self._last_loaded_ckpt = ckpt
        if reset_step:
            return 0.0
        self.step = ckpt["step"]
        return ckpt.get("elapsed_time", 0.0)
