import os
import argparse
import types

import wandb
import yaml
import torch
import torch.multiprocessing as mp

from data_utils import build_dataloaders
from models.beta_vae import BetaVAE
from models.unet import UNet
from scripts.diffusion_trainer import DiffusionTrainer
from utils import (
    discover_device, count_model_params, make_run_tag, build_lr_scheduler,
    setup_distributed, cleanup_distributed,
)


def _load_frozen_vae(cfg, device: torch.device) -> BetaVAE:
    with open(cfg.betavae_config_path) as f:
        vae_cfg = types.SimpleNamespace(**yaml.safe_load(f))
    vae = BetaVAE(
        input_channels=vae_cfg.input_channels,
        input_height=vae_cfg.input_height,
        input_width=vae_cfg.input_width,
        latent_dim=vae_cfg.latent_dim,
        hidden_dim=vae_cfg.hidden_dim,
        beta=vae_cfg.beta,
        device=device,
    )
    ckpt = torch.load(cfg.betavae_ckpt_path, map_location=device)
    vae.load_state_dict(ckpt["model_state_dict"])
    vae.eval()
    print(f"Loaded frozen β-VAE from {cfg.betavae_ckpt_path} (step {ckpt['step']})")
    return vae


def main_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    """Entry point for one training process. rank/world_size are (0, 1) for plain
    single-GPU/CPU/MPS runs; for multi-GPU runs this is spawned once per visible CUDA
    device by __main__ below, each pinned to its own GPU via DDP."""
    is_main = rank == 0
    if world_size > 1:
        try:
            device = setup_distributed(rank, world_size)
        except Exception as e:
            print(f"[GPU {rank}] FAILED to join process group: {e}", flush=True)
            raise
        print(f"[GPU {rank}] {torch.cuda.get_device_name(device)} — joined process group successfully", flush=True)
        # Every rank must reach this barrier for it to pass — if it hangs, some rank's
        # setup_distributed above either failed silently or never got here at all.
        torch.distributed.barrier()
        if is_main:
            print(f"All {world_size} GPUs launched successfully — starting DDP training\n")
    else:
        device = torch.device(discover_device())
    if is_main:
        print(f"Device: {device}" + (f" | distributed: {world_size} GPUs (DDP)" if world_size > 1 else ""))

    with open(args.config) as f:
        cfg = types.SimpleNamespace(**yaml.safe_load(f))

    cfg.output_dir = f"{cfg.output_dir}_{make_run_tag(cfg)}"

    resume_ckpt_path = args.resume or getattr(cfg, "resume_ckpt_path", None)
    if resume_ckpt_path and is_main:
        print(f"Will resume training from {resume_ckpt_path}")

    unconditional = getattr(cfg, "unconditional", False)
    if unconditional:
        vae = None
        if is_main:
            print("Train for unconditional generation")
            print("Unconditional — no VAE, z is a zero vector")
    elif hasattr(cfg, "betavae_config_path"):
        vae = _load_frozen_vae(cfg, device)
    elif getattr(cfg, "encoding_model", None) is not None:
        # Conditional on precomputed latents — the dataloader provides (image, latent)
        # pairs from {dataset_type}_{encoding_model}_pca_latents (see
        # data_utils.resolve_latents_paths, scripts/extract_image_embeddings.py).
        vae = None
        if is_main:
            print(f"Conditional on precomputed {cfg.encoding_model} latents — no VAE to load")
    else:
        raise ValueError(
            "Conditional run with nothing to condition on: set encoding_model (precomputed "
            "latents), betavae_config_path (on-the-fly VAE latents), or unconditional: true"
        )

    model = UNet(
        in_channels=cfg.in_channels,
        image_size=cfg.image_size,
        model_channels=cfg.model_channels,
        channel_mult=cfg.channel_mult,
        num_res_blocks=cfg.num_res_blocks,
        attention_resolutions=cfg.attention_resolutions,
        dropout=cfg.dropout,
        num_head_channels=getattr(cfg, "num_head_channels", 64),
        latent_dim=cfg.latent_dim,
    )
    if is_main:
        count_model_params(model)

    train_loader, val_loader = build_dataloaders(cfg, rank=rank, world_size=world_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = build_lr_scheduler(optimizer, cfg)

    if is_main:
        run_name = f"{cfg.experiment_name}_{args.run_name}" if args.run_name else cfg.experiment_name
        wandb.login(key=os.environ["WANDB_API_KEY"])
        wandb.init(project="diffusion_ban_map", name=run_name, notes=args.notes, config=vars(cfg))
        
        # # This is one time thing ... for resuming uncond training
        # wandb.init(project="diffusion_ban_map", resume_from=f"{run_name}?_step=13600")

    trainer = DiffusionTrainer(
        model=model,
        vae_model=vae,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=cfg,
        device=device,
        rank=rank,
        world_size=world_size,
    )
    trainer.train(resume_ckpt_path=resume_ckpt_path)

    if world_size > 1:
        cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--run_name", type=str, default=None, help="Appended to cfg.experiment_name for the wandb run name")
    parser.add_argument("--notes", type=str, default=None, help="wandb run notes")
    args = parser.parse_args()

    # Auto-detect all visible GPUs and train on all of them via DDP — no torchrun needed,
    # `python scripts/diffusion_train.py --config ...` alone picks up every GPU on the
    # machine. Falls back to the previous single-process path (device via discover_device,
    # cuda/mps/cpu) when there's 0 or 1 GPU, so single-GPU/CPU/MPS runs are unaffected.
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Found {num_gpus} GPUs — launching distributed data-parallel training")
        mp.spawn(main_worker, args=(num_gpus, args), nprocs=num_gpus, join=True)
    else:
        main_worker(0, 1, args)
