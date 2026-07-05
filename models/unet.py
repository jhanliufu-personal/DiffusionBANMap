"""
Conditional UNet for pixel-space diffusion / flow matching.

Architecture follows ADM (Dhariwal & Nichol 2021, "Diffusion Models Beat GANs"),
the same U-Net used in the original Flow Matching paper (Lipman et al. 2023)
for its ImageNet experiments:
  - Sinusoidal timestep embeddings
  - ResBlocks with GroupNorm + SiLU, AdaGN conditioning (scale+shift from the
    combined time + latent embedding), and BigGAN-style up/down resampling
    (parameter-free resample applied to both the main and skip paths, folded
    into the ResBlock instead of a separate strided conv)
  - Multi-head self-attention at specified resolutions (fixed channels/head,
    so head count grows with channel width like ADM)
  - β-VAE latent conditioning: project z → time_emb_dim, add to time embedding
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding, shape [B, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1)
    )
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class Upsample(nn.Module):
    """Parameter-free nearest-neighbor 2x upsample (BigGAN-style resample used inside ResBlock)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2.0, mode='nearest')


class Downsample(nn.Module):
    """Parameter-free 2x average-pool downsample (BigGAN-style resample used inside ResBlock)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, kernel_size=2, stride=2)


class ResBlock(nn.Module):
    """
    ADM-style ResBlock.

    Conditioning (time + latent, pre-combined into t_emb by the caller) is injected
    via AdaGN: the block predicts a per-channel scale and shift from t_emb and applies
    them at the second GroupNorm, rather than adding a plain bias.

    If up/down is set, a parameter-free resample (nearest-upsample or avg-pool-downsample)
    is applied to both the main path and the skip path before the main-path convolutions,
    matching BigGAN/ADM's approach of folding resampling into the residual block instead
    of using a dedicated strided/transposed conv between blocks.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        t_dim: int,
        dropout: float = 0.0,
        up: bool = False,
        down: bool = False,
    ):
        super().__init__()
        assert not (up and down)
        self.in_norm = nn.GroupNorm(32, in_ch)
        self.in_act = nn.SiLU()
        self.in_conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        if up:
            self.h_upd = Upsample()
            self.x_upd = Upsample()
        elif down:
            self.h_upd = Downsample()
            self.x_upd = Downsample()
        else:
            self.h_upd = nn.Identity()
            self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, 2 * out_ch))
        self.out_norm = nn.GroupNorm(32, out_ch)
        self.out_rest = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.in_act(self.in_norm(x))
        h = self.h_upd(h)
        x = self.x_upd(x)
        h = self.in_conv(h)

        scale, shift = self.emb_layers(t_emb)[:, :, None, None].chunk(2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_rest(h)
        return self.skip(x) + h


class AttentionBlock(nn.Module):
    """Multi-head spatial self-attention with residual."""

    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W)
        qkv = self.qkv(h).reshape(B * self.num_heads, 3 * self.head_dim, H * W)
        q, k, v = qkv.chunk(3, dim=1)                        # each [B*h, d, HW]
        attn = torch.bmm(q.transpose(1, 2), k) * self.scale  # [B*h, HW, HW]
        attn = attn.softmax(dim=-1)
        out = torch.bmm(attn, v.transpose(1, 2)).transpose(1, 2)  # [B*h, d, HW]
        out = out.reshape(B, C, H * W)
        out = self.proj_out(out).view(B, C, H, W)
        return x + out


class UNet(nn.Module):
    """
    Conditional UNet for pixel-space diffusion / flow matching.

    Args:
        in_channels: image channels (e.g. 3)
        model_channels: base channel count (e.g. 64 or 192)
        channel_mult: channel multipliers per resolution level (e.g. [1, 2, 3, 4])
        num_res_blocks: ResBlocks per resolution level in the encoder
        attention_resolutions: spatial resolutions at which to apply self-attention
        dropout: dropout probability inside ResBlocks
        num_head_channels: channels per attention head (ADM default: 64); the
            number of heads at each resolution is channels // num_head_channels
        latent_dim: β-VAE latent dimension used for conditioning
        image_size: input spatial size (must be divisible by 2^(len(channel_mult)-1))
    """

    def __init__(
        self,
        in_channels: int = 3,
        model_channels: int = 64,
        channel_mult: List[int] = (1, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = (16,),
        dropout: float = 0.1,
        num_head_channels: int = 64,
        latent_dim: int = 10,
        image_size: int = 64,
    ):
        super().__init__()
        self._model_channels = model_channels
        self._num_head_channels = num_head_channels
        t_dim = model_channels * 4
        ch_list = [model_channels * m for m in channel_mult]

        def num_heads(channels: int) -> int:
            return max(1, channels // num_head_channels)

        # ── Time + conditioning ────────────────────────────────────────────────
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )
        self.cond_proj = nn.Linear(latent_dim, t_dim)

        # ── Encoder ───────────────────────────────────────────────────────────
        self.input_conv = nn.Conv2d(in_channels, ch_list[0], 3, padding=1)

        # Build encoder levels; track skip-connection channel sizes
        self.encoder = nn.ModuleList()   # List[nn.ModuleList] — one per level
        self.downsamples = nn.ModuleList()

        skip_ch_list: List[int] = [ch_list[0]]  # input_conv output
        in_ch = ch_list[0]
        cur_res = image_size

        for level, out_ch in enumerate(ch_list):
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(ResBlock(in_ch, out_ch, t_dim, dropout))
                if cur_res in attention_resolutions:
                    level_blocks.append(AttentionBlock(out_ch, num_heads(out_ch)))
                skip_ch_list.append(out_ch)
                in_ch = out_ch
            self.encoder.append(level_blocks)
            if level < len(ch_list) - 1:
                self.downsamples.append(ResBlock(in_ch, in_ch, t_dim, dropout, down=True))
                cur_res //= 2
                skip_ch_list.append(in_ch)

        # ── Bottleneck ─────────────────────────────────────────────────────────
        self.mid1 = ResBlock(in_ch, in_ch, t_dim, dropout)
        self.mid_attn = AttentionBlock(in_ch, num_heads(in_ch))
        self.mid2 = ResBlock(in_ch, in_ch, t_dim, dropout)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.decoder = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level, out_ch in enumerate(reversed(ch_list)):
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):  # +1 to consume level-boundary skip
                skip_ch = skip_ch_list.pop()
                level_blocks.append(ResBlock(in_ch + skip_ch, out_ch, t_dim, dropout))
                if cur_res in attention_resolutions:
                    level_blocks.append(AttentionBlock(out_ch, num_heads(out_ch)))
                in_ch = out_ch
            self.decoder.append(level_blocks)
            if level < len(ch_list) - 1:
                self.upsamples.append(ResBlock(in_ch, in_ch, t_dim, dropout, up=True))
                cur_res *= 2

        # ── Output ─────────────────────────────────────────────────────────────
        self.out_norm = nn.GroupNorm(32, in_ch)
        self.out_conv = nn.Conv2d(in_ch, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: noisy image [B, C, H, W]
            t: integer timesteps [B]
            z: β-VAE conditioning latent [B, latent_dim]
        Returns:
            predicted noise ε (or velocity, depending on the noise process) [B, C, H, W]
        """
        t_emb = self.time_embed(timestep_embedding(t, self._model_channels))
        t_emb = t_emb + self.cond_proj(z)

        h = self.input_conv(x)
        skips = [h]

        # Encoder
        for level, level_blocks in enumerate(self.encoder):
            for block in level_blocks:
                if isinstance(block, ResBlock):
                    h = block(h, t_emb)
                    skips.append(h)
                else:
                    h = block(h)
                    skips[-1] = h   # attention doesn't add a new skip; update last
            if level < len(self.encoder) - 1:
                h = self.downsamples[level](h, t_emb)
                skips.append(h)

        # Bottleneck
        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        # Decoder
        for level, level_blocks in enumerate(self.decoder):
            for block in level_blocks:
                if isinstance(block, ResBlock):
                    h = block(torch.cat([h, skips.pop()], dim=1), t_emb)
                else:
                    h = block(h)
            if level < len(self.decoder) - 1:
                h = self.upsamples[level](h, t_emb)

        return self.out_conv(F.silu(self.out_norm(h)))
