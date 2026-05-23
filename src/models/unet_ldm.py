from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ResBlockUNet(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        h = h + self.time(t_emb)[:, :, None, None]
        h = self.conv2(torch.nn.functional.silu(self.norm2(h)))
        return h + self.skip(x)


class LatentUNet(nn.Module):
    """Compact U-Net for epsilon prediction in latent space (32x32)."""

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 64,
        channel_mult: tuple[int, ...] = (1, 2, 4, 4),
        time_dim: int = 256,
    ) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        chs = [base_channels * m for m in channel_mult]
        self.in_conv = nn.Conv2d(in_channels, chs[0], 3, padding=1)

        self.down1 = ResBlockUNet(chs[0], chs[0], time_dim)
        self.down2 = ResBlockUNet(chs[0], chs[1], time_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.down3 = ResBlockUNet(chs[1], chs[1], time_dim)
        self.down4 = ResBlockUNet(chs[1], chs[2], time_dim)
        self.pool2 = nn.MaxPool2d(2)
        self.mid1 = ResBlockUNet(chs[2], chs[2], time_dim)
        self.mid2 = ResBlockUNet(chs[2], chs[2], time_dim)

        self.up1 = ResBlockUNet(chs[2] + chs[2], chs[1], time_dim)
        self.up2 = ResBlockUNet(chs[1] + chs[1], chs[0], time_dim)
        self.up3 = ResBlockUNet(chs[0] + chs[0], chs[0], time_dim)
        self.out_conv = nn.Conv2d(chs[0], in_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        h0 = self.in_conv(x)
        h1 = self.down1(h0, t_emb)
        h2 = self.down2(h1, t_emb)
        h3 = self.pool1(h2)
        h4 = self.down3(h3, t_emb)
        h5 = self.down4(h4, t_emb)
        h6 = self.pool2(h5)
        m = self.mid2(self.mid1(h6, t_emb), t_emb)
        u = torch.nn.functional.interpolate(m, scale_factor=2, mode="nearest")
        u = self.up1(torch.cat([u, h5], dim=1), t_emb)
        u = torch.nn.functional.interpolate(u, scale_factor=2, mode="nearest")
        u = self.up2(torch.cat([u, h2], dim=1), t_emb)
        u = self.up3(torch.cat([u, h1], dim=1), t_emb)
        return self.out_conv(u)
