from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class VAE(nn.Module):
    """Convolutional VAE for 256x256 grayscale (3-channel) patches."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        c = base_channels

        # 3x downsample: 256 -> 32 spatial (8x per side)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, c, 4, 2, 1),
            ResBlock(c),
            nn.Conv2d(c, c * 2, 4, 2, 1),
            ResBlock(c * 2),
            nn.Conv2d(c * 2, c * 4, 4, 2, 1),
            ResBlock(c * 4),
        )
        self.mu = nn.Conv2d(c * 4, latent_channels, 1)
        self.logvar = nn.Conv2d(c * 4, latent_channels, 1)

        self.dec_in = nn.Conv2d(latent_channels, c * 4, 1)
        self.decoder = nn.Sequential(
            ResBlock(c * 4),
            nn.ConvTranspose2d(c * 4, c * 2, 4, 2, 1),
            ResBlock(c * 2),
            nn.ConvTranspose2d(c * 2, c, 4, 2, 1),
            ResBlock(c),
            nn.ConvTranspose2d(c, in_channels, 4, 2, 1),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_in(z)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @torch.no_grad()
    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x)
        return mu
