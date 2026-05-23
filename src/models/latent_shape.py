from __future__ import annotations

import torch

from src.models.vae import VAE


@torch.no_grad()
def latent_spatial_size(vae: VAE, image_size: int = 256) -> int:
    dummy = torch.zeros(1, 3, image_size, image_size, device=next(vae.parameters()).device)
    z = vae.encode_latent(dummy)
    return z.shape[-1]
