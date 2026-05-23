from __future__ import annotations

import torch

from src.diffusion.schedule import DiffusionSchedule
from src.models.unet_ldm import LatentUNet
from src.models.vae import VAE


@torch.no_grad()
def ddim_sample(
    unet: LatentUNet,
    schedule: DiffusionSchedule,
    shape: tuple[int, ...],
    steps: int = 50,
    eta: float = 0.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    device = device or next(unet.parameters()).device
    schedule = schedule.to(device)
    b = shape[0]
    x = torch.randn(shape, device=device)

    times = torch.linspace(
        schedule.timesteps - 1, 0, steps, device=device
    ).long()

    for i, t in enumerate(times):
        t_batch = t.expand(b)
        eps = unet(x, t_batch)

        alpha_bar = schedule.alpha_bar[t]
        alpha_bar_prev = (
            schedule.alpha_bar[times[i + 1]]
            if i + 1 < len(times)
            else torch.tensor(1.0, device=device)
        )

        pred_x0 = (x - schedule.sqrt_one_minus_alpha_bar[t] * eps) / (
            schedule.sqrt_alpha_bar[t] + 1e-8
        )

        if i + 1 >= len(times):
            x = pred_x0
            break

        sigma = (
            eta
            * torch.sqrt(
                (1 - alpha_bar_prev)
                / (1 - alpha_bar + 1e-8)
                * (1 - alpha_bar / (alpha_bar_prev + 1e-8))
            )
        )
        dir_xt = torch.sqrt(alpha_bar_prev) * pred_x0
        noise = torch.randn_like(x) if eta > 0 else 0
        x = dir_xt + torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps + sigma * noise

    return x


@torch.no_grad()
def decode_latents(vae: VAE, z: torch.Tensor) -> torch.Tensor:
    return vae.decode(z).clamp(-1, 1)
