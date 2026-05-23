from __future__ import annotations

import torch


class DiffusionSchedule:
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.timesteps = timesteps
        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    def to(self, device: torch.device) -> DiffusionSchedule:
        for name in ("betas", "alphas", "alpha_bar", "sqrt_alpha_bar", "sqrt_one_minus_alpha_bar"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        sa = self.sqrt_alpha_bar[t][:, None, None, None]
        so = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        xt = sa * x0 + so * noise
        return xt, noise
