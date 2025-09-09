# Esegui in /content/ddpm-birds
cat > diffusion.py <<'PY'
# diffusion.py
from __future__ import annotations
from typing import Tuple, Optional
import math
import torch
import torch.nn.functional as F

@torch.no_grad()
def make_beta_schedule(T: int, mode: str = "cosine") -> torch.Tensor:
    if mode == "linear":
        beta_start, beta_end = 1e-4, 0.02
        return torch.linspace(beta_start, beta_end, T)
    elif mode == "cosine":
        s = 0.008
        steps = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos(((steps / T + s) / (1 + s)) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        betas = torch.clip(1 - (alpha_bar[1:] / alpha_bar[:-1]), 0.0001, 0.999)
        return betas.float()
    else:
        raise ValueError(f"Unknown schedule mode: {mode}")

class Diffusion:
    def __init__(self, T: int, schedule: str = "cosine", device: str = "cpu"):
        self.T = T
        self.device = device
        betas = make_beta_schedule(T, schedule).to(device)
        alphas = 1.0 - betas
        alphabars = torch.cumprod(alphas, dim=0)
        self.register_buffers(betas, alphas, alphabars)

    def register_buffers(self, betas: torch.Tensor, alphas: torch.Tensor, alphabars: torch.Tensor) -> None:
        self.betas = betas
        self.alphas = alphas
        self.alphabars = alphabars
        self.sqrt_alphabars = torch.sqrt(self.alphabars)
        self.sqrt_one_minus_alphabars = torch.sqrt(1.0 - self.alphabars)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        # posterior variance: \tilde{beta}_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t), with alpha_bar_{-1}=1
        alpha_bar_prev = torch.cat([torch.ones(1, device=self.device), self.alphabars[:-1]], dim=0)
        self.posterior_variance = self.betas * (1.0 - alpha_bar_prev) / (1.0 - self.alphabars + 1e-8)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alphabars.gather(0, t).reshape(-1, 1, 1, 1)
        sqrt_omb = self.sqrt_one_minus_alphabars.gather(0, t).reshape(-1, 1, 1, 1)
        return sqrt_ab * x0 + sqrt_omb * noise

    def p_losses(self, model, x0: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None,
                 p_uncond: float = 0.0) -> torch.Tensor:
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        y_in = None if (y is None or torch.rand(()) < p_uncond) else y
        eps_pred = model(x_t, t, y_in)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def p_sample(self, model, x_t: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None,
                 guidance_scale: float = 0.0) -> torch.Tensor:
        eps_cond = model(x_t, t, y)
        if guidance_scale > 0.0:
            eps_uncond = model(x_t, t, None)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps = eps_cond
        alpha_t = self.alphas.gather(0, t).reshape(-1, 1, 1, 1)
        alpha_bar_t = self.alphabars.gather(0, t).reshape(-1, 1, 1, 1)
        mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * eps)
        noise = torch.randn_like(x_t)
        mask = (t > 0).float().reshape(-1, 1, 1, 1)
        var_t = self.posterior_variance.gather(0, t).reshape(-1, 1, 1, 1)
        return mean + mask * torch.sqrt(var_t.clamp(min=1e-20)) * noise

    @torch.no_grad()
    def sample(self, model, shape: Tuple[int, int, int, int], y: Optional[torch.Tensor] = None,
               guidance_scale: float = 0.0) -> torch.Tensor:
        x = torch.randn(shape, device=self.device)
        for t_step in reversed(range(self.T)):
            t = torch.full((shape[0],), t_step, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t, y=y, guidance_scale=guidance_scale)
        return x
PY

