# diffusion.py
# Line-by-line commented diffusion utilities: schedules, forward process, loss, and sampler.
from __future__ import annotations  # Future annotations

from typing import Tuple, Optional  # Type hints

import math  # For cosine schedule and numeric stability
import torch  # Tensors and math
import torch.nn.functional as F  # Loss function

@torch.no_grad()
def make_beta_schedule(T: int, mode: str = "cosine") -> torch.Tensor:
    """Create a schedule of betas for T steps.
    'linear' uses a simple linear increase; 'cosine' follows the improved schedule from Nichol & Dhariwal.
    Returns tensor of shape (T,) on CPU; move to device as needed.
    """
    if mode == "linear":
        beta_start, beta_end = 1e-4, 0.02  # Common linear schedule bounds
        return torch.linspace(beta_start, beta_end, T)
    elif mode == "cosine":
        s = 0.008  # Small offset to avoid singularities at t=0
        steps = torch.arange(T + 1, dtype=torch.float64)  # T+1 points to define alpha_bar
        f = torch.cos(((steps / T + s) / (1 + s)) * math.pi / 2) ** 2  # Cosine curve
        alpha_bar = f / f[0]  # Normalize to 1 at t=0
        betas = torch.clip(1 - (alpha_bar[1:] / alpha_bar[:-1]), 0.0001, 0.999)  # Derive betas
        return betas.float()  # Return float32
    else:
        raise ValueError(f"Unknown schedule mode: {mode}")

class Diffusion:
    """Encapsulates forward diffusion (q) and reverse sampling (p_theta)."""
    def __init__(self, T: int, schedule: str = "cosine", device: str = "cpu"):
        self.T = T  # Number of diffusion steps
        self.device = device  # Device for buffers
        betas = make_beta_schedule(T, schedule).to(device)  # (T,)
        alphas = 1.0 - betas  # α_t
        alphabars = torch.cumprod(alphas, dim=0)  # ᾱ_t = ∏_{s≤t} α_s
        self.register_buffers(betas, alphas, alphabars)  # Precompute helpers

    def register_buffers(self, betas: torch.Tensor, alphas: torch.Tensor, alphabars: torch.Tensor) -> None:
        """Keep schedule tensors on device and expose convenient views."""
        self.betas = betas  # β_t
        self.alphas = alphas  # α_t
        self.alphabars = alphabars  # ᾱ_t
        self.sqrt_alphabars = torch.sqrt(self.alphabars)  # √ᾱ_t
        self.sqrt_one_minus_alphabars = torch.sqrt(1.0 - self.alphabars)  # √(1-ᾱ_t)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)  # √(1/α_t)

        # Posterior variance (DDPM):
        #   \tilde{\beta}_t = \beta_t * (1 - \bar{\alpha}_{t-1}) / (1 - \bar{\alpha}_t)
        # with \bar{\alpha}_{-1} := 1 to align shapes.
        alpha_bar_prev = torch.cat([torch.ones(1, device=self.device), self.alphabars[:-1]], dim=0)  # (T,)
        self.posterior_variance = self.betas * (1.0 - alpha_bar_prev) / (1.0 - self.alphabars + 1e-8)  # (T,)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample x_t ~ q(x_t | x_0) = √ᾱ_t x_0 + √(1-ᾱ_t) ε.
        x0: (N,C,H,W) in [-1,1]; t: (N,) in [0, T-1].
        """
        if noise is None:
            noise = torch.randn_like(x0)  # Gaussian noise
        sqrt_ab = self.sqrt_alphabars.gather(0, t).reshape(-1, 1, 1, 1)  # √ᾱ_t per sample
        sqrt_omb = self.sqrt_one_minus_alphabars.gather(0, t).reshape(-1, 1, 1, 1)  # √(1-ᾱ_t)
        return sqrt_ab * x0 + sqrt_omb * noise  # Noised sample

    def p_losses(self, model, x0: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None,
                 p_uncond: float = 0.0) -> torch.Tensor:
        """Compute ε-prediction loss (MSE) with optional classifier-free guidance dropout."""
        noise = torch.randn_like(x0)  # Ground-truth noise
        x_t = self.q_sample(x0, t, noise)  # Forward diffuse
        y_in = None if (y is None or torch.rand(()) < p_uncond) else y  # Drop labels w.p. p_uncond
        eps_pred = model(x_t, t, y_in)  # Predict ε_t
        loss = torch.nn.functional.mse_loss(eps_pred, noise)  # MSE loss
        return loss  # Scalar tensor

    @torch.no_grad()
    def p_sample(self, model, x_t: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None,
                 guidance_scale: float = 0.0) -> torch.Tensor:
        """One reverse diffusion step x_{t-1} from x_t using DDPM update; supports classifier-free guidance."""
        eps_cond = model(x_t, t, y)  # Conditional prediction
        if guidance_scale > 0.0:
            eps_uncond = model(x_t, t, None)  # Unconditional prediction
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)  # CFG mixing
        else:
            eps = eps_cond  # No guidance

        alpha_t = self.alphas.gather(0, t).reshape(-1, 1, 1, 1)  # α_t
        alpha_bar_t = self.alphabars.gather(0, t).reshape(-1, 1, 1, 1)  # ᾱ_t
        # DDPM mean of p(x_{t-1} | x_t)
        mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * eps)

        noise = torch.randn_like(x_t)  # Random noise for sampling
        mask = (t > 0).float().reshape(-1, 1, 1, 1)  # No noise at t=0
        var_t = self.posterior_variance.gather(0, t).reshape(-1, 1, 1, 1)  # \tilde{\beta}_t
        sample = mean + mask * torch.sqrt(var_t.clamp(min=1e-20)) * noise  # Sample x_{t-1}
        return sample  # Next sample

    @torch.no_grad()
    def sample(self, model, shape: Tuple[int, int, int, int], y: Optional[torch.Tensor] = None,
               guidance_scale: float = 0.0) -> torch.Tensor:
        """Generate samples by iterating p_sample from T-1 down to 0."""
        x = torch.randn(shape, device=self.device)  # Start from Gaussian noise
        for t_step in reversed(range(self.T)):  # T-1 ... 0
            t = torch.full((shape[0],), t_step, device=self.device, dtype=torch.long)  # Current t per item
            x = self.p_sample(model, x, t, y=y, guidance_scale=guidance_scale)  # One reverse step
        return x  # Final x_0 estimate in [-1,1]
