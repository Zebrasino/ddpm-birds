# diffusion.py
# Line-by-line commented diffusion utilities: schedules, forward process, loss, and sampler.
from __future__ import annotations  # Future annotations

from typing import Tuple, Optional  # For type hints

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
        # Linearly increase beta from small to larger value
        beta_start, beta_end = 1e-4, 0.02
        return torch.linspace(beta_start, beta_end, T)
    elif mode == "cosine":
        # Cosine schedule as in improved DDPM: define alpha_bar via cosine, then derive betas
        s = 0.008  # Small offset to avoid singularities at t=0
        steps = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos(((steps / T + s) / (1 + s)) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        betas = torch.clip(1 - (alpha_bar[1:] / alpha_bar[:-1]), 0.0001, 0.999)
        return betas.float()
    else:
        raise ValueError(f"Unknown schedule mode: {mode}")

class Diffusion:
    """Encapsulates forward diffusion (q) and reverse sampling (p_theta)."""
    def __init__(self, T: int, schedule: str = "cosine", device: str = "cpu"):
        self.T = T  # Number of diffusion steps
        self.device = device  # Device on which tensors should live
        # Precompute and store schedule-related buffers
        betas = make_beta_schedule(T, schedule).to(device)  # (T,)
        alphas = 1.0 - betas  # (T,)
        alphabars = torch.cumprod(alphas, dim=0)  # Cumulative product ᾱ_t
        self.register_buffers(betas, alphas, alphabars)  # Store as buffers in a runtime-friendly way

    def register_buffers(self, betas: torch.Tensor, alphas: torch.Tensor, alphabars: torch.Tensor) -> None:
        """Keep schedule tensors on device and expose convenient views."""
        self.betas = betas  # β_t per step
        self.alphas = alphas  # α_t = 1 - β_t
        self.alphabars = alphabars  # ᾱ_t = ∏_{s<=t} α_s
        # Precompute sqrt terms used frequently
        self.sqrt_alphabars = torch.sqrt(self.alphabars)
        self.sqrt_one_minus_alphabars = torch.sqrt(1.0 - self.alphabars)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        # σ_t choice for DDPM sampling; default variance
        self.posterior_variance = self.betas * (1.0 - self.alphabars[:-1].roll(1, 0).fill_(1.0))  # Unused placeholder

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample x_t ~ q(x_t | x_0) = √ᾱ_t x_0 + √(1-ᾱ_t) ε.

        x0: clean images in [-1,1], shape (N,C,H,W); t: integer timesteps in [0, T-1], shape (N,).

        """
        if noise is None:
            noise = torch.randn_like(x0)  # Sample standard Gaussian noise
        # Gather the appropriate ᾱ_t for each element in the batch and reshape for broadcasting
        sqrt_ab = self.sqrt_alphabars.gather(0, t).reshape(-1, 1, 1, 1)
        sqrt_omb = self.sqrt_one_minus_alphabars.gather(0, t).reshape(-1, 1, 1, 1)
        # Mix clean signal and noise to obtain a noisy sample at time t
        return sqrt_ab * x0 + sqrt_omb * noise

    def p_losses(self, model, x0: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None,
                 p_uncond: float = 0.0) -> torch.Tensor:
        """Compute the standard ε-prediction loss (MSE) with optional classifier-free guidance dropout.

        model: U-Net that predicts ε; x0: clean images; t: timesteps; y: labels; p_uncond: drop prob for cond.

        """
        # Sample Gaussian noise that we want the network to predict
        noise = torch.randn_like(x0)
        # Obtain x_t by diffusing x_0 with the sampled noise and selected timestep
        x_t = self.q_sample(x0, t, noise)
        # With probability p_uncond, drop the labels to train an unconditional branch
        y_in = None if (y is None or torch.rand(()) < p_uncond) else y
        # Predict the noise component ε_t using the model
        eps_pred = model(x_t, t, y_in)
        # Mean-squared error between predicted and true noise
        loss = F.mse_loss(eps_pred, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, model, x_t: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None,
                 guidance_scale: float = 0.0) -> torch.Tensor:
        """One reverse diffusion step x_{t-1} from x_t using DDPM update; supports classifier-free guidance.

        guidance_scale: s; output uses ε = ε_u + s (ε_c − ε_u).

        """
        # Predict noise from the conditional branch (using provided labels)
        eps_cond = model(x_t, t, y)
        if guidance_scale > 0.0:
            # Predict noise from the unconditional branch by passing y=None
            eps_uncond = model(x_t, t, None)
            # Combine predictions according to CFG formula
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            # No guidance: just use the conditional prediction
            eps = eps_cond

        # Retrieve α_t and β_t for this timestep
        alpha_t = self.alphas.gather(0, t).reshape(-1, 1, 1, 1)
        alpha_bar_t = self.alphabars.gather(0, t).reshape(-1, 1, 1, 1)
        beta_t = self.betas.gather(0, t).reshape(-1, 1, 1, 1)

        # Compute the DDPM mean for p(x_{t-1} | x_t)
        mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - ( (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) ) * eps)

        # For t > 0, add noise; for t=0, produce the mean only
        noise = torch.randn_like(x_t)
        mask = (t > 0).float().reshape(-1, 1, 1, 1)  # 1 if t>0 else 0
        # Standard DDPM choice of variance: β_t
        sample = mean + mask * torch.sqrt(beta_t) * noise
        return sample

    @torch.no_grad()
    def sample(self, model, shape: Tuple[int, int, int, int], y: Optional[torch.Tensor] = None,
               guidance_scale: float = 0.0) -> torch.Tensor:
        """Generate samples by iterating p_sample from T-1 down to 0."""
        # Start from pure Gaussian noise
        x = torch.randn(shape, device=self.device)
        # Prepare a vector of timesteps that will be broadcasted across the batch
        for t_step in reversed(range(self.T)):
            t = torch.full((shape[0],), t_step, device=self.device, dtype=torch.long)
            # Perform one reverse step
            x = self.p_sample(model, x, t, y=y, guidance_scale=guidance_scale)
        return x  # Final x_0 estimate in [-1,1]
