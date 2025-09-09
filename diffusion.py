# diffusion.py
# DDPM core with linear/cosine beta schedules, stable posterior, and CFG sampling.
from __future__ import annotations  # deve restare la prima riga

import math, torch, numpy as np
from torch import nn
import torch.nn.functional as F  # <-- AGGIUNTO: serve per F.mse_loss


Schedule = Literal["linear", "cosine"]


def make_beta_schedule(T: int, schedule: Schedule = "cosine") -> torch.Tensor:
    """
    Create a beta schedule of length T.
    - 'linear': linearly spaced betas.
    - 'cosine': cosine alphas (per Nichol & Dhariwal), numerically stable.
    """
    if schedule == "linear":
        # small betas for 64x64 works fine
        return torch.linspace(1e-4, 0.02, T)
    elif schedule == "cosine":
        # cosine schedule -> alphas_bar; convert to betas
        s = 0.008
        t = torch.linspace(0, T, T + 1)
        f = torch.cos(((t / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_bar = f / f[0]
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        return betas.clamp(1e-6, 0.999)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


class Diffusion(nn.Module):
    """
    Minimal DDPM wrapper that:
    - registers buffers for {betas, alphas, alphas_cumprod}
    - provides loss and sampling routines
    """
    def __init__(self, T: int = 400, schedule: Schedule = "cosine", device: Optional[torch.device] = None):
        super().__init__()
        # set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # build schedule tensors
        betas = make_beta_schedule(T, schedule).to(self.device)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # compute helpers with correct shapes
        alphas_bar_prev = torch.cat([torch.ones(1, device=self.device), alphas_bar[:-1]], dim=0)

        # register buffers used in training/sampling
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", alphas_bar)
        self.register_buffer("alphas_bar_prev", alphas_bar_prev)

        # posterior variance per Ho et al. (equation 7): beta_t * (1 - alphabar_{t-1}) / (1 - alphabar_t)
        posterior_var = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
        self.register_buffer("posterior_variance", posterior_var.clamp(min=1e-20))
        self.register_buffer("sqrt_alphas", torch.sqrt(alphas))
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1.0 - alphas_bar))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer("T_int", torch.tensor(T, dtype=torch.long, device=self.device))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample x_t ~ q(x_t | x_0) = sqrt(alphabar_t) * x0 + sqrt(1 - alphabar_t) * eps
        """
        if eps is None:
            eps = torch.randn_like(x0)
        a_bar = self.alphas_bar[t]                                 # (B,)
        return self.sqrt_alphas_bar[t, None, None, None] * x0 + self.sqrt_one_minus_alphas_bar[t, None, None, None] * eps

    def p_losses(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        p_uncond: float = 0.1,
    ) -> torch.Tensor:
        """
        DDPM training loss: MSE between true noise and predicted noise at random t.
        Supports classifier-free guidance training by randomly dropping y.
        """
        b = x0.size(0)
        t = torch.randint(0, int(self.T_int.item()), (b,), device=x0.device)  # uniform t
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)

        # Classifier-free guidance dropout
        y_in = None
        if (getattr(model, "num_classes", None) is not None) and (y is not None):
            drop_mask = (torch.rand(b, device=x0.device) < p_uncond)          # True -> drop
            y_in = y.clone()
            y_in[drop_mask] = -1                                              # mark dropped entries with -1
            # Model will treat y=None when it sees label -1
        eps_pred = model(x_t, t, y_in if (y_in is not None and (y_in >= 0).any()) else None)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        img_size: int,
        n: int,
        y: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate n samples at resolution img_size.
        - If model is conditional and y is provided, can use CFG with guidance_scale>1.
        - If unconditional or guidance_scale==1, single pass per step.
        Returns images in [-1, 1].
        """
        device = next(model.parameters()).device
        x = torch.randn(n, 3, img_size, img_size, device=device)

        # Determine if we can do CFG
        cond_model = (getattr(model, "num_classes", None) is not None) and (y is not None)
        for t_int in reversed(range(int(self.T_int.item()))):
            t = torch.full((n,), t_int, device=device, dtype=torch.long)
            if cond_model and guidance_scale != 1.0:
                # conditional pass
                eps_cond = model(x, t, y)
                # unconditional pass (y=None)
                eps_uncond = model(x, t, None)
                # combine
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps = model(x, t, y if cond_model else None)

            # DDPM x_{t-1} step
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alphas_bar[t]
            sqrt_recip_alpha = self.sqrt_recip_alphas[t, None, None, None]
            beta_t = self.betas[t, None, None, None]

            # predict x0
            x0_pred = (x - torch.sqrt(1 - alpha_bar_t)[:, None, None, None] * eps) / torch.sqrt(alpha_bar_t)[:, None, None, None]
            # compute mean of posterior q(x_{t-1} | x_t, x0)
            mean = sqrt_recip_alpha * (x - beta_t / torch.sqrt(1 - alpha_bar_t)[:, None, None, None] * eps)

            if t_int > 0:
                noise = torch.randn_like(x)
                var = self.posterior_variance[t, None, None, None]
                x = mean + torch.sqrt(var) * noise
            else:
                x = mean

        return x.clamp(-1, 1)

