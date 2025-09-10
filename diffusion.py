import math, torch, numpy as np                 # standard modules
from typing import Optional, Literal            # typing helpers
import torch.nn.functional as F                 # functional ops
from torch import nn                            # nn base

Schedule = Literal["linear", "cosine"]          # allowed schedules


class Diffusion(nn.Module):                     # DDPM core module
    def __init__(self, T: int = 400, schedule: Schedule = "cosine", device: Optional[torch.device] = None):
        """Precompute all schedules/buffers for T steps."""
        super().__init__()                                                          # init superclass
        self.T = int(T)                                                             # diffusion steps (int)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ----- build beta schedule -----
        if schedule == "linear":                                                    # linear beta schedule
            betas = torch.linspace(1e-4, 0.02, T, device=self.device)
        elif schedule == "cosine":                                                  # cosine ᾱ_t (Nichol & Dhariwal)
            s = 0.008
            x = torch.linspace(0, T, T + 1, dtype=torch.float64, device=self.device)  # 0..T (T+1 points)
            alphas_bar = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2      # ᾱ(t)
            alphas_bar = alphas_bar / alphas_bar[0]                                    # normalize so ᾱ_0 = 1
            betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])                             # convert to β_t
            betas = betas.float().clamp(1e-4, 0.999)                                    # numeric safety
        else:
            raise ValueError("Unknown schedule")                                       # guard invalid name

        alphas = 1.0 - betas                                                          # α_t
        alphas_bar = torch.cumprod(alhas := alphas, dim=0) if False else torch.cumprod(alphas, dim=0)  # ᾱ_t
        alphas_bar_prev = torch.cat([torch.ones(1, device=self.device), alphas_bar[:-1]], dim=0)       # ᾱ_{t-1}

        # register buffers used by training/sampling (kept on same device)
        self.register_buffer("betas", betas)                                          # β_t
        self.register_buffer("alphas", alphas)                                        # α_t
        self.register_buffer("alphas_bar", alphas_bar)                                # ᾱ_t
        self.register_buffer("alphas_bar_prev", alphas_bar_prev)                      # ᾱ_{t-1}
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))               # √ᾱ_t
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1.0 - alphas_bar)) # √(1-ᾱ_t)
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))           # √(1/α_t)
        posterior_var = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)          # posterior variance (Eq.7)
        self.register_buffer("posterior_variance", posterior_var.clamp(min=1e-20))    # clamp for stability

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward (noising) process q(x_t | x_0)."""
        if eps is None:                                                               # draw noise if not provided
            eps = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alphas_bar[t][:, None, None, None]                        # broadcast √ᾱ_t
        sqrt_mab = self.sqrt_one_minus_alphas_bar[t][:, None, None, None]            # broadcast √(1-ᾱ_t)
        return sqrt_ab * x0 + sqrt_mab * eps                                         # x_t = √ᾱ_t x0 + √(1-ᾱ_t) ε

    def p_losses(self, model: nn.Module, x0: torch.Tensor, y: Optional[torch.Tensor] = None, p_uncond: float = 0.1) -> torch.Tensor:
        """Training objective: predict ε on a random timestep."""
        b = x0.size(0)                                                                # batch size
        t = torch.randint(0, self.T, (b,), device=x0.device)                          # random timesteps
        noise = torch.randn_like(x0)                                                  # ground-truth ε
        x_t = self.q_sample(x0, t, noise)                                             # build x_t
        use_cond = (getattr(model, "num_classes", None) is not None) and (y is not None)  # conditional?
        if use_cond:                                                                  # classifier-free dropout
            mask = torch.rand(b, device=x0.device) < p_uncond                         # which samples are unconditional
            y_in = y.clone()                                                          # copy labels
            y_in[mask] = -1                                                           # mark dropped labels with -1 (ignored in UNet)
        else:
            y_in = None                                                               # unconditional
        eps_pred = model(x_t, t, y_in)                                                # predict noise
        return F.mse_loss(eps_pred, noise)                                            # MSE(ε̂, ε)

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """One reverse DDPM step p(x_{t-1} | x_t)."""
        eps = model(x, t, y)                                                          # ε̂(x_t, t, y)
        sqrt_recip_alpha = self.sqrt_recip_alphas[t][:, None, None, None]             # √(1/α_t)
        beta_t = self.betas[t][:, None, None, None]                                   # β_t
        sqrt_one_minus_ab = self.sqrt_one_minus_alphas_bar[t][:, None, None, None]    # √(1-ᾱ_t)
        mean = sqrt_recip_alpha * (x - beta_t / sqrt_one_minus_ab * eps)              # DDPM mean
        var = self.posterior_variance[t][:, None, None, None]                         # DDPM variance
        nonzero_mask = (t != 0).float().view(x.size(0), 1, 1, 1)                      # add noise except at t=0
        noise = torch.randn_like(x)                                                   # fresh noise
        return mean + nonzero_mask * torch.sqrt(var) * noise                          # update

    @torch.no_grad()
    def sample(self, model: nn.Module, n: int, img_size: int, y: Optional[torch.Tensor] = None, guidance_scale: float = 0.0) -> torch.Tensor:
        """Full ancestral sampling loop (optionally with classifier-free guidance)."""
        was_training = model.training                                                 # remember mode
        model.eval()                                                                  # eval for sampling
        ch = getattr(getattr(model, "in_conv", None), "in_channels", 3)              # infer channels (3)
        x = torch.randn(n, ch, img_size, img_size, device=self.device)               # start from noise
        for i in reversed(range(self.T)):                                            # t = T-1..0
            t = torch.full((n,), i, device=self.device, dtype=torch.long)            # current timestep
            if guidance_scale > 0 and (y is not None):                                # CFG branch
                eps_uc = model(x, t, None)                                            # unconditional ε̂
                eps_c  = model(x, t, y)                                               # conditional ε̂
                eps    = eps_uc + guidance_scale * (eps_c - eps_uc)                   # blended ε̂
                sqrt_recip_alpha = self.sqrt_recip_alphas[t][:, None, None, None]     # factors (same as p_sample)
                beta_t = self.betas[t][:, None, None, None]
                sqrt_one_minus_ab = self.sqrt_one_minus_alphas_bar[t][:, None, None, None]
                mean = sqrt_recip_alpha * (x - beta_t / sqrt_one_minus_ab * eps)      # DDPM mean with guided ε̂
                var  = self.posterior_variance[t][:, None, None, None]                # variance
                noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)         # no noise at t=0
                x = mean + torch.sqrt(var) * noise                                    # update
            else:                                                                      # no guidance
                x = self.p_sample(model, x, t, y)                                     # standard step
        model.train(was_training)                                                     # restore original mode
        return x.clamp(-1, 1)                                                         # clamp range
