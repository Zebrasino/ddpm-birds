# diffusion.py
# DDPM scheduler + training loss + ancestral sampler (with optional CFG).
# Every line is commented.

from typing import Optional                       # typing
import torch                                      # PyTorch
import torch.nn as nn                             # nn modules
import torch.nn.functional as F                   # functional ops

def make_beta_schedule(T: int, schedule: str = "cosine"):
    # Builds beta_t for t=1..T following a stable schedule (cosine/linear).
    if schedule == "linear":                      # simple linear schedule
        beta_start, beta_end = 1e-4, 0.02         # standard DDPM linear
        return torch.linspace(beta_start, beta_end, T)
    elif schedule == "cosine":                    # improved cosine schedule
        s = 0.008                                 # small offset
        t = torch.linspace(0, T, T+1) / T         # [0..1]
        f = torch.cos((t + s) / (1 + s) * torch.pi * 0.5) ** 2
        a_bar = f / f[0]                          # normalize so a_bar(0)=1
        betas = 1 - (a_bar[1:] / a_bar[:-1])      # derive betas
        return betas.clamp(1e-6, 0.999)           # keep numerically safe
    else:
        raise ValueError(f"Unknown schedule={schedule}")

class Diffusion(nn.Module):
    # Container for schedules and convenience sampling funcs.
    def __init__(self, T: int = 200, schedule: str = "cosine", device=None):
        super().__init__()                        # init
        betas = make_beta_schedule(T, schedule)   # beta_t sequence
        alphas = 1.0 - betas                      # alpha_t
        alphas_bar = torch.cumprod(alphas, dim=0) # prod_k alpha_k
        # Register as buffers so they move with .to(device) and save in ckpt
        self.register_buffer("betas", betas)                      # (T,)
        self.register_buffer("alphas", alphas)                    # (T,)
        self.register_buffer("alphas_bar", alphas_bar)            # (T,)
        self.register_buffer("sqrt_alphas", torch.sqrt(alphas))   # (T,)
        self.register_buffer("sqrt_one_minus_alphas_bar",
                             torch.sqrt(1 - alphas_bar))          # (T,)
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))  # (T,)
        self.T = T                                                # store T

        if device is not None:                                   # optional move
            self.to(device)

    # ------- Forward diffusion (q) helpers -------

    def q_sample(self, x0, t, noise=None):
        # Sample x_t ~ q(x_t | x_0) = sqrt(a_bar_t)*x0 + sqrt(1-a_bar_t)*ε
        if noise is None: noise = torch.randn_like(x0)            # ε ~ N(0,I)
        sqrt_ab = self.alphas_bar[t].sqrt().view(-1,1,1,1)        # (B,1,1,1)
        sqrt_omb = self.sqrt_one_minus_alphas_bar[t].view(-1,1,1,1)
        return sqrt_ab * x0 + sqrt_omb * noise                    # (B,3,H,W)

    # ------- Training loss -------

    def p_losses(self, model, x0, y: Optional[torch.Tensor] = None, p_uncond: float = 0.0):
        # Implements E_{t,ε}[ || ε - ε̂(x_t,t,y) ||^2 ]
        B = x0.size(0)                                            # batch size
        t = torch.randint(0, self.T, (B,), device=x0.device)      # sample t
        noise = torch.randn_like(x0)                              # ε ~ N(0,I)
        xt = self.q_sample(x0, t, noise)                          # x_t
        # Classifier-free guidance: randomly drop conditioning with prob p_uncond
        y_in = None                                               # default: uncond
        if y is not None:                                         # if dataset is class-cond
            if p_uncond > 0.0:                                    # drop with probability
                drop = torch.rand(B, device=x0.device) < p_uncond # mask
                y_mod = y.clone()
                y_mod[drop] = -1                                  # mark "null" (we'll pass None)
                y_in = torch.where(drop, torch.full_like(y, -1), y)  # keep interface
                # we actually pass None for unconditional positions by masking below
                y_in = torch.where(y_in >= 0, y_in, torch.tensor(-1, device=y.device))
            else:
                y_in = y
        # Convert -1 to None at call-site by splitting the mini-batch:
        if y_in is not None and (y_in >= 0).sum() != B:           # mixed cond/uncond
            # Run two forwards to keep shapes simple (tiny overhead)
            cond_mask = (y_in >= 0)                               # mask cond items
            eps = torch.zeros_like(xt)                            # output buffer
            if cond_mask.any():                                   # run cond part
                eps[cond_mask] = model(xt[cond_mask], t[cond_mask], y_in[cond_mask])
            if (~cond_mask).any():                                # run uncond part
                eps[~cond_mask] = model(xt[~cond_mask], t[~cond_mask], None)
        else:
            # Pure cond or pure uncond
            eps = model(xt, t, None if (y_in is None or (y_in is not None and (y_in < 0).all())) else y_in)
        return F.mse_loss(eps, noise)                              # L2 on noise

    # ------- Reverse diffusion (p) sampler -------

    @torch.no_grad()
    def sample(self, model, shape, y: Optional[torch.Tensor] = None,
               guidance_scale: float = 0.0, deterministic: bool = False):
        # DDPM ancestral sampler with optional classifier-free guidance (CFG).
        B, C, H, W = shape                                       # unpack shape
        x = torch.randn(shape, device=self.betas.device)         # start from noise
        for i in reversed(range(self.T)):                        # t = T-1..0
            t = torch.full((B,), i, device=x.device, dtype=torch.long)   # (B,)
            # unconditional ε̂
            eps_u = model(x, t, None)
            if y is not None and guidance_scale > 0:             # conditional branch (CFG)
                eps_c = model(x, t, y)                           # conditional ε̂
                eps = eps_u + guidance_scale * (eps_c - eps_u)   # CFG combine
            else:
                eps = eps_u                                      # just uncond
            # Compute mean of p(x_{t-1}|x_t)
            sra = self.sqrt_recip_alphas[t].view(-1,1,1,1)       # sqrt(1/α_t)
            bt  = self.betas[t].view(-1,1,1,1)                   # β_t
            somab = self.sqrt_one_minus_alphas_bar[t].view(-1,1,1,1)  # √(1-ᾱ_t)
            mean = sra * (x - bt / somab * eps)                  # DDPM mean
            if i > 0 and not deterministic:                      # add noise except last step
                noise = torch.randn_like(x)                      # ζ ~ N(0,I)
                var = self.betas[t].view(-1,1,1,1)               # Var = β_t
                x = mean + var.sqrt() * noise                    # ancestral step
            else:
                x = mean                                         # last step or deterministic
        return x.clamp(-1, 1)                                    # clamp to valid range
