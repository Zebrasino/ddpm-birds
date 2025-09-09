# diffusion.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

def betas_linear(T: int, beta_start=1e-4, beta_end=2e-2) -> Tensor:
    # classico schedule lineare usato nel paper DDPM
    return torch.linspace(beta_start, beta_end, T)

def betas_cosine(T: int, s: float = 0.008) -> Tensor:
    # Improved DDPM cosine schedule (Nichol & Dhariwal)
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-8, 0.999)

def extract_at(tensor: Tensor, t: Tensor, shape) -> Tensor:
    # helper: prende tensor[t] e lo ridimensiona per broadcast sul batch
    out = tensor.gather(0, t)
    return out.view(-1, *([1] * (len(shape) - 1)))

class Diffusion(nn.Module):
    """
    Implementazione DDPM (ε-prediction).
    Training: loss = ||ε_pred − ε||²
    Sampling: x_{t-1} = 1/sqrt(α_t) * (x_t − (1−α_t)/sqrt(1−\bar{α}_t) * ε̂) + σ_t z
              con σ_t^2 = β̃_t = β_t * (1−\bar{α}_{t-1})/(1−\bar{α}_t)
    """
    def __init__(self, T: int = 400, schedule: str = "cosine", device: str = "cuda"):
        super().__init__()
        self.T = T
        self.device = device

        if schedule == "cosine":
            betas = betas_cosine(T)
        elif schedule == "linear":
            betas = betas_linear(T)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        # registra tutti i tensori necessari come buffer
        self.register_buffer("betas", betas.to(device))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("alphas_cumprod_prev",
                             torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]]))

        # termini ricorrenti utili
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / self.alphas))
        self.register_buffer("posterior_variance",
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.register_buffer("posterior_log_variance_clipped",
            torch.log(self.posterior_variance.clamp(min=1e-20)))
        self.register_buffer("posterior_mean_coef1",
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.register_buffer("posterior_mean_coef2",
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod))

    @torch.no_grad()
    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
        # x_t = sqrt(\bar{α}_t) x0 + sqrt(1 - \bar{α}_t) ε
        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_at(self.sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_at(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    def p_losses(self, model: nn.Module, x0: Tensor, t: Tensor,
                 y: Tensor | None = None, p_uncond: float = 0.0) -> Tensor:
        """
        Loss di training: il modello predice ε.
        Per CFG durante il training, a caso si "droppa" l'etichetta (null/None) con p_uncond.
        """
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)

        # classifier-free: droppa alcune label
        y_in = None
        if y is not None:
            if p_uncond > 0.0:
                mask = (torch.rand(y.shape[0], device=y.device) < p_uncond)
                y_in = y.clone()
                y_in[mask] = -1  # usa -1 come "no label" (assicurati che il tuo UNet lo gestisca)
            else:
                y_in = y

        eps_pred = model(x_t, t, y_in)  # il tuo UNet deve restituire ε_pred con stessa shape di x_t
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def p_sample_step(self, model: nn.Module, x_t: Tensor, t_scalar: int,
                      y: Tensor | None = None, guidance_scale: float = 0.0) -> Tensor:
        """
        Un singolo step di reverse diffusion.
        Implementa la posterior mean corretta + variance corretta.
        Supporta CFG (richiede che il modello accetti y=None / y=-1 come unconditional).
        """
        b = x_t.size(0)
        t = torch.full((b,), t_scalar, device=x_t.device, dtype=torch.long)

        # predizione ε con (opzionale) CFG
        if guidance_scale and y is not None:
            y_uncond = torch.full_like(y, -1)  # etichetta "vuota"
            eps_uncond = model(x_t, t, y_uncond)
            eps_cond = model(x_t, t, y)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps = model(x_t, t, y if y is not None else None)

        # posterior mean: μ = 1/sqrt(α_t) * (x_t − (1−α_t)/sqrt(1−\bar{α}_t) * ε)
        sqrt_recip_alpha_t = extract_at(self.sqrt_recip_alphas, t, x_t.shape)
        one_minus_alphabar_t = extract_at(1.0 - self.alphas_cumprod, t, x_t.shape)
        beta_t = extract_at(self.betas, t, x_t.shape)
        alpha_t = extract_at(self.alphas, t, x_t.shape)

        mean = sqrt_recip_alpha_t * (x_t - ((1 - alpha_t) / torch.sqrt(one_minus_alphabar_t)) * eps)

        # variance: β̃_t
        var = extract_at(self.posterior_variance, t, x_t.shape)
        if t_scalar == 0:
            noise = torch.zeros_like(x_t)
        else:
            noise = torch.randn_like(x_t)
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, model: nn.Module, n: int, img_size: int,
               y: Tensor | None = None, guidance_scale: float = 0.0) -> Tensor:
        """
        Campiona n immagini da N(0, I) e fa T step di reverse.
        Ritorna tensore in [-1, 1].
        """
        x_t = torch.randn(n, 3, img_size, img_size, device=self.device)
        for t_scalar in reversed(range(self.T)):
            x_t = self.p_sample_step(model, x_t, t_scalar, y=y, guidance_scale=guidance_scale)
        return x_t.clamp(-1, 1)


