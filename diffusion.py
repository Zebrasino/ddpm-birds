import math, torch, numpy as np  # core modules
from typing import Optional, Literal  # typing helpers
import torch.nn.functional as F  # functional ops
from torch import nn  # neural network base

Schedule = Literal["linear", "cosine"]  # schedule enum type


class Diffusion(nn.Module):  # DDPM core
    def __init__(self, T: int = 400, schedule: Schedule = "cosine", device: Optional[torch.device] = None):  # ctor
        super().__init__()  # init parent
        self.T_int = torch.tensor(int(T), dtype=torch.long)  # store steps as tensor
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")  # choose device

        # ----- make beta schedule -----
        if schedule == "linear":  # linear betas
            betas = torch.linspace(1e-4, 0.02, T, device=self.device)  # mild linear schedule
        elif schedule == "cosine":  # cosine alphas_bar (Nichol & Dhariwal)
            s = 0.008  # small offset
            x = torch.linspace(0, T, T + 1, dtype=torch.float64, device=self.device)  # 0..T
            alphas_bar = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2  # cosine curve
            alphas_bar = alphas_bar / alphas_bar[0]  # normalize to 1 at t=0
            betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])  # convert to betas
            betas = betas.float().clamp(1e-4, 0.999)  # clip safety
        else:  # unknown schedule name
            raise ValueError("Unknown schedule")  # error

        alphas = 1.0 - betas  # per-step alpha
        alphas_bar = torch.cumprod(alphas, dim=0)  # cumulative product
        alphas_bar_prev = torch.cat([torch.ones(1, device=self.device), alphas_bar[:-1]], dim=0)  # shifted cumprod

        # register constant buffers used at train/sample time
        self.register_buffer("betas", betas)  # beta_t
        self.register_buffer("alphas", alphas)  # alpha_t
        self.register_buffer("alphas_bar", alphas_bar)  # \bar{alpha}_t
        self.register_buffer("alphas_bar_prev", alphas_bar_prev)  # \bar{alpha}_{t-1}
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))  # sqrt(\bar{alpha}_t)
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1.0 - alphas_bar))  # sqrt(1-\bar{alpha}_t)
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))  # sqrt(1/alpha_t)
        posterior_var = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)  # posterior variance (Eq. 7)
        self.register_buffer("posterior_variance", posterior_var.clamp(min=1e-20))  # numerical stability

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None) -> torch.Tensor:  # forward (noised) sample
        if eps is None:  # if no noise given
            eps = torch.randn_like(x0)  # sample std normal noise
        sqrt_ab = self.sqrt_alphas_bar[t][:, None, None, None]  # broadcast sqrt(\bar{alpha}_t)
        sqrt_mab = self.sqrt_one_minus_alphas_bar[t][:, None, None, None]  # broadcast sqrt(1-\bar{alpha}_t)
        return sqrt_ab * x0 + sqrt_mab * eps  # construct x_t

    def p_losses(self, model: nn.Module, x0: torch.Tensor, y: Optional[torch.Tensor] = None, p_uncond: float = 0.1) -> torch.Tensor:  # training loss
        b = x0.size(0)  # batch size
        t = torch.randint(0, int(self.T_int.item()), (b,), device=x0.device)  # uniform timestep per sample
        noise = torch.randn_like(x0)  # target noise
        x_t = self.q_sample(x0, t, noise)  # noised input
        use_cond = (getattr(model, "num_classes", None) is not None) and (y is not None)  # check conditioning
        if use_cond and (torch.rand((), device=x0.device) < p_uncond):  # batch-level dropout of labels (CFG train)
            y_in = None  # unconditional pass
        else:  # otherwise
            y_in = (y if use_cond else None)  # keep labels or None
        eps_pred = model(x_t, t, y_in)  # predict noise
        return F.mse_loss(eps_pred, noise)  # MSE on noise

    @torch.no_grad()  # no grad for sampling
    def p_sample(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:  # one reverse step
        eps = model(x, t, y)  # predicted noise
        sqrt_recip_alpha = self.sqrt_recip_alphas[t][:, None, None, None]  # factor
        beta_t = self.betas[t][:, None, None, None]  # beta_t
        sqrt_one_minus_ab = self.sqrt_one_minus_alphas_bar[t][:, None, None, None]  # sqrt(1-\bar{alpha}_t)
        mean = sqrt_recip_alpha * (x - beta_t / sqrt_one_minus_ab * eps)  # DDPM mean
        var = self.posterior_variance[t][:, None, None, None]  # DDPM variance
        nonzero_mask = (t != 0).float().view(x.size(0), 1, 1, 1)  # add noise except at t=0
        noise = torch.randn_like(x)  # fresh noise
        return mean + nonzero_mask * torch.sqrt(var) * noise  # x_{t-1}

    @torch.no_grad()  # sampling loop
    def sample(self, model: nn.Module, n: int, img_size: int, y: Optional[torch.Tensor] = None, guidance_scale: float = 0.0) -> torch.Tensor:  # full sampling
        model.eval()  # eval mode
        x = torch.randn(n, 3, img_size, img_size, device=self.device)  # start from noise
        for i in reversed(range(int(self.T_int.item()))):  # loop t=T-1..0
            t = torch.full((n,), i, device=self.device, dtype=torch.long)  # current timestep batch
            if guidance_scale > 0 and (y is not None):  # classifier-free guidance
                eps_uc = model(x, t, None)  # unconditional prediction
                eps_c = model(x, t, y)  # conditional prediction
                eps = eps_uc + guidance_scale * (eps_c - eps_uc)  # guided noise
                sqrt_recip_alpha = self.sqrt_recip_alphas[t][:, None, None, None]  # factor
                beta_t = self.betas[t][:, None, None, None]  # beta_t
                sqrt_one_minus_ab = self.sqrt_one_minus_alphas_bar[t][:, None, None, None]  # sqrt(1-\bar{alpha}_t)
                mean = sqrt_recip_alpha * (x - beta_t / sqrt_one_minus_ab * eps)  # mean
                var = self.posterior_variance[t][:, None, None, None]  # variance
                noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)  # last step noiseless
                x = mean + torch.sqrt(var) * noise  # update x
            else:  # no guidance
                x = self.p_sample(model, x, t, y)  # standard DDPM step
        return x.clamp(-1, 1)  # clamp to valid range


