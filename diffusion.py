import math, torch, numpy as np  # import modules
from typing import Optional, Literal  # import names from module
import torch.nn.functional as F  # import names from module
from torch import nn  # import names from module

Schedule = Literal["linear", "cosine"]  # variable assignment

class Diffusion(nn.Module):  # define class Diffusion
    def __init__(self, T: int = 400, schedule: Schedule = "cosine", device: Optional[torch.device] = None):  # define function __init__
        super().__init__()  # call parent constructor  # statement
        self.T_int = torch.tensor(T, dtype=torch.long)  # variable assignment
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # variable assignment

        # set up beta schedule  # comment  # statement
        if schedule == "linear":  # control flow
            betas = torch.linspace(1e-4, 0.02, T, device=self.device)  # PyTorch operation
        elif schedule == "cosine":  # control flow
            s = 0.008  # variable assignment
            steps = T + 1  # variable assignment
            x = torch.linspace(0, T, steps, dtype=torch.float64, device=self.device)  # PyTorch operation
            alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2  # PyTorch operation
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # PyTorch operation
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])  # PyTorch operation
            betas = betas.float().clamp(0.0001, 0.999)  # PyTorch operation
        else:  # control flow
            raise ValueError("Unknown schedule")  # raise exception

        alphas = 1.0 - betas  # PyTorch operation
        alphas_bar = torch.cumprod(alphas, dim=0)  # PyTorch operation
        alphas_bar_prev = torch.cat([torch.ones(1, device=self.device), alphas_bar[:-1]], dim=0)  # PyTorch operation

        self.register_buffer('betas', betas)  # PyTorch operation
        self.register_buffer('alphas', alphas)  # PyTorch operation
        self.register_buffer('alphas_bar', alphas_bar)  # PyTorch operation
        self.register_buffer('alphas_bar_prev', alphas_bar_prev)  # PyTorch operation
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))  # PyTorch operation
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1.0 - alphas_bar))  # PyTorch operation
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))  # PyTorch operation
        posterior_var = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)  # PyTorch operation
        self.register_buffer('posterior_variance', posterior_var.clamp(min=1e-20))  # PyTorch operation

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None) -> torch.Tensor:  # define function q_sample
        """Sample x_t ~ q(x_t | x_0)."""  # docstring
        if eps is None:  # control flow
            eps = torch.randn_like(x0)  # PyTorch operation
        sqrt_ab = self.sqrt_alphas_bar[t][:, None, None, None]  # PyTorch operation
        sqrt_mab = self.sqrt_one_minus_alphas_bar[t][:, None, None, None]  # PyTorch operation
        return sqrt_ab * x0 + sqrt_mab * eps  # return value

    def p_losses(  # define function p_losses
        self,
        model: nn.Module,
        x0: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        p_uncond: float = 0.1,
    ) -> torch.Tensor:
        """
        DDPM training loss: MSE between true noise and predicted noise at random t.
        Uses batch-level dropout of labels for classifier-free guidance.
        """  # docstring
        b = x0.size(0)  # PyTorch operation
        t = torch.randint(0, int(self.T_int.item()), (b,), device=x0.device)  # PyTorch operation
        noise = torch.randn_like(x0)  # PyTorch operation
        x_t = self.q_sample(x0, t, noise)  # PyTorch operation

        use_cond = (getattr(model, 'num_classes', None) is not None) and (y is not None)  # variable assignment
        if use_cond and (torch.rand((), device=x0.device) < p_uncond):  # control flow
            y_in = None  # variable assignment
        else:  # control flow
            y_in = (y if use_cond else None)  # variable assignment

        eps_pred = model(x_t, t, y_in)  # PyTorch operation
        return F.mse_loss(eps_pred, noise)  # return value

    @torch.no_grad()  # decorator
    def p_sample(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:  # define function p_sample
        eps = model(x, t, y)  # PyTorch operation
        sqrt_recip_alpha = self.sqrt_recip_alphas[t][:, None, None, None]  # PyTorch operation
        beta_t = self.betas[t][:, None, None, None]  # PyTorch operation
        sqrt_one_minus_ab = self.sqrt_one_minus_alphas_bar[t][:, None, None, None]  # PyTorch operation
        mean = sqrt_recip_alpha * (x - beta_t / sqrt_one_minus_ab * eps)  # PyTorch operation
        var = self.posterior_variance[t][:, None, None, None]  # PyTorch operation
        nonzero_mask = (t != 0).float().view(x.size(0), 1, 1, 1)  # PyTorch operation
        noise = torch.randn_like(x)  # PyTorch operation
        return mean + nonzero_mask * torch.sqrt(var) * noise  # return value

    @torch.no_grad()  # decorator
    def sample(self, model: nn.Module, n: int, img_size: int, y: Optional[torch.Tensor] = None, guidance_scale: float = 0.0) -> torch.Tensor:  # define function sample
        model.eval()  # PyTorch operation
        x = torch.randn(n, 3, img_size, img_size, device=self.device)  # PyTorch operation
        for i in reversed(range(int(self.T_int.item()))):  # loop
            t = torch.full((n,), i, device=self.device, dtype=torch.long)  # PyTorch operation
            if guidance_scale > 0 and (y is not None):  # control flow
                eps_uc = model(x, t, None)  # PyTorch operation
                eps_c = model(x, t, y)  # PyTorch operation
                eps = eps_uc + guidance_scale * (eps_c - eps_uc)  # PyTorch operation
                sqrt_recip_alpha = self.sqrt_recip_alphas[t][:, None, None, None]  # PyTorch operation
                beta_t = self.betas[t][:, None, None, None]  # PyTorch operation
                sqrt_one_minus_ab = self.sqrt_one_minus_alphas_bar[t][:, None, None, None]  # PyTorch operation
                mean = sqrt_recip_alpha * (x - beta_t / sqrt_one_minus_ab * eps)  # PyTorch operation
                var = self.posterior_variance[t][:, None, None, None]  # PyTorch operation
                noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)  # PyTorch operation
                x = mean + torch.sqrt(var) * noise  # PyTorch operation
            else:  # control flow
                x = self.p_sample(model, x, t, y)  # PyTorch operation
        return x.clamp(-1, 1)  # return value

