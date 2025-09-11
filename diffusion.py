from __future__ import annotations                # future annotations
import math                                       # math ops
import torch                                      # tensors
import torch.nn.functional as F                   # mse_loss

# --------------------------------
# Beta schedules for DDPM
# --------------------------------
def make_beta_schedule(T: int, schedule: str = "cosine") -> torch.Tensor:
    """Return beta_t (length T) in (0,1)."""
    if schedule == "linear":                                      # linear beta
        beta_start, beta_end = 1e-4, 0.02                         # standard linear
        betas = torch.linspace(beta_start, beta_end, T)           # linear ramp
    elif schedule == "cosine":                                    # cosine alpha_bar
        s = 0.008                                                 # small offset
        steps = torch.arange(T + 1, dtype=torch.float32)          # t=0..T
        alphas_bar = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2  # cosine
        alphas_bar = alphas_bar / alphas_bar[0]                   # normalize
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])            # derive betas
        betas = betas.clamp(1e-5, 0.999)                          # clamp for stability
    else:
        raise ValueError(f"Unknown schedule: {schedule}")         # invalid option
    return betas                                                  # (T,)

# --------------------------------
# Diffusion core container
# --------------------------------
class Diffusion:
    """Holds precomputed diffusion buffers and offers loss/sampling APIs."""
    def __init__(self, T: int = 1000, schedule: str = "cosine", device: torch.device | str = "cpu"):
        self.T = T                                                # total timesteps
        self.device = torch.device(device)                        # device
        # Precompute schedule buffers
        self.betas = make_beta_schedule(T, schedule).to(self.device)     # (T,)
        self.alphas = 1.0 - self.betas                                   # (T,)
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)              # (T,)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)               # (T,)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alphas_bar)  # (T,)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)           # (T,)

    # -----------------------------
    # Forward noising q(x_t|x_0)
    # -----------------------------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Sample x_t from q(x_t | x_0)."""
        if noise is None:                                           # default noise
            noise = torch.randn_like(x0)                            # Gaussian ε
        # Gather scalar coeffs per batch element and reshape as (B,1,1,1)
        s_ab = self.sqrt_alphas_bar[t].view(-1, 1, 1, 1)            # sqrt(alpha_bar_t)
        s_om = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1, 1)  # sqrt(1 - alpha_bar_t)
        return s_ab * x0 + s_om * noise                             # broadcasted mix

    # -----------------------------
    # Training loss (ε-MSE)
    # -----------------------------
    def p_losses(
        self,
        model,
        x0: torch.Tensor,                  # clean images in [-1,1]
        y: torch.Tensor | None,            # class labels or None
        fg_mask: torch.Tensor | None = None,  # foreground mask (1,H,W) or None
        fg_weight: float = 1.0,            # weight multiplier on foreground pixels
    ) -> torch.Tensor:
        """Compute MSE(ε_pred, ε) with optional foreground reweighting."""
        B = x0.size(0)                                            # batch size
        t = torch.randint(0, self.T, (B,), device=x0.device)      # random timesteps
        eps = torch.randn_like(x0)                                # target noise
        xt = self.q_sample(x0, t, eps)                            # noisy input

        # Predict noise with the model
        eps_pred = model(xt, t, y)                                # ε̂(x_t, t, y)

        # If we have a foreground mask, upweight its pixels
        if fg_mask is not None and fg_weight != 1.0:              # apply weighting
            w = torch.ones_like(x0)                               # base weight 1
            w = w * (1.0 + (fg_weight - 1.0) * fg_mask)          # 1 outside, fg_weight inside
            loss = (w * (eps_pred - eps) ** 2).mean()             # weighted MSE
        else:
            loss = F.mse_loss(eps_pred, eps)                      # plain MSE

        return loss                                               # scalar

    # -----------------------------
    # DDPM ancestral sampler
    # -----------------------------
    @torch.no_grad()
    def sample_ddpm(
        self,
        model,
        shape: tuple,                         # (B,3,H,W)
        y: torch.Tensor | None = None,        # labels or None
        guidance_scale: float = 0.0,          # CFG scale (0 disables)
    ) -> torch.Tensor:
        """Standard DDPM sampler, optionally with classifier-free guidance."""
        B, C, H, W = shape                                    # unpack shape
        x = torch.randn(B, C, H, W, device=self.device)       # start from noise

        # Loop T-1 ... 0
        for i in reversed(range(self.T)):                     # time descending
            t = torch.full((B,), i, device=self.device, dtype=torch.long)  # timestep batch

            if guidance_scale > 0 and y is not None:          # CFG branch
                # Conditional pass
                eps_c = model(x, t, y)                        # ε̂ cond
                # Unconditional pass (y = -1 -> mapped to NULL in UNet)
                eps_u = model(x, t, torch.full_like(y, -1))   # ε̂ uncond
                # Linear blend
                eps = eps_u + guidance_scale * (eps_c - eps_u)# ε̂ guided
            else:
                eps = model(x, t, y)                          # ε̂ simple

            # Compute posterior mean (DDPM update)
            sra  = self.sqrt_recip_alphas[i]                  # sqrt(1/alpha_t)
            beta = self.betas[i]                              # beta_t
            somab= self.sqrt_one_minus_alphas_bar[i]          # sqrt(1 - alpha_bar_t)

            mean = sra * (x - (beta / somab) * eps)           # posterior mean

            if i > 0:                                         # add noise except last
                z = torch.randn_like(x)                       # fresh noise
                sigma = torch.sqrt(beta)                      # variance term
                x = mean + sigma * z                          # sample
            else:
                x = mean                                      # final x_0

        return x.clamp(-1, 1)                                 # clamp to valid range

    # -----------------------------
    # DDIM sampler (eta=0 -> deterministic)
    # -----------------------------
    @torch.no_grad()
    def sample_ddim(
        self,
        model,
        shape: tuple,                         # (B,3,H,W)
        steps: int = 50,                      # number of DDIM steps (<= T)
        eta: float = 0.0,                     # stochasticity; 0 -> deterministic
        y: torch.Tensor | None = None,        # labels or None
        guidance_scale: float = 0.0,          # CFG scale
        skip_first: int = 0,                  # optionally skip early steps
    ) -> torch.Tensor:
        """DDIM sampler with optional CFG."""
        B, C, H, W = shape                                    # unpack
        x = torch.randn(B, C, H, W, device=self.device)       # start from noise
        # Pick a schedule of indices T-1..0 with given stride
        idxs = torch.linspace(self.T - 1 - skip_first, 0, steps, dtype=torch.long, device=self.device)

        for s, t_idx in enumerate(idxs):                      # iterate selected times
            t = t_idx.repeat(B)                               # (B,) current timestep
            at = self.alphas_bar[t_idx]                       # alpha_bar_t

            # Predict noise (with CFG if requested)
            if guidance_scale > 0 and y is not None:
                eps_c = model(x, t, y)                        # cond
                eps_u = model(x, t, torch.full_like(y, -1))   # uncond
                eps = eps_u + guidance_scale * (eps_c - eps_u)# guided
            else:
                eps = model(x, t, y)                          # simple

            # Predict x0 from current x and eps
            x0 = (x - torch.sqrt(1 - at) * eps) / torch.sqrt(at)   # x̂0

            if s == len(idxs) - 1:                            # last step -> output x0
                x = x0
                break

            # Compute next alpha_bar (t_prev)
            t_prev = idxs[s + 1]                              # next time
            a_prev = self.alphas_bar[t_prev]                  # alpha_bar_{t-1}

            # DDIM update
            sigma = eta * torch.sqrt((1 - a_prev) / (1 - at) * (1 - at / a_prev))  # variance
            dir_xt = torch.sqrt(a_prev) * x0                  # deterministic part
            noise = sigma * torch.randn_like(x)               # stochastic part
            x = dir_xt + torch.sqrt(1 - a_prev - sigma**2) * eps + noise  # update

        return x.clamp(-1, 1)                                 # clamp
