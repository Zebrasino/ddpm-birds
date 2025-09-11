from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def _cosine_alphas_bar(T: int, device: torch.device) -> torch.Tensor:
    """Compute Nichol&Dhariwal cosine cumulative alphas (alpha_bar) over 0..T."""
    s = 0.008                                        # small offset for stability
    t = torch.linspace(0, T, T + 1, device=device) / T  # normalized time grid
    ab = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2  # cos^2 schedule
    ab = ab / ab[0].clamp(min=1e-8)                  # normalize so ab[0] == 1
    return ab                                        # shape [T+1] (cumulative)


def _linear_betas(T: int, device: torch.device) -> torch.Tensor:
    """Classic linear beta schedule, small→large."""
    beta_start, beta_end = 1e-4, 2e-2                # bounds from original DDPM
    return torch.linspace(beta_start, beta_end, T, device=device)  # [T]


class Diffusion:
    """Small helper object that stores schedules and provides loss/samplers."""
    def __init__(self, T: int = 200, schedule: str = "cosine", device: torch.device = torch.device("cpu")):
        self.T = int(T)                               # number of diffusion steps
        self.device = device                          # default device for buffers

        # Build schedule: betas, alphas, cumulative alpha_bar, and handy roots
        if schedule == "cosine":
            ab = _cosine_alphas_bar(self.T, device)   # cumulative ᾱ for 0..T
            self.alphas_bar = ab[1:].clamp(1e-8, 1 - 1e-8)   # ᾱ_t for t=1..T
            prev = ab[:-1].clamp(1e-8, 1 - 1e-8)             # ᾱ_{t-1}
            self.betas = (1.0 - (self.alphas_bar / prev)).clamp(1e-8, 0.999)  # β_t
        elif schedule == "linear":
            self.betas = _linear_betas(self.T, device)       # β_t
            self.alphas_bar = torch.cumprod(1.0 - self.betas, dim=0)  # ᾱ_t
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.alphas = 1.0 - self.betas                 # α_t
        self.sqrt_alphas = torch.sqrt(self.alphas)     # √α_t
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)       # √(1/α_t)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)           # √ᾱ_t
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alphas_bar)  # √(1-ᾱ_t)

    # ---------- Forward process q(x_t|x_0) ----------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample x_t = √ᾱ_t x0 + √(1-ᾱ_t) ε  with ε~N(0,I)."""
        if noise is None:                               # lazily sample noise
            noise = torch.randn_like(x0)                # ε
        s_ab = self.sqrt_alphas_bar[t].view(-1, 1, 1, 1)          # per-sample √ᾱ_t
        s_om = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1, 1) # per-sample √(1-ᾱ_t)
        return s_ab * x0 + s_om * noise                 # noisy x_t

    # ---------- Training loss (P2 + optional foreground weighting) ----------
    def p_losses(
        self,
        model,                                          # UNet predicting ε
        x0: torch.Tensor,                               # clean batch
        y: Optional[torch.Tensor] = None,               # labels or None
        p_uncond: float = 0.0,                          # CFG dropout prob
        p2_gamma: float = 0.5,                          # P2 exponent γ
        p2_k: float = 1.0,                              # P2 offset k
        fg_mask: Optional[torch.Tensor] = None,         # (B,1,H,W) mask in [0,1]
        fg_weight: float = 1.0,                         # >1 emphasizes bbox region
    ) -> torch.Tensor:
        B = x0.size(0)                                  # batch size
        t = torch.randint(0, self.T, (B,), device=x0.device, dtype=torch.long)  # t~U{0..T-1}
        eps = torch.randn_like(x0)                      # ε ~ N(0,I)
        xt = self.q_sample(x0, t, eps)                  # produce x_t

        # Classifier-free guidance dropout (batch-wise for simplicity)
        y_in = None                                     # unconditional by default
        if (y is not None) and (torch.rand((), device=x0.device) > p_uncond):
            y_in = y                                    # keep labels this step

        eps_pred = model(xt, t, y_in)                   # predict ε̂(x_t, t, y)

        # Per-pixel squared error
        se = (eps - eps_pred) ** 2                      # (B,3,H,W)
        if fg_mask is not None:                         # if bbox mask is provided
            if fg_mask.dim() == 3:                      # (B,H,W) → (B,1,H,W)
                fg_mask = fg_mask.unsqueeze(1)
            w_pix = 1.0 + (fg_weight - 1.0) * fg_mask   # 1 outside, fg_weight inside
            se = se * w_pix                             # emphasize foreground
        mse_per = se.flatten(1).mean(dim=1)             # per-sample MSE

        # P2 weighting: w_t = (k + SNR(t))^{-γ},  SNR(t)=ᾱ_t/(1-ᾱ_t)
        snr_t = self.alphas_bar[t] / (1.0 - self.alphas_bar[t])     # (B,)
        w = (p2_k + snr_t).pow(-p2_gamma)                           # (B,)

        return (w * mse_per).mean()                    # scalar loss

    # ---------- DDPM (ancestral / deterministic) ----------
    @torch.no_grad()
    def sample_ddpm(
        self,
        model,
        shape: Tuple[int, int, int, int],
        y: Optional[torch.Tensor] = None,
        guidance_scale: float = 0.0,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """DDPM x_T→…→x_0; deterministic=True sets sigma=0."""
        B, C, H, W = shape
        x = torch.randn(B, C, H, W, device=self.device)            # start from noise
        for i in reversed(range(self.T)):
            t = torch.full((B,), i, device=self.device, dtype=torch.long)
            # Classifier-free guidance
            if guidance_scale > 0.0 and y is not None:
                eps_c = model(x, t, y)                             # conditional
                eps_u = model(x, t, None)                          # unconditional
                eps = eps_u + guidance_scale * (eps_c - eps_u)
            else:
                eps = model(x, t, y)
            sra = self.sqrt_recip_alphas[i]                        # √(1/α_t)
            bt = self.betas[i]                                     # β_t
            somab = self.sqrt_one_minus_alphas_bar[i]              # √(1-ᾱ_t)
            mean = sra * (x - (bt / somab) * eps)                  # μ_t(x_t, ε̂)
            if i > 0:
                sigma = 0.0 if deterministic else torch.sqrt(bt)   # σ_t
                x = mean + sigma * torch.randn_like(x)             # x_{t-1}
            else:
                x = mean                                           # x_0
        return x                                                   # ~[-1,1]

    # ---------- DDIM (robust & fast) ----------
    @torch.no_grad()
    def sample_ddim(
        self,
        model,
        shape: Tuple[int, int, int, int],
        y: Optional[torch.Tensor] = None,
        steps: int = 50,
        eta: float = 0.0,
        guidance_scale: float = 0.0,
        skip_first: int = 0,
    ) -> torch.Tensor:
        """DDIM x_T→…→x_0 with optional truncation of very noisy initial steps."""
        B, C, H, W = shape
        x = torch.randn(B, C, H, W, device=self.device)            # x_T
        start = max(0, self.T - 1 - int(skip_first))               # trim earliest steps
        idxs = torch.linspace(start, 0, steps, dtype=torch.long, device=self.device)
        a_bar = self.alphas_bar                                    # ᾱ lookup

        for i, t in enumerate(idxs):
            t = int(t.item())
            tb = torch.full((B,), t, device=self.device, dtype=torch.long)
            # guidance
            if guidance_scale > 0.0 and y is not None:
                eps_c = model(x, tb, y)
                eps_u = model(x, tb, None)
                eps = eps_u + guidance_scale * (eps_c - eps_u)
            else:
                eps = model(x, tb, y)
            at = a_bar[t]
            x0 = (x - torch.sqrt(1 - at) * eps) / torch.sqrt(at)   # predict x0
            if i == len(idxs) - 1:
                x = x0                                             # final step → x0
                break
            t_prev = int(idxs[i + 1].item())
            a_prev = a_bar[t_prev]
            sigma = eta * torch.sqrt((1 - a_prev) / (1 - at) * (1 - at / a_prev))
            dir_xt = torch.sqrt(a_prev) * x0
            noise = sigma * torch.randn_like(x)
            x = dir_xt + torch.sqrt(1 - a_prev - sigma ** 2) * eps + noise
        return x                                                   # ~[-1,1]

    # ---------- Back-compat default ----------
    @torch.no_grad()
    def sample(self, model, shape, y=None, guidance_scale: float = 0.0, deterministic: bool = False):
        """Default sampler = DDPM (kept for training previews)."""
        return self.sample_ddpm(model, shape, y=y, guidance_scale=guidance_scale, deterministic=deterministic)
