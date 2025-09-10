# diffusion.py
# Minimal, robust DDPM utilities:
# - beta schedules (cosine / linear)
# - q(x_t | x_0) forward process
# - training loss with P2 weighting
# - samplers: DDPM (ancestral & deterministic) and DDIM (with skip_first)
# Every line is commented for clarity.

from typing import Optional, Tuple               # typing helpers
import torch                                    # PyTorch core
import torch.nn.functional as F                 # functional ops


def _cosine_alphas_bar(T: int, device: torch.device) -> torch.Tensor:
    """Compute cosine cumulative alphas (Nichol & Dhariwal) on [0..T].
    We sample at step centers for stability."""
    # create T+1 time grid from 0 to 1 (inclusive)
    s = 0.008                                     # small offset from paper
    t = torch.linspace(0, T, T + 1, device=device) / T  # [0..1]
    # cosine schedule for alpha_bar
    alphas_bar = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
    # normalize to start at 1.0 exactly
    alphas_bar = alphas_bar / alphas_bar[0]
    # return the *interval* values between steps: we need T values at half-steps
    # alpha_bar_t is taken at grid[1:], but we keep the cumulative array for gathers.
    return alphas_bar                              # shape [T+1]


def _linear_betas(T: int, device: torch.device) -> torch.Tensor:
    """Classic linear schedule: small betas -> large betas."""
    beta_start = 1e-4                               # lower bound
    beta_end = 2e-2                                 # upper bound
    betas = torch.linspace(beta_start, beta_end, T, device=device)  # [T]
    return betas


class Diffusion:
    """DDPM helper that stores schedules and provides training loss + samplers."""
    def __init__(self, T: int = 200, schedule: str = "cosine", device: torch.device = torch.device("cpu")):
        super().__init__()                          # initialize object
        self.T = int(T)                             # number of diffusion steps
        self.device = device                        # device to keep buffers

        # ---- build schedule ----
        if schedule == "cosine":                    # cosine (preferred)
            ab = _cosine_alphas_bar(self.T, device) # cumulative ᾱ from t=0..T
            # derive betas from ᾱ_t and ᾱ_{t-1}: β_t = 1 - ᾱ_t / ᾱ_{t-1}
            self.alphas_bar = ab[1:].clamp(1e-8, 1-1e-8)            # ᾱ_t for t=1..T  (T values)
            prev = ab[:-1].clamp(1e-8, 1-1e-8)                       # ᾱ_{t-1}
            self.betas = (1.0 - (self.alphas_bar / prev)).clamp(1e-8, 0.999)  # [T]
        elif schedule == "linear":                  # linear β
            self.betas = _linear_betas(self.T, device)               # [T]
            # cumulative ᾱ from β
            alphas = 1.0 - self.betas                               # α_t
            self.alphas_bar = torch.cumprod(alphas, dim=0)           # ᾱ_t
        else:
            raise ValueError(f"Unknown schedule: {schedule}")        # guard invalid

        # ---- precompute a few commonly used buffers ----
        self.alphas = 1.0 - self.betas                               # α_t
        self.sqrt_alphas = torch.sqrt(self.alphas)                   # √α_t
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)       # √(1/α_t)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)           # √ᾱ_t
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alphas_bar)  # √(1-ᾱ_t)

    # -------------------------------
    # Forward process q(x_t | x_0)
    # -------------------------------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Draw x_t ~ q(x_t | x_0) = N( √ᾱ_t x_0, (1-ᾱ_t)I ).
        x0: (B,3,H,W), t: (B,) long in [0..T-1], noise: optional ε."""
        if noise is None:                                            # if no external noise
            noise = torch.randn_like(x0)                             # sample ε ~ N(0,I)
        # gather per-sample scaling factors and match spatial dims via view
        sqrt_ab = self.sqrt_alphas_bar[t].view(-1, 1, 1, 1)          # √ᾱ_t
        sqrt_om = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1, 1)  # √(1-ᾱ_t)
        # combine x0 and noise
        return sqrt_ab * x0 + sqrt_om * noise                        # x_t

    # --------------------------------
    # Training loss with P2 weighting
    # --------------------------------
    def p_losses(
        self,
        model,                                  # ε-predictor UNet
        x0: torch.Tensor,                       # clean image batch
        y: Optional[torch.Tensor] = None,       # class labels or None
        p_uncond: float = 0.0,                  # batch-level CFG dropout prob
        p2_gamma: float = 0.5,                  # P2 exponent γ (0.5 works well)
        p2_k: float = 1.0,                      # P2 shift k (1.0 by default)
    ) -> torch.Tensor:
        """Compute MSE(ε̂, ε) weighted by P2 (Nichol & Dhariwal, 2021)."""
        B = x0.size(0)                                              # batch size
        # sample t ~ Uniform{0..T-1}
        t = torch.randint(0, self.T, (B,), device=x0.device, dtype=torch.long)  # (B,)
        # sample Gaussian noise
        eps = torch.randn_like(x0)                                  # ε ~ N(0,I)
        # produce a noised input x_t
        x_t = self.q_sample(x0, t, eps)                             # x_t

        # classifier-free guidance dropout (whole-batch toggle for safety)
        y_in = None                                                 # default: unconditional
        if (y is not None) and (p_uncond < 1.0):                    # if conditional training
            # with prob (1-p_uncond) we keep labels for the entire batch
            if torch.rand((), device=x0.device) > p_uncond:         # keep labels this step
                y_in = y                                            # else leave as None

        # predict noise with the model
        eps_pred = model(x_t, t, y_in)                              # ε̂(x_t, t, y)

        # per-sample MSE (flatten spatial & channel dims)
        mse_per = (eps - eps_pred).pow(2).flatten(1).mean(dim=1)    # (B,)

        # P2 weighting: w_t = (k + SNR(t))^{-γ}, SNR = ᾱ_t / (1-ᾱ_t)
        snr_t = self.alphas_bar[t] / (1.0 - self.alphas_bar[t])     # (B,)
        w = (p2_k + snr_t).pow(-p2_gamma)                           # (B,)

        # final weighted loss
        loss = (w * mse_per).mean()                                 # scalar
        return loss                                                 # return loss

    # --------------------------
    # DDPM ancestral sampler
    # --------------------------
    @torch.no_grad()
    def sample_ddpm(
        self,
        model,                                      # ε-predictor UNet
        shape: Tuple[int, int, int, int],           # (B,3,H,W)
        y: Optional[torch.Tensor] = None,           # labels or None
        guidance_scale: float = 0.0,                # CFG scale (0 = off)
        deterministic: bool = False,                # sigma=0 if True
    ) -> torch.Tensor:
        """Classic DDPM sampling x_T -> ... -> x_0."""
        B, C, H, W = shape                           # unpack shape
        x = torch.randn(B, C, H, W, device=self.device)  # start from noise
        for i in reversed(range(self.T)):            # iterate t=T-1..0
            t = torch.full((B,), i, device=self.device, dtype=torch.long)  # (B,)
            # ε̂ conditional / unconditional for CFG
            if guidance_scale > 0.0 and y is not None:
                eps_c = model(x, t, y)               # conditional
                eps_u = model(x, t, None)            # unconditional
                eps = eps_u + guidance_scale * (eps_c - eps_u)  # combine
            else:
                eps = model(x, t, y)                 # plain ε̂

            # DDPM update: mean = 1/√α_t * (x_t - (β_t/√(1-ᾱ_t)) ε̂)
            sra = self.sqrt_recip_alphas[i]          # √(1/α_t)
            bt = self.betas[i]                       # β_t
            somab = self.sqrt_one_minus_alphas_bar[i]  # √(1-ᾱ_t)
            mean = sra * (x - (bt / somab) * eps)    # μ_t(x_t, ε̂)

            if i > 0:
                # σ_t = √β_t for ancestral; or 0 for deterministic
                sigma = (0.0 if deterministic else torch.sqrt(bt))
                x = mean + sigma * torch.randn_like(x)     # sample next x_{t-1}
            else:
                x = mean                                    # final step uses mean
        return x                                            # in [-1,1], approximately

    # --------------------------
    # DDIM sampler (deterministic when eta=0)
    # --------------------------
    @torch.no_grad()
    def sample_ddim(
        self,
        model,                                      # ε-predictor
        shape: Tuple[int, int, int, int],           # (B,3,H,W)
        y: Optional[torch.Tensor] = None,           # labels or None
        steps: int = 50,                            # DDIM steps
        eta: float = 0.0,                           # 0 => deterministic
        guidance_scale: float = 0.0,                # CFG scale
        skip_first: int = 0,                        # skip earliest noisy steps
    ) -> torch.Tensor:
        """DDIM sampling with optional truncation of earliest steps."""
        B, C, H, W = shape                           # unpack shape
        x = torch.randn(B, C, H, W, device=self.device)        # start from noise
        # choose indices from (T-1 - skip_first) down to 0 inclusive
        start = max(0, self.T - 1 - int(skip_first))            # clamp
        idxs = torch.linspace(start, 0, steps, dtype=torch.long, device=self.device)

        a_bar = self.alphas_bar                       # ᾱ_t lookup
        for i, t in enumerate(idxs):                  # iterate chosen DDIM steps
            t = int(t.item())                         # python int for gathers
            tb = torch.full((B,), t, device=self.device, dtype=torch.long)  # (B,)

            # ε̂ with guidance if requested
            if guidance_scale > 0.0 and y is not None:
                eps_c = model(x, tb, y)               # conditional
                eps_u = model(x, tb, None)            # unconditional
                eps = eps_u + guidance_scale * (eps_c - eps_u)  # CFG combine
            else:
                eps = model(x, tb, y)                 # plain

            at = a_bar[t]                             # ᾱ_t scalar
            # predict x0 from current x_t and ε̂:   x0 = (x - √(1-ᾱ_t) ε̂)/√ᾱ_t
            x0 = (x - torch.sqrt(1 - at) * eps) / torch.sqrt(at)

            if i == len(idxs) - 1:                    # last step -> return x0
                x = x0
                break

            t_prev = int(idxs[i + 1].item())          # next target timestep
            a_prev = a_bar[t_prev]                     # ᾱ_{t'}
            # DDIM parameters
            sigma = eta * torch.sqrt((1 - a_prev) / (1 - at) * (1 - at / a_prev))
            dir_xt = torch.sqrt(a_prev) * x0           # deterministic direction
            noise = sigma * torch.randn_like(x)        # optional noise if eta>0
            x = dir_xt + torch.sqrt(1 - a_prev - sigma ** 2) * eps + noise
        return x                                       # in [-1,1], approximately

    # --------------
    # Convenience API
    # --------------
    @torch.no_grad()
    def sample(
        self,
        model,                                      # ε-predictor
        shape: Tuple[int, int, int, int],           # (B,3,H,W)
        y: Optional[torch.Tensor] = None,           # labels or None
        guidance_scale: float = 0.0,                # CFG scale
        deterministic: bool = False,                # DDPM sigma=0 if True
    ) -> torch.Tensor:
        """Default to DDPM sampler (kept for backward-compat in training previews)."""
        return self.sample_ddpm(
            model, shape, y=y, guidance_scale=guidance_scale, deterministic=deterministic
        )
