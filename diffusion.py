import torch  # PyTorch
import torch.nn as nn  # neural nets
import torch.nn.functional as F  # functionals
from typing import Optional  # typing helpers

class Diffusion:
    """Wraps beta schedule, forward training loss, and samplers (DDPM & DDIM)."""

    def __init__(self, T: int = 1000, schedule: str = "cosine", device: torch.device = torch.device("cuda")):
        """
        T: number of diffusion steps.
        schedule: 'cosine' or 'linear' beta schedule.
        device: torch device.
        """
        self.T = T  # total steps
        self.device = device  # target device

        # Build beta schedule
        if schedule == "linear":  # linear beta
            beta_start = 1e-4  # small start
            beta_end = 2e-2    # larger end
            self.betas = torch.linspace(beta_start, beta_end, T, device=device)  # (T,)
        elif schedule == "cosine":  # cosine schedule as in Nichol & Dhariwal
            # Construct alphas_bar via cosine, then derive betas
            s = 0.008  # small offset
            steps = torch.arange(T + 1, device=device, dtype=torch.float32)  # 0..T
            alphas_bar = torch.cos(((steps / T) + s) / (1 + s) * torch.pi * 0.5) ** 2  # cosine curve
            alphas_bar = alphas_bar / alphas_bar[0]  # normalize to 1 at t=0
            self.alphas_bar = alphas_bar[1:]  # (T,) drop t=0
            prev = alphas_bar[:-1].clamp(min=1e-8)  # previous cumulative
            curr = alphas_bar[1:].clamp(min=1e-8)   # current cumulative
            self.betas = (1 - curr / prev).clamp(1e-8, 0.999)  # betas from consecutive alphas_bar
        else:
            raise ValueError("Unknown schedule; choose 'linear' or 'cosine'")  # invalid schedule

        # If we didn't compute alphas_bar (linear case), compute now from betas
        if not hasattr(self, "alphas_bar"):  # if attribute missing
            alphas = 1.0 - self.betas  # per-step alphas
            self.alphas_bar = torch.cumprod(alphas, dim=0)  # cumulative product

        # Precompute useful terms for speed
        self.alphas = 1.0 - self.betas  # (T,)
        self.sqrt_alphas = torch.sqrt(self.alphas)  # sqrt alpha_t
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alphas_bar)  # sqrt(1 - ā_t)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)  # 1/sqrt(alpha_t)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Diffusion forward: sample x_t given x_0 and noise ε ~ N(0,I).
        x0: (B,3,H,W)
        t:  (B,) int timesteps
        noise: optional noise tensor (B,3,H,W)
        returns x_t with same shape as x0.
        """
        if noise is None:  # if no external noise is provided
            noise = torch.randn_like(x0)  # draw Gaussian noise
        # Gather ā_t for each item in the batch and reshape for broadcast
        a_bar = self.alphas_bar[t].view(-1, 1, 1, 1)  # (B,1,1,1)
        # Apply x_t = sqrt(ā_t) x0 + sqrt(1-ā_t) ε
        return torch.sqrt(a_bar) * x0 + self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1, 1) * noise  # noisy sample

    def p_losses(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        p_uncond: float = 0.0,
        mask: Optional[torch.Tensor] = None,
        fg_weight: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute the standard ε-prediction MSE loss, with optional foreground-weighted pixels.
        model: UNet mapping (x_t, t, y) -> ε̂
        x0: clean images in [-1,1], shape (B,3,H,W)
        y: optional integer labels (B,)
        p_uncond: prob. to drop labels (classifier-free guidance training)
        mask: optional foreground mask in [0,1], shape (B,1,H,W)
        fg_weight: extra weight for mask==1 pixels (>=1.0)
        returns: scalar loss
        """
        B = x0.shape[0]  # batch size
        device = x0.device  # device reference

        # Sample random timesteps uniformly
        t = torch.randint(0, self.T, (B,), device=device, dtype=torch.long)  # (B,)

        # Sample noise ε and create x_t
        eps = torch.randn_like(x0)  # true noise
        xt = self.q_sample(x0, t, noise=eps)  # forward diffusion

        # Classifier-free guidance training: randomly drop labels
        if y is not None and p_uncond > 0.0:  # only if label exists and drop prob > 0
            # For some items, replace label with -1 (the UNet maps it to null class internally)
            drop = torch.rand(B, device=device) < p_uncond  # boolean mask of which to drop
            y_in = y.clone()  # copy labels
            y_in[drop] = -1   # use -1 to indicate "null" to the UNet
        else:
            y_in = y  # keep labels as-is (or None)

        # Predict noise using the model
        eps_pred = model(xt, t, y_in)  # ε̂ (B,3,H,W)

        # Per-pixel squared error (no reduction yet)
        mse = (eps - eps_pred) ** 2  # (B,3,H,W)

        # If a mask is provided and fg_weight > 1, weight the loss accordingly
        if mask is not None and fg_weight is not None and fg_weight > 1.0:
            # Make sure mask can broadcast to (B,3,H,W)
            if mask.shape[1] == 1:  # (B,1,H,W)
                mask3 = mask  # keep single-channel mask
            else:
                # If mask already (B,3,H,W), reduce to (B,1,H,W) by max over channel
                mask3 = mask.mean(dim=1, keepdim=True)  # ensure (B,1,H,W)

            # Compute weighted per-pixel loss:
            # L = mean( fg_weight * mask * mse + (1 - mask) * mse )
            w = (1.0 - mask3) + fg_weight * mask3  # weights in [1, fg_weight]
            loss = (w * mse).mean()  # scalar
        else:
            # Standard uniform average MSE
            loss = mse.mean()  # scalar

        return loss  # return scalar loss

    @torch.no_grad()
    def sample(self, model: nn.Module, shape, y=None, guidance_scale: float = 0.0) -> torch.Tensor:
        """
        DDPM ancestral sampling (stochastic).
        model: UNet
        shape: (B,3,H,W)
        y: optional labels (B,)
        guidance_scale: CFG scale (>=0). If >0 and class-conditional, do two model passes per step.
        return: samples in [-1,1]
        """
        B, C, H, W = shape  # unpack shape
        x = torch.randn(B, C, H, W, device=self.device)  # start from pure noise
        model.eval()  # eval mode
        for i in reversed(range(self.T)):  # iterate t = T-1 ... 0
            t = torch.full((B,), i, device=self.device, dtype=torch.long)  # current timestep vector

            if guidance_scale > 0.0 and y is not None:
                # Classifier-free guidance: unconditional pass
                eps_u = model(x, t, torch.full_like(y, -1))  # ε̂_uncond with null class
                # Conditional pass
                eps_c = model(x, t, y)  # ε̂_cond
                # Combine
                eps = eps_u + guidance_scale * (eps_c - eps_u)  # ε̂_cfg
            else:
                eps = model(x, t, y)  # single pass

            # DDPM update
            sra = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1)  # 1/sqrt(alpha_t)
            bt = self.betas[t].view(-1, 1, 1, 1)  # beta_t
            somab = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1, 1)  # sqrt(1-ā_t)
            mean = sra * (x - (bt / somab) * eps)  # posterior mean
            if i > 0:
                noise = torch.randn_like(x)  # sample new noise
                x = mean + torch.sqrt(bt) * noise  # add noise term
            else:
                x = mean  # final step: no noise
        return x  # return samples in [-1,1]

    @torch.no_grad()
    def sample_ddim(self, model: nn.Module, shape, steps: int = 50, eta: float = 0.0, y=None, guidance_scale: float = 0.0):
        """
        DDIM sampling (deterministic if eta=0).
        steps: number of DDIM steps (<= T).
        eta: stochasticity (0 = deterministic).
        y: optional labels for conditional sampling.
        guidance_scale: classifier-free guidance scale.
        return: samples in [-1,1]
        """
        B, C, H, W = shape  # unpack
        x = torch.randn(B, C, H, W, device=self.device)  # init noise
        model.eval()  # eval mode

        # Choose a subset of timesteps in descending order
        idxs = torch.linspace(self.T - 1, 0, steps, dtype=torch.long, device=self.device)  # (steps,)

        a_bar = self.alphas_bar  # shortcut

        for i, t in enumerate(idxs):  # iterate over selected timesteps
            tb = t.repeat(B)  # (B,) vector filled with current t

            if guidance_scale > 0.0 and y is not None:
                # Unconditional
                eps_u = model(x, tb, torch.full_like(y, -1))  # ε̂_uncond
                # Conditional
                eps_c = model(x, tb, y)  # ε̂_cond
                # Combine
                eps = eps_u + guidance_scale * (eps_c - eps_u)  # ε̂_cfg
            else:
                eps = model(x, tb, y)  # ε̂ without CFG

            at = a_bar[t]  # ā_t scalar
            x0 = (x - torch.sqrt(1 - at) * eps) / torch.sqrt(at)  # predict x0

            if i == steps - 1:
                x = x0  # at last step we output x0
                break

            # Compute ā_{t-1} for the next index
            t_prev = idxs[i + 1]  # next timestep in the sequence (lower)
            a_prev = a_bar[t_prev]  # ā_{t-1}

            # DDIM update
            sigma = eta * torch.sqrt((1 - a_prev) / (1 - at) * (1 - at / a_prev))  # noise scale
            dir_xt = torch.sqrt(a_prev) * x0  # deterministic part
            noise = sigma * torch.randn_like(x)  # stochastic part
            x = dir_xt + torch.sqrt(1 - a_prev - sigma**2) * eps + noise  # new x

        return x  # samples in [-1,1]

