# unet.py
# Line-by-line commented U-Net backbone with timestep and optional class conditioning.
from __future__ import annotations  # Future annotations

from typing import Optional  # For optional label type hints

import torch  # Core tensor library
import torch.nn as nn  # Neural network modules
import torch.nn.functional as F  # Functional API for activations etc.

def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embeddings as in DDPM.
    t: (B,) timesteps; dim: embedding dimension; returns (B, dim).
    """
    half = dim // 2  # Half sine, half cosine
    emb = torch.log(torch.tensor(10000.0, device=t.device)) / (half - 1)  # Frequency bases
    emb = torch.exp(torch.arange(half, device=t.device) * -emb)  # Exponents
    emb = t.float().unsqueeze(1) * emb.unsqueeze(0)  # Outer product
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # Sin/Cos concat
    if dim % 2 == 1:  # Pad if odd
        emb = F.pad(emb, (0,1))
    return emb  # Shape (B, dim)

class FiLM(nn.Module):
    """Feature-wise Linear Modulation to inject conditioning."""
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()  # Init base class
        self.fc = nn.Linear(cond_dim, 2 * dim)  # Map cond to scale/shift
    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.fc(cond)  # Compute FiLM parameters
        gamma, beta = gamma_beta.chunk(2, dim=1)  # Split into scale/shift
        return h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]  # Apply per-channel

class ResBlock(nn.Module):
    """Residual block with GroupNorm, SiLU, optional FiLM conditioning."""
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, cond_dim: Optional[int] = None):
        super().__init__()  # Init nn.Module
        self.norm1 = nn.GroupNorm(32, in_ch)  # First normalization
        self.act = nn.SiLU()  # Activation
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)  # First 3x3 conv
        self.norm2 = nn.GroupNorm(32, out_ch)  # Second normalization
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)  # Second 3x3 conv
        self.time_emb = nn.Linear(t_dim, out_ch)  # Time embedding projection
        self.film = FiLM(out_ch, cond_dim) if cond_dim is not None else None  # Optional FiLM
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()  # Skip path

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, y_emb: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))  # Norm→Act→Conv
        h = h + self.time_emb(t_emb)[:, :, None, None]  # Add time embedding
        if self.film is not None and y_emb is not None:  # Apply FiLM if available
            h = self.film(h, y_emb)
        h = self.conv2(self.act(self.norm2(h)))  # Norm→Act→Conv
        return h + self.skip(x)  # Residual add

class SelfAttention(nn.Module):
    """Simple 2D self-attention block."""
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()  # Init
        self.num_heads = num_heads  # Heads count
        self.scale = (dim // num_heads) ** -0.5  # Scaling for dot product
        self.qkv = nn.Conv2d(dim, dim * 3, 1)  # 1x1 conv to produce QKV
        self.proj = nn.Conv2d(dim, dim, 1)  # Output projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape  # Unpack shape
        qkv = self.qkv(x)  # Compute Q,K,V
        q, k, v = qkv.chunk(3, dim=1)  # Split along channels
        q = q.reshape(B, self.num_heads, C // self.num_heads, H * W)  # Reshape
        k = k.reshape(B, self.num_heads, C // self.num_heads, H * W)
        v = v.reshape(B, self.num_heads, C // self.num_heads, H * W)
        attn = torch.softmax((q.transpose(2,3) @ k) * self.scale, dim=-1)  # Attention
        out = (attn @ v.transpose(2,3)).transpose(2,3).reshape(B, C, H, W)  # Aggregate
        return self.proj(out) + x  # Residual

class UNet(nn.Module):
    """U-Net with timestep and optional class conditioning (via embeddings and FiLM)."""
    def __init__(self, img_channels: int = 3, base: int = 128, ch_mults=(1,2,2,4),
                 attn_res=(16,), num_classes: int = 200, cond_mode: str = "class", t_dim: int = 256):
        super().__init__()  # Init nn.Module
        self.cond_mode = cond_mode  # 'none' or 'class'
        self.t_dim = t_dim  # Time embedding dimension
        self.time_mlp = nn.Sequential(  # Timestep MLP
            nn.Linear(t_dim, t_dim * 4),
            nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim),
        )
        self.y_embed = nn.Embedding(num_classes, t_dim) if cond_mode == "class" else None  # Class embedding
        self.in_conv = nn.Conv2d(img_channels, base, 3, padding=1)  # Input projection

        downs = []  # Down path modules
        chans = base  # Current channels
        self.attn_res = set(attn_res)  # Apply attention at these sizes
        self.feature_maps = []  # Track channels for skip connections

        for i, mult in enumerate(ch_mults):
            out_ch = base * mult  # Output channels at this stage
            downs.append(ResBlock(chans, out_ch, t_dim, cond_dim=t_dim if self.y_embed is not None else None))  # ResBlock
            downs.append(SelfAttention(out_ch) if 2**(len(ch_mults)-i) in self.attn_res else nn.Identity())  # Optional attn
            downs.append(nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1))  # Downsample
            self.feature_maps.append(out_ch)  # Record channels
            chans = out_ch  # Update channels

        self.down = nn.ModuleList(downs)  # Register down path

        self.mid1 = ResBlock(chans, chans, t_dim, cond_dim=t_dim if self.y_embed is not None else None)  # Mid block 1
        self.mid_attn = SelfAttention(chans)  # Mid attention
        self.mid2 = ResBlock(chans, chans, t_dim, cond_dim=t_dim if self.y_embed is not None else None)  # Mid block 2

        ups = []  # Up path modules
        for out_ch in reversed(self.feature_maps):
            ups.append(ResBlock(chans + out_ch, out_ch, t_dim, cond_dim=t_dim if self.y_embed is not None else None))  # ResBlock
            ups.append(SelfAttention(out_ch) if out_ch in self.feature_maps[-1:] else nn.Identity())  # Optional attn
            ups.append(nn.Upsample(scale_factor=2, mode='nearest'))  # Upsample
            ups.append(nn.Conv2d(out_ch, out_ch, 3, padding=1))  # Smooth conv
            chans = out_ch  # Update channels

        self.up = nn.ModuleList(ups)  # Register up path
        self.out_norm = nn.GroupNorm(32, chans)  # Final norm
        self.out_act = nn.SiLU()  # Final activation
        self.out_conv = nn.Conv2d(chans, img_channels, 3, padding=1)  # Predict noise ε

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        t_emb = self.time_mlp(timestep_embedding(t, self.t_dim))  # Build time embedding
        y_emb = self.y_embed(y) if (self.y_embed is not None and y is not None) else None  # Optional class embedding

        h = self.in_conv(x)  # Input projection
        skips = []  # Collect skip connections

        size = x.shape[-1]  # Track size
        it = iter(self.down)  # Iterate down path (triplets)
        while True:
            try:
                rb = next(it)  # ResBlock
                h = rb(h, t_emb, y_emb)
                skips.append(h)  # Save for skip
                attn = next(it)  # Attention or identity
                h = attn(h)
                ds = next(it)  # Downsample
                h = ds(h)
                size //= 2  # Update size
            except StopIteration:
                break  # End down path

        h = self.mid1(h, t_emb, y_emb)  # Mid block 1
        h = self.mid_attn(h)  # Mid attention
        h = self.mid2(h, t_emb, y_emb)  # Mid block 2

        it = iter(self.up)  # Iterate up path
        for skip in reversed(skips):  # Mirror skips
            rb = next(it)  # ResBlock
            h = torch.cat([h, skip], dim=1)  # Concatenate skip
            h = rb(h, t_emb, y_emb)  # Apply block
            attn = next(it)  # Attention or identity
            h = attn(h)
            upsample = next(it)  # Upsample
            h = upsample(h)
            conv = next(it)  # Smooth conv
            h = conv(h)

        h = self.out_conv(self.out_act(self.out_norm(h)))  # Final projection to noise ε
        return h  # Shape (N, C, H, W)

