# unet.py
# Line-by-line commented U-Net backbone with timestep and optional class conditioning.
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embeddings as in DDPM.
    t: (B,) timesteps; dim: embedding dimension; returns (B, dim).
    """
    half = dim // 2
    emb = torch.log(torch.tensor(10000.0, device=t.device)) / (half - 1)
    emb = torch.exp(torch.arange(half, device=t.device) * -emb)
    emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

class FiLM(nn.Module):
    """Feature-wise Linear Modulation to inject conditioning."""
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.fc = nn.Linear(cond_dim, 2 * dim)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.fc(cond)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        return h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

class ResBlock(nn.Module):
    """Residual block with GroupNorm, SiLU, optional FiLM conditioning."""
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, cond_dim: Optional[int] = None):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_emb = nn.Linear(t_dim, out_ch)
        self.film = FiLM(out_ch, cond_dim) if cond_dim is not None else None
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, y_emb: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_emb(t_emb)[:, :, None, None]
        if self.film is not None and y_emb is not None:
            h = self.film(h, y_emb)
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)

class SelfAttention(nn.Module):
    """Simple 2D self-attention block."""
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(B, self.num_heads, C // self.num_heads, H * W)
        k = k.reshape(B, self.num_heads, C // self.num_heads, H * W)
        v = v.reshape(B, self.num_heads, C // self.num_heads, H * W)
        attn = torch.softmax((q.transpose(2, 3) @ k) * self.scale, dim=-1)
        out = (attn @ v.transpose(2, 3)).transpose(2, 3).reshape(B, C, H, W)
        return self.proj(out) + x

class UNet(nn.Module):
    """U-Net with timestep and optional class conditioning (via embeddings and FiLM)."""
    def __init__(self, img_channels: int = 3, base: int = 128, ch_mults=(1, 2, 2, 4),
                 attn_res=(16,), num_classes: int = 200, cond_mode: str = "class", t_dim: int = 256):
        super().__init__()
        self.cond_mode = cond_mode
        self.t_dim = t_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim * 4),
            nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim),
        )
        self.y_embed = nn.Embedding(num_classes, t_dim) if cond_mode == "class" else None
        self.in_conv = nn.Conv2d(img_channels, base, 3, padding=1)

        # Down path
        downs = []
        chans = base
        self.attn_res = set(attn_res)
        self.feature_maps = []
        for i, mult in enumerate(ch_mults):
            out_ch = base * mult
            downs.append(ResBlock(chans, out_ch, t_dim, cond_dim=t_dim if self.y_embed is not None else None))
            downs.append(SelfAttention(out_ch) if 2 ** (len(ch_mults) - i) in self.attn_res else nn.Identity())
            downs.append(nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1))  # Downsample
            self.feature_maps.append(out_ch)
            chans = out_ch
        self.down = nn.ModuleList(downs)

        # Middle
        self.mid1 = ResBlock(chans, chans, t_dim, cond_dim=t_dim if self.y_embed is not None else None)
        self.mid_attn = SelfAttention(chans)
        self.mid2 = ResBlock(chans, chans, t_dim, cond_dim=t_dim if self.y_embed is not None else None)

        # Up path ( order: Upsample-Conv-ResBlock(after concat)-Attention)
        ups = []
        for out_ch in reversed(self.feature_maps):
            ups.append(nn.Upsample(scale_factor=2, mode='nearest'))                        # 1) upsample h
            ups.append(nn.Conv2d(chans, chans, 3, padding=1))                              # 2) smooth
            ups.append(ResBlock(chans + out_ch, out_ch, t_dim,                             # 3) concat with skip THEN block
                                 cond_dim=t_dim if self.y_embed is not None else None))
            ups.append(SelfAttention(out_ch) if out_ch in self.feature_maps[-1:] else nn.Identity())
            chans = out_ch
        self.up = nn.ModuleList(ups)

        self.out_norm = nn.GroupNorm(32, chans)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(chans, img_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        t_emb = self.time_mlp(timestep_embedding(t, self.t_dim))
        y_emb = self.y_embed(y) if (self.y_embed is not None and y is not None) else None

        h = self.in_conv(x)
        skips = []

        # Down
        it = iter(self.down)
        while True:
            try:
                rb = next(it);       h = rb(h, t_emb, y_emb);     skips.append(h)
                attn = next(it);     h = attn(h)
                ds = next(it);       h = ds(h)
            except StopIteration:
                break

        # Mid
        h = self.mid1(h, t_emb, y_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb, y_emb)

        # Up (upsample before concatenation with the matching skip)
        it = iter(self.up)
        for skip in reversed(skips):
            upsample = next(it); h = upsample(h)
            conv = next(it);     h = conv(h)
            rb = next(it);       h = torch.cat([h, skip], dim=1); h = rb(h, t_emb, y_emb)
            attn = next(it);     h = attn(h)

        h = self.out_conv(self.out_act(self.out_norm(h)))
        return h

