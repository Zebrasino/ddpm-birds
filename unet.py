# unet.py
# A compact, T4-friendly UNet for 64x64 images with optional class conditioning.
# Every block is commented for clarity; the design aims at stability and low VRAM.

import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------- small helpers ---------

def make_gn(num_channels: int) -> nn.GroupNorm:
    """
    Create a GroupNorm layer with a group number that divides num_channels.
    Prefer 32, else 16/8/4/1 to ensure divisibility.
    """
    for g in (32, 16, 8, 4, 2, 1):
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Standard sinusoidal timestep embedding.
    t: (B,) integer or float timesteps
    dim: embedding size (must be even)
    """
    device = t.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device).float() / half)
    # shape (B, half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    # concat sin and cos -> (B, dim)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        # pad if odd
        emb = F.pad(emb, (0, 1))
    return emb


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (scale/shift) predicted from a conditioning vector.
    Used for time and (optional) class conditioning.
    """
    def __init__(self, in_dim: int, out_channels: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 2 * out_channels)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond -> (B, 2*C)
        gamma_beta = self.fc(cond)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        # reshape to (B, C, 1, 1) and apply
        return h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]


class ResBlock(nn.Module):
    """
    A simple residual block with GroupNorm+SiLU+Conv, FiLM for conditioning.
    """
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, cond_dim: Optional[int]):
        super().__init__()
        self.norm1 = make_gn(in_ch)                # GroupNorm with valid groups
        self.act1 = nn.SiLU()                      # Stable activation
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)  # 3x3 conv

        self.norm2 = make_gn(out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)

        # time FiLM is always present
        self.time_film = FiLM(t_dim, out_ch)

        # optional class FiLM (if cond_dim is not None)
        self.class_film = FiLM(cond_dim, out_ch) if cond_dim is not None else None

        # skip projection if channel dims differ
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, y_emb: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))      # norm-act-conv
        h = self.time_film(h, t_emb)                  # FiLM from time
        if self.class_film is not None and y_emb is not None:
            h = self.class_film(h, y_emb)             # FiLM from class (if provided)

        h = self.conv2(self.act2(self.norm2(h)))      # norm-act-conv
        return h + self.skip(x)                       # residual add


class SelfAttention(nn.Module):
    """
    Lightweight 2D self-attention implemented via 1x1 convolutions.
    Head dim stays small to keep memory low.
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by heads"
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.qkv = nn.Conv2d(channels, channels * 3, 1)  # compute Q,K,V
        self.proj = nn.Conv2d(channels, channels, 1)     # output projection
        self.norm = make_gn(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)                           # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)                    # each (B, C, H, W)

        # reshape to (B, heads, HW, dim)
        q = q.view(b, self.num_heads, self.head_dim, h * w).transpose(2, 3)
        k = k.view(b, self.num_heads, self.head_dim, h * w).transpose(2, 3)
        v = v.view(b, self.num_heads, self.head_dim, h * w).transpose(2, 3)

        # scaled dot-product attention (B, heads, HW, HW)
        attn = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        out = attn @ v                                   # (B, heads, HW, dim)

        # back to (B, C, H, W)
        out = out.transpose(2, 3).contiguous().view(b, c, h, w)
        out = self.proj(out)
        return x + out                                   # residual


class UNet(nn.Module):
    """
    Compact UNet:
    - base channels configurable (default 64)
    - ch_mult defines channel multipliers per resolution stage
    - attention at selected spatial sizes (e.g., 16 for 64x64 models)
    - optional class conditioning via an embedding + FiLM in ResBlocks
    """
    def __init__(
        self,
        img_channels: int = 3,
        base: int = 64,
        ch_mult: List[int] = (1, 2, 2, 4),
        attn_resolutions: List[int] = (16,),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
        num_classes: Optional[int] = None,
    ):
        super().__init__()

        # store config for checkpoint
        self.num_classes = num_classes

        # initial conv
        self.in_conv = nn.Conv2d(img_channels, base, 3, 1, 1)

        # time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # optional class embedding to the same dim as time embedding
        self.y_embed = nn.Embedding(num_classes, time_emb_dim) if num_classes is not None else None

        # Down path: keep track of feature maps for U-Net skips
        downs = []
        chans = base
        in_ch = chans
        self.down_channels = []  # store channels for skips
        for i, mult in enumerate(ch_mult):
            out_ch = base * mult
            for _ in range(num_res_blocks):
                downs.append(ResBlock(in_ch, out_ch, t_dim=time_emb_dim,
                                      cond_dim=(time_emb_dim if self.y_embed is not None else None)))
                self.down_channels.append(out_ch)
                in_ch = out_ch
            # add attention at chosen resolution(s)
            # resolution after i-th stage for 64x64 is 64>>(i) before pooling
            stage_res = 64 // (2 ** i)
            if stage_res in attn_resolutions:
                downs.append(SelfAttention(in_ch))
            # downsample except for last stage
            if i != len(ch_mult) - 1:
                downs.append(nn.Conv2d(in_ch, in_ch, 4, 2, 1))  # stride-2 conv for downsample

        self.down = nn.ModuleList(downs)

        # Middle (bottleneck): ResBlock -> Attn -> ResBlock
        mid_ch = in_ch
        self.mid_block1 = ResBlock(mid_ch, mid_ch, time_emb_dim, cond_dim=(time_emb_dim if self.y_embed is not None else None))
        self.mid_attn = SelfAttention(mid_ch)
        self.mid_block2 = ResBlock(mid_ch, mid_ch, time_emb_dim, cond_dim=(time_emb_dim if self.y_embed is not None else None))

        # Up path: mirror the down path, concatenate skip features (channel doubles)
        ups = []
        skip_iter = reversed(self.down_channels)  # iterate skips in reverse
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = base * mult
            for _ in range(num_res_blocks):
                skip_ch = next(skip_iter)        # channels of the skip to concatenate
                ups.append(ResBlock(in_ch + skip_ch, out_ch, time_emb_dim,
                                    cond_dim=(time_emb_dim if self.y_embed is not None else None)))
                in_ch = out_ch
            # attention at corresponding resolution
            stage_res = 64 // (2 ** i)
            if stage_res in attn_resolutions:
                ups.append(SelfAttention(in_ch))
            # upsample except for last stage (top)
            if i != 0:
                ups.append(nn.ConvTranspose2d(in_ch, in_ch, 4, 2, 1))  # stride-2 deconv

        self.up = nn.ModuleList(ups)

        # output head
        self.out_norm = make_gn(in_ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(in_ch, img_channels, 3, 1, 1)

        # store model hyperparams for checkpoint convenience
        self.model_cfg = dict(
            img_channels=img_channels,
            base=base,
            ch_mult=tuple(ch_mult),
            attn_resolutions=tuple(attn_resolutions),
            num_res_blocks=num_res_blocks,
            time_emb_dim=time_emb_dim,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, 3, H, W)
        t: (B,)
        y: (B,) class labels or None
        """
        # compute embeddings
        t_emb = self.time_mlp(sinusoidal_time_embedding(t, self.time_mlp[0].in_features))
        y_emb = self.y_embed(y) if (self.y_embed is not None and y is not None) else None

        # input stem
        h = self.in_conv(x)

        # down path with skip collection
        skips = []
        i = 0
        while i < len(self.down):
            m = self.down[i]
            if isinstance(m, ResBlock):
                h = m(h, t_emb, y_emb)
                skips.append(h)      # store after resblock
                i += 1
            elif isinstance(m, SelfAttention):
                h = m(h)
                i += 1
            else:
                # downsample
                h = m(h)
                i += 1

        # middle
        h = self.mid_block1(h, t_emb, y_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb, y_emb)

        # up path: concatenate with corresponding skip (in reverse order)
        i = 0
        skip_ptr = len(skips) - 1
        while i < len(self.up):
            m = self.up[i]
            if isinstance(m, ResBlock):
                skip_h = skips[skip_ptr]
                skip_ptr -= 1
                h = torch.cat([h, skip_h], dim=1)
                h = m(h, t_emb, y_emb)
                i += 1
            elif isinstance(m, SelfAttention):
                h = m(h)
                i += 1
            else:
                # upsample
                h = m(h)
                i += 1

        # output head
        h = self.out_conv(self.out_act(self.out_norm(h)))
        return h


