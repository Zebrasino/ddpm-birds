impoimport math                             # math helpers
from typing import Optional             # typing
import torch                            # torch core
from torch import nn                    # neural modules
import torch.nn.functional as F         # functional ops

# ---- sinusoidal time embedding -------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    """Standard sinusoidal embedding for discrete timesteps t."""
    def __init__(self, dim: int):
        super().__init__()              # base init
        self.dim = dim                  # store dimension
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2            # half dimension
        device = t.device               # device of t
        freq = math.log(10000) / (half - 1)          # frequency scale
        freq = torch.exp(torch.arange(half, device=device) * -freq)  # [half]
        ang = t.float()[:, None] * freq[None, :]      # (B, half)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)   # (B, dim)


# ---- residual block with time + optional class conditioning -------------------
class ResBlock(nn.Module):
    """Two 3×3 convs, GroupNorm, SiLU, residual skip, plus t/y conditioning."""
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, y_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()                                      # init
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)     # 3x3 conv (in→out)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)    # 3x3 conv (out→out)
        self.emb  = nn.Linear(t_dim, out_ch)                    # project time embedding to out_ch
        # pick a GroupNorm group count that divides out_ch (≤32)
        g = min(32, out_ch)                                     # initial groups
        while out_ch % g != 0:                                  # ensure divisibility
            g -= 1
        self.norm1 = nn.GroupNorm(g, out_ch)                    # GN after conv1
        self.norm2 = nn.GroupNorm(g, out_ch)                    # GN after conv2
        self.dropout = nn.Dropout(dropout)                      # dropout (regularization)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()  # channel-matching skip
        self.y_embed = nn.Embedding(y_dim, out_ch) if y_dim is not None else None      # class embedding (optional)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(self.conv1(x))                           # conv1 → norm
        h = F.silu(h)                                           # activation
        t_emb = self.emb(t)                                     # (B, out_ch)
        h = h + t_emb[:, :, None, None]                         # add time conditioning (broadcast over H,W)
        # safe class conditioning (supports y=None for CFG)
        y_emb = None                                            # init
        if (self.y_embed is not None) and (y is not None):      # if conditional model and labels provided
            y_long = y.long()                                   # int64
            mask = (y_long >= 0)                                # dropped labels (CFG) are negative
            if mask.all():                                      # all valid ids
                y_emb = self.y_embed(y_long)                    # (B, out_ch)
            else:                                               # some invalid → zero them
                y_clamped = y_long.clamp_min(0)
                y_all = self.y_embed(y_clamped)
                y_emb = torch.where(mask[:, None], y_all, torch.zeros_like(y_all))
        if y_emb is not None:                                   # add class conditioning
            h = h + y_emb[:, :, None, None]
        h = self.norm2(self.conv2(h))                           # conv2 → norm
        h = self.dropout(F.silu(h))                             # activation + dropout
        return h + self.skip(x)                                 # residual sum


# ---- lightweight self-attention at 16×16 --------------------------------------
class AttentionBlock(nn.Module):
    """Single-head self-attention on spatial tokens (C × H × W)."""
    def __init__(self, ch: int):
        super().__init__()                     # init
        self.norm = nn.GroupNorm(32, ch)       # normalize
        self.q = nn.Conv2d(ch, ch, 1)          # 1x1 query
        self.k = nn.Conv2d(ch, ch, 1)          # 1x1 key
        self.v = nn.Conv2d(ch, ch, 1)          # 1x1 value
        self.proj = nn.Conv2d(ch, ch, 1)       # output projection
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape                   # shapes
        h_in = self.norm(x)                    # normalize activations
        q = self.q(h_in).view(b, c, -1)        # (B,C,HW)
        k = self.k(h_in).view(b, c, -1)        # (B,C,HW)
        v = self.v(h_in).view(b, c, -1)        # (B,C,HW)
        attn = torch.softmax(torch.bmm(q.permute(0,2,1), k) / math.sqrt(c), dim=-1)  # (B,HW,HW)
        out = torch.bmm(v, attn.permute(0,2,1)).view(b, c, h, w)                     # apply attention
        out = self.proj(out)                   # final 1x1 conv
        return out + x                         # residual connection


# ---- U-Net backbone ------------------------------------------------------------
class UNet(nn.Module):
    """Small UNet for ε-prediction (supports class conditioning via embeddings)."""
    def __init__(self, base: int = 64, num_classes: Optional[int] = None, img_ch: int = 3):
        super().__init__()                                     # init
        self.num_classes = num_classes                         # store number of classes (or None)
        t_dim = base * 4                                       # time embedding width
        self.t_pos = SinusoidalPosEmb(t_dim)                   # positional embedding of t
        self.t_mlp = nn.Sequential(                            # MLP on t-embedding
            nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )

        # encoder (two stages at 64×, one at 32×, one at 16×)
        self.in_conv = nn.Conv2d(img_ch, base, 3, padding=1)   # input stem
        self.down1 = ResBlock(base, base, t_dim, y_dim=num_classes)           # 64×
        self.down2 = ResBlock(base, base * 2, t_dim, y_dim=num_classes)       # 64×
        self.pool1 = nn.AvgPool2d(2)                                          # 64→32
        self.down3 = ResBlock(base * 2, base * 2, t_dim, y_dim=num_classes)   # 32×
        self.pool2 = nn.AvgPool2d(2)                                          # 32→16
        self.down4 = ResBlock(base * 2, base * 4, t_dim, y_dim=num_classes)   # 16×
        self.attn = AttentionBlock(base * 4)                                   # attention at 16×16

        # decoder with skip connections (concat encoder features)
        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)        # 16→32
        self.up_block1 = ResBlock(base * 4, base * 2, t_dim, y_dim=num_classes)  # concat (2+2)*base
        self.up2 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)            # 32→64
        self.up_block2 = ResBlock(base * 3, base, t_dim, y_dim=num_classes)   # concat (1+2)*base  **FIXED in_ch**
        self.out_conv = nn.Conv2d(base, img_ch, 3, padding=1)                 # output conv → ε̂

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        t_emb = self.t_mlp(self.t_pos(t))                 # (B, t_dim) time embedding
        x0 = F.silu(self.in_conv(x))                      # stem + SiLU
        d1 = self.down1(x0, t_emb, y)                     # 64× stage 1
        d2 = self.down2(d1, t_emb, y)                     # 64× stage 2 (more channels)
        p1 = self.pool1(d2)                               # 64→32
        d3 = self.down3(p1, t_emb, y)                     # 32× stage
        p2 = self.pool2(d3)                               # 32→16
        d4 = self.down4(p2, t_emb, y)                     # 16× stage
        b  = self.attn(d4)                                # attention bottleneck
        u1 = self.up1(b)                                  # 16→32
        u1 = torch.cat([u1, d3], dim=1)                   # concat skip (channels: 2*base + 2*base = 4*base)
        u1 = self.up_block1(u1, t_emb, y)                 # resblock at 32×
        u2 = self.up2(u1)                                 # 32→64
        u2 = torch.cat([u2, d2], dim=1)                   # concat skip (base + 2*base = 3*base)
        u2 = self.up_block2(u2, t_emb, y)                 # resblock at 64×  **channels fixed**
        out = self.out_conv(F.silu(u2))                   # final conv to 3 channels (ε̂)
        return out                                        # return noise prediction



