import math  # math helpers
from typing import Optional  # optional types
import torch  # torch core
from torch import nn  # nn modules
import torch.nn.functional as F  # functional ops

# --- time embedding ---
class SinusoidalPosEmb(nn.Module):  # sinusoidal positional embedding (t)
    def __init__(self, dim: int):  # ctor
        super().__init__()  # parent init
        self.dim = dim  # store dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # forward
        half = self.dim // 2  # half-dim
        device = t.device  # device
        emb = math.log(10000) / (half - 1)  # scale factor
        emb = torch.exp(torch.arange(half, device=device) * -emb)  # frequency vector
        emb = t.float()[:, None] * emb[None, :]  # outer product (B, half)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # concat sin/cos → (B, dim)
        return emb  # return embedding


class ResBlock(nn.Module):  # residual block with t & optional y cond
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, y_dim: Optional[int] = None, dropout: float = 0.1):  # ctor
        super().__init__()  # init
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)  # conv3x3 in→out
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)  # conv3x3 out→out
        self.emb = nn.Linear(t_dim, out_ch)  # linear for time embedding
        self.norm1 = nn.GroupNorm(32, out_ch)  # GN after conv1
        self.norm2 = nn.GroupNorm(32, out_ch)  # GN after conv2
        self.dropout = nn.Dropout(dropout)  # dropout
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()  # skip conv if channels differ
        self.y_embed = nn.Embedding(y_dim, out_ch) if y_dim is not None else None  # class embedding if conditional

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:  # forward
        h = self.norm1(self.conv1(x))  # conv1 + norm
        h = F.silu(h)  # SiLU act
        t_emb = self.emb(t)  # project time emb to out_ch
        h = h + t_emb[:, :, None, None]  # add time conditioning
        # safe class embedding: allow None or negative labels by zeroing
        y_emb = None  # init
        if (self.y_embed is not None) and (y is not None):  # if cond used
            y_long = y.long()  # ensure int64
            mask = (y_long >= 0)  # valid class ids
            if mask.all():  # all valid
                y_emb = self.y_embed(y_long)  # lookup all
            else:  # some invalid (e.g., dropped for CFG)
                y_clamped = y_long.clamp_min(0)  # clamp to 0
                y_all = self.y_embed(y_clamped)  # lookup anyway
                y_emb = torch.where(mask[:, None], y_all, torch.zeros_like(y_all))  # zero-out invalid
        if y_emb is not None:  # if present
            h = h + y_emb[:, :, None, None]  # add class conditioning
        h = self.norm2(self.conv2(h))  # conv2 + norm
        h = self.dropout(F.silu(h))  # act + dropout
        return h + self.skip(x)  # residual add


class AttentionBlock(nn.Module):  # simple self-attention (spatial)
    def __init__(self, ch: int):  # ctor
        super().__init__()  # init
        self.norm = nn.GroupNorm(32, ch)  # normalize
        self.q = nn.Conv2d(ch, ch, 1)  # 1x1 q
        self.k = nn.Conv2d(ch, ch, 1)  # 1x1 k
        self.v = nn.Conv2d(ch, ch, 1)  # 1x1 v
        self.proj = nn.Conv2d(ch, ch, 1)  # 1x1 projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # forward
        b, c, h, w = x.shape  # shape
        h_in = self.norm(x)  # norm
        q = self.q(h_in).view(b, c, -1)  # (B,C,HW)
        k = self.k(h_in).view(b, c, -1)  # (B,C,HW)
        v = self.v(h_in).view(b, c, -1)  # (B,C,HW)
        attn = torch.softmax(torch.bmm(q.permute(0, 2, 1), k) / math.sqrt(c), dim=-1)  # (B,HW,HW) attention
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(b, c, h, w)  # apply attn → (B,C,H,W)
        out = self.proj(out)  # final proj
        return out + x  # residual add


class UNet(nn.Module):  # U-Net backbone for epsilon prediction
    def __init__(self, base: int = 64, num_classes: Optional[int] = None, img_ch: int = 3):  # ctor
        super().__init__()  # init
        self.num_classes = num_classes  # store classes (or None)
        t_dim = base * 4  # time embedding dim
        self.t_pos = SinusoidalPosEmb(t_dim)  # positional embedder
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))  # time MLP

        # encoder
        self.in_conv = nn.Conv2d(img_ch, base, 3, padding=1)  # stem conv
        self.down1 = ResBlock(base, base, t_dim, y_dim=num_classes)  # res block @64x
        self.down2 = ResBlock(base, base * 2, t_dim, y_dim=num_classes)  # res block @64x
        self.pool1 = nn.AvgPool2d(2)  # 64→32
        self.down3 = ResBlock(base * 2, base * 2, t_dim, y_dim=num_classes)  # res block @32x
        self.pool2 = nn.AvgPool2d(2)  # 32→16
        self.down4 = ResBlock(base * 2, base * 4, t_dim, y_dim=num_classes)  # res block @16x
        self.attn = AttentionBlock(base * 4)  # attn at 16x16

        # decoder
        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)  # 16→32
        self.up_block1 = ResBlock(base * 4, base * 2, t_dim, y_dim=num_classes)  # cat with down3 (2+2=4)
        self.up2 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)  # 32→64
        self.up_block2 = ResBlock(base * 3, base, t_dim, y_dim=num_classes)  # cat with down2 (1+2=3) **fixed**
        self.out_conv = nn.Conv2d(base, img_ch, 3, padding=1)  # final conv

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:  # forward
        t_emb = self.t_mlp(self.t_pos(t))  # build time embedding (B,t_dim)
        x0 = F.silu(self.in_conv(x))  # stem activation
        d1 = self.down1(x0, t_emb, y)  # 64x
        d2 = self.down2(d1, t_emb, y)  # 64x
        p1 = self.pool1(d2)  # 32x
        d3 = self.down3(p1, t_emb, y)  # 32x
        p2 = self.pool2(d3)  # 16x
        d4 = self.down4(p2, t_emb, y)  # 16x
        b = self.attn(d4)  # 16x attn
        u1 = self.up1(b)  # up to 32x
        u1 = torch.cat([u1, d3], dim=1)  # concat skip (2+2=4)*base
        u1 = self.up_block1(u1, t_emb, y)  # res block
        u2 = self.up2(u1)  # up to 64x
        u2 = torch.cat([u2, d2], dim=1)  # concat skip (1+2=3)*base
        u2 = self.up_block2(u2, t_emb, y)  # res block (fixed in_ch)
        out = self.out_conv(F.silu(u2))  # final conv
        return out  # predict epsilon


