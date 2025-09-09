import math  # import modules
from typing import Optional  # import names from module
import torch  # import modules
from torch import nn  # import names from module
import torch.nn.functional as F  # import names from module

# Positional (time) embedding utilities  # comment  # statement
class SinusoidalPosEmb(nn.Module):  # define class SinusoidalPosEmb
    def __init__(self, dim: int):  # define function __init__
        super().__init__()  # call parent constructor  # statement
        self.dim = dim  # variable assignment

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # define function forward
        half = self.dim // 2  # variable assignment
        device = t.device  # variable assignment
        emb = math.log(10000) / (half - 1)  # variable assignment
        emb = torch.exp(torch.arange(half, device=device) * -emb)  # PyTorch operation
        emb = t.float()[:, None] * emb[None, :]  # PyTorch operation
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # PyTorch operation
        return emb  # return value

# Residual block with optional class/text conditioning  # comment  # statement
class ResBlock(nn.Module):  # define class ResBlock
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, y_dim: Optional[int] = None, dropout: float = 0.1):  # define function __init__
        super().__init__()  # call parent constructor  # statement
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)  # PyTorch operation
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)  # PyTorch operation
        self.emb = nn.Linear(t_dim, out_ch)  # PyTorch operation
        self.norm1 = nn.GroupNorm(32, out_ch)  # PyTorch operation
        self.norm2 = nn.GroupNorm(32, out_ch)  # PyTorch operation
        self.dropout = nn.Dropout(dropout)  # PyTorch operation
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()  # PyTorch operation
        self.y_embed = nn.Embedding(y_dim, out_ch) if y_dim is not None else None  # PyTorch operation

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:  # define function forward
        h = self.norm1(self.conv1(x))  # PyTorch operation
        h = F.silu(h)  # PyTorch operation
        t_emb = self.emb(t)  # PyTorch operation
        h = h + t_emb[:, :, None, None]  # PyTorch operation
        # Safe class embedding: allow None or negative labels by zeroing their embeddings  # comment  # statement
        y_emb = None  # variable assignment
        if (self.y_embed is not None and y is not None):  # control flow
            y_long = y.long()  # PyTorch operation
            mask = (y_long >= 0)  # PyTorch operation
            if mask.all():  # control flow
                y_emb = self.y_embed(y_long)  # PyTorch operation
            else:  # control flow
                y_clamped = y_long.clamp_min(0)  # PyTorch operation
                y_emb_all = self.y_embed(y_clamped)  # PyTorch operation
                y_emb = torch.where(mask[:, None], y_emb_all, torch.zeros_like(y_emb_all))  # PyTorch operation
        if y_emb is not None:  # control flow
            h = h + y_emb[:, :, None, None]  # PyTorch operation
        h = self.norm2(self.conv2(h))  # PyTorch operation
        h = self.dropout(F.silu(h))  # PyTorch operation
        return h + self.skip(x)  # return value

class AttentionBlock(nn.Module):  # define class AttentionBlock
    def __init__(self, ch: int):  # define function __init__
        super().__init__()  # call parent constructor  # statement
        self.norm = nn.GroupNorm(32, ch)  # PyTorch operation
        self.q = nn.Conv2d(ch, ch, 1)  # PyTorch operation
        self.k = nn.Conv2d(ch, ch, 1)  # PyTorch operation
        self.v = nn.Conv2d(ch, ch, 1)  # PyTorch operation
        self.proj = nn.Conv2d(ch, ch, 1)  # PyTorch operation

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # define function forward
        b, c, h, w = x.shape  # variable assignment
        h_in = self.norm(x)  # PyTorch operation
        q = self.q(h_in).view(b, c, -1)  # PyTorch operation
        k = self.k(h_in).view(b, c, -1)  # PyTorch operation
        v = self.v(h_in).view(b, c, -1)  # PyTorch operation
        attn = torch.softmax(torch.bmm(q.permute(0, 2, 1), k) / math.sqrt(c), dim=-1)  # PyTorch operation
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(b, c, h, w)  # PyTorch operation
        out = self.proj(out)  # PyTorch operation
        return out + x  # return value

class UNet(nn.Module):  # define class UNet
    def __init__(self, base: int = 64, num_classes: Optional[int] = None, img_ch: int = 3):  # define function __init__
        super().__init__()  # call parent constructor  # statement
        self.num_classes = num_classes  # variable assignment
        t_dim = base * 4  # variable assignment
        self.t_pos = SinusoidalPosEmb(t_dim)  # PyTorch operation
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))  # PyTorch operation

        # Encoder  # comment  # statement
        self.in_conv = nn.Conv2d(img_ch, base, 3, padding=1)  # PyTorch operation
        self.down1 = ResBlock(base, base, t_dim, y_dim=num_classes)  # PyTorch operation
        self.down2 = ResBlock(base, base * 2, t_dim, y_dim=num_classes)  # PyTorch operation
        self.pool1 = nn.AvgPool2d(2)  # PyTorch operation
        self.down3 = ResBlock(base * 2, base * 2, t_dim, y_dim=num_classes)  # PyTorch operation
        self.pool2 = nn.AvgPool2d(2)  # PyTorch operation
        self.down4 = ResBlock(base * 2, base * 4, t_dim, y_dim=num_classes)  # PyTorch operation
        self.attn = AttentionBlock(base * 4)  # PyTorch operation

        # Decoder  # comment  # statement
        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)  # PyTorch operation
        self.up_block1 = ResBlock(base * 4, base * 2, t_dim, y_dim=num_classes)  # PyTorch operation
        self.up2 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)  # PyTorch operation
        self.up_block2 = ResBlock(base * 2, base, t_dim, y_dim=num_classes)  # PyTorch operation
        self.out_conv = nn.Conv2d(base, img_ch, 3, padding=1)  # PyTorch operation

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:  # define function forward
        t_emb = self.t_mlp(self.t_pos(t))  # PyTorch operation
        x0 = F.silu(self.in_conv(x))  # PyTorch operation
        d1 = self.down1(x0, t_emb, y)  # PyTorch operation
        d2 = self.down2(d1, t_emb, y)  # PyTorch operation
        p1 = self.pool1(d2)  # PyTorch operation
        d3 = self.down3(p1, t_emb, y)  # PyTorch operation
        p2 = self.pool2(d3)  # PyTorch operation
        d4 = self.down4(p2, t_emb, y)  # PyTorch operation
        b = self.attn(d4)  # PyTorch operation
        u1 = self.up1(b)  # PyTorch operation
        u1 = torch.cat([u1, d3], dim=1)  # PyTorch operation
        u1 = self.up_block1(u1, t_emb, y)  # PyTorch operation
        u2 = self.up2(u1)  # PyTorch operation
        u2 = torch.cat([u2, d2], dim=1)  # PyTorch operation
        u2 = self.up_block2(u2, t_emb, y)  # PyTorch operation
        out = self.out_conv(F.silu(u2))  # PyTorch operation
        return out  # return value

