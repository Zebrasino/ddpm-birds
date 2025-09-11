from __future__ import annotations               # future annotations
import torch                                     # PyTorch
import torch.nn as nn                            # layers
import torch.nn.functional as F                  # functional ops

# -----------------------------------------------------------------------------
# Sinusoidal timestep embedding (as in DDPM)
# -----------------------------------------------------------------------------
def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of integer timesteps -> (B, dim)."""
    half = dim // 2                                              # half dimension
    t = t.float().unsqueeze(1)                                   # (B,) -> (B,1)
    freqs = torch.exp(
        torch.arange(half, device=t.device, dtype=t.dtype)
        * (-torch.log(torch.tensor(10000.0, device=t.device)) / max(half - 1, 1))
    )                                                            # geometric frequencies
    args = t * freqs                                             # (B, half)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)   # concat cos/sin
    if dim % 2 == 1:                                             # pad if odd
        emb = F.pad(emb, (0, 1))
    return emb                                                   # (B, dim)

# -----------------------------------------------------------------------------
# Residual block with conditioning on [t_emb || y_emb]
# -----------------------------------------------------------------------------
class ResBlock(nn.Module):
    """Residual block that injects conditioning via per-channel bias."""
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, y_dim: int):
        super().__init__()                                       # init parent
        self.in_ch = in_ch                                       # store in channels
        self.out_ch = out_ch                                     # store out channels
        self.y_dim = y_dim                                       # class embedding size (0 if unconditional)

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)      # 3x3 conv
        self.norm1 = nn.GroupNorm(8, out_ch)                     # group norm

        emb_in = t_dim + y_dim                                   # cond vector size
        self.emb = nn.Linear(emb_in, out_ch)                     # project cond -> bias

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)     # 3x3 conv
        self.norm2 = nn.GroupNorm(8, out_ch)                     # group norm

        # Skip path to match channels if needed
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, y_emb: torch.Tensor | None):
        h = self.conv1(x)                                        # conv
        h = self.norm1(h)                                        # norm
        h = F.silu(h)                                            # activation

        # Build conditioning vector
        if self.y_dim > 0:                                       # conditional
            assert (y_emb is not None) and (y_emb.shape[1] == self.y_dim), \
                f"Expected y_emb dim {self.y_dim}, got {None if y_emb is None else y_emb.shape}"
            cond = torch.cat([t_emb, y_emb], dim=1)              # [t||y]
        else:                                                     # unconditional
            cond = t_emb                                         # just t

        bias = self.emb(cond).unsqueeze(-1).unsqueeze(-1)        # (B,out_ch,1,1)
        h = h + bias                                             # add as bias

        h = self.conv2(h)                                        # conv
        h = self.norm2(h)                                        # norm
        h = F.silu(h)                                            # activation

        return h + self.skip(x)                                  # residual add

# -----------------------------------------------------------------------------
# U-Net backbone
# -----------------------------------------------------------------------------
class UNet(nn.Module):
    """Light U-Net used by the DDPM trainer."""
    def __init__(self, base: int = 64, num_classes: int | None = None):
        super().__init__()                                       # init
        self.base = base                                         # width multiplier
        self.num_classes = num_classes                           # number of classes (None if unconditional)

        t_dim = base * 8                                         # t embedding dim
        y_dim = (base * 8) if num_classes is not None else 0     # y embedding dim

        # MLP to process time embedding
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim),                             # affine
            nn.SiLU(),                                           # nonlinearity
            nn.Linear(t_dim, t_dim),                             # affine
        )

        # Optional class embedding with a NULL token for CFG
        if num_classes is not None:
            self.null_y_id = num_classes                         # null class id = last
            self.y_embed = nn.Embedding(num_classes + 1, y_dim)  # table size (C+1, y_dim)
        else:
            self.null_y_id = None                                # unused
            self.y_embed = None                                  # no class embedding

        # Stem (RGB -> base)
        self.in_conv = nn.Conv2d(3, base, 3, padding=1)          # 3x3 conv

        # Encoder (three downsamples via avg-pool)
        self.down1 = ResBlock(base,     base,     t_dim, y_dim)  # H,   W
        self.down2 = ResBlock(base,     base * 2, t_dim, y_dim)  # H/2, W/2
        self.down3 = ResBlock(base * 2, base * 4, t_dim, y_dim)  # H/4, W/4
        self.down4 = ResBlock(base * 4, base * 8, t_dim, y_dim)  # H/8, W/8

        # Decoder (three upsamples with skip-concat)
        self.up1 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)        # H/8 -> H/4
        self.up_block1 = ResBlock(base * 4 + base * 4, base * 4, t_dim, y_dim)# cat with d3

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)        # H/4 -> H/2
        self.up_block2 = ResBlock(base * 2 + base * 2, base * 2, t_dim, y_dim)# cat with d2

        self.up3 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)            # H/2 -> H
        self.up_block3 = ResBlock(base + base, base, t_dim, y_dim)            # cat with d1

        # Output (base -> RGB)
        self.out_conv = nn.Conv2d(base, 3, 3, padding=1)          # final conv

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None):
        B = x.size(0)                                             # batch size

        # Time embedding
        t_emb = timestep_embedding(t, self.base * 8)              # (B,t_dim)
        t_emb = self.t_mlp(t_emb)                                 # (B,t_dim)

        # Class embedding (or NULL)
        if self.y_embed is not None:                              # if conditional
            if y is None:                                         # no labels provided
                y_long = torch.full((B,), self.null_y_id, device=x.device, dtype=torch.long)  # NULL
            else:
                y_long = y.long().view(-1)                        # to long shape (B,)
                y_long = torch.where(                             # negative labels -> NULL id
                    y_long >= 0, y_long, torch.full_like(y_long, self.null_y_id)
                )
            y_emb = self.y_embed(y_long)                          # (B,y_dim)
        else:
            y_emb = None                                          # unconditional

        # Encoder path
        x0 = self.in_conv(x)                                      # (B, B,  H,   W)
        d1 = self.down1(x0, t_emb, y_emb)                         # (B, B,  H,   W)
        d2 = self.down2(F.avg_pool2d(d1, 2), t_emb, y_emb)        # (B,2B,  H/2, W/2)
        d3 = self.down3(F.avg_pool2d(d2, 2), t_emb, y_emb)        # (B,4B,  H/4, W/4)
        d4 = self.down4(F.avg_pool2d(d3, 2), t_emb, y_emb)        # (B,8B,  H/8, W/8)

        # Decoder with skip connections
        u1 = self.up1(d4)                                         # (B,4B, H/4, W/4)
        u1 = torch.cat([u1, d3], dim=1)                           # concat skip
        u1 = self.up_block1(u1, t_emb, y_emb)                     # (B,4B, H/4, W/4)

        u2 = self.up2(u1)                                         # (B,2B, H/2, W/2)
        u2 = torch.cat([u2, d2], dim=1)                           # concat skip
        u2 = self.up_block2(u2, t_emb, y_emb)                     # (B,2B, H/2, W/2)

        u3 = self.up3(u2)                                         # (B,B,  H,   W)
        u3 = torch.cat([u3, d1], dim=1)                           # concat skip
        u3 = self.up_block3(u3, t_emb, y_emb)                     # (B,B,  H,   W)

        out = self.out_conv(u3)                                   # (B,3,H,W) predicted ε
        return out                                                # return noise prediction
