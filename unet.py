from __future__ import annotations                    # postponed annotations
import torch                                          # tensors
import torch.nn as nn                                 # modules/layers
import torch.nn.functional as F                       # activations/pool


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding (B,)->(B,dim) used in diffusion models."""
    half = dim // 2                                    # half dimension
    t = t.float().unsqueeze(1)                         # (B,1)
    freqs = torch.exp(                                 # frequencies
        torch.arange(half, device=t.device, dtype=t.dtype)
        * (-torch.log(torch.tensor(10000.0)) / max(1, half - 1))
    )
    args = t * freqs                                   # (B,half)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)  # (B,dim or dim-1)
    if dim % 2 == 1:                                   # pad if odd
        emb = F.pad(emb, (0, 1))                       # add one zero column
    return emb                                         # (B,dim)


class ResBlock(nn.Module):
    """Residual block that adds a learned bias from [t_emb || y_emb]."""
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, y_dim: int):
        super().__init__()                             # init base
        self.in_ch = in_ch                             
        self.out_ch = out_ch                           
        self.y_dim = y_dim                             # remember y size

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)   # first conv
        self.norm1 = nn.GroupNorm(8, out_ch)                  # group norm
        self.emb = nn.Linear(t_dim + y_dim, out_ch)           # (t||y)->bias
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)  # second conv
        self.norm2 = nn.GroupNorm(8, out_ch)                  # group norm
        self.skip = (nn.Identity() if in_ch == out_ch         # channel skip
                     else nn.Conv2d(in_ch, out_ch, 1))        # 1×1 match

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, y_emb: torch.Tensor | None):
        """Forward with (t_emb, y_emb) → bias; y_emb must be provided if y_dim>0."""
        h = F.silu(self.norm1(self.conv1(x)))         # conv-norm-activation
        if self.y_dim > 0:                            # expect y if conditional
            assert y_emb is not None and y_emb.shape[1] == self.y_dim, \
                "Missing/incorrect y_emb in conditional ResBlock."
            sty = torch.cat([t_emb, y_emb], dim=1)    # concat (B,t+y)
        else:
            sty = t_emb                               # unconditional
        h = h + self.emb(sty).unsqueeze(-1).unsqueeze(-1)  # add bias (broadcast)
        h = F.silu(self.norm2(self.conv2(h)))         # second conv block
        return h + self.skip(x)                       # residual add


class UNet(nn.Module):
    """3-level UNet with skips; up/down by 2×, base width configurable."""
    def __init__(self, base: int = 64, num_classes: int | None = None):
        super().__init__()                             # init nn.Module
        self.base = int(base)                          # base channels
        self.num_classes = num_classes                 # None = unconditional

        # Embedding sizes for time and class (keep them equal for simplicity)
        t_dim = base * 8                               # time embedding width
        y_dim = (base * 8) if (num_classes is not None) else 0  # class emb

        # Time MLP (sinusoid → MLP) — keeps dimension t_dim
        self.t_mlp = nn.Sequential(                    
            nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )

        # Optional class embedding with one extra NULL id for CFG
        if num_classes is not None:
            self.null_y_id = num_classes               # extra slot as NULL
            self.y_embed = nn.Embedding(num_classes + 1, y_dim)  # (num+1, y_dim)
        else:
            self.null_y_id = None                      # mark as uncond
            self.y_embed = None                        # no embedding

        # Stem conv (3→B)
        B = self.base                                  # shorthand
        self.in_conv = nn.Conv2d(3, B, 3, padding=1)   # initial conv

        # Encoder: keep, then ×2, ×4, ×8 channels with average pooling
        self.down1 = ResBlock(B, B, t_dim, y_dim)      # B → B
        self.down2 = ResBlock(B, 2 * B, t_dim, y_dim)  # B → 2B
        self.down3 = ResBlock(2 * B, 4 * B, t_dim, y_dim)  # 2B → 4B
        self.down4 = ResBlock(4 * B, 8 * B, t_dim, y_dim)  # 4B → 8B

        # Decoder: upsample and fuse with skip via concatenation
        self.up1 = nn.ConvTranspose2d(8 * B, 4 * B, 2, stride=2)     # up 8B→4B
        self.up_block1 = ResBlock(8 * B, 4 * B, t_dim, y_dim)        # cat(4B,4B)

        self.up2 = nn.ConvTranspose2d(4 * B, 2 * B, 2, stride=2)     # up 4B→2B
        self.up_block2 = ResBlock(4 * B, 2 * B, t_dim, y_dim)        # cat(2B,2B)

        self.up3 = nn.ConvTranspose2d(2 * B, B, 2, stride=2)         # up 2B→B
        self.up_block3 = ResBlock(2 * B, B, t_dim, y_dim)            # cat(B,B)

        self.out_conv = nn.Conv2d(B, 3, 3, padding=1)                # B→3

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None):
        """Predict noise ε given x_t, t (and optional y)."""
        # Build time embedding (sinusoid → MLP)
        t_emb = self.t_mlp(timestep_embedding(t, self.base * 8))     # (B,t_dim)

        # Build class embedding, falling back to NULL class when y is None/negative
        if self.y_embed is not None:                                  # conditional
            B = x.size(0)                                            # batch size
            if y is None:                                            # missing labels
                y_long = torch.full((B,), self.null_y_id, device=x.device, dtype=torch.long)
            else:                                                    # provided labels
                y_long = y.long().view(-1)                           # ensure shape
                y_long = torch.where(                                # CFG: negatives → NULL
                    y_long >= 0, y_long, torch.full_like(y_long, self.null_y_id)
                )
            y_emb = self.y_embed(y_long)                             # (B,y_dim)
        else:
            y_emb = None                                             # unconditional path

        # Encoder path with average pooling for downsampling
        x0 = self.in_conv(x)                                         # (B,B,H,W)
        d1 = self.down1(x0, t_emb, y_emb)                            # (B,B,H,W)
        d2 = self.down2(F.avg_pool2d(d1, 2), t_emb, y_emb)           # (B,2B,H/2,W/2)
        d3 = self.down3(F.avg_pool2d(d2, 2), t_emb, y_emb)           # (B,4B,H/4,W/4)
        d4 = self.down4(F.avg_pool2d(d3, 2), t_emb, y_emb)           # (B,8B,H/8,W/8)

        # Decoder with skip connections
        u1 = self.up1(d4)                                            # (B,4B,H/4,W/4)
        u1 = self.up_block1(torch.cat([u1, d3], dim=1), t_emb, y_emb)# (B,4B,*,*)
        u2 = self.up2(u1)                                            # (B,2B,H/2,W/2)
        u2 = self.up_block2(torch.cat([u2, d2], dim=1), t_emb, y_emb)# (B,2B,*,*)
        u3 = self.up3(u2)                                            # (B,B,H,W)
        u3 = self.up_block3(torch.cat([u3, d1], dim=1), t_emb, y_emb)# (B,B,*,*)

        return self.out_conv(u3)                                     # predict ε
