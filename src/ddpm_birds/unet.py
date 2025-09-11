from __future__ import annotations  # enable future annotations in Python
import torch                        # PyTorch main package
import torch.nn as nn               # neural network layers
import torch.nn.functional as F     # functional ops (activations, conv, etc.)

# -----------------------------------------------------------------------------
# Helper: sinusoidal timestep embedding (like in DDPM)
# -----------------------------------------------------------------------------
def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Return sinusoidal timestep embeddings of shape (B, dim)."""
    half = dim // 2                                        # half of the embedding size
    t = t.float().unsqueeze(1)                             # (B,) -> (B, 1) for broadcasting
    # frequencies spaced geometrically between 1 and 1/10000
    freqs = torch.exp(
        torch.arange(half, device=t.device, dtype=t.dtype)
        * (-torch.log(torch.tensor(10000.0, device=t.device)) / max(half - 1, 1))
    )
    args = t * freqs                                       # (B, half)
    emb = torch.cat([torch.cos(args), torch.sin(args)], 1) # (B, 2*half) = (B, dim or dim-1)
    if dim % 2 == 1:                                       # if odd, pad one column of zeros
        emb = F.pad(emb, (0, 1))
    return emb                                             # (B, dim)

# -----------------------------------------------------------------------------
# Residual block with conditioning on [t_emb || y_emb]
# -----------------------------------------------------------------------------
class ResBlock(nn.Module):
    """A simple residual block that injects conditioning via an MLP."""
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, y_dim: int):
        super().__init__()                                 # init parent nn.Module
        self.in_ch = in_ch                                 # input channels
        self.out_ch = out_ch                               # output channels
        self.y_dim = y_dim                                 # class-embedding dimensionality

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1) # first 3x3 conv
        self.norm1 = nn.GroupNorm(8, out_ch)                # group norm for stability

        emb_in = t_dim + y_dim                              # conditioning vector size
        self.emb = nn.Linear(emb_in, out_ch)                # project cond to per-channel bias

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)# second 3x3 conv
        self.norm2 = nn.GroupNorm(8, out_ch)                # another group norm

        # if channel count changes, align with a 1x1 conv; else identity
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(
        self,
        x: torch.Tensor,           # (B, C, H, W)
        t_emb: torch.Tensor,       # (B, t_dim)
        y_emb: torch.Tensor | None # (B, y_dim) or None if model is unconditional
    ) -> torch.Tensor:
        h = self.conv1(x)                                      # conv #1
        h = self.norm1(h)                                      # norm #1
        h = F.silu(h)                                          # activation

        if self.y_dim > 0:                                     # if conditional model
            # y_emb must exist and match the expected size (safety check)
            assert (y_emb is not None) and (y_emb.shape[1] == self.y_dim), \
                f"Expected y_emb dim {self.y_dim}, got {None if y_emb is None else y_emb.shape}"
            emb_cat = torch.cat([t_emb, y_emb], dim=1)         # concatenate time and class embeddings
        else:                                                  # unconditional model
            emb_cat = t_emb                                    # use only time embedding

        bias = self.emb(emb_cat).unsqueeze(-1).unsqueeze(-1)   # (B, out_ch, 1, 1) broadcast bias
        h = h + bias                                           # add conditioning as bias

        h = self.conv2(h)                                      # conv #2
        h = self.norm2(h)                                      # norm #2
        h = F.silu(h)                                          # activation

        return h + self.skip(x)                                # residual connection

# -----------------------------------------------------------------------------
# U-Net backbone
# -----------------------------------------------------------------------------
class UNet(nn.Module):
    """
    A light U-Net used in the DDPM project.
    - base:   width multiplier (e.g., 64, 96)
    - num_classes: if not None, enables class conditioning with a NULL class for CFG.
    """
    def __init__(self, base: int = 64, num_classes: int | None = None):
        super().__init__()                                     # init parent nn.Module
        self.base = base                                       # store base width
        self.num_classes = num_classes                         # number of classes or None

        t_dim = base * 8                                       # time embedding dimension
        y_dim = t_dim if num_classes is not None else 0        # class embedding same size for simplicity

        # small MLP that refines the sinusoidal time embedding
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim),                           # project t_emb
            nn.SiLU(),                                         # nonlinearity
            nn.Linear(t_dim, t_dim),                           # keep size
        )

        # If conditional: allocate an embedding table with an extra NULL token
        if num_classes is not None:
            self.null_y_id = num_classes                       # index for the null class (last row)
            self.y_embed = nn.Embedding(num_classes + 1, y_dim)# (C+1, y_dim)
        else:
            self.null_y_id = None                              # not used in unconditional case
            self.y_embed = None                                # no class embeddings

        # Stem: first conv maps RGB to base channels
        self.in_conv = nn.Conv2d(3, base, 3, padding=1)        # (B, 3, H, W) -> (B, B, H, W)

        # Encoder: four residual stages, downsampled via AvgPool between them
        self.down1 = ResBlock(base,     base,     t_dim, y_dim) # keep spatial size
        self.down2 = ResBlock(base,     base * 2, t_dim, y_dim) # after pool: H/2
        self.down3 = ResBlock(base * 2, base * 4, t_dim, y_dim) # after pool: H/4
        self.down4 = ResBlock(base * 4, base * 8, t_dim, y_dim) # after pool: H/8

        # Decoder: upsample and concatenate skips, then ResBlock
        self.up1 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)    # H/8 -> H/4
        self.up_block1 = ResBlock(base * 4 + base * 4, base * 4, t_dim, y_dim)  # cat with d3

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)    # H/4 -> H/2
        self.up_block2 = ResBlock(base * 2 + base * 2, base * 2, t_dim, y_dim)  # cat with d2

        self.up3 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)        # H/2 -> H
        self.up_block3 = ResBlock(base + base, base, t_dim, y_dim)        # cat with d1

        # Final 3x3 conv back to RGB
        self.out_conv = nn.Conv2d(base, 3, 3, padding=1)       # (B, B, H, W) -> (B, 3, H, W)

    def forward(
        self,
        x: torch.Tensor,                # input image/noise (B, 3, H, W)
        t: torch.Tensor,                # integer timesteps (B,)
        y: torch.Tensor | None = None   # class labels (B,) or None
    ) -> torch.Tensor:
        B = x.size(0)                                        # batch size

        # Build time embedding and pass through the MLP
        t_emb = timestep_embedding(t, self.base * 8)         # (B, t_dim)
        t_emb = self.t_mlp(t_emb)                            # (B, t_dim)

        # Build class embedding (or null class) if the model is conditional
        if self.y_embed is not None:                         # conditional case
            if y is None:                                    # if no y is provided...
                y_long = torch.full(                         # ...use NULL class id everywhere
                    (B,), self.null_y_id, device=x.device, dtype=torch.long
                )
            else:                                            # y provided
                y_long = y.long().view(-1)                  # ensure long dtype and flat shape
                # negative labels (e.g., -1 for uncond pass in CFG) -> map to NULL class
                y_long = torch.where(
                    y_long >= 0, y_long,
                    torch.full_like(y_long, self.null_y_id)
                )
            y_emb = self.y_embed(y_long)                    # (B, y_dim)
        else:                                                # unconditional model
            y_emb = None                                     # no class embedding

        # --------- Encoder ---------
        x0 = self.in_conv(x)                                 # (B, B,  H,   W)
        d1 = self.down1(x0, t_emb, y_emb)                    # (B, B,  H,   W)

        x2 = F.avg_pool2d(d1, 2)                             # (B, B,  H/2, W/2)
        d2 = self.down2(x2, t_emb, y_emb)                    # (B, 2B, H/2, W/2)

        x3 = F.avg_pool2d(d2, 2)                             # (B, 2B, H/4, W/4)
        d3 = self.down3(x3, t_emb, y_emb)                    # (B, 4B, H/4, W/4)

        x4 = F.avg_pool2d(d3, 2)                             # (B, 4B, H/8, W/8)
        d4 = self.down4(x4, t_emb, y_emb)                    # (B, 8B, H/8, W/8)

        # --------- Decoder ---------
        u1 = self.up1(d4)                                    # upsample: (B, 4B, H/4, W/4)
        u1 = torch.cat([u1, d3], dim=1)                      # skip: concat with encoder feat
        u1 = self.up_block1(u1, t_emb, y_emb)                # (B, 4B, H/4, W/4)

        u2 = self.up2(u1)                                    # upsample: (B, 2B, H/2, W/2)
        u2 = torch.cat([u2, d2], dim=1)                      # skip: concat with encoder feat
        u2 = self.up_block2(u2, t_emb, y_emb)                # (B, 2B, H/2, W/2)

        u3 = self.up3(u2)                                    # upsample: (B, B,  H,   W)
        u3 = torch.cat([u3, d1], dim=1)                      # skip: concat with encoder feat
        u3 = self.up_block3(u3, t_emb, y_emb)                # (B, B,  H,   W)

        out = self.out_conv(u3)                              # final RGB: (B, 3, H, W)
        return out                                           # predicted noise ε̂ (or x̂₀ depending on training)
