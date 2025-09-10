# unet.py
# A compact, stable UNet for DDPM epsilon-prediction with optional class conditioning.
# This version fixes decoder/skip channel sizes:
#   - After up1: concat( up1(2B), d4(4B) ) -> 6B  -> ResBlock(6B -> 2B)
#   - After up2: concat( up2(B),  d2(2B) ) -> 3B  -> ResBlock(3B -> B)
# Every line is commented for clarity.

from typing import Optional                          # typing hints
import torch                                         # PyTorch core
import torch.nn as nn                                # neural network modules
import torch.nn.functional as F                      # functional ops

# -----------------------------
# Small helpers / building bits
# -----------------------------

class SiLU(nn.Module):                                # SiLU activation module
    def forward(self, x):                             # forward method
        return x * torch.sigmoid(x)                  # SiLU(x) = x * sigmoid(x)

def conv3x3(in_ch: int, out_ch: int) -> nn.Conv2d:   # 3×3 conv with padding=1
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

def conv1x1(in_ch: int, out_ch: int) -> nn.Conv2d:   # 1×1 conv (for projections)
    return nn.Conv2d(in_ch, out_ch, kernel_size=1)

class TimeMLP(nn.Module):
    # Projects sinusoidal timestep embeddings to a feature dimension.
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()                           # init parent
        self.net = nn.Sequential(                    # simple 2-layer MLP
            nn.Linear(in_dim, out_dim), SiLU(),      # linear + SiLU
            nn.Linear(out_dim, out_dim),             # keep same size
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        # t_emb: (B, in_dim) -> (B, out_dim)
        return self.net(t_emb)

class PositionalEmbedding(nn.Module):
    # Standard sinusoidal embedding for integer timesteps t in [0..T-1].
    def __init__(self, dim: int):
        super().__init__()                           # init parent
        self.dim = dim                               # target embedding dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) integer timesteps
        half = self.dim // 2                         # half for sin, half for cos
        # frequencies: exp(- log(1e4) * i / half), i=0..half-1
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0, device=t.device)) *
            torch.arange(0, half, device=t.device, dtype=torch.float32) / half
        )
        # angles: (B,1) * (half,) -> (B, half)
        angles = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        # concat sin and cos -> (B, 2*half) == (B, self.dim) if even
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        # if dim is odd, pad one channel with zero
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]), value=0.0)
        return emb                                   # (B, dim)

class ResBlock(nn.Module):
    """
    Residual block with GroupNorm + SiLU, and additive (t, y) embedding.
    - in_ch -> out_ch via two 3×3 convs
    - If in_ch != out_ch, a 1×1 skip aligns channels
    - Embedding (t_emb [+ y_emb]) is projected to out_ch and added after conv1
    """
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, y_dim: int = 0):
        super().__init__()                           # init parent
        self.conv1 = conv3x3(in_ch, out_ch)          # first 3×3 conv
        self.conv2 = conv3x3(out_ch, out_ch)         # second 3×3 conv
        self.norm1 = nn.GroupNorm(8, out_ch)         # GroupNorm robust for small batches
        self.norm2 = nn.GroupNorm(8, out_ch)         # second GroupNorm
        self.act   = SiLU()                          # SiLU activation
        emb_in = t_dim + y_dim                       # total embedding input dim
        self.emb = nn.Linear(emb_in, out_ch) if emb_in > 0 else None  # project (t,y)->out_ch
        self.skip = conv1x1(in_ch, out_ch) if in_ch != out_ch else nn.Identity()  # channel align

    def forward(self, x: torch.Tensor,
                t_emb: torch.Tensor,
                y_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, in_ch, H, W), t_emb: (B, t_dim), y_emb: (B, y_dim) or None
        h = self.conv1(x)                            # first conv: (B, out_ch, H, W)
        if self.emb is not None:                     # if we have an embedding to inject
            # concat time and class embeddings if class available
            emb_cat = torch.cat([t_emb, y_emb], dim=-1) if y_emb is not None else t_emb
            # project to out_ch and broadcast to (B, out_ch, 1, 1)
            h = h + self.emb(emb_cat).unsqueeze(-1).unsqueeze(-1)
        h = self.act(self.norm1(h))                  # norm + activation
        h = self.act(self.norm2(self.conv2(h)))      # second conv + norm + act
        return h + self.skip(x)                      # residual add (with 1×1 if needed)

class UpSample(nn.Module):
    # ConvTranspose 2× upsampling (learned upsample).
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()                           # init parent
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)  # 2× upsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)                            # (B, out_ch, 2H, 2W)

# -----------------------------
# The actual UNet backbone
# -----------------------------

class UNet(nn.Module):
    """
    Compact UNet for DDPM ε-prediction.
    - base: base channel multiplier B (default 64)
    - num_classes: if not None, enables class conditioning via embedding table
    Encoder:
      stem: 3 -> B
      d1:   B -> B
      d2:   B -> 2B
      pool
      d3:   2B -> 2B
      d4:   2B -> 4B
      pool
      mid:  4B -> 4B
    Decoder:
      up1:     4B -> 2B (×2),  concat with d4(4B)  => 6B  -> up_block1(6B -> 2B)
      up2:     2B -> B  (×2),  concat with d2(2B)  => 3B  -> up_block2(3B -> B)
      out_conv: B -> 3
    """
    def __init__(self, base: int = 64, num_classes: Optional[int] = None):
        super().__init__()                           # init parent
        self.base = base                              # store base
        self.num_classes = num_classes                # store num classes (or None)

        # ---- Embeddings ----
        t_dim = base * 4                              # timestep embedding size
        self.pos_emb = PositionalEmbedding(t_dim)     # sinusoidal t embedding
        self.t_mlp  = TimeMLP(t_dim, t_dim)          # project t embedding to t_dim
        if num_classes is not None:                   # optional class embedding
            self.y_embed = nn.Embedding(num_classes, t_dim)
            y_dim = t_dim                             # class embed dim equals t_dim
        else:
            self.y_embed = None
            y_dim = 0

        # ---- Stem ----
        B = base                                      # shorthand
        self.in_conv = conv3x3(3, B)                  # input conv: 3 -> B

        # ---- Encoder ----
        self.down1 = ResBlock(B,     B,     t_dim, y_dim)  # keep B
        self.down2 = ResBlock(B,     2*B,  t_dim, y_dim)   # B -> 2B
        self.pool1 = nn.AvgPool2d(2)                       # /2
        self.down3 = ResBlock(2*B,  2*B,  t_dim, y_dim)    # 2B -> 2B
        self.down4 = ResBlock(2*B,  4*B,  t_dim, y_dim)    # 2B -> 4B
        self.pool2 = nn.AvgPool2d(2)                       # /2

        # ---- Bottleneck ----
        self.mid   = ResBlock(4*B,  4*B,  t_dim, y_dim)    # 4B -> 4B

        # ---- Decoder ----
        self.up1        = UpSample(4*B, 2*B)               # 4B -> 2B (×2)
        self.up_block1  = ResBlock(6*B, 2*B, t_dim, y_dim) # concat(2B,4B)=6B -> 2B
        self.up2        = UpSample(2*B, B)                 # 2B -> B (×2)
        self.up_block2  = ResBlock(3*B, B,   t_dim, y_dim) # concat(B,2B)=3B  -> B

        # ---- Head ----
        self.out_conv   = conv3x3(B, 3)                    # B -> 3 (RGB)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B,3,H,W)   t: (B,) integer timesteps   y: (B,) class ids or None
        B = x.size(0)                                      # batch size

        # --- build embeddings ---
        t_emb = self.pos_emb(t)                            # (B, t_dim) sinusoidal
        t_emb = self.t_mlp(t_emb)                          # (B, t_dim) projected
        y_emb = self.y_embed(y) if (self.y_embed is not None and y is not None) else None  # (B, t_dim) or None

        # --- encoder path with skips ---
        x0 = self.in_conv(x)                               # stem: 3 -> B               (B, B, H,   W)
        d1 = self.down1(x0, t_emb, y_emb)                  # level1 keep B              (B, B, H,   W)
        d2 = self.down2(d1, t_emb, y_emb)                  # B -> 2B                    (B, 2B, H,  W)
        p1 = self.pool1(d2)                                # downsample /2              (B, 2B, H/2,W/2)
        d3 = self.down3(p1, t_emb, y_emb)                  # keep 2B                    (B, 2B, H/2,W/2)
        d4 = self.down4(d3, t_emb, y_emb)                  # 2B -> 4B                   (B, 4B, H/2,W/2)
        p2 = self.pool2(d4)                                # downsample /2              (B, 4B, H/4,W/4)

        # --- bottleneck ---
        m  = self.mid(p2, t_emb, y_emb)                    # 4B -> 4B                   (B, 4B, H/4,W/4)

        # --- decoder path with correct concat channel sizes ---
        u1 = self.up1(m)                                   # 4B -> 2B (upsample ×2)     (B, 2B, H/2,W/2)
        u1 = torch.cat([u1, d4], dim=1)                    # concat with skip d4(4B)    (B, 6B, H/2,W/2)
        u1 = self.up_block1(u1, t_emb, y_emb)              # 6B -> 2B                   (B, 2B, H/2,W/2)

        u2 = self.up2(u1)                                  # 2B -> B  (upsample ×2)     (B, B,  H,  W)
        u2 = torch.cat([u2, d2], dim=1)                    # concat with skip d2(2B)    (B, 3B, H,  W)
        u2 = self.up_block2(u2, t_emb, y_emb)              # 3B -> B                    (B, B,  H,  W)

        out = self.out_conv(u2)                            # B -> 3                      (B, 3,  H,  W)
        return out                                         # ε̂(x_t, t, y)
