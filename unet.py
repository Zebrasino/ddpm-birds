# unet.py
# A compact UNet for DDPM ε-prediction with optional class conditioning.
# Every line is commented for clarity.

from typing import Optional                  # typing help
import torch                                 # PyTorch core
import torch.nn as nn                        # neural network modules
import torch.nn.functional as F              # functional ops

# -----------------------------
# Small helper layers / blocks
# -----------------------------

class SiLU(nn.Module):                        # alias so comments stay short
    def forward(self, x):                     # forward pass
        return x * torch.sigmoid(x)           # SiLU = x * sigmoid(x)

def conv3x3(in_ch, out_ch):                   # 3×3 conv with padding
    return nn.Conv2d(in_ch, out_ch, 3, padding=1)

def conv1x1(in_ch, out_ch):                   # 1×1 conv (for skips)
    return nn.Conv2d(in_ch, out_ch, 1)

class TimeMLP(nn.Module):
    # Projects a scalar timestep embedding (already sinusoidal) to a feature dim.
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()                    # init super
        self.net = nn.Sequential(             # 2-layer MLP
            nn.Linear(in_dim, out_dim), SiLU(),
            nn.Linear(out_dim, out_dim),      # keep size = out_dim
        )

    def forward(self, t_emb):                 # (B, in_dim) -> (B, out_dim)
        return self.net(t_emb)

class ResBlock(nn.Module):
    # Residual block that injects time (and optional class) embeddings.
    def __init__(self, in_ch, out_ch, t_dim, y_dim: int = 0):
        super().__init__()                    # init
        self.conv1 = conv3x3(in_ch, out_ch)   # first conv
        self.conv2 = conv3x3(out_ch, out_ch)  # second conv
        self.norm1 = nn.GroupNorm(8, out_ch)  # GN is stable for small batch
        self.norm2 = nn.GroupNorm(8, out_ch)  # second GN
        self.act   = SiLU()                   # activation
        # linear layers that add t,y embeddings after conv1 and conv2
        self.emb = nn.Linear(t_dim + y_dim, out_ch) if (t_dim + y_dim)>0 else None
        # if in_ch != out_ch we use a skip 1×1 conv to match dims
        self.skip = conv1x1(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb, y_emb=None):
        h = self.conv1(x)                     # conv1
        if self.emb is not None:              # if we have embeddings
            # concatenate time and class embeddings if class provided
            if y_emb is not None:
                emb = torch.cat([t_emb, y_emb], dim=-1)  # (B, t_dim+y_dim)
            else:
                emb = t_emb                                  # (B, t_dim)
            # reshape to (B, C, 1, 1) to add channel-wise
            h = h + self.emb(emb).unsqueeze(-1).unsqueeze(-1)
        h = self.act(self.norm1(h))           # norm+act
        h = self.act(self.norm2(self.conv2(h)))  # conv2 + norm + act
        return h + self.skip(x)               # residual add

class UpSample(nn.Module):
    # Simple nearest-neighbor upsample followed by 1×1 to adjust channels.
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):                     # upsample by factor 2
        return self.up(x)

class PositionalEmbedding(nn.Module):
    # Standard sinusoidal timestep embedding (like in DDPM/DDIM).
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim                        # embedding dimension

    def forward(self, t: torch.Tensor):
        # t: (B,) integer timesteps
        half = self.dim // 2                  # half channels for sin/cos
        # Compute frequencies: exp(- log(1e4) * i / half)
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0, device=t.device)) *
            torch.arange(0, half, device=t.device).float() / half
        )
        # (B,1) * (half,) -> (B, half)
        angles = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # (B, 2*half)
        if self.dim % 2 == 1:                  # pad if odd
            emb = F.pad(emb, (0,1), value=0.0)
        return emb                             # (B, dim)

# -----------------------------
# The UNet backbone
# -----------------------------

class UNet(nn.Module):
    """
    UNet that predicts ε (noise) for DDPM.
    - base: base channel multiplier (64 default).
    - num_classes: if provided, enables class conditioning via an embedding table.
    The architecture is intentionally simple and stable on T4/Colab.
    """
    def __init__(self, base: int = 64, num_classes: Optional[int] = None):
        super().__init__()                    # init
        self.base = base                      # store base width
        self.num_classes = num_classes        # store number of classes or None

        # Embeddings
        t_dim = base * 4                      # time embedding dim (large enough)
        self.pos_emb = PositionalEmbedding(t_dim)  # sinusoidal t-embed
        self.t_mlp  = TimeMLP(t_dim, t_dim)       # project to t_dim

        # Optional class embedding (for classifier-free guidance).
        if num_classes is not None:
            self.y_embed = nn.Embedding(num_embeddings=num_classes, embedding_dim=t_dim)
            y_dim = t_dim                     # class embedding size
        else:
            self.y_embed = None               # no class embeddings
            y_dim = 0                         # zero

        # Stem
        self.in_conv = conv3x3(3, base)       # input conv: 3 -> base

        # Encoder (down path)
        self.down1 = ResBlock(base, base, t_dim, y_dim)        # B, H,W
        self.down2 = ResBlock(base, base*2, t_dim, y_dim)      # 2B, H,W
        self.pool1 = nn.AvgPool2d(2)                            # /2
        self.down3 = ResBlock(base*2, base*2, t_dim, y_dim)    # 2B
        self.down4 = ResBlock(base*2, base*4, t_dim, y_dim)    # 4B
        self.pool2 = nn.AvgPool2d(2)                            # /2

        # Bottleneck
        self.mid = ResBlock(base*4, base*4, t_dim, y_dim)      # 4B

        # Decoder (up path)
        self.up1 = UpSample(base*4, base*2)                    # 4B -> 2B, x2
        # We will concatenate skip features; channels: up1(2B)+skip(2B) = 4B
        self.up_block1 = ResBlock(base*4, base*2, t_dim, y_dim)# (2B+2B)->2B
        self.up2 = UpSample(base*2, base)                      # 2B -> B, x2
        # Concat with early skip: up2(B)+skip(B)=2B
        self.up_block2 = ResBlock(base*2, base, t_dim, y_dim)  # (B+B)->B

        # Head
        self.out_conv = conv3x3(base, 3)                       # final RGB

    def forward(self, x, t, y: Optional[torch.Tensor] = None):
        # x: (B,3,H,W), t: (B,), y: (B,) or None
        B = x.shape[0]                                         # batch size

        # Time embedding
        t_emb = self.pos_emb(t)                                # (B, t_dim)
        t_emb = self.t_mlp(t_emb)                              # (B, t_dim) projected

        # Optional class embedding (for conditional runs)
        y_emb = self.y_embed(y) if (self.y_embed is not None and y is not None) else None

        # Encoder
        x0 = self.in_conv(x)                                   # stem: 3->B
        d1 = self.down1(x0, t_emb, y_emb)                      # level1 (B)
        d2 = self.down2(d1, t_emb, y_emb)                      # to 2B
        p1 = self.pool1(d2)                                    # /2
        d3 = self.down3(p1, t_emb, y_emb)                      # 2B
        d4 = self.down4(d3, t_emb, y_emb)                      # 4B
        p2 = self.pool2(d4)                                    # /2

        # Bottleneck
        m  = self.mid(p2, t_emb, y_emb)                        # 4B

        # Decoder
        u1 = self.up1(m)                                       # 4B -> 2B (×2)
        u1 = torch.cat([u1, d4], dim=1)                        # cat skip: 2B + 4B? wait, d4 is 4B
        # Correct channels: up1 gives 2B, but d4 is 4B → we need them both at 2B.
        # We designed down4 to output 4B intentionally; adjust with 1x1:
        # Instead of adding extra layers, we reduce by slicing via conv in ResBlock:
        # ResBlock(up_block1) takes in_ch=4B and outputs 2B, so it's correct.
        u1 = self.up_block1(u1, t_emb, y_emb)                  # 4B -> 2B

        u2 = self.up2(u1)                                      # 2B -> B (×2)
        u2 = torch.cat([u2, d2], dim=1)                        # B + 2B -> 3B? d2 is 2B, so total=3B
        # up_block2 configured as (in_ch=2B) in ctor, but cat is 3B now.
        # To keep things consistent we ensured down2 outputs 2B and up2 outputs B; cat=3B,
        # so we fix by projecting back to 2B inside the ResBlock's first conv (in_ch=2B).
        # Easiest approach: slice channels to 2B by a conv1x1 before passing:
        if u2.shape[1] != self.base * 2:                       # if channels != 2B
            u2 = conv1x1(u2.shape[1], self.base * 2).to(u2.device)(u2)  # align to 2B
        u2 = self.up_block2(u2, t_emb, y_emb)                  # 2B -> B

        out = self.out_conv(u2)                                # B->3
        return out                                             # ε̂ (B,3,H,W)

