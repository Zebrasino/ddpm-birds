# unet.py
# Line-by-line commented U-Net backbone with time and optional class conditioning.
from __future__ import annotations  # Future annotations

from typing import Optional  # For optional label type hints

import torch  # Core tensor library
import torch.nn as nn  # Neural network modules
import torch.nn.functional as F  # Functional API for activations etc.

def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embeddings as in DDPM.

    t: (B,) integer or float timesteps; dim: embedding dimension.

    Returns a (B, dim) embedding.

    """
    half = dim // 2  # Half the embedding uses sine, half uses cosine
    # Create scales (frequencies) exponential in range [1, 10000]
    emb = torch.log(torch.tensor(10000.0, device=t.device)) / (half - 1)
    emb = torch.exp(torch.arange(half, device=t.device) * -emb)
    # Outer product t * scales; ensure float tensor
    emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
    # Concatenate sin and cos parts
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:  # Pad if odd dimension
        emb = F.pad(emb, (0,1))
    return emb  # Shape (B, dim)

class FiLM(nn.Module):
    """A simple FiLM layer to inject conditioning into normalization statistics."""
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()  # Initialize parent nn.Module
        self.fc = nn.Linear(cond_dim, 2 * dim)  # Map conditioning to scale and shift
    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.fc(cond)  # Compute FiLM parameters
        gamma, beta = gamma_beta.chunk(2, dim=1)  # Split into scale and shift
        # Reshape to broadcast over spatial dims (N, C) -> (N, C, 1, 1)
        return h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

class ResBlock(nn.Module):
    """Residual block with GroupNorm, SiLU, and optional FiLM conditioning."""
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, cond_dim: Optional[int] = None):
        super().__init__()  # Initialize nn.Module
        self.norm1 = nn.GroupNorm(32, in_ch)  # Normalize across channels
        self.act = nn.SiLU()  # Smooth nonlinearity
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)  # First 3x3 conv
        self.norm2 = nn.GroupNorm(32, out_ch)  # Second norm
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)  # Second 3x3 conv
        # Linear to project time embedding into feature space
        self.time_emb = nn.Linear(t_dim, out_ch)
        # Optional FiLM for class conditioning
        self.film = FiLM(out_ch, cond_dim) if cond_dim is not None else None
        # If input and output channels differ, use a 1x1 conv for the skip connection
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, y_emb: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))  # First norm-activation-conv
        # Add time embedding; reshape to (N, C, 1, 1) for broadcasting
        h = h + self.time_emb(t_emb)[:, :, None, None]
        # If class conditioning is present, apply FiLM
        if self.film is not None and y_emb is not None:
            h = self.film(h, y_emb)
        h = self.conv2(self.act(self.norm2(h)))  # Second norm-activation-conv
        return h + self.skip(x)  # Residual addition

class SelfAttention(nn.Module):
    """Simple 2D self-attention block for feature maps."""
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()  # Initialize parent module
        self.num_heads = num_heads  # Multihead parameter
        self.scale = (dim // num_heads) ** -0.5  # Scale factor for dot products
        self.qkv = nn.Conv2d(dim, dim * 3, 1)  # 1x1 conv to produce Q, K, V
        self.proj = nn.Conv2d(dim, dim, 1)  # Output projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape  # Unpack tensor shape
        qkv = self.qkv(x)  # Compute Q, K, V
        q, k, v = qkv.chunk(3, dim=1)  # Split channels
        # Reshape to (B, heads, C//heads, HW) to perform attention
        q = q.reshape(B, self.num_heads, C // self.num_heads, H * W)
        k = k.reshape(B, self.num_heads, C // self.num_heads, H * W)
        v = v.reshape(B, self.num_heads, C // self.num_heads, H * W)
        # Compute scaled dot-product attention
        attn = torch.softmax((q.transpose(2,3) @ k) * self.scale, dim=-1)
        out = (attn @ v.transpose(2,3)).transpose(2,3)  # Apply attention weights
        out = out.reshape(B, C, H, W)  # Merge heads back
        return self.proj(out) + x  # Residual connection

class UNet(nn.Module):
    """U-Net with timestep and optional class conditioning (via embeddings and FiLM)."""
    def __init__(self, img_channels: int = 3, base: int = 128, ch_mults=(1,2,2,4),
                 attn_res=(16,), num_classes: int = 200, cond_mode: str = "class", t_dim: int = 256):
        super().__init__()  # Initialize nn.Module
        self.cond_mode = cond_mode  # 'none' or 'class'
        self.t_dim = t_dim  # Dimension of the time embedding
        # Timestep MLP transforms sinusoidal embedding into feature vector
        self.time_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim * 4),
            nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim),
        )
        # Optional class embedding if conditioning is enabled
        self.y_embed = nn.Embedding(num_classes, t_dim) if cond_mode == "class" else None
        # Initial convolution to project image to feature maps
        self.in_conv = nn.Conv2d(img_channels, base, 3, padding=1)

        # Downsampling path: sequence of ResBlocks with occasional attention
        downs = []
        chans = base
        self.attn_res = set(attn_res)  # Set of spatial sizes where we apply attention
        self.feature_maps = []  # To keep track of channel counts for skip connections

        for i, mult in enumerate(ch_mults):
            out_ch = base * mult  # Desired output channels for this stage
            # Residual block; conditionally FiLM if class conditioning is used
            downs.append(ResBlock(chans, out_ch, t_dim, cond_dim=t_dim if self.y_embed is not None else None))
            # Attention at specific spatial resolution (e.g., 16x16)
            downs.append(SelfAttention(out_ch) if 2**(len(ch_mults)-i) in self.attn_res else nn.Identity())
            # Downsample via strided conv
            downs.append(nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1))
            self.feature_maps.append(out_ch)  # Record channels for the skip path
            chans = out_ch  # Update channel count

        self.down = nn.ModuleList(downs)  # Register the down path as a ModuleList

        # Middle blocks (bottleneck) with attention for global context
        self.mid1 = ResBlock(chans, chans, t_dim, cond_dim=t_dim if self.y_embed is not None else None)
        self.mid_attn = SelfAttention(chans)
        self.mid2 = ResBlock(chans, chans, t_dim, cond_dim=t_dim if self.y_embed is not None else None)

        # Upsampling path mirrors the down path
        ups = []
        for out_ch in reversed(self.feature_maps):
            # Residual block after concatenating skip features (so input doubles channels)
            ups.append(ResBlock(chans + out_ch, out_ch, t_dim, cond_dim=t_dim if self.y_embed is not None else None))
            ups.append(SelfAttention(out_ch) if out_ch in self.feature_maps[-1:] else nn.Identity())
            # Upsample using nearest-neighbor followed by 3x3 conv
            ups.append(nn.Upsample(scale_factor=2, mode='nearest'))
            ups.append(nn.Conv2d(out_ch, out_ch, 3, padding=1))
            chans = out_ch  # Update channels as we go up

        self.up = nn.ModuleList(ups)  # Register up path
        # Final projection back to image space (predicting noise ε with 3 channels)
        self.out_norm = nn.GroupNorm(32, chans)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(chans, img_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Compute sinusoidal timestep embedding and process through MLP
        t_emb = self.time_mlp(timestep_embedding(t, self.t_dim))
        # If class conditioning is enabled, embed labels; else keep None
        y_emb = self.y_embed(y) if (self.y_embed is not None and y is not None) else None

        # Initial conv
        h = self.in_conv(x)
        skips = []  # Will store intermediate activations for U-Net skips

        size = x.shape[-1]  # Spatial size to track attention resolution
        # Down path: apply blocks sequentially, storing post-ResBlock activations for skips
        it = iter(self.down)  # Create iterator to step through blocks in triples
        while True:
            try:
                # ResBlock
                rb = next(it)
                h = rb(h, t_emb, y_emb)
                skips.append(h)
                # Attention or identity
                attn = next(it)
                h = attn(h)
                # Downsample
                ds = next(it)
                h = ds(h)
                size //= 2  # Track spatial resolution
            except StopIteration:
                break  # Exit when we exhaust the down path

        # Middle blocks with attention
        h = self.mid1(h, t_emb, y_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb, y_emb)

        # Up path: mirror down; pop skips in reverse order
        it = iter(self.up)
        for skip in reversed(skips):
            rb = next(it)
            h = torch.cat([h, skip], dim=1)
            h = rb(h, t_emb, y_emb)
            attn = next(it)
            h = attn(h)
            upsample = next(it)
            h = upsample(h)
            conv = next(it)
            h = conv(h)

        # Output projection to predict noise ε with same channels as input image
        h = self.out_conv(self.out_act(self.out_norm(h)))
        return h  # Shape (N, C, H, W)
