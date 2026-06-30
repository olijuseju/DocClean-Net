"""
model/unet.py
=============
Lightweight U-Net (~660k parameters) for structured background removal
from scanned handwritten documents.

Architecture summary:
    Input:      (B, 1, 256, 256)  — grayscale, float32 in [0, 1]
    Encoder:    3 ConvBlock + MaxPool stages
    Bottleneck: 1 ConvBlock at lowest resolution
    Decoder:    3 ConvTranspose2d + skip connection + ConvBlock stages
    Output:     (B, 1, 256, 256) — Sigmoid activation, values in [0, 1]

Design decisions (do not change without reopening them):
    - Input is raw grayscale, NOT the synthetic B-R channel. The model
      learns its own feature representation.
    - No dropout, no attention — keeps parameter count near 660k.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two consecutive Conv2d → BatchNorm2d → ReLU sequences.

    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.

    Input tensor:  (B, in_channels,  H, W)
    Output tensor: (B, out_channels, H, W)
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Lightweight U-Net for document background removal.

    Args:
        in_channels (int):  Number of input channels. Default: 1 (grayscale).
        out_channels (int): Number of output channels. Default: 1 (clean mask).

    Forward pass:
        Input:  (B, 1, 256, 256)  float32, values in [0, 1]
        Output: (B, 1, 256, 256)  float32, values in [0, 1]

    Parameter count: ~660k
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1) -> None:
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        # Each stage: ConvBlock then MaxPool (pool applied in forward())
        self.enc1 = ConvBlock(in_channels, 16)   # → (B, 16,  H,   W  )
        self.enc2 = ConvBlock(16, 32)             # → (B, 32,  H/2, W/2)
        self.enc3 = ConvBlock(32, 64)             # → (B, 64,  H/4, W/4)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Bottleneck ────────────────────────────────────────────────────────
        # Operates at H/8, W/8 (32×32 for 256 input)
        self.bottleneck = ConvBlock(64, 128)      # → (B, 128, H/8, W/8)

        # ── Decoder ───────────────────────────────────────────────────────────
        # ConvTranspose2d doubles spatial dims; skip connection doubles channels
        # before the following ConvBlock.
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(128, 64)   # 64 (up) + 64 (skip) → 64

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64, 32)    # 32 (up) + 32 (skip) → 32

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(32, 16)    # 16 (up) + 16 (skip) → 16

        # ── Output head ───────────────────────────────────────────────────────
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input patch, shape (B, 1, 256, 256),
                              float32, values in [0, 1].

        Returns:
            torch.Tensor: Cleaned patch, shape (B, 1, 256, 256),
                         float32, values in [0, 1].
        """
        # Encoder
        s1 = self.enc1(x)             # (B, 16,  256, 256)
        s2 = self.enc2(self.pool(s1)) # (B, 32,  128, 128)
        s3 = self.enc3(self.pool(s2)) # (B, 64,   64,  64)

        # Bottleneck
        b = self.bottleneck(self.pool(s3))  # (B, 128,  32,  32)

        # Decoder — concatenate skip connections along channel dim
        d3 = self.dec3(torch.cat([self.up3(b),  s3], dim=1))  # (B, 64,  64,  64)
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))  # (B, 32, 128, 128)
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))  # (B, 16, 256, 256)

        return self.sigmoid(self.out_conv(d1))                 # (B,  1, 256, 256)
