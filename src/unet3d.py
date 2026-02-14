"""
3D UNet with Monte Carlo Dropout for volumetric medical image segmentation.

Architecture
------------
Encoder:  4 down-sampling stages, each with two 3x3x3 conv + BN + ReLU + Dropout,
          followed by 2x2x2 max-pool.
Bottleneck: Two 3x3x3 conv + BN + ReLU + Dropout.
Decoder:  4 up-sampling stages via 2x2x2 transposed-conv, skip-connections from
          the encoder, each followed by two 3x3x3 conv + BN + ReLU + Dropout.
Output:   1x1x1 conv -> num_classes channels.

Monte Carlo Dropout
-------------------
Standard dropout is disabled during eval by default. MC Dropout keeps dropout
**active** at inference time so that multiple stochastic forward passes produce a
distribution over predictions.  Call ``model.enable_mc_dropout()`` before
running T forward passes; call ``model.disable_mc_dropout()`` to restore
normal eval behaviour.

Reference
---------
Gal, Y. & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation:
Representing Model Uncertainty in Deep Learning." ICML.

Cicek, O. et al. (2016). "3D U-Net: Learning Dense Volumetric Segmentation
from Sparse Annotation." MICCAI.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock3D(nn.Module):
    """Two consecutive 3x3x3 conv layers, each followed by BN, ReLU, Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.drop1 = nn.Dropout3d(p=dropout_rate)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.drop2 = nn.Dropout3d(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = self.drop2(F.relu(self.bn2(self.conv2(x))))
        return x


class EncoderBlock(nn.Module):
    """ConvBlock3D followed by 2x2x2 max-pool."""

    def __init__(self, in_ch: int, out_ch: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.conv = ConvBlock3D(in_ch, out_ch, dropout_rate)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        down = self.pool(skip)
        return down, skip


class DecoderBlock(nn.Module):
    """Transposed-conv upsample + skip-connection + ConvBlock3D."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(out_ch + skip_ch, out_ch, dropout_rate)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if spatial dims don't match exactly (non-even input sizes)
        if x.shape != skip.shape:
            diff_d = skip.shape[2] - x.shape[2]
            diff_h = skip.shape[3] - x.shape[3]
            diff_w = skip.shape[4] - x.shape[4]
            x = F.pad(x, [
                diff_w // 2, diff_w - diff_w // 2,
                diff_h // 2, diff_h - diff_h // 2,
                diff_d // 2, diff_d - diff_d // 2,
            ])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Full 3D UNet
# ---------------------------------------------------------------------------

class UNet3D(nn.Module):
    """
    3D UNet with Monte Carlo Dropout.

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g. 1 for single-modality MRA).
    num_classes : int
        Number of segmentation classes (including background).
    base_features : int
        Number of feature maps in the first encoder stage (doubled each stage).
    depth : int
        Number of encoder/decoder stages (default 4).
    dropout_rate : float
        Dropout probability applied throughout the network.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_features: int = 32,
        depth: int = 4,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.dropout_rate = dropout_rate
        self._mc_dropout_active = False

        # Encoder
        self.encoders = nn.ModuleList()
        ch_in = in_channels
        encoder_channels: List[int] = []
        for i in range(depth):
            ch_out = base_features * (2 ** i)
            self.encoders.append(EncoderBlock(ch_in, ch_out, dropout_rate))
            encoder_channels.append(ch_out)
            ch_in = ch_out

        # Bottleneck
        bottleneck_ch = base_features * (2 ** depth)
        self.bottleneck = ConvBlock3D(ch_in, bottleneck_ch, dropout_rate)

        # Decoder
        self.decoders = nn.ModuleList()
        ch_in = bottleneck_ch
        for i in reversed(range(depth)):
            ch_out = base_features * (2 ** i)
            skip_ch = encoder_channels[i]
            self.decoders.append(DecoderBlock(ch_in, skip_ch, ch_out, dropout_rate))
            ch_in = ch_out

        # Final 1x1x1 conv
        self.final_conv = nn.Conv3d(ch_in, num_classes, kernel_size=1)

        # Weight initialization
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Monte Carlo Dropout helpers
    # ------------------------------------------------------------------

    def enable_mc_dropout(self) -> None:
        """Keep dropout active even in eval mode (for MC sampling)."""
        self._mc_dropout_active = True
        for m in self.modules():
            if isinstance(m, (nn.Dropout3d, nn.Dropout)):
                m.train()

    def disable_mc_dropout(self) -> None:
        """Restore normal eval behaviour (dropout off during eval)."""
        self._mc_dropout_active = False
        self.eval()

    def train(self, mode: bool = True) -> "UNet3D":
        """Override train() so that MC Dropout persists across .eval() calls."""
        super().train(mode)
        if not mode and self._mc_dropout_active:
            # Re-enable dropout layers even though we're in eval mode
            for m in self.modules():
                if isinstance(m, (nn.Dropout3d, nn.Dropout)):
                    m.train()
        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return self.final_conv(x)


# ---------------------------------------------------------------------------
# Convenience constructor with common presets
# ---------------------------------------------------------------------------

def build_unet3d(
    in_channels: int = 1,
    num_classes: int = 2,
    preset: str = "base",
    dropout_rate: float = 0.1,
) -> UNet3D:
    """
    Build a 3D UNet with a named preset.

    Presets
    -------
    - ``base``  : 32 base features, depth 4  (~4.5 M params)
    - ``small`` : 16 base features, depth 3  (~0.3 M params, fits in 8 GB)
    - ``large`` : 64 base features, depth 4  (~18 M params, needs >=16 GB)
    """
    presets = {
        "base":  {"base_features": 32, "depth": 4},
        "small": {"base_features": 16, "depth": 3},
        "large": {"base_features": 64, "depth": 4},
    }
    if preset not in presets:
        raise ValueError(f"Unknown preset '{preset}'. Choose from {list(presets.keys())}")
    cfg = presets[preset]
    return UNet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        base_features=cfg["base_features"],
        depth=cfg["depth"],
        dropout_rate=dropout_rate,
    )
