import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


class h_swish(nn.Module):
    """
    Hard-Swish activation function.

    This is the efficient approximation of the Swish activation used in
    MobileNetV3 and lightweight attention modules.

    Forward pass:
        h-swish(x) = x * ReLU6(x + 3) / 6

    Returns:
        Tensor: Activated feature map.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Apply h-swish activation."""
        return x * F.relu6(x + 3, inplace=True) / 6


class CoordAtt(nn.Module):
    """
    Coordinate Attention (CoordAtt) block compatible with YOLOv8.

    This module enhances spatial attention by separately encoding height-wise
    and width-wise context. Unlike SE or CBAM, CoordAtt preserves positional
    information and is lightweight, making it suitable for real-time detectors.

    Args:
        c1 (int): Input channel dimension (kept for YOLO config compatibility).
        c2 (int): Output channel dimension (unused; output = input).
        reduction (int, optional): Channel reduction ratio. Default is 32.

    Notes:
        - Internal layers are built lazily during the first forward pass
          because channel count is not always known during __init__ in YOLO.
        - Input and output shapes are identical: (B, C, H, W)
    """

    def __init__(self, c1: int, c2: int, reduction: int = 32):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.reduction = reduction
        self.built = False

    def _build(self, channels: int) -> None:
        """
        Lazily create convolution layers based on runtime channel count.

        Args:
            channels (int): Number of input channels (C).

        Builds:
            - Height & width pooling paths
            - Shared 1×1 transform: Conv → BN → h_swish
            - Separate projections conv_h, conv_w
        """
        mip = max(8, channels // self.reduction)

        # Spatial pooling along width and height
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # Shared transform
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        # Separate projections
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, bias=False)

        self.built = True

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of CoordAtt.

        Steps:
            1. Pool feature map along H and W separately.
            2. Concatenate pooled maps → shared Conv-BN-h_swish.
            3. Split back into height & width attention maps.
            4. Apply sigmoid gates and reweight input.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            Tensor: Output tensor of shape (B, C, H, W)
        """
        if not self.built:
            self._build(x.size(1))

        # Pool along height and width
        x_h: Tensor = self.pool_h(x)                     # (B, C, H, 1)
        x_w: Tensor = self.pool_w(x).permute(0, 1, 3, 2) # (B, C, 1, W)

        # Merge & transform
        y: Tensor = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))

        # Split back
        x_h_att, x_w_att = torch.split(y, [x.size(2), x.size(3)], dim=2)
        x_w_att = x_w_att.permute(0, 1, 3, 2)

        # Apply attention
        out: Tensor = x * self.conv_h(x_h_att).sigmoid() * self.conv_w(x_w_att).sigmoid()
        return out
