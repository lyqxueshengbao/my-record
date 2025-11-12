# rodnet/core/models/layers/rms_norm.py

import torch
import torch.nn as nn


class RMSNorm2d(nn.Module):
    """
    RMSNorm for (B, C, H, W) tensors
    Compatible with torch.compile
    """

    def __init__(self, num_channels, eps=1e-6):
        super(RMSNorm2d, self).__init__()
        self.eps = eps
        self.num_channels = num_channels
        self.weight = nn.Parameter(torch.ones(num_channels))

    def forward(self, x):
        """x shape: (B, C, H, W)"""
        # 在 (B, H, W) 维度计算 RMS，每个通道独立
        variance = x.pow(2).mean(dim=(0, 2, 3), keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight.view(1, -1, 1, 1)

    def extra_repr(self):
        return f'{self.num_channels}, eps={self.eps}'
