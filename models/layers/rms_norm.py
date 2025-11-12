# rodnet/core/models/layers/rms_norm.py

import torch
import torch.nn as nn

# 安装 Apex
# pip install apex  # 或从源码安装

from apex.normalization import FusedRMSNorm


class RMSNorm2d(nn.Module):
    """
    使用 Apex 融合内核的高效 RMSNorm
    """

    def __init__(self, num_channels, eps=1e-6):
        super(RMSNorm2d, self).__init__()
        self.num_channels = num_channels
        # Apex 的 FusedRMSNorm 需要 1D 输入
        self.norm = FusedRMSNorm(num_channels, eps=eps)

    def forward(self, x):
        """
        x shape: (B, C, H, W)
        """
        B, C, H, W = x.shape

        # 重塑为 (B*H*W, C) 进行归一化
        x_reshaped = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x_reshaped = x_reshaped.view(-1, C)  # (B*H*W, C)

        # 使用融合 RMSNorm
        x_norm = self.norm(x_reshaped)  # (B*H*W, C)

        # 重塑回 (B, C, H, W)
        x_norm = x_norm.view(B, H, W, C)
        x_norm = x_norm.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        return x_norm

