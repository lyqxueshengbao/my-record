"""
RMSNorm 实现
"""
import torch
import torch.nn as nn


class RMSNorm2d(nn.Module):
    """
    用于 (B, C, H, W) 张量的 RMSNorm, 行为等效于 nn.GroupNorm(1, C).
    在 (C, H, W) 维度上进行归一化.
    """

    def __init__(self, num_channels, eps=1e-6):
        super(RMSNorm2d, self).__init__()
        self.eps = eps
        # 增益参数, 形状 (1, C, 1, 1) 以便广播
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))

    def forward(self, x):
        """x shape: (B, C, H, W)"""
        # 原来的代码（显存占用高）：
        # variance = x.pow(2).mean(dim=(1, 2, 3), keepdim=True)
        # hidden_states = x * torch.rsqrt(variance + self.eps)
        # return self.weight * hidden_states

        # 优化后（显存占用低）：
        variance = (x * x).mean(dim=(1, 2, 3), keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight