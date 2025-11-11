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

    # ============================================================
    # ⬇️ 关键修改：替换为更高效的 forward 方法
    # ============================================================
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (B, C, H, W)
        """
        # 原来的实现:
        # variance = x.pow(2).mean(dim=(1, 2, 3), keepdim=True)
        # hidden_states = x * torch.rsqrt(variance + self.eps)
        # return self.weight * hidden_states

        # 更高效的实现 (来自我们的分析):
        # (B, C, H, W) -> (B, 1, 1, 1)
        norm = x.norm(2, dim=(1, 2, 3), keepdim=True)
        # 归一化因子
        rms = norm * (x.shape[1] * x.shape[2] * x.shape[3]) ** (-0.5)
        # 归一化
        return self.weight * x / (rms + self.eps)
    # ============================================================
    # ⬆️ 关键修改结束
    # ============================================================