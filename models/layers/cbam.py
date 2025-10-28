import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        """
        Channel Attention Module
        @param channels: number of input channels
        @param reduction: reduction ratio for the bottleneck
        """
        super(ChannelAttention, self).__init__()

        # ⭐ 关键修复：确保 reduced_channels 至少为 1
        reduced_channels = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP (使用 1x1 卷积实现)
        self.fc1 = nn.Conv2d(channels, reduced_channels, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(reduced_channels, channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average pooling branch
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        # Max pooling branch
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        # Combine and apply sigmoid
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        """
        Spatial Attention Module
        @param kernel_size: kernel size for the convolutional layer (default: 7)
        """
        super(SpatialAttention, self).__init__()

        # 断言 kernel_size 必须是奇数
        assert kernel_size % 2 == 1, 'kernel size must be an odd number'
        assert kernel_size >= 1, 'kernel size must be at least 1'

        # 计算 padding
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 计算通道维度的平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # 拼接并通过卷积层
        attention_input = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv(attention_input))

        return x * attention_map


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel_size=7):
        """
        Convolutional Block Attention Module (CBAM)
        @param channels: number of input channels
        @param reduction: reduction ratio for channel attention
        @param spatial_kernel_size: kernel size for spatial attention
        """
        super(CBAM, self).__init__()

        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        # 先应用通道注意力
        x = self.channel_attention(x)
        # 再应用空间注意力
        x = self.spatial_attention(x)
        return x
