import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module
    Applies channel-wise attention using both average pooling and max pooling
    """

    def __init__(self, in_channels, reduction_ratio=16):
        """
        @param in_channels: number of input channels
        @param reduction_ratio: reduction ratio for the MLP (default: 16)
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        @param x: input tensor with shape (B, C, H, W)
        @return: channel attention map with shape (B, C, 1, 1)
        """
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    Applies spatial attention using both average pooling and max pooling along the channel axis
    """

    def __init__(self, kernel_size=7):
        """
        @param kernel_size: kernel size for the convolutional layer (default: 7)
        """
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        @param x: input tensor with shape (B, C, H, W)
        @return: spatial attention map with shape (B, 1, H, W)
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    Combines both channel and spatial attention mechanisms
    Reference: "CBAM: Convolutional Block Attention Module" (ECCV 2018)
    """

    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        """
        @param in_channels: number of input channels
        @param reduction_ratio: reduction ratio for channel attention (default: 16)
        @param kernel_size: kernel size for spatial attention (default: 7)
        """
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """
        Apply CBAM attention to input tensor
        @param x: input tensor with shape (B, C, H, W)
        @return: attention-refined tensor with shape (B, C, H, W)
        """
        # Channel attention
        x = x * self.channel_attention(x)
        # Spatial attention
        x = x * self.spatial_attention(x)
        return x
