from torch import nn
from .bottleneck_lstm import BottleneckLSTM


class Conv3x3ReLUNorm(nn.Module):
    def __init__(self, in_channels, out_channels, stride, norm='layer', num_groups=8):
        """
        Conv 3x3 + GroupNorm + LeakyReLU activation function module
        @param in_channels: number of input channels
        @param out_channels: number of output channels
        @param stride: stride of the convolution
        @param norm: normalisation to use (default: 'layer' uses GN with num_groups). Set to None to disable normalisation.
        @param num_groups: number of groups for GroupNorm (default: 8). Only used when norm is not None.
        """
        super(Conv3x3ReLUNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=1, kernel_size=3, stride=stride)
        self.acti = nn.LeakyReLU(inplace=True)

        if norm is not None:
            # Ensure num_groups divides out_channels
            effective_num_groups = min(num_groups, out_channels)
            while out_channels % effective_num_groups != 0:
                effective_num_groups -= 1
            self.norm = nn.GroupNorm(effective_num_groups, out_channels)
        else:
            self.norm = None

    def forward(self, x):
        """
        Forward pass Conv3x3ReLUNorm module
        @param x: input tensor with shape (B, Cin, H, W)
        @return: output tensor with shape (B, Cout, H/s, W/s)
        """
        x = self.conv(x)
        x = self.acti(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride, norm='layer', num_groups=8):
        """
        Modified MobileNetV2 Inverted Residual bottleneck layer with Group Norm and
        LeakyReLU activation function.
        @param in_channels: number of input channels
        @param out_channels: number of output channels
        @param expansion_factor: expansion factor for hidden dimension
        @param stride: stride of the convolution
        @param norm: normalisation to use (default: 'layer' uses GN). Set to None to disable normalisation.
        @param num_groups: number of groups for GroupNorm (default: 8)
        """
        super(InvertedResidual, self).__init__()
        hidden_dim = round(in_channels * expansion_factor)
        self.identity = stride == 1 and in_channels == out_channels

        # Helper function to create GroupNorm with proper num_groups
        def _get_norm(channels):
            if norm is None:
                return None
            effective_num_groups = min(num_groups, channels)
            while channels % effective_num_groups != 0:
                effective_num_groups -= 1
            return nn.GroupNorm(effective_num_groups, channels)

        if expansion_factor == 1:
            layers = [
                # dw
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3,
                          stride=1, padding=1, groups=hidden_dim),
                nn.LeakyReLU(inplace=True),
                # pw-linear
                nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=1,
                          stride=1, padding=0),
            ]

            # Add norms if needed
            if norm is not None:
                layers.insert(1, _get_norm(hidden_dim))
                layers.append(_get_norm(out_channels))

            self.conv = nn.Sequential(*layers)
        else:
            layers = [
                # pw
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1,
                          stride=1, padding=0),
                nn.LeakyReLU(inplace=True),
                # dw
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3,
                          stride=stride, padding=1, groups=hidden_dim),
                nn.LeakyReLU(inplace=True),
                # pw-linear
                nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=1,
                          stride=1, padding=0),
            ]

            # Add norms if needed
            if norm is not None:
                layers.insert(1, _get_norm(hidden_dim))
                layers.insert(4, _get_norm(hidden_dim))
                layers.append(_get_norm(out_channels))

            self.conv = nn.Sequential(*layers)

    def forward(self, x):
        """
        InvertedResidual bottleneck block forward pass
        @param x: input tensor with shape (B, Cin, H, W)
        @return: output tensor with shape (B, Cout, H/s, W/s)
        """
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)
