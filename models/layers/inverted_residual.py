from torch import nn
from .bottleneck_lstm import BottleneckLSTM
from .cbam import CBAM


class Conv3x3ReLUNorm(nn.Module):
    def __init__(self, in_channels, out_channels, stride, norm='layer'):
        """
        Conv 3x3 + LayerNorm + LeakyReLU activation function module
        @param in_channels: number of input channels
        @param out_channels: number of output channels
        @param stride: stride of the convolution
        @param norm: normalisation to use (default: LayerNorm). Set to None to disable normalisation.
        """
        super(Conv3x3ReLUNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=1, kernel_size=3, stride=stride)
        self.acti = nn.LeakyReLU(inplace=True)

        if norm is not None:
            self.norm = nn.GroupNorm(1, out_channels)
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
    def __init__(self, in_channels, out_channels, expansion_factor, stride, norm='layer',
                 use_cbam=False, cbam_reduction=16, cbam_kernel_size=7):
        """
        Modified MobileNetV2 Inverted Residual bottleneck layer with layer norm and
        LeakyReLU activation function. Optionally includes CBAM attention module.
        @param in_channels: number of input channels
        @param out_channels: number of output channels
        @param expansion_factor: round the number of channels in each layer to be a multiple of this number
        @param stride: stride of the convolution
        @param norm: normalisation to use (default: LayerNorm). Set to None to disable normalisation.
        @param use_cbam: whether to use CBAM attention module (default: False)
        @param cbam_reduction: reduction ratio for CBAM channel attention (default: 16)
        @param cbam_kernel_size: kernel size for CBAM spatial attention (default: 7)
        """
        super(InvertedResidual, self).__init__()
        hidden_dim = round(in_channels * expansion_factor)
        self.identity = stride == 1 and in_channels == out_channels
        self.use_cbam = use_cbam

        if norm is not None:
            n_group = 1
            norm_op = nn.GroupNorm
        else:
            norm_op = None

        if expansion_factor == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3,
                          stride=1, padding=1, groups=hidden_dim),

                norm_op(num_groups=n_group, num_channels=hidden_dim),
                nn.LeakyReLU(inplace=True),
                # pw-linear
                nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=1,
                          stride=1, padding=0),
                norm_op(num_groups=n_group, num_channels=out_channels)
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1,
                          stride=1, padding=0),
                norm_op(num_groups=n_group, num_channels=hidden_dim),
                nn.LeakyReLU(inplace=True),
                # dw
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3,
                          stride=stride, padding=1, groups=hidden_dim),
                norm_op(num_groups=n_group, num_channels=hidden_dim),
                nn.LeakyReLU(inplace=True),
                # pw-linear
                nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=1,
                          stride=1, padding=0),
                norm_op(num_groups=n_group, num_channels=out_channels)
            )

        # Add CBAM attention module if specified
        if self.use_cbam:
            self.cbam = CBAM(out_channels, reduction_ratio=cbam_reduction, kernel_size=cbam_kernel_size)

    def forward(self, x):
        """
        InvertedResidual bottleneck block forward pass with optional CBAM attention
        @param x: input tensor with shape (B, Cin, H, W)
        @return: output tensor with shape (B, Cout, H/s, W/s)
        """
        if self.identity:
            out = x + self.conv(x)
        else:
            out = self.conv(x)

        # Apply CBAM attention if enabled
        if self.use_cbam:
            out = self.cbam(out)

        return out
