import torch
import torch.nn as nn
import math


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original GhostNet paper's implementation.
    It ensures that all layers have a channel number that is divisible by 8
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _get_norm_op(num_channels, norm='layer'):
    """
    Crea un'operazione di normalizzazione basata sulla stringa 'norm'.
    Utilizza GroupNorm(1, ...) per 'layer' come nel tuo inverted_residual.py
    """
    if norm == 'layer':
        return nn.GroupNorm(1, num_channels)
    elif norm is None:
        return nn.Identity()
    else:
        # Fallback o errore, ma per ora seguiamo la tua convenzione
        return nn.GroupNorm(1, num_channels)


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_kernel_size=3, stride=1, relu=True, norm='layer'):
        """
        Ghost Module
        @param inp: number of input channels
        @param oup: number of output channels
        @param kernel_size: kernel size of the primary convolution
        @param ratio: ratio to define the number of intrinsic features
        @param dw_kernel_size: kernel size of the cheap operation (depthwise conv)
        @param stride: stride of the cheap operation
        @param relu: whether to use ReLU activation
        @param norm: normalization type
        """
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        # Primary convolution
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            _get_norm_op(init_channels, norm),
            nn.LeakyReLU(inplace=True) if relu else nn.Identity(),
        )

        # Cheap operation (depthwise convolution)
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_kernel_size, 1, dw_kernel_size // 2, groups=init_channels,
                      bias=False),
            _get_norm_op(new_channels, norm),
            nn.LeakyReLU(inplace=True) if relu else nn.Identity(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, dw_kernel_size, stride, norm='layer'):
        """
        Ghost Bottleneck layer
        @param in_channels: number of input channels
        @param hidden_dim: number of hidden channels (expansion)
        @param out_channels: number of output channels
        @param dw_kernel_size: kernel size of the depthwise convolution
        @param stride: stride
        @param norm: normalization type
        """
        super(GhostBottleneck, self).__init__()
        self.identity = stride == 1 and in_channels == out_channels

        # Ghost Module 1 (Pointwise expansion)
        self.ghost1 = GhostModule(in_channels, hidden_dim, kernel_size=1, relu=True, norm=norm)

        # Depthwise Convolution (if stride=2)
        if stride == 2:
            self.dw_conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, dw_kernel_size, stride=stride, padding=dw_kernel_size // 2,
                          groups=hidden_dim, bias=False),
                _get_norm_op(hidden_dim, norm)
            )
        else:
            self.dw_conv = nn.Identity()

        # Ghost Module 2 (Pointwise linear reduction)
        self.ghost2 = GhostModule(hidden_dim, out_channels, kernel_size=1, relu=False, norm=norm)

        # Shortcut connection
        if self.identity:
            self.shortcut = nn.Identity()
        else:
            # If stride=2, downsample the shortcut connection
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, dw_kernel_size, stride=stride, padding=dw_kernel_size // 2,
                          groups=in_channels, bias=False),
                _get_norm_op(in_channels, norm),
                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
                _get_norm_op(out_channels, norm),
            )

    def forward(self, x):
        x_residual = x

        x = self.ghost1(x)
        x = self.dw_conv(x)
        x = self.ghost2(x)

        if self.identity:
            return x + x_residual
        else:
            return x + self.shortcut(x_residual)