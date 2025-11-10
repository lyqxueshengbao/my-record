from torch import nn
# from .bottleneck_lstm import BottleneckLSTM  # <--- 移除这个未使用的导入
from .rms_norm import RMSNorm2d  # <--- 添加 RMSNorm 导入
import torch  # <--- 添加 torch 导入


# ++++++ 添加辅助函数 ++++++
def _get_norm_layer(norm_type, num_channels):
    """根据 'norm_type' 字符串返回一个归一化层实例"""
    if norm_type == 'layer':
        return nn.GroupNorm(1, num_channels)
    elif norm_type == 'rms':
        return RMSNorm2d(num_channels)
    elif norm_type is None:
        return nn.Identity()  # 返回一个空操作层
    else:
        raise ValueError(f"未知的 norm 类型: {norm_type}")


# +++++++++++++++++++++++++


class Conv3x3ReLUNorm(nn.Module):
    def __init__(self, in_channels, out_channels, stride, norm='rms'):  # <-- 默认改为 'rms'
        """
        Conv 3x3 + LayerNorm/RMSNorm + LeakyReLU activation function module
        @param in_channels: number of input channels
        @param out_channels: number of output channels
        @param stride: stride of the convolution
        @param norm: normalisation to use (default: RMSNorm). Set to None to disable normalisation.
        """
        super(Conv3x3ReLUNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=1, kernel_size=3, stride=stride)
        self.acti = nn.LeakyReLU(inplace=True)

        # --- 使用辅助函数 ---
        if norm is not None:
            self.norm = _get_norm_layer(norm, out_channels)
        else:
            self.norm = None
        # ---------------------

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
    def __init__(self, in_channels, out_channels, expansion_factor, stride, norm='rms'):  # <-- 默认改为 'rms'
        """
        Modified MobileNetV2 Inverted Residual bottleneck layer with layer norm/RMSNorm and
        LeakyReLU activation function.
        @param in_channels: number of input channels
        @param out_channels: number of output channels
        @param expansion_factor: round the number of channels in each layer to be a multiple of this number
        @param stride: stride of the convolution
        @param norm: normalisation to use (default: RMSNorm). Set to None to disable normalisation.
        """
        super(InvertedResidual, self).__init__()
        hidden_dim = round(in_channels * expansion_factor)
        self.identity = stride == 1 and in_channels == out_channels

        # --- 移除旧的 norm_op 逻辑 ---
        # ---------------------------

        if expansion_factor == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3,
                          stride=1, padding=1, groups=hidden_dim),
                # --- 使用辅助函数 ---
                _get_norm_layer(norm, hidden_dim),
                # ---------------------
                nn.LeakyReLU(inplace=True),
                # pw-linear
                nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=1,
                          stride=1, padding=0),
                # --- 使用辅助函数 ---
                _get_norm_layer(norm, out_channels)
                # ---------------------
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1,
                          stride=1, padding=0),
                # --- 使用辅助函数 ---
                _get_norm_layer(norm, hidden_dim),
                # ---------------------
                nn.LeakyReLU(inplace=True),
                # dw
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3,
                          stride=stride, padding=1, groups=hidden_dim),
                # --- 使用辅助函数 ---
                _get_norm_layer(norm, hidden_dim),
                # ---------------------
                nn.LeakyReLU(inplace=True),
                # pw-linear
                nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=1,
                          stride=1, padding=0),
                # --- 使用辅助函数 ---
                _get_norm_layer(norm, out_channels)
                # ---------------------
            )

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