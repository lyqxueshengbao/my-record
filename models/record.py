import pytorch_lightning as pl
import torch
from torch import nn
import math  # <-- 新增导入

from .layers.inverted_residual import Conv3x3ReLUNorm, InvertedResidual
from .layers.bottleneck_lstm import BottleneckLSTM
from utils.models_utils import _make_divisible


def build_model(model_config, alpha=1.0, norm_type='layer'):
    layers = []
    for layer_name in model_config:
        layer = model_config[layer_name]
        layer_type = layer['type']
        in_channels = layer['in_channels']
        # If addition (for skip connections), then evaluate expression
        if isinstance(in_channels, str):
            in_channels = eval(in_channels)
        in_channels = _make_divisible(in_channels * alpha, 8.0)
        out_channels = layer['out_channels']
        # If addition (for skip connection), then evaluate expression
        if isinstance(out_channels, str):
            out_channels = eval(out_channels)
        out_channels = _make_divisible(out_channels * alpha, 8.0)
        stride = layer['stride']
        if layer['use_norm']:
            norm = norm_type
        else:
            norm = None
        print('Build layer {}...'.format(layer_name))
        if layer_type == 'conv':
            layers.append(Conv3x3ReLUNorm(in_channels=in_channels, out_channels=out_channels, stride=stride, norm=norm))
        elif layer_type == 'conv2d':
            kernel_size = layer['kernel_size']
            padding = layer['padding']
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding))
        elif layer_type == 'inverted_residual':
            expansion_factor = layer['expansion_factor']
            num_block = layer['num_block']
            layers.append(
                InvertedResidual(in_channels=in_channels, out_channels=out_channels, expansion_factor=expansion_factor,
                                 stride=stride, norm=norm))
            for i in range(1, num_block):
                layers.append(InvertedResidual(in_channels=out_channels, out_channels=out_channels,
                                               expansion_factor=expansion_factor,
                                               stride=stride, norm=norm))
        elif layer_type == 'bottleneck_lstm':
            num_block = layer['num_block']
            layers.append(BottleneckLSTM(input_channels=in_channels, hidden_channels=out_channels, norm=norm))
            for i in range(1, num_block):
                layers.append(BottleneckLSTM(input_channels=in_channels, hidden_channels=out_channels, norm=norm))
        elif layer_type == 'conv_transpose':
            kernel_size = layer['kernel_size']
            padding = layer['padding']
            output_padding = layer['output_padding']
            layers.append(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, output_padding=output_padding, stride=stride))

    return layers


class Record(nn.Module):
    def __init__(self, config, in_channels=8, norm='layer', n_class=3):
        """
        RECurrent Online object detectOR (RECORD) model class
        @param config: configuration file of the model
        @param alpha: expansion factor to modify the size of the model (default: 1.0)
        @param in_channels: number of input channels (default: 8)
        @param norm: type of normalisation (default: LayerNorm). Other normalisation are not supported yet.
        @param n_class: number of classes (default: 3)
        @param shallow: load a shallow version of RECORD (fewer channels in the decoder)
        """
        super(Record, self).__init__()
        self.encoder = RecordEncoder(config=config['encoder_config'], in_channels=in_channels, norm=norm)
        self.decoder = RecordDecoder(config=config['decoder_config'], n_class=n_class)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass RECORD model
        @param x: input tensor with shape (B, C, T, H, W) where T is the number of timesteps
        @return: ConfMap prediction of the last time step with shape (B, n_classes, H, W)
        """
        time_steps = x.shape[2]
        assert len(x.shape) == 5
        for t in range(time_steps):
            if t == 0:
                # Init hidden states if first time step of sliding window
                self.encoder.__init_hidden__()
            st_features_lstm1, st_features_lstm2, st_features_backbone = self.encoder(x[:, :, t])

        confmap_pred = self.decoder(st_features_lstm1, st_features_lstm2, st_features_backbone)
        return self.sigmoid(confmap_pred)


class RecordEncoder(nn.Module):
    def __init__(self, in_channels, config, norm='layer'):
        """
        RECurrent Online object detectOR (RECORD) features extractor.
        @param in_channels: number of input channels (default: 8)
        @param config: number of input channels per block
        @param norm: type of normalisation (default: LayerNorm). Other normalisation are not supported yet.
        """
        super(RecordEncoder, self).__init__()
        self.norm = norm
        # Set the number of input channels in the configuration file
        config['in_conv']['in_channels'] = in_channels

        # config_tmp = **config['in_conv'] --> dict with arguments

        # Input convolution (expands the number of input channels)
        self.in_conv = Conv3x3ReLUNorm(in_channels=config['in_conv']['in_channels'],
                                       out_channels=config['in_conv']['out_channels'],
                                       stride=config['in_conv']['stride'], norm=norm)

        # IR block 1 (acts as a bottleneck)
        self.ir_block1 = self._make_ir_block(in_channels=config['ir_block1']['in_channels'],
                                             out_channels=config['ir_block1']['out_channels'],
                                             num_block=config['ir_block1']['num_block'],
                                             expansion_factor=config['ir_block1']['expansion_factor'],
                                             stride=config['ir_block1']['stride'],
                                             use_norm=config['ir_block1']['use_norm'])

        # IR block 2 (extracts spatial features and decrease spatial dimension by a factor of 2)
        self.ir_block2 = self._make_ir_block(in_channels=config['ir_block2']['in_channels'],
                                             out_channels=config['ir_block2']['out_channels'],
                                             num_block=config['ir_block2']['num_block'],
                                             expansion_factor=config['ir_block2']['expansion_factor'],
                                             stride=config['ir_block2']['stride'],
                                             use_norm=config['ir_block2']['use_norm'])

        # Bottleneck LSTM 1 (extract spatial and temporal features)
        lstm_norm = None if not config['bottleneck_lstm1']['use_norm'] else self.norm
        self.bottleneck_lstm1 = BottleneckLSTM(input_channels=config['bottleneck_lstm1']['in_channels'],
                                               hidden_channels=config['bottleneck_lstm1']['out_channels'],
                                               norm=lstm_norm)

        # IR block 3 (extracts spatial features and decrease spatial dimension by a factor of 2)
        self.ir_block3 = self._make_ir_block(in_channels=config['ir_block3']['in_channels'],
                                             out_channels=config['ir_block3']['out_channels'],
                                             num_block=config['ir_block3']['num_block'],
                                             expansion_factor=config['ir_block3']['expansion_factor'],
                                             stride=config['ir_block3']['stride'],
                                             use_norm=config['ir_block3']['use_norm'])

        # Bottleneck LSTM 2 (extract spatial and temporal features)
        lstm_norm = None if not config['bottleneck_lstm2']['use_norm'] else self.norm
        self.bottleneck_lstm2 = BottleneckLSTM(input_channels=config['bottleneck_lstm2']['in_channels'],
                                               hidden_channels=config['bottleneck_lstm2']['out_channels'],
                                               norm=lstm_norm)

        # IR block 4 (extracts spatial features and decrease spatial dimension by a factor of 2)
        self.ir_block4 = self._make_ir_block(in_channels=config['ir_block4']['in_channels'],
                                             out_channels=config['ir_block4']['out_channels'],
                                             num_block=config['ir_block4']['num_block'],
                                             expansion_factor=config['ir_block4']['expansion_factor'],
                                             stride=config['ir_block4']['stride'],
                                             use_norm=config['ir_block4']['use_norm'])

    def forward(self, x):
        """
        @param x: input tensor for timestep t with shape (B, C, H, W)
        @return: list of features maps and hidden states (spatio-temporal features)
        """
        # Extracts spatial information
        x = self.in_conv(x)
        x = self.ir_block1(x)
        x = self.ir_block2(x)
        # Extract spatial and temporal representation at a first scale + update hidden states and cell states
        self.h_list[0], self.c_list[0] = self.bottleneck_lstm1(x, self.h_list[0], self.c_list[0])
        # Use last hidden state as input for the next convolutional layer
        st_features_lstm1 = self.h_list[0]
        x = self.ir_block3(st_features_lstm1)
        # Extract spatial and temporal representation at a second scale + update hidden states and cell states
        self.h_list[1], self.c_list[1] = self.bottleneck_lstm2(x, self.h_list[1], self.c_list[1])
        # Use last hidden state as input for the next convolutional layer
        st_features_lstm2 = self.h_list[1]
        st_features_backbone = self.ir_block4(st_features_lstm2)

        return st_features_backbone, st_features_lstm2, st_features_lstm1

    def _make_ir_block(self, in_channels, out_channels, num_block, expansion_factor, stride, use_norm):
        """
        Build an Inverted Residual bottleneck block
        @param in_channels: number of input channels
        @param out_channels: number of output channels
        @param num_block: number of IR layer in the block
        @param expansion_factor: expansion factor of each IR layer
        @param stride: stride of the first convolution
        @return a torch.nn.Sequential layer
        """
        if use_norm:
            norm = self.norm
        else:
            norm = None
        layers = [InvertedResidual(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                   expansion_factor=expansion_factor, norm=norm)]
        for i in range(1, num_block):
            layers.append(InvertedResidual(in_channels=out_channels, out_channels=out_channels, stride=1,
                                           expansion_factor=expansion_factor, norm=norm))
        return nn.Sequential(*layers)

    def __init_hidden__(self):
        """
        Init hidden states and cell states list
        """
        # List of 2 hidden/cell states as we use 2 Bottleneck LSTM. The initialisation is done inside a Bottleneck LSTM cell.
        self.h_list = [None, None]
        self.c_list = [None, None]


# vvvvvvvvvvvvvvvv 新增 ECALayer vvvvvvvvvvvvvvvv
class ECALayer(nn.Module):
    """Constructs an ECA-Net layer.
    Args:
        channels (int): Number of channels of the input feature map
        gamma (int): Parameter for kernel size calculation
        b (int): Parameter for kernel size calculation
    """

    def __init__(self, channels, gamma=2, b=1):
        super(ECALayer, self).__init__()
        # 根据通道数 C 动态计算卷积核大小 k
        # k = |log2(C) + b| / gamma
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1  # 确保 k_size 是奇数

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (b, c, h, w)
        b, c, h, w = x.size()

        # 全局平均池化: (b, c, h, w) -> (b, c, 1, 1)
        y = self.avg_pool(x)

        # 维度变换: (b, c, 1, 1) -> (b, c, 1) -> (b, 1, c)
        y = y.view(b, c, 1).transpose(-1, -2)

        # 1D 卷积: (b, 1, c) -> (b, 1, c)
        y = self.conv(y)

        # 维度变换: (b, 1, c) -> (b, c, 1) -> (b, c, 1, 1)
        y = y.transpose(-1, -2).view(b, c, 1, 1)

        # Sigmoid 激活
        y = self.sigmoid(y)

        # (b, c, h, w) * (b, c, 1, 1) -> (b, c, h, w)
        return x + (x * y.expand_as(x)) * 0.5


# ^^^^^^^^^^^^^^ 新增 ECALayer ^^^^^^^^^^^^^^


class RecordDecoder(nn.Module):
    def __init__(self, config, n_class, norm_decoder="layer"):
        """
        RECurrent Online object detectOR (RECORD) decoder.

        @param config: config list to build the decoder
        @param n_class: number of output class
        @param alpha: expansion factor to modify the size of the model (default: 1.0)
        @param round_nearest: Round the number of channels in each layer to be a multiple of this number
        @param norm_decoder: type of normalisation (default: LayerNorm). Other normalisation are not supported yet.
        """
        super(RecordDecoder, self).__init__()

        # Set the number of classes as the number of output channel of the last convolution
        config['conv_head2']['out_channels'] = n_class

        self.up_conv1 = nn.ConvTranspose2d(in_channels=config['conv_transpose1']['in_channels'],
                                           out_channels=config['conv_transpose1']['out_channels'],
                                           kernel_size=config['conv_transpose1']['kernel_size'],
                                           stride=config['conv_transpose1']['stride'],
                                           output_padding=config['conv_transpose1']['output_padding'],
                                           padding=config['conv_transpose1']['padding'])

        # vvvvvvvvvvvvvvvv 修改部分 vvvvvvvvvvvvvvvv
        # 评估上采样和第一个LSTM隐藏状态拼接后的通道数
        conv_norm = None if not config['conv_skip1']['use_norm'] else norm_decoder
        # 从配置中读取拼接后的通道数
        skip1_in_channels = config['conv_skip1']['in_channels']
        # 1. 新增ECA层，输入通道为拼接后的通道数
        self.eca_skip1 = ECALayer(channels=skip1_in_channels)
        # 2. InvertedResidual 层的输入通道不变
        self.conv_skip_connection1 = InvertedResidual(in_channels=skip1_in_channels,
                                                      out_channels=config['conv_skip1']['out_channels'],
                                                      expansion_factor=config['conv_skip1']['expansion_factor'],
                                                      stride=config['conv_skip1']['stride'],
                                                      norm=conv_norm)
        # ^^^^^^^^^^^^^^ 修改部分 ^^^^^^^^^^^^^^

        self.up_conv2 = nn.ConvTranspose2d(in_channels=config['conv_transpose2']['in_channels'],
                                           out_channels=config['conv_transpose2']['out_channels'],
                                           kernel_size=config['conv_transpose2']['kernel_size'],
                                           stride=config['conv_transpose2']['stride'],
                                           output_padding=config['conv_transpose2']['output_padding'],
                                           padding=config['conv_transpose2']['padding'])

        # vvvvvvvvvvvvvvvv 修改部分 vvvvvvvvvvvvvvvv
        # 评估上采样和第二个LSTM隐藏状态拼接后的通道数
        conv_norm = None if not config['conv_skip2']['use_norm'] else norm_decoder
        # 从配置中读取拼接后的通道数
        skip2_in_channels = config['conv_skip2']['in_channels']
        # 1. 新增ECA层，输入通道为拼接后的通道数
        self.eca_skip2 = ECALayer(channels=skip2_in_channels)
        # 2. InvertedResidual 层的输入通道不变
        self.conv_skip_connection2 = InvertedResidual(in_channels=skip2_in_channels,
                                                      out_channels=config['conv_skip2']['out_channels'],
                                                      expansion_factor=config['conv_skip2']['expansion_factor'],
                                                      stride=config['conv_skip2']['stride'],
                                                      norm=conv_norm)
        # ^^^^^^^^^^^^^^ 修改部分 ^^^^^^^^^^^^^^

        self.up_conv3 = nn.ConvTranspose2d(in_channels=config['conv_transpose3']['in_channels'],
                                           out_channels=config['conv_transpose3']['out_channels'],
                                           kernel_size=config['conv_transpose3']['kernel_size'],
                                           stride=config['conv_transpose3']['stride'],
                                           output_padding=config['conv_transpose3']['output_padding'],
                                           padding=config['conv_transpose3']['padding'])

        conv_norm = None if not config['conv_skip3']['use_norm'] else norm_decoder
        self.conv_skip_connection3 = InvertedResidual(in_channels=config['conv_skip3']['in_channels'],
                                                      out_channels=config['conv_skip3']['out_channels'],
                                                      expansion_factor=config['conv_skip3']['expansion_factor'],
                                                      stride=config['conv_skip3']['stride'],
                                                      norm=conv_norm)

        conv_norm = None if not config['conv_head1']['use_norm'] else norm_decoder
        self.conv_head1 = Conv3x3ReLUNorm(in_channels=config['conv_head1']['in_channels'],
                                          out_channels=config['conv_head1']['out_channels'],
                                          stride=config['conv_head1']['stride'], norm=conv_norm)
        self.conv_head2 = nn.Conv2d(in_channels=config['conv_head2']['in_channels'],
                                    out_channels=config['conv_head2']['out_channels'],
                                    kernel_size=config['conv_head2']['kernel_size'],
                                    stride=config['conv_head2']['stride'], padding=config['conv_head2']['padding'])

    def forward(self, st_features_backbone, st_features_lstm2, st_features_lstm1):
        """
        Forward pass RECORD decoder
        @param st_features_backbone: Last features map
        @param st_features_lstm2: Spatio-temporal features map from the second Bottleneck LSTM
        @param st_features_lstm1: Spatio-temporal features map from the first Bottleneck LSTM
        @return: ConfMap prediction (B, n_class, H, W)
        """
        # Spatio-temporal skip connection 1
        # 1. 拼接 (Concatenate)
        skip_connection1_out = torch.cat((self.up_conv1(st_features_backbone), st_features_lstm2), dim=1)
        # 2. 应用ECA注意力 (新增)
        skip_connection1_out = self.eca_skip1(skip_connection1_out)
        # 3. 卷积 (IR Block)
        x = self.conv_skip_connection1(skip_connection1_out)

        # Spatio-temporal skip connection 2
        # 1. 拼接 (Concatenate)
        skip_connection2_out = torch.cat((self.up_conv2(x), st_features_lstm1), dim=1)
        # 2. 应用ECA注意力 (新增)
        skip_connection2_out = self.eca_skip2(skip_connection2_out)
        # 3. 卷积 (IR Block)
        x = self.conv_skip_connection2(skip_connection2_out)

        # 保持不变
        x = self.up_conv3(x)
        x = self.conv_skip_connection3(x)

        x = self.conv_head1(x)
        x = self.conv_head2(x)
        return x