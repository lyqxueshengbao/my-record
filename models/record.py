#
# 完整文件: models/record.py
# (此版本包含 Hybrid Norm 架构 + ECA in Decoder)
#
import pytorch_lightning as pl
import torch
from torch import nn
import math  # <-- ECA 需要的导入

from .layers.inverted_residual import Conv3x3ReLUNorm, InvertedResidual
from .layers.bottleneck_lstm import BottleneckLSTM
from utils.models_utils import _make_divisible


# build_model 函数在这个文件中没有被 Record 类使用，所以我们跳过它。
# (如果你也用到了它, 记得去修改它)


class Record(nn.Module):
    def __init__(self, config, in_channels=8, norm='layer', n_class=3):
        """
        RECurrent Online object detectOR (RECORD) model class
        *** 最终修改版 (Hybrid Norm + ECA) ***

        @param config: configuration file of the model
        @param alpha: expansion factor to modify the size of the model (default: 1.0)
        @param in_channels: number of input channels (default: 8)
        @param norm: type of normalisation FOR RECURRENT/DECODER parts (default: 'layer' for GN(1)).
                     The STEM part (pre-LSTM) is hardcoded to 'bn' (BatchNorm2d).
        @param n_class: number of classes (default: 3)
        @param shallow: load a shallow version of RECORD (fewer channels in the decoder)
        """
        super(Record, self).__init__()

        # 混合归一化：
        # norm_stem: 'bn' (BatchNorm) - 用于 LSTM 之前，处理 (B*T) 批次
        # norm_recurrent: 'layer' (GroupNorm(1)) - 用于 LSTM 之后和 LSTM 内部
        self.encoder = RecordEncoder(config=config['encoder_config'],
                                     in_channels=in_channels,
                                     norm_stem='bn',
                                     norm_recurrent=norm)

        # Decoder 必须使用 'layer' (GN(1)) 因为它只在最后一个时间步运行 (批次为 B)
        self.decoder = RecordDecoder(config=config['decoder_config'],
                                     n_class=n_class,
                                     norm_decoder=norm)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass RECORD model (MODIFIED FOR HYBRID NORM)
        *** 这是快速的 forward 版本 ***

        @param x: input tensor with shape (B, C, T, H, W) where T is the number of timesteps
        @return: ConfMap prediction of the last time step with shape (B, n_classes, H, W)
        """
        B, C, T, H, W = x.shape
        assert len(x.shape) == 5

        # 1. Reshape for Stem (BN part)
        # (B, C, T, H, W) -> (B, T, C, H, W) -> (B*T, C, H, W)
        x_reshaped = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)

        # 2. Run Stem (BN part) - 在循环外执行 1 次
        # self.encoder.train() vs eval() mode will be handled by pytorch_lightning
        # stem_features shape: (B*T, C_feat, H_feat, W_feat)
        stem_features = self.encoder.forward_stem(x_reshaped)

        # 3. Reshape for Recurrent (GN/LayerNorm part)
        # (B*T, C_feat, H_feat, W_feat) -> (B, T, C_feat, H_feat, W_feat)
        _, C_feat, H_feat, W_feat = stem_features.shape
        recurrent_input = stem_features.view(B, T, C_feat, H_feat, W_feat)

        # 4. Initialize hidden states
        # (这会设置 self.encoder.h_list = [None, None], self.encoder.c_list = [None, None])
        self.encoder.__init_hidden__()
        h_list = self.encoder.h_list
        c_list = self.encoder.c_list

        # 5. Loop over time (Recurrent part) - 只执行循环必要的
        for t in range(T):
            # Get features for this timestep
            x_t = recurrent_input[:, t, ...]

            # (st_features_backbone,
            #  st_features_lstm2,
            #  st_features_lstm1) 存储最后一个时间步的输出
            (st_features_backbone,
             st_features_lstm2,
             st_features_lstm1), h_list, c_list = self.encoder.forward_recurrent_step(x_t, h_list, c_list)

        # Decoder 仅使用最后一个时间步的特征
        # 6. Run Decoder
        confmap_pred = self.decoder(st_features_backbone, st_features_lstm2, st_features_lstm1)
        return self.sigmoid(confmap_pred)


class RecordEncoder(nn.Module):
    def __init__(self, in_channels, config, norm_stem='bn', norm_recurrent='layer'):
        """
        RECurrent Online object detectOR (RECORD) features extractor.
        MODIFIED FOR HYBRID NORM

        @param in_channels: number of input channels (default: 8)
        @param config: number of input channels per block
        @param norm_stem: norm type for pre-LSTM layers (default: 'bn')
        @param norm_recurrent: norm type for post-LSTM layers (default: 'layer')
        """
        super(RecordEncoder, self).__init__()
        self.norm_recurrent = norm_recurrent
        # Set the number of input channels in the configuration file
        config['in_conv']['in_channels'] = in_channels

        # --- STEM (BatchNorm) ---
        # Input convolution (expands the number of input channels)
        self.in_conv = Conv3x3ReLUNorm(in_channels=config['in_conv']['in_channels'],
                                       out_channels=config['in_conv']['out_channels'],
                                       stride=config['in_conv']['stride'],
                                       norm=norm_stem)  # 使用 norm_stem

        # IR block 1 (acts as a bottleneck)
        self.ir_block1 = self._make_ir_block(in_channels=config['ir_block1']['in_channels'],
                                             out_channels=config['ir_block1']['out_channels'],
                                             num_block=config['ir_block1']['num_block'],
                                             expansion_factor=config['ir_block1']['expansion_factor'],
                                             stride=config['ir_block1']['stride'],
                                             use_norm=config['ir_block1']['use_norm'],
                                             norm_type=norm_stem)  # 使用 norm_stem

        # IR block 2 (extracts spatial features and decrease spatial dimension by a factor of 2)
        self.ir_block2 = self._make_ir_block(in_channels=config['ir_block2']['in_channels'],
                                             out_channels=config['ir_block2']['out_channels'],
                                             num_block=config['ir_block2']['num_block'],
                                             expansion_factor=config['ir_block2']['expansion_factor'],
                                             stride=config['ir_block2']['stride'],
                                             use_norm=config['ir_block2']['use_norm'],
                                             norm_type=norm_stem)  # 使用 norm_stem

        # --- RECURRENT (LayerNorm/GN(1)) ---
        # Bottleneck LSTM 1 (extract spatial and temporal features)
        lstm_norm = None if not config['bottleneck_lstm1']['use_norm'] else self.norm_recurrent
        self.bottleneck_lstm1 = BottleneckLSTM(input_channels=config['bottleneck_lstm1']['in_channels'],
                                               hidden_channels=config['bottleneck_lstm1']['out_channels'],
                                               norm=lstm_norm)

        # IR block 3 (extracts spatial features and decrease spatial dimension by a factor of 2)
        self.ir_block3 = self._make_ir_block(in_channels=config['ir_block3']['in_channels'],
                                             out_channels=config['ir_block3']['out_channels'],
                                             num_block=config['ir_block3']['num_block'],
                                             expansion_factor=config['ir_block3']['expansion_factor'],
                                             stride=config['ir_block3']['stride'],
                                             use_norm=config['ir_block3']['use_norm'],
                                             norm_type=self.norm_recurrent)  # 使用 norm_recurrent

        # Bottleneck LSTM 2 (extract spatial and temporal features)
        lstm_norm = None if not config['bottleneck_lstm2']['use_norm'] else self.norm_recurrent
        self.bottleneck_lstm2 = BottleneckLSTM(input_channels=config['bottleneck_lstm2']['in_channels'],
                                               hidden_channels=config['bottleneck_lstm2']['out_channels'],
                                               norm=lstm_norm)

        # IR block 4 (extracts spatial features and decrease spatial dimension by a factor of 2)
        self.ir_block4 = self._make_ir_block(in_channels=config['ir_block4']['in_channels'],
                                             out_channels=config['ir_block4']['out_channels'],
                                             num_block=config['ir_block4']['num_block'],
                                             expansion_factor=config['ir_block4']['expansion_factor'],
                                             stride=config['ir_block4']['stride'],
                                             use_norm=config['ir_block4']['use_norm'],
                                             norm_type=self.norm_recurrent)  # 使用 norm_recurrent

        # --- Create Stem Sequential module ---
        self.stem = nn.Sequential(self.in_conv, self.ir_block1, self.ir_block2)

    def forward_stem(self, x):
        """
        Forward pass for the STEM part (BN)
        @param x: input tensor with shape (B*T, C, H, W)
        @return: features tensor with shape (B*T, C_feat, H_feat, W_feat)
        """
        return self.stem(x)

    def forward_recurrent_step(self, x, h_list, c_list):
        """
        Forward pass for ONE TIMESTEP of the RECURRENT part (GN/LayerNorm)
        @param x: input tensor for timestep t with shape (B, C_feat, H_feat, W_feat)
        @param h_list: list of hidden states [h1, h2]
        @param c_list: list of cell states [c1, c2]
        @return: tuple of (features, new_h_list, new_c_list)
                    features: (st_features_backbone, st_features_lstm2, st_features_lstm1)
                    new_h_list: [new_h1, new_h2]
                    new_c_list: [new_c1, new_c2]
        """
        new_h_list = [None, None]
        new_c_list = [None, None]

        # Extract spatial and temporal representation at a first scale
        # h_list[0] 和 c_list[0] 初始为 None, BottleneckLSTM 会自动初始化
        new_h_list[0], new_c_list[0] = self.bottleneck_lstm1(x, h_list[0], c_list[0])
        st_features_lstm1 = new_h_list[0]

        x_rec = self.ir_block3(st_features_lstm1)

        # Extract spatial and temporal representation at a second scale
        new_h_list[1], new_c_list[1] = self.bottleneck_lstm2(x_rec, h_list[1], c_list[1])
        st_features_lstm2 = new_h_list[1]

        st_features_backbone = self.ir_block4(st_features_lstm2)

        output_features = (st_features_backbone, st_features_lstm2, st_features_lstm1)

        return output_features, new_h_list, new_c_list

    def _make_ir_block(self, in_channels, out_channels, num_block, expansion_factor, stride, use_norm, norm_type):
        """
        Build an Inverted Residual bottleneck block
        @param in_channels: number of input channels
        @param out_channels: number of output channels
        @param num_block: number of IR layer in the block
        @param expansion_factor: expansion factor of each IR layer
        @param stride: stride of the first convolution
        @param use_norm: whether to use norm or not
        @param norm_type: 'bn' or 'layer'
        @return a torch.nn.Sequential layer
        """
        if use_norm:
            norm = norm_type
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
        return x * y.expand_as(x)


# ^^^^^^^^^^^^^^ 新增 ECALayer ^^^^^^^^^^^^^^


class RecordDecoder(nn.Module):
    def __init__(self, config, n_class, norm_decoder="layer"):
        """
        RECurrent Online object detectOR (RECORD) decoder.
        *** 最终修改版 (Hybrid Norm + ECA) ***

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

        # vvvvvvvvvvvvvvvv ECA 修改 vvvvvvvvvvvvvvvv
        # Evaluate the sum of channels of the # channels of up_conv1 and # channels of the last hidden states of second
        # LSTM for the skip connection
        conv_norm = None if not config['conv_skip1']['use_norm'] else norm_decoder
        # 从配置中读取拼接后的通道数
        skip1_in_channels = config['conv_skip1']['in_channels']
        # 1. 新增ECA层
        self.eca_skip1 = ECALayer(channels=skip1_in_channels)
        # 2. InvertedResidual 层 (保持不变, 它将使用 'layer' norm)
        self.conv_skip_connection1 = InvertedResidual(in_channels=skip1_in_channels,
                                                      out_channels=config['conv_skip1']['out_channels'],
                                                      expansion_factor=config['conv_skip1']['expansion_factor'],
                                                      stride=config['conv_skip1']['stride'],
                                                      norm=conv_norm)
        # ^^^^^^^^^^^^^^ ECA 修改 ^^^^^^^^^^^^^^

        self.up_conv2 = nn.ConvTranspose2d(in_channels=config['conv_transpose2']['in_channels'],
                                           out_channels=config['conv_transpose2']['out_channels'],
                                           kernel_size=config['conv_transpose2']['kernel_size'],
                                           stride=config['conv_transpose2']['stride'],
                                           output_padding=config['conv_transpose2']['output_padding'],
                                           padding=config['conv_transpose2']['padding'])

        # vvvvvvvvvvvvvvvv ECA 修改 vvvvvvvvvvvvvvvv
        # Evaluate the sum of channels of the # channels of up_conv2 and # channels of the last hidden states of first
        # LSTM for the skip connection
        conv_norm = None if not config['conv_skip2']['use_norm'] else norm_decoder
        # 从配置中读取拼接后的通道数
        skip2_in_channels = config['conv_skip2']['in_channels']
        # 1. 新增ECA层
        self.eca_skip2 = ECALayer(channels=skip2_in_channels)
        # 2. InvertedResidual 层 (保持不变, 它将使用 'layer' norm)
        self.conv_skip_connection2 = InvertedResidual(in_channels=skip2_in_channels,
                                                      out_channels=config['conv_skip2']['out_channels'],
                                                      expansion_factor=config['conv_skip2']['expansion_factor'],
                                                      stride=config['conv_skip2']['stride'],
                                                      norm=conv_norm)
        # ^^^^^^^^^^^^^^ ECA 修改 ^^^^^^^^^^^^^^

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
        *** 最终修改版 (Hybrid Norm + ECA) ***

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