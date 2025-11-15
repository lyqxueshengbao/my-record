from torch import nn
from .record import RecordEncoder, RecordDecoder
from .layers.inverted_residual import Conv3x3ReLUNorm, InvertedResidual
from .layers.bottleneck_lstm import BottleneckLSTM
from utils.models_utils import _make_divisible

"""
Online Inference version of RECORD model with Hybrid Norm + ECA
"""


class RecordOI(nn.Module):
    def __init__(self, config, in_channels=8, norm='layer', n_class=3):
        """
        RECurrent Online object detectOR (RECORD) model class for online inference
        *** 修改版 (Hybrid Norm + ECA) ***

        @param config: configuration file of the model
        @param in_channels: number of input channels (default: 8)
        @param norm: type of normalisation for recurrent parts (default: 'layer')
        @param n_class: number of classes (default: 3)
        """
        super(RecordOI, self).__init__()

        # Hybrid Norm: Stem 用 BN，Recurrent 用 LayerNorm/GN(1)
        self.encoder = RecordEncoder(
            config=config['encoder_config'],
            in_channels=in_channels,
            norm_stem='bn',  # Stem 部分使用 BatchNorm
            norm_recurrent=norm  # Recurrent 部分使用 LayerNorm
        )

        # Decoder 使用 LayerNorm (因为只在最后一个时间步运行)
        self.decoder = RecordDecoder(
            config=config['decoder_config'],
            n_class=n_class,
            norm_decoder=norm  # 添加 norm_decoder 参数
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass RECORD-OI model for SINGLE TIMESTEP
        *** 修改为支持 Hybrid Norm 的版本 ***

        @param x: input tensor with shape (B, C, H, W) - SINGLE timestep
        @return: ConfMap prediction with shape (B, n_classes, H, W)
        """
        # x shape: (B, C, H, W) - 单个时间步

        # 1. Stem part (BN) - 直接处理
        stem_features = self.encoder.forward_stem(x)

        # 2. Recurrent part (GN/LayerNorm) - 单步处理
        # 获取当前的隐藏状态
        h_list = self.encoder.h_list
        c_list = self.encoder.c_list

        # 执行单步 recurrent forward
        (st_features_backbone,
         st_features_lstm2,
         st_features_lstm1), new_h_list, new_c_list = self.encoder.forward_recurrent_step(
            stem_features, h_list, c_list
        )

        # 更新隐藏状态
        self.encoder.h_list = new_h_list
        self.encoder.c_list = new_c_list

        # 3. Decoder (使用最后一个时间步的特征)
        confmap_pred = self.decoder(st_features_backbone, st_features_lstm2, st_features_lstm1)

        return self.sigmoid(confmap_pred)