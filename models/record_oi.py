from torch import nn
from .record import RecordEncoder, RecordDecoder
from .layers.inverted_residual import Conv3x3ReLUNorm, InvertedResidual
from .layers.bottleneck_lstm import BottleneckLSTM
from utils.models_utils import _make_divisible

"""
Online Inference 版本 - 使用 Hybrid Norm + ECA
每次只处理单帧,保持LSTM状态
"""


class RecordOI(nn.Module):
    def __init__(self, config, in_channels=8, norm='layer', n_class=3):
        """
        RECurrent Online object detectOR (RECORD) model class for online inference
        *** 修改版: Hybrid Norm + ECA ***

        @param config: configuration file of the model
        @param in_channels: number of input channels (default: 8)
        @param norm: type of normalisation for recurrent/decoder parts (default: 'layer')
        @param n_class: number of classes (default: 3)
        """
        super(RecordOI, self).__init__()

        # Encoder: Hybrid Norm (BN for stem, GN for recurrent)
        # 注意：参数顺序和名称必须匹配RecordEncoder的定义
        self.encoder = RecordEncoder(
            in_channels=in_channels,
            config=config['encoder_config'],
            norm_stem='bn',  # Stem部分使用BatchNorm
            norm_recurrent=norm  # Recurrent部分使用LayerNorm/GroupNorm
        )

        # Decoder: 使用LayerNorm/GroupNorm + ECA
        self.decoder = RecordDecoder(
            config=config['decoder_config'],
            n_class=n_class,
            norm_decoder=norm
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass RECORD-OI model (single timestep)
        *** 修改版: 支持Hybrid Norm架构 ***

        @param x: input tensor with shape (B, C, H, W) - 单个时间步
        @return: ConfMap prediction with shape (B, n_classes, H, W)
        """
        B, C, H, W = x.shape
        assert len(x.shape) == 4, "Online模式输入应该是4D: (B, C, H, W)"

        # 1. Stem forward (BatchNorm part)
        stem_features = self.encoder.forward_stem(x)

        # 2. Recurrent forward (GroupNorm/LayerNorm part)
        # 获取当前hidden states
        h_list = self.encoder.h_list
        c_list = self.encoder.c_list

        # 单步recurrent forward
        (st_features_backbone,
         st_features_lstm2,
         st_features_lstm1), new_h_list, new_c_list = self.encoder.forward_recurrent_step(
            stem_features, h_list, c_list
        )

        # 更新encoder的hidden states
        self.encoder.h_list = new_h_list
        self.encoder.c_list = new_c_list

        # 3. Decoder forward (使用ECA注意力机制)
        confmap_pred = self.decoder(st_features_backbone, st_features_lstm2, st_features_lstm1)

        return self.sigmoid(confmap_pred)
