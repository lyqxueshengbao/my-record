#
# 已修改： models/record_oi.py
#
import torch
from torch import nn
from .record import RecordEncoder, RecordDecoder
from .layers.inverted_residual import Conv3x3ReLUNorm, InvertedResidual
from .layers.bottleneck_lstm import BottleneckLSTM
from utils.models_utils import _make_divisible

"""
Entrainement en initialisant les états sur les k premières frames puis prédire les suivantes

Online learning LSTM biblio 

"""


class RecordOI(nn.Module):
    def __init__(self, config, in_channels=8, norm='layer', n_class=3):
        """
        RECurrent Online object detectOR (RECORD) model class for online inference
        """
        super(RecordOI, self).__init__()

        # 1. Encoder 设置
        # 注意：如果你想复用 Buffer 模式训练的 Mixed Norm 权重（前半部分 BN），
        # 这里建议把 norm_stem 设为 'bn'。
        # norm_recurrent 使用传入的 norm (通常是 'layer')
        self.encoder = RecordEncoder(config=config['encoder_config'],
                                     in_channels=in_channels,
                                     norm_stem='bn',  # 如果你的权重是 BN 训练的，这里必须是 'bn'
                                     norm_recurrent=norm)

        # 2. Decoder 设置 (补全了这里缺少的代码)
        self.decoder = RecordDecoder(config=config['decoder_config'],
                                     n_class=n_class,
                                     norm_decoder=norm)

        self.sigmoid = nn.Sigmoid()

        # 3. 修复作者遗漏的初始化 Bug (只在初始化时调用一次)
        self.encoder.__init_hidden__()

    def forward(self, x):
        # 3. 不要在这里加循环，也不要加 __init_hidden__
        # 外部喂进来的 x 应该是 (B, C, H, W) 的单帧
        st_features_lstm1, st_features_lstm2, st_features_backbone = self.encoder(x)
        confmap_pred = self.decoder(st_features_lstm1, st_features_lstm2, st_features_backbone)
        return self.sigmoid(confmap_pred)

    def reset_hidden(self):
        # 提供一个手动重置接口，换视频的时候调用
        self.encoder.__init_hidden__()