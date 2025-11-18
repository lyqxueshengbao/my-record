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
        super(RecordOI, self).__init__()
        # 1. 这里保持 'bn' 以匹配你 Buffer 模式训练好的权重
        self.encoder = RecordEncoder(config=config['encoder_config'],
                                     in_channels=in_channels,
                                     norm_stem='bn',  # 关键点
                                     norm_recurrent=norm)

        self.decoder = RecordDecoder(...)
        self.sigmoid = nn.Sigmoid()

        # 2. 修复作者的 Bug：只在初始化时重置一次，而不是每次推理都重置
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