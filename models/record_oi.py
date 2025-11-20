from torch import nn
from .record import RecordEncoder, RecordDecoder
from .layers.inverted_residual import Conv3x3ReLUNorm, InvertedResidual
from .layers.bottleneck_lstm import BottleneckLSTM
from utils.models_utils import _make_divisible

"""
Entrainement en initialisant les états sur les k premières frames puis prédire les suivantes

Online learning LSTM biblio 

"""

import torch
import torch.nn as nn


# 假设 RecordEncoder, RecordDecoder 在同级目录或正确路径下
# from .record import RecordEncoder, RecordDecoder

class RecordOI(nn.Module):
    def __init__(self, config, in_channels=8, norm='layer', n_class=3):
        super(RecordOI, self).__init__()
        # ... (你的初始化代码保持不变) ...
        self.encoder = RecordEncoder(config=config['encoder_config'],
                                     in_channels=in_channels,
                                     norm_stem='bn',
                                     norm_recurrent=norm)
        self.decoder = RecordDecoder(config=config['decoder_config'],
                                     n_class=n_class,
                                     norm_decoder=norm)
        self.sigmoid = nn.Sigmoid()

        # 内部状态
        self.h_list = None
        self.c_list = None

    def reset_memory(self):
        """
        彻底重置记忆，切断计算图
        """
        self.h_list = None
        self.c_list = None
        # 同时调用 encoder 的重置
        if hasattr(self.encoder, '__init_hidden__'):
            self.encoder.__init_hidden__()

    def forward(self, x):
        B, C, T, H, W = x.shape

        # 1. Stem
        x_reshaped = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        stem_features = self.encoder.forward_stem(x_reshaped)
        _, C_feat, H_feat, W_feat = stem_features.shape
        recurrent_input = stem_features.view(B, T, C_feat, H_feat, W_feat)

        # 2. 状态管理 (关键修复)
        # 如果 h_list 为 None，或者 Batch Size 变了，必须重置
        if self.h_list is None or (self.h_list[0] is not None and self.h_list[0].size(0) != B):
            self.encoder.__init_hidden__()
            self.h_list = self.encoder.h_list
            self.c_list = self.encoder.c_list

        # 3. 循环
        st_features_backbone = None
        st_features_lstm2 = None
        st_features_lstm1 = None

        # 必须使用局部变量传递状态，最后再写回 self
        curr_h_list = self.h_list
        curr_c_list = self.c_list

        for t in range(T):
            x_t = recurrent_input[:, t, ...]
            (st_features_backbone,
             st_features_lstm2,
             st_features_lstm1), curr_h_list, curr_c_list = self.encoder.forward_recurrent_step(
                x_t, curr_h_list, curr_c_list
            )

        # 4. 将更新后的状态写回 self (用于下一次 Inference)
        # 但在 Training 时，下一次 forward 前会被 reset_memory 清除
        self.h_list = curr_h_list
        self.c_list = curr_c_list

        confmap_pred = self.decoder(st_features_backbone, st_features_lstm2, st_features_lstm1)
        return self.sigmoid(confmap_pred)
