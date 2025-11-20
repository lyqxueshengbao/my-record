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

        # 1. Stem (保持不变)
        x_reshaped = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        stem_features = self.encoder.forward_stem(x_reshaped)
        _, C_feat, H_feat, W_feat = stem_features.shape
        recurrent_input = stem_features.view(B, T, C_feat, H_feat, W_feat)

        # 2. 状态初始化 (保持不变)
        if self.h_list is None or (self.h_list[0] is not None and self.h_list[0].size(0) != B):
            self.encoder.__init_hidden__()
            self.h_list = self.encoder.h_list
            self.c_list = self.encoder.c_list

        # 3. 循环 (关键修改！)
        curr_h_list = self.h_list
        curr_c_list = self.c_list

        # 用于收集每一帧的预测结果
        confmap_preds = []

        for t in range(T):
            x_t = recurrent_input[:, t, ...]

            # 更新 LSTM 状态
            (st_features_backbone,
             st_features_lstm2,
             st_features_lstm1), curr_h_list, curr_c_list = self.encoder.forward_recurrent_step(
                x_t, curr_h_list, curr_c_list
            )

            # [新增] 每一帧都立即解码！
            # 这样我们可以监控每一帧的质量
            pred_t = self.decoder(st_features_backbone, st_features_lstm2, st_features_lstm1)
            pred_t = self.sigmoid(pred_t)

            # pred_t shape: (B, n_class, H, W) -> unsqueeze -> (B, n_class, 1, H, W)
            confmap_preds.append(pred_t.unsqueeze(2))

        # 更新内部状态
        self.h_list = curr_h_list
        self.c_list = curr_c_list

        # [修改返回值]
        # 将列表拼接成 5D 张量: (B, n_class, T, H, W)
        # 这样训练时可以计算整个序列的 Loss
        full_pred = torch.cat(confmap_preds, dim=2)

        # 为了兼容现有的评估代码 (它们可能只想要最后一帧)，我们做个判断
        # 或者让 Trainer 去处理。
        # 建议：训练时用 full_pred，推理时其实这一步比较慢（多了解码过程），但为了微调必须这么做。
        # 这里的修改对 Inference 速度有轻微影响，但在可接受范围内。
        return full_pred
