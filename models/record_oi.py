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
        MODIFIED FOR HYBRID NORM
        @param config: configuration file of the model
        @param alpha: expansion factor to modify the size of the model (default: 1.0)
        @param in_channels: number of input channels (default: 8)
        @param norm: type of normalisation FOR RECURRENT/DECODER parts (default: 'layer' for GN(1)).
                     The STEM part (pre-LSTM) is hardcoded to 'bn' (BatchNorm2d).
        @param n_class: number of classes (default: 3)
        @param shallow: load a shallow version of RECORD (fewer channels in the decoder)
        """
        super(RecordOI, self).__init__()

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
        # 初始化隐藏状态（修复原作者遗漏的 bug）
        self.encoder.__init_hidden__()

    def forward(self, x):
        """
        Forward pass RECORD-OI model (MODIFIED FOR HYBRID NORM)
        @param x: input tensor with shape (B, C, T, H, W) where T is the number of timesteps
        @return: ConfMap prediction of the last time step with shape (B, n_classes, H, W)
        """
        B, C, T, H, W = x.shape
        assert len(x.shape) == 5

        # 1. Reshape for Stem (BN part)
        # (B, C, T, H, W) -> (B, T, C, H, W) -> (B*T, C, H, W)
        x_reshaped = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)

        # 2. Run Stem (BN part)
        # stem_features shape: (B*T, C_feat, H_feat, W_feat)
        stem_features = self.encoder.forward_stem(x_reshaped)

        # 3. Reshape for Recurrent (GN/LayerNorm part)
        # (B*T, C_feat, H_feat, W_feat) -> (B, T, C_feat, H_feat, W_feat)
        _, C_feat, H_feat, W_feat = stem_features.shape
        recurrent_input = stem_features.view(B, T, C_feat, H_feat, W_feat)

        # 4. Initialize hidden states
        self.encoder.__init_hidden__()
        h_list = self.encoder.h_list
        c_list = self.encoder.c_list

        # 5. Loop over time (Recurrent part)
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

    def reset_hidden(self):
        """
        重置隐藏状态
        用于：
        1. 开始处理新的视频序列
        2. 处理独立的、不相关的帧
        """
        self.encoder.__init_hidden__()