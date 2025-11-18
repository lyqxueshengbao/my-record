import torch
from torch import nn
from .record import RecordEncoder, RecordDecoder


class RecordOI(nn.Module):
    def __init__(self, config, in_channels=8, norm='layer', n_class=3):
        super(RecordOI, self).__init__()

        # 1. Encoder 配置 (保持 bn 以加载权重)
        self.encoder = RecordEncoder(config=config['encoder_config'],
                                     in_channels=in_channels,
                                     norm_stem='bn',
                                     norm_recurrent=norm)

        # 2. Decoder 配置
        self.decoder = RecordDecoder(config=config['decoder_config'],
                                     n_class=n_class,
                                     norm_decoder=norm)

        self.sigmoid = nn.Sigmoid()

        # 3. 初始化状态 (修复 Bug)
        self.encoder.__init_hidden__()

    def forward(self, x):
        """
        Forward pass for Online Inference (Single Frame)
        x shape: (B, C, H, W)
        """
        # === 关键修改开始 ===
        # 不再调用 self.encoder(x)，而是分步调用以适配 Mixed Norm Encoder

        # 1. 执行 Stem 部分 (BN)
        # x: (B, C, H, W)
        stem_features = self.encoder.forward_stem(x)

        # 2. 执行 Recurrent 部分 (GN + LSTM)
        # 使用 encoder 内部保存的 h_list 和 c_list
        # 如果是刚开始 (None)，则初始化
        if self.encoder.h_list[0] is None:
            self.encoder.__init_hidden__()

        (st_features_backbone,
         st_features_lstm2,
         st_features_lstm1), h_list, c_list = self.encoder.forward_recurrent_step(
            stem_features,
            self.encoder.h_list,
            self.encoder.c_list
        )

        # 3. 更新状态
        # 注意：这里将更新后的状态存回 self.encoder，供下一帧使用
        self.encoder.h_list = h_list
        self.encoder.c_list = c_list
        # === 关键修改结束 ===

        # 4. Decoder
        confmap_pred = self.decoder(st_features_backbone, st_features_lstm2, st_features_lstm1)
        return self.sigmoid(confmap_pred)

    def reset_hidden(self):
        self.encoder.__init_hidden__()