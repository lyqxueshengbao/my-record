from torch import nn
# 确保 record.py 和 record_oi.py 在同一个 models 包下
from .record import RecordEncoder, RecordDecoder


class RecordOI(nn.Module):
    def __init__(self, config, in_channels=8, norm='layer', n_class=3):
        """
        RECurrent Online object detectOR (RECORD) model class for online inference
        适配 Hybrid Norm: Stem 使用 BN, Recurrent/Decoder 使用 LayerNorm/GN
        """
        super(RecordOI, self).__init__()

        # 1. 修改 Encoder 初始化
        # 注意：record.py 中 RecordEncoder 的参数顺序变了，且增加了 norm_stem/norm_recurrent
        # 我们强制 stem 用 'bn' (与训练保持一致), recurrent 部分用传入的 norm (通常是 'layer')
        self.encoder = RecordEncoder(
            in_channels=in_channels,
            config=config['encoder_config'],
            norm_stem='bn',  # 关键：保持 Stem 为 BN
            norm_recurrent=norm  # Recurrent 部分跟随配置
        )

        # 2. 修改 Decoder 初始化
        # 传入 norm_decoder 参数，确保 Decoder 里的 Skip Connection 也能用到正确的归一化
        self.decoder = RecordDecoder(
            config=config['decoder_config'],
            n_class=n_class,
            norm_decoder=norm
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass RECORD-OI model
        @param x: input tensor with shape (B, C, H, W) (单帧)
        """
        # 兼容性处理：如果是 (B, C, T, H, W) 且 T=1，则压缩维度
        if x.dim() == 5:
            x = x.squeeze(2)

        # 1. 运行 Stem (BN 部分)
        # 在 eval 模式下，BN 使用训练好的 running stats，单帧输入也没问题
        stem_features = self.encoder.forward_stem(x)

        # 2. 获取当前的 LSTM 状态
        h_list = self.encoder.h_list
        c_list = self.encoder.c_list

        # 3. 运行 Recurrent Step (LayerNorm 部分)
        # 这会计算这一帧的特征，并返回新的状态
        (st_features_backbone, st_features_lstm2, st_features_lstm1), new_h_list, new_c_list = \
            self.encoder.forward_recurrent_step(stem_features, h_list, c_list)

        # 4. 更新状态以便下一帧使用
        self.encoder.h_list = new_h_list
        self.encoder.c_list = new_c_list

        # 5. Decoder (使用 LayerNorm)
        confmap_pred = self.decoder(st_features_backbone, st_features_lstm2, st_features_lstm1)

        return self.sigmoid(confmap_pred)